"""Live inference: load each gauge's CNN checkpoint and produce a 12-hour-ahead
forecast using the most recent USGS observations + Open-Meteo recent and
forecast hourly data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .dataset import PAST_STEPS, FUTURE_STEPS, Scaler, encode_window
from .model import FloodCNN
from .sites import BY_ID, DEFAULT_REGION, REGIONS, sites_in

REPO = Path(__file__).resolve().parents[1]
CKPT_DIR = Path(__file__).resolve().parent / 'checkpoints'


def out_path(region: str) -> Path:
    """outputs/<model_id>/preds.json — one file per region."""
    return REPO / 'outputs' / REGIONS[region]['model_id'] / 'preds.json'


def predict_gauge(gauge_id: str) -> dict | None:
    ckpt_path = CKPT_DIR / f'{gauge_id}.pt'
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, weights_only=False)
    scaler = Scaler.from_dict(ckpt['scaler'])
    cfg = ckpt['config']
    model = FloodCNN(hidden=cfg['hidden'],
                     past_steps=cfg['past_steps'],
                     future_steps=cfg['future_steps'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    from .fetch import fetch_hourly_live
    # One fetch serves both the live window and the 7-day backtest below.
    # Reaching back 10 days also clears ERA5-Land's ~5-day lag, so the
    # soil-moisture channel carries real antecedent values instead of an
    # all-NaN tail that collapses to the training mean.
    df_live = fetch_hourly_live(gauge_id, days_back=10)

    flow_valid = df_live['flow_m3s'].dropna()
    if len(flow_valid) < PAST_STEPS:
        print(f'  {gauge_id}: not enough recent USGS hours ({len(flow_valid)})')
        return None
    issue_time = flow_valid.index[-1]   # last hour with a real observation

    past_idx = pd.date_range(end=issue_time, periods=PAST_STEPS, freq='h')
    fut_idx = pd.date_range(start=issue_time + pd.Timedelta(hours=1),
                            periods=FUTURE_STEPS, freq='h')

    # Fetch already bridged small interior flow gaps (limit=4); no second
    # fill pass here, or an 8-hour dropout would sneak past the NaN guard
    # inside encode_window.
    win = encode_window(df_live, past_idx, fut_idx, scaler)
    if win is None:
        print(f'  {gauge_id}: missing values in input window')
        return None
    past_enc, fut_enc, raw = win

    with torch.no_grad():
        pred = model(torch.from_numpy(past_enc[None]),
                     torch.from_numpy(fut_enc[None])).numpy()[0]
    pred_m3s = scaler.decode_flow(pred)

    # Single combined series matching the schema used by every other model's
    # preds.json — each entry has a date/time and may carry observed (o),
    # predicted (p), and/or precip values. The CNN's hourly nature is encoded
    # by using full ISO datetimes; the chart auto-detects.
    series = []
    for t, v, p in zip(past_idx, raw['flow'], raw['precip_past']):
        series.append({'d': t.strftime('%Y-%m-%dT%H:00Z'),
                       'o': round(float(v), 3),
                       'precip_mm': round(float(p), 2)})
    for t, v, p in zip(fut_idx, pred_m3s, raw['precip_fut']):
        series.append({'d': t.strftime('%Y-%m-%dT%H:00Z'),
                       'p': round(float(v), 3),
                       'precip_mm': round(float(p), 2)})

    # Historical backtest — rolling predictions over the last 7 days (the
    # window the UI renders), replayed on the frame fetched above.
    backtest_series = backtest_gauge(model, scaler, cfg, df_live,
                                     days_back=7, issue_anchor=issue_time)

    return {
        'id': gauge_id,
        'issue_time': issue_time.strftime('%Y-%m-%dT%H:00Z'),
        'series': series,                  # 24h obs + 12h forecast (active window)
        'backtest': backtest_series,        # rolling 1h/6h/12h-ahead over last 7 days
        'metrics': ckpt['metrics'],
    }


def backtest_gauge(model, scaler, cfg, df: pd.DataFrame, days_back: int = 7,
                   issue_anchor: pd.Timestamp | None = None) -> list[dict]:
    """For each hour in the last `days_back` days, generate the CNN's 12-hour
    forecast that *would* have been issued at that hour. Return a list of
    {t, o, p1, p6, p12} rows for plotting model trace vs reality.

    Replays over the same frame predict_gauge fetched (one fetch per gauge
    per run), through the same encode_window path as the live forecast, so
    the published skill is scored on the pipeline the live path actually uses.
    """
    flow_valid = df['flow_m3s'].dropna()
    if flow_valid.empty:
        return []
    end_t = flow_valid.index.max()
    if issue_anchor is not None:
        # Don't backtest past the live issue time (no point — that's the live forecast region)
        end_t = min(end_t, issue_anchor - pd.Timedelta(hours=1))
    start_t = end_t - pd.Timedelta(days=days_back)

    past_steps = cfg['past_steps']
    future_steps = cfg['future_steps']
    times = pd.date_range(start=start_t, end=end_t, freq='h')
    past_batch = []
    fut_batch = []
    valid_t = []
    obs_at_t = []
    for cur in times:
        past_idx = pd.date_range(end=cur - pd.Timedelta(hours=1), periods=past_steps, freq='h')
        fut_idx = pd.date_range(start=cur, periods=future_steps, freq='h')
        win = encode_window(df, past_idx, fut_idx, scaler)
        if win is None:
            continue
        past_enc, fut_enc, _ = win
        past_batch.append(past_enc)
        fut_batch.append(fut_enc)
        valid_t.append(cur)
        obs_now = df['flow_m3s'].get(cur, float('nan'))
        obs_at_t.append(obs_now)

    if not past_batch:
        return []
    pred_batch = []
    with torch.no_grad():
        # Process in chunks to keep memory bounded
        chunk = 256
        for i in range(0, len(past_batch), chunk):
            past = torch.from_numpy(np.stack(past_batch[i:i + chunk]))
            fut = torch.from_numpy(np.stack(fut_batch[i:i + chunk]))
            pred_batch.append(model(past, fut).numpy())
    preds = np.concatenate(pred_batch)
    preds_m3s = scaler.decode_flow(preds)

    out = []
    for cur, obs, p in zip(valid_t, obs_at_t, preds_m3s):
        out.append({
            't': cur.strftime('%Y-%m-%dT%H:00Z'),
            'o': round(float(obs), 3) if np.isfinite(obs) else None,
            'p1': round(float(max(0, p[0])), 3),     # 1h-ahead nowcast
            'p6': round(float(max(0, p[5])), 3),     # 6h-ahead
            'p12': round(float(max(0, p[11])), 3),   # 12h-ahead
        })
    return out


def run_all(region: str = DEFAULT_REGION):
    sites = [s for s in sites_in(region)
             if (CKPT_DIR / f'{s["id"]}.pt').exists()]
    if not sites:
        # Region registered but not trained yet: write nothing, so the map
        # keeps whatever it had instead of switching to an empty region.
        print(f'No trained checkpoints for region {region!r} — nothing to '
              f'predict (run `python -m flood_warning.train {region}`)')
        return
    out = out_path(region)
    out.parent.mkdir(parents=True, exist_ok=True)
    from . import thresholds
    thresh = thresholds.load()
    if not thresh:
        print('  (no thresholds.json — run `python -m flood_warning.thresholds`)')
    preds = []
    print(f'Live CNN inference ({REGIONS[region]["name"]}):')
    for site in sites:
        try:
            p = predict_gauge(site['id'])
        except Exception as e:
            print(f'  ! {site["id"]}: {e}')
            p = None
        if p is None:
            # Keep the gauge on the map as an explicit "offline" marker — a
            # silently missing site looks like a UI bug (Rock Creek's USGS
            # feed has been dark since before launch and nobody could tell).
            p = {'id': site['id'], 'status': 'offline',
                 'series': [], 'backtest': []}
        # Site metadata so the UI can place + label markers without a
        # second file.
        p['name'] = site['name']
        p['short'] = site['short']
        p['lat'] = site['lat']
        p['lon'] = site['lon']
        p['drainage_sqmi'] = site['drainage_sqmi']
        p['kind'] = site['kind']
        if site['id'] in thresh:
            p['thresholds'] = thresh[site['id']]
        preds.append(p)
        forecasts = [e for e in p['series'] if 'p' in e]
        if forecasts:
            f1, f12 = forecasts[0], forecasts[-1]
            print(f'  {site["id"]} {site["short"]:<16}  +1h={f1["p"]:6.2f}  '
                  f'+12h={f12["p"]:6.2f} m³/s  (issued {p["issue_time"]})')

    # NOAA / NWS comparison overlays — fetched once per gauge and bundled into
    # each prediction record so the UI can draw them alongside the CNN forecast.
    # These are NOT fed back into the model; they are reference series:
    #   noaa_nwm           NWM short-range streamflow  (next ~18h, m³/s)
    #   noaa_nwm_medium    NWM medium-range blend      (next ~10d, m³/s)
    #   noaa_nwm_analysis  NWM analysis "observed"     (recent, m³/s)
    #   noaa_qpf           NWS forecast rainfall       (hourly, mm)
    #   noaa_mrms_precip   MRMS observed rainfall      (daily, mm)
    print('NOAA / NWS overlays:')
    from . import noaa
    mrms_end = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=1)
    mrms_start = mrms_end - pd.Timedelta(days=7)
    for p in preds:
        site = BY_ID[p['id']]
        n = noaa.enrich_site(
            p['id'], lat=site['lat'], lon=site['lon'],
            mrms_start=mrms_start.strftime('%Y-%m-%d'),
            mrms_end=mrms_end.strftime('%Y-%m-%d'),
        )
        if n['nwm_short']:
            p['noaa_nwm'] = n['nwm_short']
        if n['nwm_medium']:
            p['noaa_nwm_medium'] = n['nwm_medium']
        if n['nwm_analysis']:
            p['noaa_nwm_analysis'] = n['nwm_analysis']
        if n['qpf']:
            p['noaa_qpf'] = n['qpf']
        if n['mrms_precip']:
            p['noaa_mrms_precip'] = n['mrms_precip']
        print(f'  {p["id"]} {site["short"]:<16}  '
              f'NWM short={len(n["nwm_short"])} medium={len(n["nwm_medium"])} '
              f'analysis={len(n["nwm_analysis"])} | QPF={len(n["qpf"])}h '
              f'MRMS={len(n["mrms_precip"])}d')

    # Google Flood Hub overlays — same rule as NOAA: display-only reference
    # data, never fed back into the model. No-op without GOOGLE_FLOOD_API_KEY
    # and the committed gauge mapping (see google_flood.py docstring).
    from . import google_flood
    goog = google_flood.enrich_sites([p['id'] for p in preds])
    if goog:
        print('Google Flood Hub overlays:')
        for p in preds:
            if p['id'] in goog:
                g = goog[p['id']]
                p['google_flood'] = g
                print(f'  {p["id"]} {BY_ID[p["id"]]["short"]:<16}  '
                      f'severity={g.get("severity")} trend={g.get("trend")} '
                      f'forecast={len(g.get("forecast", []))}pts unit={g.get("unit")}')
    elif google_flood.API_KEY:
        print('Google Flood Hub overlays: no mapped gauges '
              f'(run `python -m flood_warning.google_flood {region}` once to discover)')

    reg = REGIONS[region]
    out.write_text(json.dumps({
        'model_id': reg['model_id'],
        'region': {'id': region, 'name': reg['name'], 'marquee': reg['marquee']},
        'updated': pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'predictions': preds,
    }, separators=(',', ':')))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    import sys
    run_all(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REGION)
