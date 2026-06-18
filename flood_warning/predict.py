"""Live inference: load each gauge's CNN checkpoint and produce a 12-hour-ahead
forecast using the most recent USGS observations + Open-Meteo recent and
forecast hourly data.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .dataset import PAST_STEPS, FUTURE_STEPS, Scaler
from .model import FloodCNN
from .sites import SITES, BY_ID
from .fetch import _http_get, USGS_API_KEY, CFS_TO_M3S

REPO = Path(__file__).resolve().parents[1]
CKPT_DIR = Path(__file__).resolve().parent / 'checkpoints'
OUT_PATH = REPO / 'outputs' / 'dmv-cnn-12h' / 'preds.json'


def fetch_recent_usgs(gauge_id: str, hours: int = 30) -> pd.Series:
    """Last N hours of USGS observed flow (15-min cadence -> hourly mean, m³/s)."""
    end = pd.Timestamp.utcnow().tz_localize(None)
    start = end - pd.Timedelta(hours=hours + 6)
    url = ('https://waterservices.usgs.gov/nwis/iv/?sites='
           + gauge_id + '&parameterCd=00060'
           + f'&startDT={start.strftime("%Y-%m-%dT%H:%MZ")}'
           + f'&endDT={end.strftime("%Y-%m-%dT%H:%MZ")}'
           + '&format=json')
    headers = {'X-Api-Key': USGS_API_KEY} if USGS_API_KEY else {}
    payload = json.loads(_http_get(url, headers))
    rows = []
    for ts in payload.get('value', {}).get('timeSeries', []):
        for v in ts.get('values', [{}])[0].get('value', []):
            try:
                cfs = float(v['value'])
                if cfs < 0:
                    continue
                rows.append((v['dateTime'], cfs * CFS_TO_M3S))
            except (ValueError, KeyError, TypeError):
                continue
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows, columns=['t', 'flow_m3s'])
    df['t'] = pd.to_datetime(df['t'], utc=True).dt.tz_convert(None)
    return df.set_index('t').sort_index()['flow_m3s'].resample('h').mean()


def fetch_openmeteo_window(lat: float, lon: float) -> pd.DataFrame:
    """Hourly precip + temp covering last ~30h past + next 16h forecast.
    Open-Meteo Forecast API has `past_days` and `forecast_days` params.
    """
    url = ('https://api.open-meteo.com/v1/forecast?'
           f'latitude={lat:.4f}&longitude={lon:.4f}'
           f'&hourly=precipitation,temperature_2m,surface_pressure'
           f'&past_days=2&forecast_days=2&timezone=GMT')
    payload = json.loads(_http_get(url))
    df = pd.DataFrame(payload['hourly']).rename(columns={'time': 't'})
    df['t'] = pd.to_datetime(df['t'])
    return df.set_index('t').sort_index()


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
    df_live = fetch_hourly_live(gauge_id, days_back=2)
    site = BY_ID[gauge_id]

    flow_valid = df_live['flow_m3s'].dropna()
    if len(flow_valid) < PAST_STEPS:
        print(f'  {gauge_id}: not enough recent USGS hours ({len(flow_valid)})')
        return None
    issue_time = flow_valid.index[-1]   # last hour with a real observation

    past_idx = pd.date_range(end=issue_time, periods=PAST_STEPS, freq='h')
    fut_idx = pd.date_range(start=issue_time + pd.Timedelta(hours=1),
                            periods=FUTURE_STEPS, freq='h')

    flow_past = df_live['flow_m3s'].reindex(past_idx).interpolate(limit=4).values
    precip_past = df_live['precip_mm'].reindex(past_idx).fillna(0).values
    precip_fut = df_live['precip_mm'].reindex(fut_idx).fillna(0).values
    temp_past = df_live['temp_c'].reindex(past_idx).interpolate(limit=4).values
    sm_past = (df_live['sm_surface'].reindex(past_idx)
                .ffill().bfill().fillna(scaler.sm_mean).values)

    if np.isnan(flow_past).any() or np.isnan(temp_past).any():
        print(f'  {gauge_id}: missing values in input window')
        return None

    # Encode and run (4-channel past stream + 1-channel future precip)
    past_enc = np.stack([
        scaler.encode_flow(flow_past),
        scaler.encode_precip(precip_past),
        scaler.encode_temp(temp_past),
        scaler.encode_sm(sm_past),
    ], axis=0).astype(np.float32)
    fut_enc = scaler.encode_precip(precip_fut)[None, :].astype(np.float32)

    with torch.no_grad():
        pred = model(torch.from_numpy(past_enc[None]),
                     torch.from_numpy(fut_enc[None])).numpy()[0]
    pred_m3s = scaler.decode_flow(pred)

    # Single combined series matching the schema used by every other model's
    # preds.json — each entry has a date/time and may carry observed (o),
    # predicted (p), and/or precip values. The CNN's hourly nature is encoded
    # by using full ISO datetimes; the chart auto-detects.
    series = []
    for t, v, p in zip(past_idx, flow_past, precip_past):
        series.append({'d': t.strftime('%Y-%m-%dT%H:00Z'),
                       'o': round(float(v), 3),
                       'precip_mm': round(float(p), 2)})
    for t, v, p in zip(fut_idx, pred_m3s, precip_fut):
        series.append({'d': t.strftime('%Y-%m-%dT%H:00Z'),
                       'p': round(float(v), 3),
                       'precip_mm': round(float(p), 2)})

    # Historical backtest — rolling 1h-ahead predictions over the last 30 days
    # so the UI can show model-vs-actual performance for the recent past.
    backtest_series = backtest_gauge(gauge_id, model, scaler, cfg,
                                      days_back=30, issue_anchor=issue_time)

    return {
        'id': gauge_id,
        'issue_time': issue_time.strftime('%Y-%m-%dT%H:00Z'),
        'series': series,                  # 24h obs + 12h forecast (active window)
        'backtest': backtest_series,        # rolling 1h-ahead over last 7 days
        'metrics': ckpt['metrics'],
    }


def backtest_gauge(gauge_id: str, model, scaler, cfg, days_back: int = 30,
                   issue_anchor: pd.Timestamp | None = None) -> list[dict]:
    """For each hour in the last `days_back` days, generate the CNN's 12-hour
    forecast that *would* have been issued at that hour. Return a list of
    {t, obs, pred_1h, pred_6h, pred_12h} for plotting model trace vs reality.

    We use the stored parquet (USGS observed flow + Open-Meteo archive ERA5
    forcings) so this is a true historical replay using the same input shape
    the model was trained on.
    """
    from .fetch import DATA_DIR
    parquet = DATA_DIR / gauge_id / 'hourly.parquet'
    if parquet.exists():
        df = pd.read_parquet(parquet)
    else:
        # CI mode — no local cache. Fetch directly.
        from .fetch import fetch_hourly_live
        df = fetch_hourly_live(gauge_id, days_back=days_back + 1)
        if df['flow_m3s'].dropna().empty:
            return []
    end_t = df['flow_m3s'].dropna().index.max()
    if issue_anchor is not None:
        # Don't backtest past the live issue time (no point — that's the live forecast region)
        end_t = min(end_t, issue_anchor - pd.Timedelta(hours=1))
    start_t = end_t - pd.Timedelta(days=days_back)

    past_steps = cfg['past_steps']
    future_steps = cfg['future_steps']
    # Pre-resolve soil-moisture from ERA5 with forward/back fill so the
    # 5-day archive lag near "now" doesn't kill recent windows.
    if 'sm_surface' in df.columns:
        sm_series = df['sm_surface'].ffill().bfill()
    else:
        sm_series = pd.Series(scaler.sm_mean, index=df.index)
    times = pd.date_range(start=start_t, end=end_t, freq='h')
    past_batch = []
    fut_batch = []
    valid_t = []
    obs_at_t = []
    for cur in times:
        past_idx = pd.date_range(end=cur - pd.Timedelta(hours=1), periods=past_steps, freq='h')
        fut_idx = pd.date_range(start=cur, periods=future_steps, freq='h')
        flow_past = df['flow_m3s'].reindex(past_idx).values
        precip_past = df['precip_mm'].reindex(past_idx).fillna(0).values
        precip_fut = df['precip_mm'].reindex(fut_idx).fillna(0).values
        temp_past = df['temp_c'].reindex(past_idx).interpolate(limit=2).values
        sm_past = sm_series.reindex(past_idx).fillna(scaler.sm_mean).values
        if np.isnan(flow_past).any() or np.isnan(temp_past).any():
            continue
        past_enc = np.stack([
            scaler.encode_flow(flow_past),
            scaler.encode_precip(precip_past),
            scaler.encode_temp(temp_past),
            scaler.encode_sm(sm_past),
        ], axis=0).astype(np.float32)
        fut_enc = scaler.encode_precip(precip_fut)[None, :].astype(np.float32)
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


def run_all():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    preds = []
    print('Live CNN inference:')
    for site in SITES:
        ckpt = CKPT_DIR / f'{site["id"]}.pt'
        if not ckpt.exists():
            continue
        try:
            p = predict_gauge(site['id'])
        except Exception as e:
            print(f'  ! {site["id"]}: {e}')
            continue
        if p:
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

    OUT_PATH.write_text(json.dumps({
        'model_id': 'dmv-cnn-12h',
        'updated': pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'predictions': preds,
    }, separators=(',', ':')))
    print(f'\nwrote {OUT_PATH}')


if __name__ == '__main__':
    run_all()
