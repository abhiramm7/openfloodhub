"""Google Flood Forecasting overlays for the CNN forecast.

Like noaa.py, these series are *comparison overlays* — they ride alongside
the CNN's own forecast in the prediction output, they are not fed back into
the model.

Source: Google Flood Forecasting API (the engine behind Google Flood Hub,
g.co/floodhub). Docs: https://developers.google.com/flood-forecasting

  POST /v1/gauges:searchGaugesByArea            find Google gauges near our sites
  GET  /v1/gauges:queryGaugeForecasts           forecast time series per gauge
  GET  /v1/gaugeModels:batchGet                 model unit + warning/danger thresholds
  GET  /v1/floodStatus:queryLatestFloodStatusByGaugeIds   severity + trend

Access is free (CC BY 4.0) but requires an API key on a Google Cloud project
that has been granted access via Google's signup form (see the docs page).
Put the key in `.env.local` as GOOGLE_FLOOD_API_KEY=... locally, and as a
repo Actions secret of the same name for CI. Everything here degrades to a
no-op when the key is absent.

The Google gauge id for each USGS site is discovered once with

    .venv/bin/python -m flood_warning.google_flood

which searches a bounding box around the gauge set, matches Google gauges to
our sites by distance, and writes flood_warning/google_gauges.json (committed,
like thresholds.json). Google's US gauges are HYBAS virtual points at basin
outlets, not USGS locations — the committed mapping was hand-verified by
comparing Google's return-period thresholds and live forecast magnitudes
against each site's own scale, so review before overwriting it with the
auto-matcher's output. Forecast units are whatever the Google gauge model
uses — CUBIC_METERS_PER_SECOND aligns with the CNN chart; METERS (stage) is
shown as status only, never drawn on the discharge chart.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

from .fetch import _http_get   # retried HTTP + .env.local side-load
from .sites import SITES

API = 'https://floodforecasting.googleapis.com/v1'
MAPPING_PATH = Path(__file__).resolve().parent / 'google_gauges.json'

API_KEY = os.environ.get('GOOGLE_FLOOD_API_KEY')

# ~2 km in degrees latitude — a Google gauge further than this from the USGS
# coordinates is a different river point, not a match.
MATCH_DEG = 0.02


def _api(path: str, params: dict | None = None, body: dict | None = None) -> dict | None:
    """One API call, GET unless a body is given. Returns None on any failure —
    overlays are optional and must never sink the prediction run."""
    if not API_KEY:
        return None
    q = {'key': API_KEY, **(params or {})}
    url = f'{API}/{path}?' + urllib.parse.urlencode(q, doseq=True)
    try:
        if body is None:
            return json.loads(_http_get(url))
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={'User-Agent': 'openfloodhub-cnn',
                     'Content-Type': 'application/json'})
        return json.loads(urllib.request.urlopen(req, timeout=60).read())
    except Exception as e:
        print(f'   ! Google {path}: {e}')
        return None


# --------------------------------------------------------------------------
# One-time gauge discovery -> google_gauges.json
# --------------------------------------------------------------------------

def discover_gauges(pad: float = 0.3) -> dict[str, str]:
    """Search a bounding box around the gauge set and match Google gauges to
    our USGS sites — by id when Google embeds the USGS number, else by
    nearest-within-2km. Returns {usgs_id: google_gauge_id}."""
    lats = [s['lat'] for s in SITES]
    lons = [s['lon'] for s in SITES]
    lo_lat, hi_lat = min(lats) - pad, max(lats) + pad
    lo_lon, hi_lon = min(lons) - pad, max(lons) + pad
    box = [(lo_lat, lo_lon), (lo_lat, hi_lon), (hi_lat, hi_lon), (hi_lat, lo_lon)]
    resp = _api('gauges:searchGaugesByArea', body={
        'loop': {'vertices': [{'latitude': la, 'longitude': lo} for la, lo in box]},
        'includeNonQualityVerified': True,
        'includeGaugesWithoutHydroModel': False,
    })
    gauges = (resp or {}).get('gauges', [])
    print(f'Google gauges in box: {len(gauges)}')

    mapping = {}
    for site in SITES:
        best, best_d = None, MATCH_DEG
        for g in gauges:
            gid = g.get('gaugeId', '')
            loc = g.get('location', {})
            if site['id'] in gid:
                best, best_d = g, 0.0
                break
            d = max(abs(loc.get('latitude', 999) - site['lat']),
                    abs(loc.get('longitude', 999) - site['lon']))
            if d < best_d:
                best, best_d = g, d
        if best:
            mapping[site['id']] = best['gaugeId']
            print(f"  {site['id']} {site['short']:<16} -> {best['gaugeId']}"
                  f"  (Δ≈{best_d:.4f}°, source={best.get('source')},"
                  f" verified={best.get('qualityVerified')})")
        else:
            print(f"  {site['id']} {site['short']:<16} -> no Google gauge within ~2 km")
    return mapping


def load_mapping() -> dict[str, str]:
    if MAPPING_PATH.exists():
        return json.loads(MAPPING_PATH.read_text())
    return {}


# --------------------------------------------------------------------------
# Per-run enrichment (batched: 3 API calls for the whole gauge set)
# --------------------------------------------------------------------------

def enrich_sites(usgs_ids: list[str]) -> dict[str, dict]:
    """Overlays for every mapped gauge in three batched calls. Returns
    {usgs_id: {gauge_id, unit, thresholds, severity, trend, issued_time,
    forecast: [{t, v}]}} — only ids that produced data are present."""
    mapping = load_mapping()
    if not API_KEY or not mapping:
        return {}
    ids = {u: g for u, g in mapping.items() if u in usgs_ids}
    if not ids:
        return {}
    gids = sorted(set(ids.values()))

    fc_resp = _api('gauges:queryGaugeForecasts', params={'gaugeIds': gids}) or {}
    models = _api('gaugeModels:batchGet',
                  params={'names': [f'gaugeModels/{g}' for g in gids]}) or {}
    status = _api('floodStatus:queryLatestFloodStatusByGaugeIds',
                  params={'gaugeIds': gids}) or {}

    model_by_gid = {m.get('gaugeId'): m for m in models.get('gaugeModels', [])}
    status_by_gid = {s.get('gaugeId'): s for s in status.get('floodStatuses', [])}
    forecasts = fc_resp.get('forecasts', {})

    out = {}
    for usgs_id, gid in ids.items():
        model = model_by_gid.get(gid, {})
        stat = status_by_gid.get(gid, {})
        entry = {
            'gauge_id': gid,
            'unit': model.get('gaugeValueUnit'),
            'quality_verified': model.get('qualityVerified'),
            'severity': stat.get('severity'),
            'trend': stat.get('forecastTrend'),
        }
        th = model.get('thresholds') or {}
        if th:
            entry['thresholds'] = {
                'warning': th.get('warningLevel'),
                'danger': th.get('dangerLevel'),
                'extreme': th.get('extremeDangerLevel'),
            }
        # Latest issued forecast only; each range becomes one point at its
        # start time (start == end for instantaneous values).
        fset = (forecasts.get(gid) or {}).get('forecasts', [])
        if fset:
            latest = max(fset, key=lambda f: f.get('issuedTime', ''))
            entry['issued_time'] = latest.get('issuedTime')
            series = []
            for r in latest.get('forecastRanges', []):
                t, v = r.get('forecastStartTime'), r.get('value')
                if t is not None and v is not None:
                    series.append({'t': t, 'v': round(float(v), 3)})
            series.sort(key=lambda r: r['t'])
            entry['forecast'] = series
        if entry.get('forecast') or entry.get('severity'):
            out[usgs_id] = entry
    return out


if __name__ == '__main__':
    if not API_KEY:
        raise SystemExit('GOOGLE_FLOOD_API_KEY not set — add it to .env.local '
                         '(see module docstring for how to get one)')
    mapping = discover_gauges()
    if mapping:
        MAPPING_PATH.write_text(json.dumps(mapping, indent=2) + '\n')
        print(f'wrote {MAPPING_PATH}')
    else:
        print('no matches found — nothing written')
