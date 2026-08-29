"""NOAA / NWS data overlays for the CNN forecast.

These series are *comparison overlays* — they ride alongside the CNN's own
forecast in the prediction output, they are not fed back into the model.

Sources, all unauthenticated JSON:

  NWPS  https://api.water.noaa.gov/nwps/v1/docs/
    /gauges/{usgsId}                 gauge metadata, official flood
                                     thresholds, current obs + NWS forecast
    /reaches/{reachId}/streamflow    NOAA National Water Model streamflow:
                                       analysisAssimilation  best-estimate "observed"
                                       shortRange            next ~18h hourly
                                       mediumRangeBlend      next ~10d hourly

  NWS   https://api.weather.gov/
    /points/{lat},{lon} -> /gridpoints/...   quantitativePrecipitation (QPF),
                                             forecast rainfall in mm

  IEM   https://mesonet.agron.iastate.edu/
    /iemre/multiday/...              MRMS radar QPE, observed daily rainfall

Flow is converted ft³/s -> m³/s and precip inches -> mm so everything lines
up with the CNN's units on the chart.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

import pandas as pd

NWPS = 'https://api.water.noaa.gov/nwps/v1'
NWS = 'https://api.weather.gov'
IEM = 'https://mesonet.agron.iastate.edu'
# NWS asks for a descriptive User-Agent with contact info.
UA = 'openfloodhub (https://github.com/abhiramm7/openfloodhub)'
CFS_TO_M3S = 0.0283168
IN_TO_MM = 25.4


def _get(url: str, timeout: int = 30, retries: int = 3) -> dict | None:
    # NWPS regularly drops or times out a single request (same flakiness as
    # USGS/Open-Meteo in fetch.py) — retry before giving the overlay up.
    req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept': 'application/json'})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as e:
            # 404 means the resource doesn't exist for this gauge — no retry.
            if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                print(f'   ! NOAA {url[-70:]}: {e}')
                return None
            if attempt == retries - 1:
                print(f'   ! NOAA {url[-70:]}: {e}')
                return None
            time.sleep(2 * (attempt + 1))


def fetch_gauge(usgs_id: str) -> dict | None:
    """One gauge record. Pulls lid, reach id, official flood thresholds, and
    the current observed + NWS-forecast stage."""
    g = _get(f'{NWPS}/gauges/{usgs_id}')
    if not g or not g.get('lid'):
        return None

    cats = g.get('flood', {}).get('categories', {}) or {}
    def _stage(name):
        c = cats.get(name, {})
        v = c.get('stage')
        return float(v) if isinstance(v, (int, float)) and v > -9000 else None
    def _flow(name):
        c = cats.get(name, {})
        v = c.get('flow')
        return float(v) if isinstance(v, (int, float)) and v > -9000 else None

    obs = g.get('status', {}).get('observed', {}) or {}
    fc = g.get('status', {}).get('forecast', {}) or {}

    return {
        'lid': g['lid'],
        'reach_id': g.get('reachId'),
        'rfc': g.get('rfc', {}).get('abbreviation'),
        'wfo': g.get('wfo', {}).get('abbreviation'),
        'name_nws': g.get('name'),
        'thresholds_stage_ft': {
            'action':   _stage('action'),
            'minor':    _stage('minor'),
            'moderate': _stage('moderate'),
            'major':    _stage('major'),
        },
        'thresholds_flow_cfs': {
            'action':   _flow('action'),
            'minor':    _flow('minor'),
            'moderate': _flow('moderate'),
            'major':    _flow('major'),
        },
        'current': {
            'stage_ft': obs.get('primary') if obs.get('primaryUnit') == 'ft' else None,
            'flow_cfs': (obs.get('secondary') * 1000 if obs.get('secondaryUnit') == 'kcfs'
                          else obs.get('secondary') if obs.get('secondaryUnit') == 'cfs'
                          else None),
            'category': obs.get('floodCategory'),
            'valid_time': obs.get('validTime'),
        },
        'nws_forecast': {
            'stage_ft': fc.get('primary') if fc.get('primaryUnit') == 'ft' else None,
            'flow_cfs': (fc.get('secondary') * 1000 if fc.get('secondaryUnit') == 'kcfs'
                          else fc.get('secondary') if fc.get('secondaryUnit') == 'cfs'
                          else None),
            'category': fc.get('floodCategory'),
            'valid_time': fc.get('validTime'),
        },
    }


# --------------------------------------------------------------------------
# NOAA National Water Model streamflow (analysis + short + medium range)
# --------------------------------------------------------------------------

def _to_m3s(flow, units: str | None) -> float:
    """NWM streamflow is published in ft³/s; convert to m³/s. If the API ever
    reports metric units we leave the value alone."""
    f = float(flow)
    if units and ('ft' in units or 'cfs' in units.lower()):
        return round(f * CFS_TO_M3S, 3)
    return round(f, 3)


def _extract_series(payload: dict, key: str) -> list[dict]:
    """Pull one NWM series out of a /streamflow response. Returns
    [{t, flow_m3s}, ...] sorted by time, or [] if absent/empty."""
    block = payload.get(key) or {}
    series = block.get('series') or {}
    units = series.get('units')
    out = []
    for row in series.get('data', []):
        try:
            out.append({'t': row['validTime'], 'flow_m3s': _to_m3s(row['flow'], units)})
        except (KeyError, ValueError, TypeError):
            continue
    out.sort(key=lambda r: r['t'])
    return out


def fetch_nwm_streamflow(reach_id: str) -> dict:
    """One request to the NWPS reach endpoint returns every NWM series at once.
    Returns {short, medium, analysis} as lists of {t, flow_m3s} in m³/s.

      short    next ~18h hourly      (short_range)
      medium   next ~10d hourly      (mediumRangeBlend, deterministic blend
                                       of the medium-range ensemble)
      analysis recent best-estimate  (analysisAssimilation) — NWM's "observed"
    """
    empty = {'short': [], 'medium': [], 'analysis': []}
    if not reach_id:
        return empty
    d = _get(f'{NWPS}/reaches/{reach_id}/streamflow')
    if not d:
        return empty
    medium = _extract_series(d, 'mediumRangeBlend') or _extract_series(d, 'mediumRange')
    return {
        'short': _extract_series(d, 'shortRange'),
        'medium': medium,
        'analysis': _extract_series(d, 'analysisAssimilation'),
    }


# --------------------------------------------------------------------------
# NWS quantitative precipitation forecast (QPF) — forecast rainfall, mm
# --------------------------------------------------------------------------

_DUR = re.compile(r'P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?)?')


def _interval_hours(iso_interval: str) -> tuple[str, int]:
    """Split an ISO8601 '<start>/<duration>' into (start_iso, n_hours).
    NWS QPF periods are multiples of an hour, e.g. 'PT6H' -> 6."""
    start, _, dur = iso_interval.partition('/')
    m = _DUR.fullmatch(dur) if dur else None
    if not m:
        return start, 1
    days, hours, mins = (int(g) if g else 0 for g in m.groups())
    return start, max(1, days * 24 + hours + round(mins / 60))


def fetch_nws_qpf(lat: float, lon: float) -> list[dict]:
    """NWS gridpoint quantitative precipitation forecast at a point. Each NWS
    period reports total mm over a multi-hour window; we spread it evenly to
    hourly so it overlays the chart's hourly precip bars. Returns
    [{t, precip_mm}, ...] hourly in mm."""
    pt = _get(f'{NWS}/points/{lat:.4f},{lon:.4f}')
    grid_url = (pt or {}).get('properties', {}).get('forecastGridData')
    if not grid_url:
        return []
    grid = _get(grid_url)
    qpf = (grid or {}).get('properties', {}).get('quantitativePrecipitation', {})
    out = []
    for v in qpf.get('values', []):
        try:
            total = float(v['value'])
        except (KeyError, ValueError, TypeError):
            continue
        start, n = _interval_hours(v.get('validTime', ''))
        if not start:
            continue
        t0 = pd.Timestamp(start)
        per_hour = round(total / n, 3)
        for h in range(n):
            t = (t0 + pd.Timedelta(hours=h)).strftime('%Y-%m-%dT%H:%M:%SZ')
            out.append({'t': t, 'precip_mm': per_hour})
    return out


# --------------------------------------------------------------------------
# MRMS observed precipitation (radar QPE) via IEM IEMRE — daily, mm
# --------------------------------------------------------------------------

def fetch_mrms_precip(lat: float, lon: float,
                      start_date: str, end_date: str) -> list[dict]:
    """Observed MRMS radar precipitation (QPE) at a point, daily totals in mm,
    from the IEM IEMRE point service. Dates are 'YYYY-MM-DD'. Returns
    [{date, precip_mm}, ...]."""
    d = _get(f'{IEM}/iemre/multiday/{start_date}/{end_date}/{lat:.4f}/{lon:.4f}/json')
    if not d:
        return []
    out = []
    for row in d.get('data', []):
        mrms_in = row.get('mrms_precip_in')
        if mrms_in is None:
            continue
        try:
            out.append({'date': row['date'], 'precip_mm': round(float(mrms_in) * IN_TO_MM, 2)})
        except (KeyError, ValueError, TypeError):
            continue
    return out


# --------------------------------------------------------------------------

def enrich_site(usgs_id: str, lat: float | None = None, lon: float | None = None,
                mrms_start: str | None = None, mrms_end: str | None = None) -> dict:
    """Gather every NOAA/NWS overlay for one gauge in one shot. Returns a dict
    with whatever could be fetched (any field may be empty/None):

      gauge          NWS official thresholds + observed/forecast stage
      nwm_short      NWM short-range streamflow (m³/s)
      nwm_medium     NWM medium-range blend streamflow (m³/s)
      nwm_analysis   NWM analysis-assimilation "observed" streamflow (m³/s)
      qpf            NWS forecast rainfall, hourly mm
      mrms_precip    MRMS observed rainfall, daily mm (needs lat/lon + dates)
    """
    gauge = fetch_gauge(usgs_id)
    reach_id = gauge.get('reach_id') if gauge else None
    if lat is None and gauge is not None:
        lat = gauge.get('latitude')
    if lon is None and gauge is not None:
        lon = gauge.get('longitude')

    nwm = fetch_nwm_streamflow(reach_id)
    out = {
        'gauge': gauge,
        'nwm_short': nwm['short'],
        'nwm_medium': nwm['medium'],
        'nwm_analysis': nwm['analysis'],
        'qpf': [],
        'mrms_precip': [],
    }
    if lat is not None and lon is not None:
        out['qpf'] = fetch_nws_qpf(lat, lon)
        if mrms_start and mrms_end:
            out['mrms_precip'] = fetch_mrms_precip(lat, lon, mrms_start, mrms_end)
    return out
