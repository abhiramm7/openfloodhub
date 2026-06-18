"""NOAA NWPS API integration — pulls official flood stages, NWS observed/
forecast stage, and National Water Model streamflow for our gauges.

  https://api.water.noaa.gov/nwps/v1/docs/

What we use:
  /gauges/{usgsId}                   gauge metadata, current obs, NWS
                                     forecast, official flood thresholds
  /reaches/{reachId}/streamflow      NOAA National Water Model series in
                                     several horizons; we use short_range
                                     (next 18 hours, hourly)

All endpoints are unauthenticated JSON. ft³/s -> m³/s conversion for the
NWM output so it lines up with our CNN's units on the chart.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

NWPS = 'https://api.water.noaa.gov/nwps/v1'
UA = 'dmv-flood-watch'
CFS_TO_M3S = 0.0283168


def _get(url: str, timeout: int = 30) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as e:
        print(f'   ! NOAA {url[-60:]}: {e}')
        return None


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


def fetch_nwm_short_range(reach_id: str) -> list[dict]:
    """NWM short-range streamflow forecast — next ~18 hours hourly, in cfs.
    Returns [{t, flow_m3s}, ...] converted to m³/s for chart parity."""
    if not reach_id:
        return []
    d = _get(f'{NWPS}/reaches/{reach_id}/streamflow?series=short_range')
    if not d:
        return []
    series = d.get('shortRange', {}).get('series', {})
    out = []
    for row in series.get('data', []):
        try:
            out.append({
                't': row['validTime'],
                'flow_m3s': round(float(row['flow']) * CFS_TO_M3S, 3),
            })
        except (KeyError, ValueError, TypeError):
            continue
    return out


def enrich_site(usgs_id: str) -> tuple[dict | None, list[dict]]:
    """Convenience: fetch_gauge + fetch_nwm_short_range in one call.
    Returns (gauge_record, nwm_series). Either may be empty."""
    g = fetch_gauge(usgs_id)
    if not g:
        return None, []
    nwm = fetch_nwm_short_range(g.get('reach_id'))
    return g, nwm
