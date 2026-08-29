"""Hourly data fetcher: USGS NWIS instantaneous flow + Open-Meteo hourly archive."""
from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .sites import SITES, BY_ID

REPO = Path(__file__).resolve().parents[1]
# Cached hourly observations + forcings. Override with $FLOOD_DATA_DIR.
DATA_DIR = Path(os.environ.get('FLOOD_DATA_DIR', REPO / 'data'))
UA = 'openfloodhub-cnn'

# Load USGS API key from .env.local
for env_path in (REPO / '.env.local',):
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
USGS_API_KEY = os.environ.get('USGS_API_KEY')

CFS_TO_M3S = 0.0283168


def _http_get(url: str, headers: dict | None = None, timeout: int = 120,
              retries: int = 3) -> bytes:
    # USGS/Open-Meteo intermittently drop TLS handshakes (seen ~daily in CI);
    # a single failed request must not cost a gauge its forecast for 2 hours.
    h = {'User-Agent': UA}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, headers=h)
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(req, timeout=timeout).read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def fetch_usgs_hourly(gauge_id: str, start: str, end: str) -> pd.Series:
    """USGS NWIS instantaneous values (15-min cadence), resampled to hourly mean
    (m³/s). Single request can cover ~1 year; we batch in 90-day chunks to keep
    payloads reasonable."""
    headers = {'X-Api-Key': USGS_API_KEY} if USGS_API_KEY else {}
    chunks = []
    cur = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while cur < end_ts:
        chunk_end = min(cur + pd.Timedelta(days=90), end_ts)
        url = ('https://waterservices.usgs.gov/nwis/iv/?sites='
               + gauge_id + '&parameterCd=00060'
               + f'&startDT={cur.strftime("%Y-%m-%d")}'
               + f'&endDT={chunk_end.strftime("%Y-%m-%d")}'
               + '&format=json')
        try:
            payload = json.loads(_http_get(url, headers))
        except Exception as e:
            print(f'   ! {gauge_id} {cur.date()}: {e}')
            cur = chunk_end + pd.Timedelta(days=1)
            continue
        for ts in payload.get('value', {}).get('timeSeries', []):
            for v in ts.get('values', [{}])[0].get('value', []):
                try:
                    cfs = float(v['value'])
                    if cfs < 0:
                        continue
                    chunks.append((v['dateTime'], cfs * CFS_TO_M3S))
                except (ValueError, KeyError, TypeError):
                    continue
        cur = chunk_end + pd.Timedelta(days=1)
        time.sleep(0.1)

    if not chunks:
        return pd.Series(dtype=float, name='flow_m3s')
    df = pd.DataFrame(chunks, columns=['t', 'flow_m3s'])
    df['t'] = pd.to_datetime(df['t'], utc=True).dt.tz_convert(None)
    df = df.set_index('t').sort_index()
    # Resample 15-min cadence to hourly mean
    hourly = df['flow_m3s'].resample('h').mean()
    hourly.name = 'flow_m3s'
    return hourly


def fetch_openmeteo_hourly(lat: float, lon: float,
                            start: str, end: str) -> pd.DataFrame:
    """Open-Meteo Archive (ERA5) hourly forcings at lat/lon.

    Hourly precip is the actual rainfall during that hour (mm). Soil moisture
    (0-7cm + 7-28cm layers, m³/m³) gives the model an antecedent-wetness
    signal — proxy for groundwater storage, drives runoff response to rain.
    """
    url = ('https://archive-api.open-meteo.com/v1/archive?'
           f'latitude={lat:.4f}&longitude={lon:.4f}'
           f'&start_date={start}&end_date={end}'
           f'&hourly=precipitation,temperature_2m,surface_pressure,'
           f'soil_moisture_0_to_7cm,soil_moisture_7_to_28cm'
           f'&timezone=GMT')
    payload = json.loads(_http_get(url))
    df = pd.DataFrame(payload['hourly']).rename(columns={'time': 't'})
    df['t'] = pd.to_datetime(df['t'])
    return df.set_index('t')


def fetch_site(site: dict, years_back: int = 3) -> pd.DataFrame:
    """Combine USGS + Open-Meteo for one site, save to disk, return the merged
    DataFrame (hourly, columns flow_m3s, precip_mm, temp_c, pressure_hpa)."""
    end = pd.Timestamp.utcnow().normalize().tz_localize(None)
    start = end - pd.DateOffset(years=years_back)
    start_s = start.strftime('%Y-%m-%d')
    end_s = (end - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    print(f'  [{site["id"]}] {site["short"]:<16}', end='', flush=True)
    print(' usgs...', end='', flush=True)
    flow = fetch_usgs_hourly(site['id'], start_s, end_s)
    print(f' ({len(flow)} hr)', end='', flush=True)

    print(' meteo...', end='', flush=True)
    weather = fetch_openmeteo_hourly(site['lat'], site['lon'], start_s, end_s)
    print(f' ({len(weather)} hr)', end='', flush=True)

    df = pd.concat(
        [flow, weather.rename(columns={
            'precipitation': 'precip_mm',
            'temperature_2m': 'temp_c',
            'surface_pressure': 'pressure_hpa',
            'soil_moisture_0_to_7cm': 'sm_surface',
            'soil_moisture_7_to_28cm': 'sm_subsurface',
        })], axis=1, join='outer')
    df = df.loc[(df.index >= start) & (df.index < end)].sort_index()
    # Forward-fill small gaps in flow (USGS sometimes missing 1-2 hours).
    # Soil moisture from ERA5 lags ~5 days; forward-fill across the gap so
    # the recent window has a usable value.
    df['flow_m3s'] = df['flow_m3s'].interpolate(limit=4)
    df['sm_surface'] = df['sm_surface'].ffill(limit=24 * 7)
    df['sm_subsurface'] = df['sm_subsurface'].ffill(limit=24 * 7)

    out_dir = DATA_DIR / site['id']
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / 'hourly.parquet')
    print(f' -> {len(df)} hr, {df["flow_m3s"].notna().sum()} valid flows')
    return df


def fetch_all(years_back: int = 3):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Fetching {len(SITES)} sites × {years_back} years hourly...')
    for site in SITES:
        try:
            fetch_site(site, years_back=years_back)
        except Exception as e:
            print(f'  ! {site["id"]} FAILED: {e}')


def fetch_openmeteo_combined(lat: float, lon: float,
                              past_days: int = 8, forecast_days: int = 2) -> pd.DataFrame:
    """Single Open-Meteo Forecast call covering past_days + forecast_days
    hourly. The Forecast endpoint doesn't expose soil moisture — that's pulled
    from the Archive endpoint separately."""
    url = ('https://api.open-meteo.com/v1/forecast?'
           f'latitude={lat:.4f}&longitude={lon:.4f}'
           f'&hourly=precipitation,temperature_2m,surface_pressure'
           f'&past_days={past_days}&forecast_days={forecast_days}&timezone=GMT')
    payload = json.loads(_http_get(url))
    df = pd.DataFrame(payload['hourly']).rename(columns={'time': 't'})
    df['t'] = pd.to_datetime(df['t'])
    return df.set_index('t').sort_index()


def fetch_hourly_live(gauge_id: str, days_back: int = 8) -> pd.DataFrame:
    """Last `days_back` days of hourly USGS + Open-Meteo for one gauge, fully
    in-memory. Used in CI where the parquet cache from fetch_all() is gone.

    Two-endpoint merge:
      - Forecast (`past_days` + `forecast_days`) for precip / temp / pressure.
        Covers up to "today + 16d" with no gap.
      - Archive (`start_date` + `end_date`) for soil moisture, since ERA5-Land
        is the only public source for that field. Lags ~5 days; we forward-fill
        the tail so the recent window has a value.
    """
    site = BY_ID[gauge_id]
    end = pd.Timestamp.utcnow().tz_localize(None)
    start = end - pd.Timedelta(days=days_back + 1)
    flow = fetch_usgs_hourly(gauge_id, start.strftime('%Y-%m-%d'),
                              end.strftime('%Y-%m-%d'))
    weather = fetch_openmeteo_combined(site['lat'], site['lon'],
                                        past_days=days_back, forecast_days=2)
    # Archive endpoint rejects end_date in the future; cap at yesterday and
    # let ffill cover the remainder.
    sm_end = min(end, pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=1))
    sm = fetch_openmeteo_hourly(
        site['lat'], site['lon'],
        start.strftime('%Y-%m-%d'),
        sm_end.strftime('%Y-%m-%d'),
    )[['soil_moisture_0_to_7cm', 'soil_moisture_7_to_28cm']]
    sm = sm.rename(columns={'soil_moisture_0_to_7cm': 'sm_surface',
                              'soil_moisture_7_to_28cm': 'sm_subsurface'})
    df = pd.concat([
        flow,
        weather.rename(columns={
            'precipitation': 'precip_mm',
            'temperature_2m': 'temp_c',
            'surface_pressure': 'pressure_hpa',
        }),
        sm,
    ], axis=1, join='outer').sort_index()
    df['flow_m3s'] = df['flow_m3s'].interpolate(limit=4)
    df['sm_surface'] = df['sm_surface'].ffill(limit=24 * 7)
    df['sm_subsurface'] = df['sm_subsurface'].ffill(limit=24 * 7)
    return df


if __name__ == '__main__':
    fetch_all(years_back=3)
