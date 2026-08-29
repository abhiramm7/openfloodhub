"""Per-gauge flood thresholds (Warning / Danger / Extreme), in m³/s.

NWS publishes flood categories mostly as *stage* (gauge height in ft), which a
discharge model can't speak to. So we derive flow thresholds the way the
hydrology does when only a short record exists: take the daily-peak flow
distribution over the multi-year record and read off high empirical quantiles
as return-period stand-ins.

  warning  ~ 0.95 quantile of daily peaks   (roughly a 2-year return flow)
  danger   ~ 0.99 quantile                  (~5-year)
  extreme  ~ 0.999 quantile                 (~10-year)

With only a few years of record these are empirical stand-ins, not fitted
return periods — but they give the UI a stable, per-site Normal/Warning/
Danger/Extreme banding that matches each gauge's own behavior.

Run once (slow — pulls multi-year USGS flow per gauge) and cache to
thresholds.json, which predict.py reads on every inference run:

    python -m flood_warning.thresholds
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .fetch import fetch_usgs_hourly
from .sites import SITES

THRESH_PATH = Path(__file__).resolve().parent / 'thresholds.json'
MIN_DAYS = 365


def compute_thresholds(gauge_id: str, years_back: int = 3) -> dict | None:
    """Daily-peak quantile thresholds for one gauge from `years_back` of hourly
    USGS flow. Returns None if the record is too short."""
    end = pd.Timestamp.utcnow().normalize().tz_localize(None)
    start = end - pd.DateOffset(years=years_back)
    flow = fetch_usgs_hourly(gauge_id, start.strftime('%Y-%m-%d'),
                             (end - pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
    daily_peaks = flow.resample('D').max().dropna()
    if len(daily_peaks) < MIN_DAYS:
        return None
    return {
        'warning':      round(float(daily_peaks.quantile(0.95)), 3),
        'danger':       round(float(daily_peaks.quantile(0.99)), 3),
        'extreme':      round(float(daily_peaks.quantile(0.999)), 3),
        'max_observed': round(float(daily_peaks.max()), 3),
        'record_years': int(len(daily_peaks) / 365),
    }


def load() -> dict:
    """Cached thresholds keyed by gauge id, or {} if not yet computed."""
    if THRESH_PATH.exists():
        return json.loads(THRESH_PATH.read_text())
    return {}


def build_all(years_back: int = 3) -> dict:
    """Compute thresholds for every site and write thresholds.json."""
    out = {}
    print(f'Computing flood thresholds from {years_back}y of daily peaks...')
    for site in SITES:
        try:
            th = compute_thresholds(site['id'], years_back=years_back)
        except Exception as e:
            print(f'  ! {site["id"]} {site["short"]:<16} FAILED: {e}')
            continue
        if th:
            out[site['id']] = th
            print(f'  {site["id"]} {site["short"]:<16} '
                  f'warn={th["warning"]:.2f} danger={th["danger"]:.2f} '
                  f'extreme={th["extreme"]:.2f} m³/s ({th["record_years"]}y)')
        else:
            print(f'  {site["id"]} {site["short"]:<16} record too short, skipped')
    THRESH_PATH.write_text(json.dumps(out, indent=2))
    print(f'\nwrote {THRESH_PATH} ({len(out)} gauges)')
    return out


if __name__ == '__main__':
    build_all()
