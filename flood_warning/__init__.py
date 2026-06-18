"""DMV hourly flood warning subsystem.

Separate from the daily LSTM pipeline. Architecture:
  - Operational USGS gauges (not CAMELS), 10 priority sites
  - Hourly time resolution
  - 24-hour input window -> 12-hour ahead flow forecast
  - 1D CNN per gauge

Files:
  sites.py       gauge registry (id, name, lat/lon, drainage, type)
  fetch.py       USGS NWIS + Open-Meteo hourly archive
  dataset.py     window builder + train/val/test split
  model.py       two-branch 1D CNN
  train.py       per-gauge training loop
  predict.py     live inference (NWS QPF + recent USGS -> 12h forecast)
"""
