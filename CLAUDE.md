# CLAUDE.md

Guidance for working in this repo. Read `README.md` for the full project rationale; this file covers what's non-obvious for making changes.

## What this is

OpenFloodHub trains a **separate small CNN per USGS river gauge** (one `.pt` per gauge in `flood_warning/checkpoints/`), instead of one global model. Each model is ~50k params, CPU-trainable in ~90s. Ships with a deployment for 7 gauges around Washington, DC. The `web/` map is a static UI driven entirely by a generated `preds.json` — no backend.

## Environment & commands

Python 3.12, managed with `uv`. Use the venv interpreter directly (`.venv/bin/python`), not a global one.

```bash
uv venv --python 3.12 .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m flood_warning.fetch        # cache ~3yr hourly data -> ./data/
.venv/bin/python -m flood_warning.train <gauge_id>   # train one gauge -> checkpoints/<id>.pt
.venv/bin/python -m flood_warning.thresholds    # -> flood_warning/thresholds.json
.venv/bin/python -m flood_warning.predict       # live inference -> outputs/dmv-cnn-12h/preds.json
```

Serve the map (use the `web` preview config, or):
```bash
cp outputs/dmv-cnn-12h/preds.json web/preds.json
cd web && python3 -m http.server 8772
```

- Secrets: `.env.local` with `USGS_API_KEY=...` (optional — USGS NWIS works without one).
- Data cache dir: `./data/` (override with `$FLOOD_DATA_DIR`). Both `data/` and `outputs/` are gitignored.
- **No test suite and no linter config** in this repo. Verify changes by running the module CLIs above.

## Layout

Everything is in the `flood_warning/` package (module scripts, each with a `__main__`):
`sites.py` (gauge registry: id/lat/lon/drainage/kind) · `fetch.py` (USGS NWIS + Open-Meteo) · `dataset.py` (windowing + `Scaler`) · `model.py` (`FloodCNN`) · `train.py` · `predict.py` · `thresholds.py` · `noaa.py` (comparison overlays) · `google_flood.py` (Google Flood Hub overlays + `google_gauges.json` mapping).

## Gotchas

- **The past stream is 4 channels, not 3.** `dataset.py` feeds `[flow, precip, temp, soil_moisture]` (24h). The 4th channel is ERA5-Land surface soil moisture (antecedent-wetness proxy). `model.py`'s docstring still says "3 channels" but its default is `n_past_features=4` — trust the code. Future stream is 1 channel (`precip`, 12h). Target is 12-step flow.
- **CPU-only. Do not enable MPS/CUDA** in training — PyTorch 2.5 MPS has flaky 1D conv kernels; `train.py` is intentionally CPU/single-process.
- **Normalization** is `log1p` on flow+precip then z-score, stats computed on the **train split only**, stored in the checkpoint alongside weights via `Scaler`. Test split is the last 15% (temporal, not random).
- **Current gauge set is 7** (`sites.py`): Potomac at Little Falls, Rock Creek, Anacostia, NE/NW Branch Anacostia, Watts Branch, Difficult Run. (Point of Rocks, Goose Creek, and Catoctin were dropped — >40 km out of the DC core.) If you add/remove a gauge, update `sites.py` then re-run fetch → train → thresholds.
- **NOAA overlays in `noaa.py` are display-only** reference series (NWM streamflow, NWS QPF, MRMS precip). They are never fed back into the model — don't wire them into `dataset.py`/`model.py`.
- **Google Flood Hub overlays (`google_flood.py`) are display-only too**, same rule. They need `GOOGLE_FLOOD_API_KEY` (`.env.local` locally, Actions secret in CI) and are silently skipped without it. `google_gauges.json` maps USGS sites to Google HYBAS basin-outlet gauges; it was **hand-verified by threshold-magnitude comparison** — the `python -m flood_warning.google_flood` auto-matcher only proposes, don't let it clobber the curated file blindly. NW Anacostia + Watts Branch have no Google gauge (basins too small). The `google_flood` entry also carries `backtest` (past week of Google's next-day forecasts) for the panel's model-comparison section.
- **`web/basins.json` is generated offline, not by the pipeline** — upstream unions of HydroBASINS level-12 polygons for each Google-mapped gauge, drawn when a site is selected. Rebuild only if `google_gauges.json` changes (needs the ~65 MB hybas_na_lev12 download + shapely; see README).
- **Anacostia at Kenilworth (01651760) is tidal — negative discharge is real** (flow reverses on flood tide, ~5-6h stretches twice a day). `fetch.py` keeps negative 15-min readings in the hourly mean and floors the hour at 0. Don't "fix" this by dropping negatives: that punches recurring multi-hour holes in the record, and the NaN guard in `dataset.py::encode_window` then refuses every live window for the gauge.
- **Flow gap-filling happens exactly once, in fetch** (`interpolate(limit=4, limit_area='inside')`). `encode_window` deliberately does no flow filling — a leftover NaN means real missing data and the window is refused. Don't add a second interpolate pass downstream (it doubles the tolerated gap) and keep `limit_area='inside'` (without it, `fetch_hourly_live`'s forecast-extended index gets the last USGS reading held forward as fake observations).
- **Sites that fail to fetch get an "offline" stub** in preds.json (`status: 'offline'`, empty `series`/`backtest`) so the map shows a gray marker — don't "clean up" the stub logic in `predict.py::run_all`. Rock Creek (01648000) has had a dead USGS feed since before launch and lives permanently in this state.
- **Thresholds are flow-based (m³/s), derived from each gauge's own daily-peak distribution** as return-period stand-ins — not NWS stage-height categories (a discharge model can't speak feet).

## Data flow & CI

`predict.py` writes `outputs/dmv-cnn-12h/preds.json` (per gauge: 12h forecast, 7-day backtest, thresholds, NOAA overlays; `model_id: dmv-cnn-12h`). The map reads `web/preds.json`, so any pipeline change must end with copying the file across.

Two GitHub Actions workflows:
- `forecast.yml` — every 2h (and manual): reinstalls deps, runs `predict`, copies to `web/preds.json`, commits it, and deploys Pages itself (a `GITHUB_TOKEN` push can't trigger `pages.yml`). 20-min timeout.
- `pages.yml` — deploys the map on pushes touching `web/**`.

The committed `web/preds.json` is the live map data; the scheduled bot commits (`data: refresh forecasts ...`) are expected churn on `main`.
