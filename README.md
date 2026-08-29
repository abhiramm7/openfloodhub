# OpenFloodHub

An open-source, self-hostable alternative to [Google Flood Hub](https://sites.research.google/floods/).

Google Flood Hub runs one global model: a single neural network, trained across thousands of basins, that predicts streamflow everywhere from one set of weights. OpenFloodHub does the opposite. It trains a separate small model for each river gauge, on that gauge's own history.

That tradeoff is the whole point of the project:

- A model that only has to learn one catchment can be tiny (about 50k parameters), train in 90 seconds on a laptop CPU, and you can actually inspect what it learned for that specific river.
- A global model generalizes to places with no gauge at all, which a per-site model cannot do. You need a few years of record for each gauge before you can train it.

So this is not a drop-in replacement for everything Flood Hub does. It is a different bet: that for a known set of gauges you care about, a specialized local model is simpler to run, cheaper to retrain, and easier to reason about than one big model trying to cover the planet.

The repo ships with a working deployment for the gauges around Washington, DC — the live map is at **[abhiramm7.github.io/openfloodhub](https://abhiramm7.github.io/openfloodhub/)**, refreshed every two hours.

Version 0.2 brings the comparison full circle: the map is now integrated with the [Google Flood Forecasting API](https://developers.google.com/flood-forecasting), the model behind Flood Hub itself, so you can see Google's predictions for these same gauges next to the local CNN's and check which of the two tracked the river better over the past week.

> **Caveat.** This is a research prototype. Do not use it to decide whether to drive through floodwater. The official sources for that are [NWS AHPS](https://water.weather.gov/ahps/) and [NOAA NWPS](https://water.noaa.gov/).

## The map

`web/` is a static map with no build step and no backend: one HTML file, one JS file, Leaflet, and the `preds.json` the pipeline writes.

![OpenFloodHub demo: select the Potomac gauge, open the model comparison](assets/demo.gif)

Click a gauge and the panel shows the last day of observed discharge with the CNN's 12-hour forecast, plus NOAA's National Water Model and Google's forecast for comparison. The catchment that drains to the gauge is outlined on the map. Thresholds are per gauge, derived from each river's own flood history, so a green Potomac and a green Watts Branch mean very different absolute flows. The compare view replays the past week of CNN and Google predictions against what the river actually did and scores both.

The GIF above comes from `scripts/make_demo_gif.py`; rerun it after UI changes (it needs `playwright` and `pillow` in the venv).

To run it locally after generating `preds.json` (see below):

```bash
cp outputs/dmv-cnn-12h/preds.json web/preds.json
cd web && python3 -m http.server 8772      # open http://localhost:8772
```

## The model

`flood_warning/model.py`. Two-branch 1D CNN: a 4-channel past stream (flow, precip, temperature, and soil moisture for the last 24 hours) and a 1-channel future stream (forecast precip for the next 12 hours). Each goes through a few Conv1D layers, gets flattened, concatenated, and projected to a 12-step output. The 4th past channel is ERA5-Land surface soil moisture, a proxy for how saturated the basin already is when a storm arrives.

```
past   (4 ch x 24h)  ->  Conv1D x 3  ->  flatten
future (1 ch x 12h)  ->  Conv1D x 2  ->  flatten
                         concat -> FC -> 12-step forecast
```

About 50k parameters. Trains per gauge in 90 seconds on CPU.

Held-out test NSE (3 years hourly, last 15% as test):

| Gauge | Drainage (mi²) | Test NSE | 12h-ahead NSE |
| --- | ---: | ---: | ---: |
| Potomac at Little Falls (DC) | 11,560 | 0.977 | 0.945 |
| Anacostia at Kenilworth (DC) | 134 | 0.694 | 0.653 |
| NE Branch Anacostia (MD) | 73 | 0.436 | 0.175 |
| Rock Creek at Sherrill Dr (DC) | 62 | 0.414 | 0.144 |
| Difficult Run (VA) | 58 | 0.352 | 0.115 |
| NW Branch Anacostia (MD) | 21 | 0.211 | 0.061 |
| Watts Branch (DC) | 3.6 | 0.123 | 0.024 |

The big mainstem gauges are basically solved at hourly resolution. The small urban catchments at the bottom of the table are hard and there's no real way around it: Watts Branch is 3.6 mi² of pavement, it responds to rain in minutes, and an hourly model is the wrong tool. Either bump to 15-minute cadence or feed it NEXRAD precip instead of point ERA5.

## Thresholds

NWS publishes flood categories mostly as gauge height (stage in feet), which a discharge model can't speak to. So `thresholds.py` derives flow thresholds (m³/s) from each gauge's own record: it takes the daily-peak distribution over the multi-year history and reads off high quantiles as return-period stand-ins (roughly 2-, 5-, and 10-year for Warning, Danger, Extreme). These are cached to `thresholds.json` and attached to every prediction.

## Layout

```
flood_warning/
├── sites.py            # the gauges + lat/lon/drainage
├── fetch.py            # USGS NWIS + Open-Meteo data fetcher
├── dataset.py          # windowing + scaler
├── model.py            # the CNN
├── train.py            # per-gauge training
├── predict.py          # live inference + 7-day backtest
├── thresholds.py       # per-gauge flood thresholds from the record
├── thresholds.json     # cached thresholds (committed)
├── noaa.py             # NOAA/NWS comparison overlays (NWM, QPF, MRMS)
├── google_flood.py     # Google Flood Hub comparison overlays
├── google_gauges.json  # USGS site -> Google (HYBAS) gauge mapping (committed)
├── checkpoints/        # pretrained .pt files (one per gauge)
└── requirements-ci.txt

web/                    # static map UI (index.html, app.js, preds.json)
```

## Setup

Python 3.12. Get a free USGS API key from [waterdata.usgs.gov](https://waterdata.usgs.gov) and put it in `.env.local` as `USGS_API_KEY=...`.

```bash
uv venv --python 3.12 .venv
.venv/bin/pip install -r requirements.txt
```

Cached data goes to `./data/` by default; override with `$FLOOD_DATA_DIR`. Inference writes to `./outputs/`.

## Train from scratch

```bash
.venv/bin/python -m flood_warning.fetch          # ~3 years of hourly data
for gid in 01646500 01648000 01651760 01649500 01650500 01651800 01646000; do
  .venv/bin/python -m flood_warning.train "$gid"
done                                              # ~90s per gauge on CPU
.venv/bin/python -m flood_warning.thresholds      # compute flood thresholds
```

## Run inference

```bash
.venv/bin/python -m flood_warning.predict        # writes outputs/dmv-cnn-12h/preds.json
```

`preds.json` contains, per gauge: the 12-hour-ahead CNN forecast, a 7-day hourly backtest (how the model has been tracking observed flow lately), the gauge's flood thresholds, and a set of NOAA/NWS comparison overlays. The overlays are reference series only. They ride alongside the CNN forecast and are never fed back into the model:

| Field | Source | What it is |
| --- | --- | --- |
| `noaa_nwm` | NWM short-range | Next ~18h streamflow, hourly (m³/s) |
| `noaa_nwm_medium` | NWM medium-range blend | Next ~10d streamflow, hourly (m³/s) |
| `noaa_nwm_analysis` | NWM analysis-assimilation | Recent best-estimate "observed" streamflow (m³/s) |
| `noaa_qpf` | NWS gridpoint forecast | Forecast rainfall, hourly (mm) |
| `noaa_mrms_precip` | MRMS radar QPE (via IEM) | Observed rainfall, daily (mm) |
| `google_flood` | Google Flood Forecasting API | Daily discharge forecast ~7d out (m³/s), flood severity + trend, 2/5/20-yr thresholds |

NWM streamflow and NWS QPF come from unauthenticated NOAA APIs ([NWPS](https://api.water.noaa.gov/nwps/v1/docs/), [api.weather.gov](https://www.weather.gov/documentation/services-web-api)); MRMS observed precip is pulled per-point from the [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/) IEMRE service.

The Google overlay comes from the [Flood Forecasting API](https://developers.google.com/flood-forecasting), the same model that powers [Flood Hub](https://g.co/floodhub). It needs a `GOOGLE_FLOOD_API_KEY` in `.env.local` (an Actions secret in CI) and is skipped quietly when the key is missing. Google's US gauges are virtual points at HYBAS basin outlets rather than USGS station locations, so `google_gauges.json` maps each site to its basin-matched Google gauge. Five of the seven sites have one; NW Anacostia and Watts Branch drain basins too small for HYBAS to model. The overlay also keeps the past week of Google's next-day forecasts, which the compare view scores against observed flow next to the CNN's own backtest.

Selecting a gauge outlines its contributing area from `web/basins.json` (about 12 KB), built once offline by dissolving upstream [HydroBASINS](https://www.hydrosheds.org/products/hydrobasins) level-12 catchments. Rebuild it if `google_gauges.json` changes.

## Adding a gauge

Add a row to `flood_warning/sites.py` with `id`, `name`, `short` (the abbreviated label used on the map and in logs), `lat`, `lon`, `drainage_sqmi`, and `kind`. All seven fields are required — `predict` builds the map data from them. Then fetch, train, and compute its thresholds.

## How to use this

The hosted map is at <https://abhiramm7.github.io/openfloodhub/> and refreshes every two hours.

1. Marker color is the gauge's risk tier — whichever is worse of the current flow and the forecast 12-hour peak, held against that gauge's own thresholds: green (normal), then warning, danger, and extreme. Gray means the gauge is offline or has no thresholds. Rock Creek has been gray since launch because USGS stopped serving its data, though Google still forecasts its basin.
2. Click a marker. The panel shows current flow, the 12-hour forecast, and that gauge's own thresholds. Five of the seven gauges also get the catchment that drains to them outlined on the map; NW Anacostia and Watts Branch drain basins too small for HydroBASINS, so they have no outline (and no Google forecast — see above).
3. "Compare model predictions" opens a week of hindsight along the bottom: what the CNN predicted, drawn over what actually happened, with the model's error over that week. Google's next-day forecasts join the chart for the five Google-mapped gauges; on the other two the Google score shows "—". The rainfall bars along the top of that chart explain most of the disagreements.

To run your own copy, fork the repo, add `GOOGLE_FLOOD_API_KEY` as an Actions secret (`USGS_API_KEY` is optional), enable GitHub Pages, and the forecast workflow handles the rest on a 2-hour schedule. To point it at different rivers, see "Adding a gauge" above.

## License

[Apache 2.0](./LICENSE).
