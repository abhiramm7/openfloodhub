# OpenFloodHub

An open-source, self-hostable flood-forecasting model — a free alternative to Google Flood Hub.

A tiny 1D CNN, trained per gauge on three years of hourly USGS streamflow paired with Open-Meteo ERA5 forcings, produces a 12-hour-ahead flow forecast. It runs on CPU, trains in ~90 seconds per gauge, and ships with pretrained checkpoints for ten gauges around Washington, DC. Everything here is just the model: fetch data, train, and run inference. No backend, no API keys beyond a free USGS key, no cloud dependency.

> **Caveat.** This is a research/prototype model. Do not use it to decide whether to drive through floodwater. The official sources for that are [NWS AHPS](https://water.weather.gov/ahps/) and [NOAA NWPS](https://water.noaa.gov/).

## The model

`flood_warning/model.py`. Two-branch 1D CNN: a 3-channel past stream (flow, precip, temperature for the last 24 hours) and a 1-channel future stream (forecast precip for the next 12 hours). Each goes through a few Conv1D layers, gets flattened, concatenated, and projected to a 12-step output.

```
past   (3 ch x 24h)  ->  Conv1D x 3  ->  flatten
future (1 ch x 12h)  ->  Conv1D x 2  ->  flatten
                         concat -> FC -> 12-step forecast
```

About 50k parameters. Trains per gauge in 90 seconds on CPU.

Held-out test NSE (3 years hourly, last 15% as test):

| Gauge | Drainage (mi²) | Test NSE | 12h-ahead NSE |
| --- | ---: | ---: | ---: |
| Potomac at Little Falls (DC) | 11,560 | 0.977 | 0.945 |
| Potomac at Point of Rocks (MD) | 9,651 | 0.965 | 0.921 |
| Goose Creek nr Leesburg (VA) | 332 | 0.700 | 0.601 |
| Anacostia at Kenilworth (DC) | 134 | 0.694 | 0.653 |
| Catoctin Creek (MD) | 67 | 0.648 | 0.547 |
| NE Branch Anacostia (MD) | 73 | 0.436 | 0.175 |
| Rock Creek at Sherrill Dr (DC) | 62 | 0.414 | 0.144 |
| Difficult Run (VA) | 58 | 0.352 | 0.115 |
| NW Branch Anacostia (MD) | 21 | 0.211 | 0.061 |
| Watts Branch (DC) | 3.6 | 0.123 | 0.024 |

The big mainstem gauges are basically solved at hourly resolution. The small urban catchments at the bottom of the table are hard and there's no real way around it: Watts Branch is 3.6 mi² of pavement, it responds to rain in minutes, and an hourly model is the wrong tool. Either bump to 15-minute cadence or feed it NEXRAD precip instead of point ERA5.

## Layout

```
flood_warning/
├── sites.py            # the gauges + lat/lon/drainage
├── fetch.py            # USGS NWIS + Open-Meteo data fetcher
├── dataset.py          # windowing + scaler
├── model.py            # the CNN
├── train.py            # per-gauge training
├── predict.py          # live inference + 7-day backtest
├── noaa.py             # NOAA National Water Model short-range overlay
├── checkpoints/        # pretrained .pt files (one per gauge)
└── requirements-ci.txt
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
```

## Run inference

```bash
.venv/bin/python -m flood_warning.predict        # writes outputs/dmv-cnn-12h/preds.json
```

`preds.json` contains, per gauge: the 12-hour-ahead forecast, a 7-day hourly backtest (how the model has been tracking observed flow lately), and a NOAA National Water Model short-range overlay for comparison.

## Adding a gauge

Add a row to `flood_warning/sites.py` with `id`, `name`, `lat`, `lon`, `drainage_sqmi`, `kind`. Then fetch and train it.

## License

[Apache 2.0](./LICENSE).
