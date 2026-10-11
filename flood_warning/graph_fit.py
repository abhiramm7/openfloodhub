"""Rain -> flow over the river graph: learn it once, then predict flow by
convolving rain through it. Research path; the live forecast doesn't use it.

    python -m flood_warning.graph_fit 05412500

Each graph node (subcatchment from basin_graph.py) gets its own hourly
Open-Meteo rain series. A gauge's flow is modelled as

    Q_g(t) = b_g + sum_i sum_tau  gain_i,tau * (E_tau * K_ig * P_i)(t)

  P_i       rain on node i as a flow rate: mm/h x area -> m³/s
  K_ig      node i -> gauge g travel-time histogram from the DEM graph
            ("how fast"), optionally stretched by a global factor alpha
  E_tau     unit-area linear-reservoir kernels (tau = 6 h .. 16 d): storage
            and recession, which pure travel time doesn't have
  gain      "how much": the fraction of the node's rain that reaches the
            gauge through each reservoir, fitted non-negative

Two ways to tie the gains down, both fitted at one gauge and then used
unchanged at the others — the graph does the transfer:

  per-node    one gain per node and tau (flexible, gauge-specific)
  land-cover  gain_i,tau = sum_c frac_i,c * gamma_c,tau, i.e. one runoff
              fraction per land-cover class: transferable by construction

Split matches train.py: first 70% train, next 15% val (alpha selection),
last 15% test. Scores are NSE from rain alone — no observed flow is used as
an input, which is what an ungauged prediction needs. The per-gauge CNN
(12h ahead, fed observed flow) is listed for reference but is a different
task: it has flow persistence, the graph model doesn't.

modpods (optional): `--modpods` also runs modpods' data-driven topology
inference on the rain nodes and gauges, and checks the edges it picks
against the DEM graph (is the inferred rain node really upstream of that
gauge?) and its lead/lag against the DEM travel time.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .basin_graph import GRAPH_DIR
from .fetch import DATA_DIR, fetch_openmeteo_hourly, fetch_usgs_hourly
from .sites import BY_ID

MM_H_KM2_TO_M3S = 1e-3 * 1e6 / 3600       # 1 mm/h over 1 km² in m³/s
TAUS_H = (6, 24, 96, 384)
ALPHAS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
MIN_CLASS_SHARE = 0.02    # land-cover classes below this share fold into 'other'
YEARS = 3


def load_graph(outlet_id: str) -> dict:
    return json.loads((GRAPH_DIR / f'{outlet_id}.json').read_text())


def _window():
    end = pd.Timestamp.utcnow().normalize().tz_localize(None)
    start = end - pd.DateOffset(years=YEARS)
    return start, end


def load_rain(graph: dict) -> pd.DataFrame:
    """Hourly precipitation (mm) per node centroid, cached per outlet."""
    path = DATA_DIR / 'graph' / graph['outlet'] / 'rain.parquet'
    if path.exists():
        return pd.read_parquet(path)
    start, end = _window()
    cols = {}
    for n in graph['nodes']:
        lon, lat = n['centroid_lonlat']
        df = fetch_openmeteo_hourly(lat, lon, start.strftime('%Y-%m-%d'),
                                    (end - pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
        cols[str(n['id'])] = df['precipitation']
        print(f'  rain node {n["id"]:>3} ({lat:.3f}, {lon:.3f})  '
              f'{df["precipitation"].sum():7.0f} mm')
        time.sleep(0.2)
    rain = pd.DataFrame(cols).sort_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    rain.to_parquet(path)
    return rain


def load_flows(graph: dict) -> pd.DataFrame:
    """Hourly flow (m³/s) per gauge: the fetch.py cache if present, else USGS."""
    start, end = _window()
    cols = {}
    for g in graph['gauges']:
        cached = DATA_DIR / g['id'] / 'hourly.parquet'
        if cached.exists():
            cols[g['id']] = pd.read_parquet(cached)['flow_m3s']
        else:
            q = fetch_usgs_hourly(g['id'], start.strftime('%Y-%m-%d'),
                                  (end - pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
            cols[g['id']] = q.interpolate(limit=4, limit_area='inside')
        print(f'  flow {g["id"]} {g["short"]:<18} {cols[g["id"]].notna().sum():6d} h')
    return pd.DataFrame(cols).sort_index()


# --------------------------------------------------------------------------
# Kernels + features
# --------------------------------------------------------------------------

def stretch(hist: np.ndarray, alpha: float) -> np.ndarray:
    """Travel-time histogram with every travel time multiplied by alpha
    (alpha > 1 = slower water), via its CDF."""
    n = len(hist)
    cdf = np.concatenate([[0.0], np.cumsum(hist)])
    edges = np.arange(n + 1, dtype=float)
    m = int(np.ceil(n * alpha))
    new_edges = np.arange(m + 1, dtype=float) / alpha
    new_cdf = np.interp(new_edges, edges, cdf)
    return np.diff(new_cdf)


def reservoir(tau_h: float) -> np.ndarray:
    t = np.arange(int(5 * tau_h) + 1)
    k = np.exp(-t / tau_h)
    return k / k.sum()


def _conv(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    n = len(x)
    size = 1 << int(np.ceil(np.log2(n + len(k))))
    return np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(k, size), size)[:n]


def node_features(graph: dict, rain: pd.DataFrame, gauge_id: str,
                  alpha: float) -> tuple[list[int], np.ndarray]:
    """Feature matrix (T, n_nodes * n_taus) for one gauge: each upstream
    node's rain, as m³/s, routed by its DEM kernel then each reservoir."""
    ids, feats = [], []
    for n in graph['nodes']:
        k = n['travel_time_to'].get(gauge_id)
        if k is None:
            continue
        p = rain[str(n['id'])].fillna(0).to_numpy() * n['local_area_km2'] * MM_H_KM2_TO_M3S
        routed = _conv(p, stretch(np.asarray(k['hist']), alpha))
        ids.append(n['id'])
        feats.append([_conv(routed, reservoir(tau)) for tau in TAUS_H])
    x = np.asarray(feats)                           # (nodes, taus, T)
    return ids, x.transpose(2, 0, 1).reshape(x.shape[2], -1)


def landcover_matrix(graph: dict, ids: list[int]) -> tuple[list[str], np.ndarray]:
    """Land-cover fractions (nodes x classes). Classes come from the whole
    graph so the columns line up at every gauge, not just the fit gauge."""
    by_id = {n['id']: n for n in graph['nodes']}
    area = {}
    for n in graph['nodes']:
        for c, f in n['landcover'].items():
            area[c] = area.get(c, 0.0) + f * n['local_area_km2']
    total = sum(area.values())
    # A class covering a sliver of the basin can't be identified from flow;
    # its gain just soaks up noise. Fold those into 'other'.
    group = {c: (c if a / total >= MIN_CLASS_SHARE else 'other') for c, a in area.items()}
    classes = sorted(set(group.values()))
    mat = np.zeros((len(ids), len(classes)))
    for row, i in enumerate(ids):
        for c, f in by_id[i]['landcover'].items():
            mat[row, classes.index(group[c])] += f
    return classes, mat


# --------------------------------------------------------------------------
# Fit + score
# --------------------------------------------------------------------------

def nse(sim, obs):
    m = np.isfinite(sim) & np.isfinite(obs)
    s, o = sim[m], obs[m]
    return float(1 - ((s - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()) if m.sum() > 10 else float('nan')


def _nnls_fit(x, y):
    """Non-negative gains + free intercept: centre, NNLS, recover b."""
    from scipy.optimize import nnls
    xm, ym = x.mean(axis=0), y.mean()
    coef, _ = nnls(x - xm, y - ym, maxiter=50 * x.shape[1])
    return coef, ym - xm @ coef


def splits(n: int, warmup: int):
    i_tr, i_va = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    return slice(warmup, i_tr), slice(i_tr, i_va), slice(i_va, n)


def fit_outlet(graph, rain, flows, fit_gauge: str, mode: str):
    """Fit gains at one gauge (alpha picked on val), then predict every gauge
    in the graph with those same gains. Returns a result dict."""
    q = flows[fit_gauge].reindex(rain.index).to_numpy()
    warmup = int(5 * max(TAUS_H))
    tr, va, te = splits(len(rain), warmup)
    best = None
    for alpha in ALPHAS:
        ids, x = node_features(graph, rain, fit_gauge, alpha)
        n_t = len(TAUS_H)
        if mode == 'landcover':
            classes, frac = landcover_matrix(graph, ids)
            # x[:, node*n_t + tau] -> z[:, class*n_t + tau] = sum_i frac_ic x_i,tau
            z = np.einsum('tnk,nc->tck', x.reshape(len(x), len(ids), n_t), frac)
            design = z.reshape(len(x), -1)
        else:
            design = x
        m = np.isfinite(q)
        mtr = np.zeros(len(q), bool); mtr[tr] = True; mtr &= m
        coef, b = _nnls_fit(design[mtr], q[mtr])
        sim = design @ coef + b
        score = nse(sim[va], q[va])
        if best is None or score > best['val_nse']:
            best = {'alpha': alpha, 'coef': coef, 'b': b, 'val_nse': score, 'ids': ids}
    out = {'fit_gauge': fit_gauge, 'mode': mode, 'alpha': best['alpha'],
           'val_nse': round(best['val_nse'], 3), 'test_nse': {}}

    # Gains per node (per-node mode) or per class (land-cover mode), then
    # applied unchanged at every gauge through that gauge's own DEM kernels.
    n_t = len(TAUS_H)
    if mode == 'landcover':
        classes, _ = landcover_matrix(graph, best['ids'])
        gamma = best['coef'].reshape(len(classes), n_t)
        out['runoff_fraction_by_class'] = {c: round(float(g.sum()), 3)
                                           for c, g in zip(classes, gamma)}
    else:
        gains = best['coef'].reshape(len(best['ids']), n_t)
        out['runoff_fraction_by_node'] = {int(i): round(float(g.sum()), 3)
                                          for i, g in zip(best['ids'], gains)}
    out['series'] = {}
    for g in graph['gauges']:
        ids, x = node_features(graph, rain, g['id'], best['alpha'])
        if mode == 'landcover':
            _, frac = landcover_matrix(graph, ids)
            node_gain = frac @ gamma                       # (nodes, taus)
        else:
            lookup = dict(zip(best['ids'], gains))
            node_gain = np.array([lookup.get(i, np.zeros(n_t)) for i in ids])
        sim = x @ node_gain.reshape(-1)
        # Intercept (baseflow) scaled by drainage area from the fit gauge.
        area_ratio = g['dem_area_km2'] / next(
            gg['dem_area_km2'] for gg in graph['gauges'] if gg['id'] == fit_gauge)
        sim = sim + best['b'] * area_ratio
        obs = flows[g['id']].reindex(rain.index).to_numpy()
        out['test_nse'][g['id']] = round(nse(sim[te], obs[te]), 3)
        out['series'][g['id']] = (sim, obs)
    out['test_index'] = rain.index[te]
    return out


def cnn_reference(graph) -> dict:
    """Per-gauge CNN test NSE from its checkpoint, if trained."""
    import torch
    ref = {}
    for g in graph['gauges']:
        p = Path(__file__).resolve().parent / 'checkpoints' / f'{g["id"]}.pt'
        if p.exists():
            ref[g['id']] = round(torch.load(p, weights_only=False)['metrics']['nse_overall'], 3)
    return ref


def run_modpods(graph, rain, flows, max_iter: int = 100) -> dict:
    """Data-driven topology on the train split, scored against the DEM graph."""
    import modpods
    n = len(rain)
    tr = slice(0, int(n * TRAIN_FRAC))
    # One column per distinct rain series — neighbouring nodes can share an
    # Open-Meteo grid cell, and identical inputs make the edges unidentifiable.
    seen, rain_cols = {}, []
    for c in rain.columns:
        key = rain[c].fillna(0).round(3).to_numpy().tobytes()
        if key not in seen:
            seen[key] = c
            rain_cols.append(c)
    gauges = [g['id'] for g in graph['gauges']]
    df = pd.concat([rain[rain_cols].fillna(0).add_prefix('rain_'),
                    flows[gauges].reindex(rain.index)], axis=1).iloc[tr]
    df = df.interpolate(limit_direction='both').reset_index(drop=True)
    t0 = time.time()
    res = modpods.infer_causative_topology(
        df, dependent_columns=gauges,
        independent_columns=[f'rain_{c}' for c in rain_cols], max_iter=max_iter)
    by_id = {str(nd['id']): nd for nd in graph['nodes']}
    rows = []
    for g in gauges:
        for src in df.columns:
            if src == g or res['edges'].loc[src, g] != 1:
                continue
            row = {'to': g, 'from': src, 'r2': round(float(res['r2_values'].loc[g, src]), 3),
                   'lead_h': res['lead_lag'].loc[g, src]}
            if src.startswith('rain_'):
                k = by_id[src[5:]]['travel_time_to'].get(g)
                row['upstream_in_dem'] = k is not None
                row['dem_mean_h'] = k['mean_h'] if k else None
            rows.append(row)
    return {'seconds': round(time.time() - t0), 'edges': rows,
            'n_rain_inputs': len(rain_cols)}


def main(outlet_id: str, with_modpods: bool = False):
    graph = load_graph(outlet_id)
    print(f'Graph above {outlet_id}: {len(graph["nodes"])} nodes, {len(graph["gauges"])} gauges')
    rain = load_rain(graph)
    flows = load_flows(graph)
    rain = rain.loc[rain.index.intersection(flows.index)]
    shorts = {g['id']: g['short'] for g in graph['gauges']}
    results = [fit_outlet(graph, rain, flows, outlet_id, m) for m in ('per-node', 'landcover')]
    cnn = cnn_reference(graph)

    print(f'\nTest NSE from rain alone (fit at {shorts[outlet_id]} only, applied to all):')
    print(f'{"gauge":<18}' + ''.join(f'{r["mode"]:>12}' for r in results) + f'{"CNN 12h*":>10}')
    for g in graph['gauges']:
        print(f'{g["short"]:<18}' + ''.join(f'{r["test_nse"][g["id"]]:>12.3f}' for r in results)
              + f'{cnn.get(g["id"], float("nan")):>10.3f}')
    print('  * CNN uses observed flow as input; different task, shown for scale.')
    for r in results:
        print(f'\n{r["mode"]}: alpha={r["alpha"]} (DEM travel times x alpha), val NSE {r["val_nse"]}')
        fr = r.get('runoff_fraction_by_class') or r.get('runoff_fraction_by_node')
        print('  runoff fraction:', fr)

    summary = {'outlet': outlet_id, 'cnn_test_nse': cnn,
               'fits': [{k: v for k, v in r.items() if k not in ('series', 'test_index')}
                        for r in results]}
    if with_modpods:
        print('\nmodpods topology inference (train split)...', flush=True)
        mp = run_modpods(graph, rain, flows)
        summary['modpods'] = mp
        rain_edges = [e for e in mp['edges'] if 'upstream_in_dem' in e]
        ok = sum(e['upstream_in_dem'] for e in rain_edges)
        print(f'  {mp["seconds"]}s, {mp["n_rain_inputs"]} distinct rain inputs; '
              f'{ok}/{len(rain_edges)} inferred rain->gauge edges are upstream in the DEM graph')
        for e in mp['edges']:
            print('  ', e)
    out = Path(__file__).resolve().parents[1] / 'outputs' / 'graph' / outlet_id
    out.mkdir(parents=True, exist_ok=True)
    (out / 'fit_summary.json').write_text(json.dumps(summary, indent=1, default=str))
    np.savez_compressed(out / 'test_series.npz', **{
        f'{r["mode"]}__{gid}': np.stack(r['series'][gid])[:, -len(r['test_index']):]
        for r in results for gid in r['series']},
        test_index=np.array(results[0]['test_index'].astype('int64')))
    print(f'\nwrote {out}/fit_summary.json')
    return summary


if __name__ == '__main__':
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    main(args[0] if args else '05412500', with_modpods='--modpods' in sys.argv)
