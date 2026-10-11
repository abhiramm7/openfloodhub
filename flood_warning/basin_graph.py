"""River-network graph above an outlet gauge, from a DEM (pysheds) plus land
cover. Research path for the rainfall -> flow graph model; the live forecast
(predict.py) does not use it.

    python -m flood_warning.basin_graph 05412500     # Turkey River at Garber

Pipeline:
  1. USGS 3DEP 1-arc-second DEM tiles (public S3) -> mosaic -> UTM 15N, 30 m
  2. ESA WorldCover 2021 (10 m, public S3) -> same grid, mode resampling.
     (NLCD would be the US-native choice, but MRLC's servers aren't reachable
     from every environment; WorldCover's classes cover what we need here.)
  3. pysheds: fill pits + depressions, resolve flats, D8 flow direction,
     flow accumulation
  4. Snap every registered USGS gauge inside the box to the channel cell
     whose accumulated area best matches its published drainage area. The
     remaining area mismatch is the delineation check: a big one means the
     DEM routes water differently from reality (tile drains, karst, a bad
     snap) and that gauge's graph shouldn't be trusted.
  5. Split the channel network (accumulated area >= node_area_km2) into
     reaches at confluences and at gauges. Every cell joins the reach it
     first drains into, so each graph node is one reach plus its local
     subcatchment, and each edge points downstream.
  6. Per-cell travel time ("how fast"): overland v = k * sqrt(slope) with k
     by land cover (NRCS NEH 630 ch. 15 velocity method); channel cells use
     v = V0 * (A / 1 km²)^0.1. Summed down the D8 path, this gives every
     cell's travel time to every gauge below it, and per node an hourly
     travel-time histogram to each downstream gauge. That histogram is a
     physics-prior convolution kernel for that node's rain at that gauge.

Writes flood_warning/graphs/<outlet>.json (nodes, edges, gauges, kernels;
committed, like thresholds.json) and caches rasters under
$FLOOD_DATA_DIR/graph/<outlet>/. Extra dependencies: requirements-graph.txt.
"""
from __future__ import annotations

import json
import math
import shutil
import urllib.request
from pathlib import Path

import numpy as np

from .fetch import DATA_DIR
from .sites import BY_ID, SITES

GRAPH_DIR = Path(__file__).resolve().parent / 'graphs'
CRS = 'EPSG:26915'          # NAD83 / UTM 15N — covers Iowa
RES = 30.0                  # m
CELL_KM2 = RES * RES / 1e6

DEM_URL = ('https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1/TIFF/'
           'current/{t}/USGS_1_{t}.tif')
WORLDCOVER_URL = ('https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/'
                  '2021/map/ESA_WorldCover_10m_2021_v200_{t}_Map.tif')

# WorldCover class -> (label, NEH 630 ch. 15 velocity coefficient k in ft/s
# for v = k * sqrt(slope), slope in ft/ft).
LANDCOVER = {
    10: ('tree', 2.5),        # forest with heavy ground litter
    20: ('shrub', 5.0),       # woodland / minimum tillage
    30: ('grass', 7.0),       # short-grass pasture, hay
    40: ('crop', 9.0),        # cultivated straight row
    50: ('built', 20.3),      # paved area, small upland gullies
    60: ('bare', 10.0),       # nearly bare and untilled
    80: ('water', 7.0),       # open water: overland k unused (channel-ish)
    90: ('wetland', 2.5),     # herbaceous wetland: slow, like heavy litter
}
K_DEFAULT = 7.0
FT = 0.3048

CHANNEL_AREA_KM2 = 1.0      # channel initiation for the velocity model
V0_CHANNEL = 0.5            # m/s at 1 km²; grows as A^0.1 (~1.1 m/s at 4,000 km²)
MIN_SLOPE = 0.001
FLAT_EPS = 1e-7             # m per step when pysheds drains flats
KERNEL_HOURS = 120

# D8 offsets in pysheds' default dirmap order: N, NE, E, SE, S, SW, W, NW.
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)
OFFSETS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


# --------------------------------------------------------------------------
# Inputs: DEM + land cover on one UTM grid
# --------------------------------------------------------------------------

def _download(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + '.part')
        with urllib.request.urlopen(url, timeout=300) as r, open(tmp, 'wb') as f:
            shutil.copyfileobj(r, f)
        tmp.rename(path)
    return path


def _dem_tiles(bbox):
    """3DEP 1-arc-second tile names covering (lon0, lat0, lon1, lat1). A tile
    nYYwXXX spans latitudes YY-1..YY and longitudes -XXX..-XXX+1."""
    lon0, lat0, lon1, lat1 = bbox
    return [f'n{la:02d}w{lo:03d}'
            for la in range(math.floor(lat0) + 1, math.floor(lat1) + 2)
            for lo in range(-math.floor(lon1), -math.floor(lon0) + 1)]


def _worldcover_tiles(bbox):
    """WorldCover 3°x3° tiles, named by their SW corner."""
    lon0, lat0, lon1, lat1 = bbox
    out = []
    for la in range(math.floor(lat0 / 3) * 3, math.floor(lat1 / 3) * 3 + 1, 3):
        for lo in range(math.floor(lon0 / 3) * 3, math.floor(lon1 / 3) * 3 + 1, 3):
            out.append(f'{"N" if la >= 0 else "S"}{abs(la):02d}'
                       f'{"E" if lo >= 0 else "W"}{abs(lo):03d}')
    return out


def build_inputs(bbox, cache: Path) -> tuple[Path, np.ndarray]:
    """DEM GeoTIFF on the UTM grid (path, for pysheds) + land-cover array
    aligned to it."""
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    tiles_dir = DATA_DIR / 'graph' / 'tiles'
    dem_paths = [_download(DEM_URL.format(t=t), tiles_dir / f'USGS_1_{t}.tif')
                 for t in _dem_tiles(bbox)]
    srcs = [rasterio.open(p) for p in dem_paths]
    mosaic, mosaic_tf = merge(srcs, bounds=bbox, nodata=-9999.0)
    src_crs = srcs[0].crs
    for s in srcs:
        s.close()
    mosaic = mosaic[0]

    h, w = mosaic.shape
    tf, width, height = calculate_default_transform(
        src_crs, CRS, w, h, *bbox, resolution=RES)
    # float64: resolve_flats drains flats with 1e-7 m steps, which float32
    # can't represent at a few hundred metres of elevation.
    dem = np.full((height, width), -9999.0, dtype=np.float64)
    reproject(mosaic, dem, src_transform=mosaic_tf, src_crs=src_crs,
              src_nodata=-9999.0, dst_transform=tf, dst_crs=CRS,
              dst_nodata=-9999.0, resampling=Resampling.bilinear)
    dem_path = cache / 'dem_utm.tif'
    with rasterio.open(dem_path, 'w', driver='GTiff', height=height, width=width,
                       count=1, dtype='float64', crs=CRS, transform=tf,
                       nodata=-9999.0) as dst:
        dst.write(dem, 1)

    lc = np.zeros((height, width), dtype=np.uint8)
    for t in _worldcover_tiles(bbox):
        p = _download(WORLDCOVER_URL.format(t=t), tiles_dir / f'worldcover_{t}.tif')
        with rasterio.open(p) as src:
            part = np.zeros_like(lc)
            reproject(rasterio.band(src, 1), part, dst_transform=tf, dst_crs=CRS,
                      dst_nodata=0, resampling=Resampling.mode)
        lc = np.where(lc == 0, part, lc)
    return dem_path, lc


# --------------------------------------------------------------------------
# Graph primitives on the flattened D8 tree
# --------------------------------------------------------------------------

def _down_index(fdir: np.ndarray) -> np.ndarray:
    """Flat index of each cell's D8 receiver, -1 for pits, flats, nodata and
    cells that drain off the grid."""
    h, w = fdir.shape
    rows, cols = np.indices((h, w))
    down = np.full(h * w, -1, dtype=np.int64)
    for code, (dr, dc) in zip(DIRMAP, OFFSETS):
        m = fdir == code
        r, c = rows[m] + dr, cols[m] + dc
        ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        idx = np.flatnonzero(m)
        down[idx[ok]] = r[ok] * w + c[ok]
    return down


def _first_hit(down: np.ndarray, stop: np.ndarray) -> np.ndarray:
    """For every cell, the first cell at or below it where `stop` is True, or
    -1 if its path ends first. Pointer jumping: O(log path length) passes."""
    n = down.size
    j = np.where(stop | (down < 0), np.arange(n), down)
    while True:
        jj = j[j]
        if np.array_equal(jj, j):
            break
        j = jj
    return np.where(stop[j], j, -1)


def _path_sum(down: np.ndarray, w: np.ndarray, root: np.ndarray) -> np.ndarray:
    """Sum of w along each cell's path, from the cell (inclusive) down to the
    first root cell (exclusive). Pointer-jumping list ranking."""
    n = down.size
    fixed = root | (down < 0)
    j = np.where(fixed, np.arange(n), down)
    t = np.where(fixed, 0.0, w)
    while True:
        jj = j[j]
        if np.array_equal(jj, j):
            break
        t = t + t[j]
        j = jj
    return t


def _snap(acc_km2, row, col, target_km2, radius_cells):
    """Cell within the radius whose accumulated area is closest to the
    published drainage area; ties go to the nearer cell."""
    h, w = acc_km2.shape
    r0, r1 = max(0, row - radius_cells), min(h, row + radius_cells + 1)
    c0, c1 = max(0, col - radius_cells), min(w, col + radius_cells + 1)
    win = acc_km2[r0:r1, c0:c1]
    rr, cc = np.indices(win.shape)
    dist = np.hypot(rr + r0 - row, cc + c0 - col)
    score = np.abs(win - target_km2) / target_km2 + 1e-4 * dist
    score[dist > radius_cells] = np.inf
    i, k = np.unravel_index(np.argmin(score), win.shape)
    return r0 + i, c0 + k


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def build_graph(outlet_id: str, node_area_km2: float = 100.0, pad_deg: float = 0.35,
                snap_radius_m: float = 600.0, max_tries: int = 4) -> dict:
    import rasterio
    from pysheds.grid import Grid
    from rasterio.warp import transform as warp_transform

    outlet = BY_ID[outlet_id]
    cache = DATA_DIR / 'graph' / outlet_id
    cache.mkdir(parents=True, exist_ok=True)
    region = [s for s in SITES if s['region'] == outlet['region']]

    for attempt in range(max_tries):
        lons = [s['lon'] for s in region]
        lats = [s['lat'] for s in region]
        pad = pad_deg + 0.25 * attempt
        bbox = (min(lons) - pad, min(lats) - pad, max(lons) + pad, max(lats) + pad)
        print(f'bbox {tuple(round(b, 2) for b in bbox)} (attempt {attempt + 1})')
        dem_path, lc = build_inputs(bbox, cache)

        grid = Grid.from_raster(str(dem_path))
        dem = grid.read_raster(str(dem_path))
        print('  conditioning DEM (pits, depressions, flats)...', flush=True)
        # pysheds' default eps (1e-5) tilts Iowa's broad filled flats by more
        # than the real relief at their edges, which plants thousands of new
        # pits (one drained >1,000 km² and cut the Turkey basin to a fifth of
        # its size). A smaller eps plus a few fill/resolve passes leaves only
        # a handful.
        cond = grid.resolve_flats(grid.fill_depressions(grid.fill_pits(dem)), eps=FLAT_EPS)
        fdir = grid.flowdir(cond, dirmap=DIRMAP)
        for _ in range(3):
            n_pits = int((np.asarray(fdir) == -2).sum())
            if n_pits == 0:
                break
            print(f'  {n_pits} pits left, another fill/resolve pass', flush=True)
            cond = grid.resolve_flats(grid.fill_depressions(cond), eps=FLAT_EPS)
            fdir = grid.flowdir(cond, dirmap=DIRMAP)
        acc = grid.accumulation(fdir, dirmap=DIRMAP)
        acc_km2 = np.asarray(acc, dtype=np.float64) * CELL_KM2
        fdir = np.asarray(fdir)
        h, w = fdir.shape

        with rasterio.open(dem_path) as src:
            tf = src.transform
        xs, ys = warp_transform('EPSG:4326', CRS, [s['lon'] for s in region],
                                [s['lat'] for s in region])
        gauges = []
        for s, x, y in zip(region, xs, ys):
            col, row = ~tf * (x, y)
            row, col = int(row), int(col)
            if not (0 <= row < h and 0 <= col < w):
                continue
            target = s['drainage_sqmi'] * 2.58999
            r, c = _snap(acc_km2, row, col, target, int(snap_radius_m / RES))
            gauges.append({
                'id': s['id'], 'short': s['short'],
                'usgs_area_km2': round(target, 1),
                'dem_area_km2': round(float(acc_km2[r, c]), 1),
                'area_error_pct': round(100 * (acc_km2[r, c] - target) / target, 1),
                'snap_dist_m': round(float(np.hypot(r - row, c - col)) * RES),
                'cell': (int(r), int(c)),
            })
        g_out = next(g for g in gauges if g['id'] == outlet_id)
        r, c = g_out['cell']
        out_cell = r * w + c
        down = _down_index(fdir)
        hit = np.zeros(h * w, dtype=bool)
        hit[out_cell] = True
        catch = (_first_hit(down, hit) == out_cell).reshape(h, w)
        # Clipped if the catchment reaches the edge of valid data — the UTM
        # grid has a nodata rim, so the grid border alone isn't the test.
        from scipy.ndimage import binary_dilation
        nodata = np.asarray(dem) == -9999.0
        nodata[[0, -1], :] = True
        nodata[:, [0, -1]] = True
        if not (binary_dilation(catch, iterations=2) & nodata).any():
            break
        print('  catchment touches the box edge — widening')
    else:
        raise RuntimeError('catchment still clipped after widening the box')

    flat_catch = catch.ravel()
    down[~flat_catch] = -1
    down[out_cell] = -1

    # ---- reaches + subcatchments -------------------------------------------
    acc_flat = acc_km2.ravel()
    gauges = [g for g in gauges if catch[g['cell']]]
    gauge_cell = {g['cell'][0] * w + g['cell'][1]: g['id'] for g in gauges}
    stream = flat_catch & (acc_flat >= node_area_km2)
    stream[list(gauge_cell)] = True       # small gauged creeks still get a node
    stream_down = np.where(stream, down, -1)
    s_idx = np.flatnonzero(stream & (stream_down >= 0))
    inflow = np.zeros(h * w, dtype=np.int32)
    np.add.at(inflow, stream_down[s_idx], 1)
    end = np.zeros(h * w, dtype=bool)
    end[s_idx[inflow[stream_down[s_idx]] >= 2]] = True   # last cell above a confluence
    end[list(gauge_cell)] = True
    end[out_cell] = True
    end &= stream
    # Any stream cell whose receiver isn't a stream cell is the bottom of its
    # path (only the outlet, if the masks agree) — close it as a reach end.
    end[np.flatnonzero(stream & (stream_down < 0))] = True

    reach_end = _first_hit(np.where(stream, down, -1), end)     # stream cells only
    end_cells = np.flatnonzero(end)
    rid = np.full(h * w, -1, dtype=np.int64)
    rid[end_cells] = np.arange(len(end_cells))
    reach_of = np.where(stream & (reach_end >= 0), rid[np.maximum(reach_end, 0)], -1)

    first_stream = _first_hit(down, stream)
    node = np.where(flat_catch & (first_stream >= 0),
                    reach_of[np.maximum(first_stream, 0)], -1)

    # ---- travel time ---------------------------------------------------------
    rows, cols = np.divmod(np.arange(h * w), w)
    dr = np.where(down >= 0, rows[np.maximum(down, 0)] - rows, 0)
    dc = np.where(down >= 0, cols[np.maximum(down, 0)] - cols, 0)
    length = np.hypot(dr, dc) * RES
    z = np.asarray(dem, dtype=np.float64).ravel()
    slope = np.where(down >= 0, (z - z[np.maximum(down, 0)]) / np.maximum(length, RES), 0)
    slope = np.clip(slope, MIN_SLOPE, None)
    lc_flat = lc.ravel()
    k = np.full(h * w, K_DEFAULT * FT)
    for code, (_, kk) in LANDCOVER.items():
        k[lc_flat == code] = kk * FT
    v = np.clip(k * np.sqrt(slope), 0.01, 3.0)
    chan = acc_flat >= CHANNEL_AREA_KM2
    v[chan] = V0_CHANNEL * (acc_flat[chan] / CHANNEL_AREA_KM2) ** 0.1
    hours = np.where(down >= 0, length / v / 3600.0, 0.0)
    root = np.zeros(h * w, dtype=bool)
    root[out_cell] = True
    t_out = _path_sum(down, hours, root)            # hours to the outlet

    # ---- per-node attributes ------------------------------------------------
    n_nodes = len(end_cells)
    end_xy = np.column_stack([cols[end_cells] + 0.5, rows[end_cells] + 0.5])
    labels = node[flat_catch]
    cells = np.flatnonzero(flat_catch)
    order = np.argsort(labels, kind='stable')
    bounds = np.searchsorted(labels[order], np.arange(n_nodes + 1))

    def lonlat(px, py):
        x, y = tf * (px, py)
        lon, lat = warp_transform(CRS, 'EPSG:4326', [x], [y])
        return round(lon[0], 4), round(lat[0], 4)

    # Gauge catchments in node terms: node -> the gauges it drains through.
    # A gauge on a creek below node_area_km2 is its own one-cell reach whose
    # receiver isn't a stream cell — follow down to the next stream cell.
    node_down = np.full(n_nodes, -1)
    for i, e in enumerate(end_cells):
        d = down[e]
        if d >= 0 and first_stream[d] >= 0:
            node_down[i] = reach_of[first_stream[d]]
    gauge_node = {gid: int(rid[cell]) for cell, gid in gauge_cell.items()}
    gauge_t = {gid: t_out[cell] for cell, gid in gauge_cell.items()}

    def downstream_gauges(i):
        out, j = [], i
        while j >= 0:
            out += [gid for gid, gn in gauge_node.items() if gn == j]
            j = node_down[j]
        return out

    nodes = []
    for i in range(n_nodes):
        mem = cells[order[bounds[i]:bounds[i + 1]]]
        if not len(mem):
            continue
        r_, c_ = np.divmod(mem, w)
        frac = {}
        for code, n in zip(*np.unique(lc_flat[mem], return_counts=True)):
            name = LANDCOVER.get(int(code), ('other',))[0]
            frac[name] = frac.get(name, 0) + n / len(mem)
        reach_len = float(length[reach_of == i].sum()) / 1000
        kernels = {}
        for gid in downstream_gauges(i):
            tt = t_out[mem] - gauge_t[gid]
            hist = np.bincount(np.clip(tt, 0, KERNEL_HOURS - 1).astype(int),
                               minlength=KERNEL_HOURS)[:KERNEL_HOURS] / len(mem)
            kernels[gid] = {
                'mean_h': round(float(tt.mean()), 2),
                'p10_h': round(float(np.percentile(tt, 10)), 2),
                'p90_h': round(float(np.percentile(tt, 90)), 2),
                'hist': [round(float(x), 5) for x in hist],
            }
        nodes.append({
            'id': i,
            'down': int(node_down[i]),
            'gauge': next((gid for gid, gn in gauge_node.items() if gn == i), None),
            'local_area_km2': round(len(mem) * CELL_KM2, 2),
            'upstream_area_km2': round(float(acc_flat[end_cells[i]]), 1),
            'reach_length_km': round(reach_len, 2),
            'outlet_lonlat': lonlat(*end_xy[i]),
            'centroid_lonlat': lonlat(c_.mean() + 0.5, r_.mean() + 0.5),
            'mean_slope': round(float(slope[mem].mean()), 4),
            'landcover': {k_: round(v_, 3) for k_, v_ in sorted(frac.items())},
            'travel_time_to': kernels,
        })

    for g in gauges:
        g['node'] = gauge_node[g['id']]
        g['lonlat'] = lonlat(g['cell'][1] + 0.5, g['cell'][0] + 0.5)
        del g['cell']

    np.savez_compressed(cache / 'rasters.npz', node=node.reshape(h, w).astype(np.int32),
                        t_out=t_out.reshape(h, w).astype(np.float32),
                        acc_km2=acc_km2.astype(np.float32), lc=lc,
                        catch=catch, fdir=fdir.astype(np.int16),
                        transform=np.array(tf)[:6])
    graph = {
        'outlet': outlet_id,
        'crs': CRS, 'res_m': RES,
        'params': {'node_area_km2': node_area_km2, 'channel_area_km2': CHANNEL_AREA_KM2,
                   'v0_channel_ms': V0_CHANNEL, 'min_slope': MIN_SLOPE,
                   'landcover_k_fts': {LANDCOVER[c][0]: LANDCOVER[c][1] for c in LANDCOVER},
                   'kernel_hours': KERNEL_HOURS},
        'catchment_area_km2': round(float(catch.sum()) * CELL_KM2, 1),
        'gauges': gauges,
        'nodes': nodes,
    }
    GRAPH_DIR.mkdir(exist_ok=True)
    path = GRAPH_DIR / f'{outlet_id}.json'
    path.write_text(json.dumps(graph, indent=1) + '\n')
    print(f'wrote {path} ({len(nodes)} nodes, {len(gauges)} gauges)')
    return graph


def report(graph: dict):
    print(f'\nCatchment above {graph["outlet"]}: {graph["catchment_area_km2"]:,.0f} km²')
    print(f'{"gauge":<18}{"USGS km²":>10}{"DEM km²":>10}{"error":>8}{"snap m":>8}')
    for g in graph['gauges']:
        print(f'{g["short"]:<18}{g["usgs_area_km2"]:>10,.0f}{g["dem_area_km2"]:>10,.0f}'
              f'{g["area_error_pct"]:>7.1f}%{g["snap_dist_m"]:>8}')
    out = graph['outlet']
    tt = [n['travel_time_to'][out]['mean_h'] for n in graph['nodes'] if out in n['travel_time_to']]
    print(f'\n{len(graph["nodes"])} nodes; mean travel time to outlet per node: '
          f'{min(tt):.0f}–{max(tt):.0f} h')


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

INK, INK_2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SEQ_BLUE = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
# Land-cover groups in fixed categorical order (reference palette slots 1-5).
LC_GROUPS = [('crop', ('crop',), '#2a78d6'), ('grass / pasture', ('grass',), '#eb6834'),
             ('tree', ('tree',), '#1baf7a'), ('built', ('built',), '#eda100'),
             ('other', ('shrub', 'bare', 'water', 'wetland', 'other'), '#e87ba4')]


def plot_graph(outlet_id: str):
    """Travel-time map with subcatchments, channels and gauges, plus the
    time-area histogram to the outlet split by land cover."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap

    graph = json.loads((GRAPH_DIR / f'{outlet_id}.json').read_text())
    r = np.load(DATA_DIR / 'graph' / outlet_id / 'rasters.npz')
    node, t_out, acc, lc, catch = r['node'], r['t_out'], r['acc_km2'], r['lc'], r['catch']
    fdir = r['fdir']
    a, b_, c, d, e, f = r['transform']
    out_dir = Path(__file__).resolve().parents[1] / 'outputs' / 'graph' / outlet_id
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.edgecolor': INK_2, 'axes.labelcolor': INK_2,
                         'xtick.color': INK_2, 'ytick.color': INK_2})

    rows, cols = np.nonzero(catch)
    r0, r1, c0, c1 = rows.min() - 5, rows.max() + 6, cols.min() - 5, cols.max() + 6
    sub = (slice(r0, r1), slice(c0, c1))
    tt = np.where(catch, t_out, np.nan)[sub]
    km = lambda col, row: ((c + a * col) / 1000, (f + e * row) / 1000)
    x0, y0 = km(c0, r1)
    x1, y1 = km(c1, r0)

    fig, ax = plt.subplots(figsize=(8, 7.2), dpi=150)
    cmap = LinearSegmentedColormap.from_list('seq', SEQ_BLUE)
    im = ax.imshow(tt, cmap=cmap, extent=(x0, x1, y0, y1), interpolation='nearest')
    nd = np.where(catch, node, -1)[sub].astype(float)
    edge = np.zeros_like(nd, dtype=bool)
    edge[:, 1:] |= nd[:, 1:] != nd[:, :-1]
    edge[1:, :] |= nd[1:, :] != nd[:-1, :]
    ax.imshow(np.ma.masked_where(~edge, edge), cmap=LinearSegmentedColormap.from_list(
        'e', ['#ffffff', '#ffffff']), extent=(x0, x1, y0, y1), interpolation='nearest', alpha=0.9)

    # Channels as segments from each cell to its D8 receiver, width by area.
    ch = catch & (acc >= 25)
    rr, cc = np.nonzero(ch)
    step = dict(zip(DIRMAP, OFFSETS))
    segs, widths = [], []
    for row, col in zip(rr, cc):
        if int(fdir[row, col]) not in step:
            continue
        dr, dc = step[int(fdir[row, col])]
        segs.append([km(col + .5, row + .5), km(col + dc + .5, row + dr + .5)])
        widths.append(0.5 + 0.9 * np.log10(acc[row, col] / 25))
    ax.add_collection(LineCollection(segs, colors=INK, linewidths=widths, capstyle='round'))

    from rasterio.warp import transform as wt
    # Labels: first of four offsets whose (approximate) box clears the boxes
    # already placed, so neighbouring gauges like Garber/Littleport don't stack.
    km_per_pt = (x1 - x0) / (fig.get_size_inches()[0] * 72 * 0.8)
    placed = []
    for g in sorted(graph['gauges'], key=lambda g: -g['usgs_area_km2']):
        gx, gy = wt('EPSG:4326', CRS, [g['lonlat'][0]], [g['lonlat'][1]])
        px, py = gx[0] / 1000, gy[0] / 1000
        ax.plot(px, py, 'o', ms=8, mfc='white', mec=INK, mew=1.6, zorder=5)
        wdt, hgt = len(g['short']) * 5.2 * km_per_pt, 11 * km_per_pt
        for ox, oy, ha in ((6, 5, 'left'), (6, -14, 'left'), (-6, 5, 'right'), (-6, -14, 'right')):
            bx = px + ox * km_per_pt - (wdt if ha == 'right' else 0)
            by = py + oy * km_per_pt
            box = (bx, by, bx + wdt, by + hgt)
            if not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3]
                       for b in placed):
                break
        placed.append(box)
        ax.annotate(g['short'], (px, py), xytext=(ox, oy), ha=ha,
                    textcoords='offset points', color=INK, fontsize=8.5, zorder=6,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xlabel('UTM 15N easting (km)'); ax.set_ylabel('northing (km)')
    ax.set_title(f'Travel time to {BY_ID[outlet_id]["name"]} (h), '
                 f'{len(graph["nodes"])} subcatchment nodes', color=INK, loc='left')
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label('hours to outlet', color=INK_2); cb.outline.set_visible(False)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / 'travel_time_map.png'); plt.close(fig)

    # Time-area histogram by land-cover group.
    names = {code: name for code, (name, _) in LANDCOVER.items()}
    t = t_out[catch]; l = lc[catch]
    bins = np.arange(0, np.ceil(t.max()) + 2)
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    stack, labels, colors = [], [], []
    for label, members, color in LC_GROUPS:
        m = np.isin([names.get(int(v), 'other') for v in np.unique(l)], members)
        codes = np.unique(l)[m]
        h_, _ = np.histogram(t[np.isin(l, codes)], bins=bins)
        stack.append(h_ * CELL_KM2); labels.append(label); colors.append(color)
    ax.stackplot(bins[:-1], *stack, labels=labels, colors=colors, step='post',
                 edgecolor='white', linewidth=0.6)
    ax.set_xlim(0, bins[-1]); ax.set_xlabel('travel time to outlet (h)')
    ax.set_ylabel('area per hour (km²)')
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s_ in ('top', 'right', 'left'):
        ax.spines[s_].set_visible(False)
    ax.legend(frameon=False, ncol=5, loc='upper left', bbox_to_anchor=(0, 1.13),
              labelcolor=INK)
    ax.set_title('Where the outlet\'s water comes from, by arrival time',
                 color=INK, loc='left', pad=26)
    fig.tight_layout()
    fig.savefig(out_dir / 'time_area.png'); plt.close(fig)
    print(f'wrote {out_dir}/travel_time_map.png, time_area.png')


if __name__ == '__main__':
    import sys
    oid = sys.argv[1] if len(sys.argv) > 1 else '05412500'
    if '--plot-only' not in sys.argv:
        report(build_graph(oid))
    plot_graph(oid)
