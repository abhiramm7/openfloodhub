"""Sparse identification of reach routing.

Cross-correlation cannot recover a travel time: it returns translation plus
attenuation lumped together, biased by the reach's storage constant (see
tests/test_identification.py). Separating them needs a model that carries
both terms at once, which is what sparse regression over a delay library
does.

The reach that generated the data is a pure delay followed by a linear
reservoir:

    dS/dt = Q_up(t - tau) - S/k        arrival = S/k

so the arrival obeys

    d(arrival)/dt = (1/k) * Q_up(t - tau) - (1/k) * arrival

A downstream basin's inflow is the sum of its arrivals plus local runoff.
The identification therefore has to select *which* lag enters, and read the
storage constant off the inflow coefficient at the same time.

Putting every candidate lag in one regression does not work: adjacent lags
of a smooth signal are near-collinear, so least squares spreads weight
across a band of them with alternating signs and the inflow term gets
swamped. Scanning one lag at a time is well-posed -- for each candidate the
model has two physical coefficients, and the residual picks the lag. The
recovered pair should satisfy a = -b = 1/k, which is a consistency check
the data can fail, and therefore worth reporting.
"""
import numpy as np



def _rain_basis(rain, dt, timescales=(600.0, 1800.0, 5400.0)):
    """Smooth kernels standing in for the unmodelled local runoff response.

    Local runoff enters a downstream basin's inflow alongside the routed
    water. Left out, it lands in the residual and biases the lag; modelled
    as raw lagged rainfall it needs too many collinear columns. Exponential
    kernels at a few timescales span the plausible catchment responses in a
    handful of well-conditioned terms.
    """
    cols = []
    for k in timescales:
        alpha = dt / k
        y = np.zeros_like(rain)
        acc = 0.0
        for i, r in enumerate(rain):
            acc += alpha * (r - acc)
            y[i] = acc
        cols.append(y)
        cols.append(np.gradient(y, dt))
    return cols


def _design(inflow, lagged_upstreams, rain_cols):
    cols = list(lagged_upstreams) + [inflow] + list(rain_cols)
    cols.append(np.ones_like(inflow))
    return np.column_stack(cols)


def _fit(theta, y):
    coef, *_ = np.linalg.lstsq(theta, y, rcond=None)
    resid = float(np.sum((theta @ coef - y) ** 2))
    return coef, resid


def scan_lags_discrete(inflow, upstream_flows, rain, dt, max_lag_s=1800.0,
                       mask=None, burn_in=60):
    """Lag scan in discrete form -- the model the planner actually rolls out.

    The reach is a discrete linear reservoir, so fitting

        y[t+1] = c0 * y[t] + sum_i c_i * Q_i[t - L_i] + (runoff terms)

    matches the data-generating process exactly. The continuous-derivative
    version below has to approximate dy/dt by finite differences and then
    invert a coefficient to get k, which biases the storage constant even
    when the lag is exact. Here k falls straight out of the pole:
    ``c0 = 1 / (1 + dt/k)``.
    """
    names = list(upstream_flows)
    lags = list(range(0, int(max_lag_s / dt) + 1))
    rain_cols = _rain_basis(rain, dt)
    y_next = np.roll(inflow, -1)

    keep = np.zeros(len(inflow), dtype=bool)
    keep[burn_in:len(inflow) - burn_in - 1] = True
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)

    def evaluate(combo):
        lagged = [_lagged(upstream_flows[n], l) for n, l in zip(names, combo)]
        theta = _design(inflow, lagged, rain_cols)
        return _fit(theta[keep], y_next[keep])

    combos = ([(l,) for l in lags] if len(names) == 1
              else [(a, b) for a in lags for b in lags])
    if len(names) > 2:
        raise NotImplementedError("scan supports at most two upstreams")

    best = None
    for combo in combos:
        coef, resid = evaluate(combo)
        if best is None or resid < best[1]:
            best = (combo, resid, coef)

    combo, resid, coef = best
    c0 = float(coef[len(names)])
    k = dt / (1.0 / c0 - 1.0) if 0.0 < c0 < 1.0 else np.nan
    out = {}
    for i, n in enumerate(names):
        out[n] = {"travel_time": combo[i] * dt, "a": float(coef[i]),
                  "b": c0, "k_from_a": np.nan, "k_from_b": k,
                  "k": k, "residual": resid}
    return out


def scan_lags(inflow, upstream_flows, rain, dt, max_lag_s=1800.0,
              mask=None, burn_in=60):
    """Find the lag for each upstream by exhaustive scan.

    One upstream is a 1-D scan; two is a 2-D scan over the pair, which is
    small enough to do exhaustively and avoids the coordinate-descent local
    minima that a greedy search falls into when two reaches into the same
    basin have similar travel times.
    """
    names = list(upstream_flows)
    lags = list(range(0, int(max_lag_s / dt) + 1))
    rain_cols = _rain_basis(rain, dt)
    dy = np.gradient(inflow, dt)

    keep = np.zeros(len(inflow), dtype=bool)
    keep[burn_in:len(inflow) - burn_in] = True
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)

    def evaluate(combo):
        lagged = [_lagged(upstream_flows[n], l) for n, l in zip(names, combo)]
        theta = _design(inflow, lagged, rain_cols)
        return _fit(theta[keep], dy[keep])

    best = None
    if len(names) == 1:
        combos = [(l,) for l in lags]
    elif len(names) == 2:
        combos = [(a, b) for a in lags for b in lags]
    else:
        raise NotImplementedError("scan supports at most two upstreams")

    for combo in combos:
        coef, resid = evaluate(combo)
        if best is None or resid < best[1]:
            best = (combo, resid, coef)

    combo, resid, coef = best
    b = float(coef[len(names)])
    k_from_b = -1.0 / b if b < -1e-12 else np.nan
    out = {}
    for i, n in enumerate(names):
        a = float(coef[i])
        out[n] = {"travel_time": combo[i] * dt,
                  "a": a,
                  "b": b,
                  "k_from_a": 1.0 / a if a > 1e-12 else np.nan,
                  "k_from_b": k_from_b,
                  "residual": resid}
    return out


def identify_network(data_logs, network, mask_fn=None, discrete=True, **kw):
    """Run the identification for every reach in a network.

    ``data_logs`` is one log or a list of them; multiple excited events are
    concatenated, which is how a real campaign would accumulate data.
    ``mask_fn(log) -> bool array`` selects which samples to use, e.g. to
    restrict the fit to dry-weather drawdown.
    """
    if isinstance(data_logs, dict):
        data_logs = [data_logs]

    merged = {"inflow": {}, "flow": {}, "rainfall": []}
    masks = []
    for log in data_logs:
        merged["rainfall"].extend(log["rainfall"])
        masks.append(np.ones(len(log["rainfall"]), dtype=bool)
                     if mask_fn is None else np.asarray(mask_fn(log), bool))
        for key in ("inflow", "flow"):
            for name, series in log[key].items():
                merged[key].setdefault(name, []).extend(series)
    merged["rainfall"] = np.asarray(merged["rainfall"])
    mask = np.concatenate(masks)
    for key in ("inflow", "flow"):
        merged[key] = {k: np.asarray(v) for k, v in merged[key].items()}

    results = {}
    for target in network.order:
        ups = network.upstream_of(target)
        if not ups:
            continue
        scan = scan_lags_discrete if discrete else scan_lags
        est = scan(np.asarray(merged["inflow"][target]),
                   {u: merged["flow"][u] for u in ups},
                   merged["rainfall"], network.dt, mask=mask, **kw)
        for u, r in est.items():
            results[(u, target)] = r
    return results


def _lagged(x, lag):
    out = np.zeros_like(x)
    if lag == 0:
        return x.copy()
    out[lag:] = x[:-lag]
    return out
