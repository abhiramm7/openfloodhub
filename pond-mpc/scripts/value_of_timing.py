"""Measure the value of timing directly.

Two controllers, both tuned with full hindsight on the very storm they are
scored on, so neither is handicapped by not knowing the future:

* constant   -- one valve setting per basin, fixed for the whole event
* piecewise  -- one setting per basin per time segment

Neither uses feedback. The only thing the second can do that the first
cannot is *change its mind at a particular time*. The gap between them is
therefore an estimate of how much of the achievable performance is
attributable to timing, and nothing else.

If that gap is small, a travel-time model cannot pay for itself on this
scenario no matter how good it is, and the forcing needs changing.
"""
import sys
import numpy as np
from pondmpc import GammaLike, back_to_back, design_storm

GRID = np.linspace(0.05, 1.0, 8)
N = 11


def make(rain, recession):
    return GammaLike(rainfall=rain, flow_threshold=1.0,
                     recession_hr=recession)


def score_constant(x, rain, recession):
    sc = make(rain, recession)
    sc.rollout(lambda s, x=x: x)
    return sc.performance()


def score_piecewise(X, rain, recession):
    """X has shape (segments, N); segments split the episode evenly."""
    sc = make(rain, recession)
    n_seg = X.shape[0]
    bounds = np.linspace(0, sc.n_steps, n_seg + 1).astype(int)
    step = {"i": 0}

    def ctrl(state):
        i = step["i"]
        step["i"] += 1
        seg = min(np.searchsorted(bounds, i, side="right") - 1, n_seg - 1)
        return X[seg]

    sc.rollout(ctrl)
    return sc.performance()


def descend(x0, scorer, sweeps=3, tag=""):
    x = x0.copy()
    cur = scorer(x)
    flat = x.reshape(-1)
    for s in range(sweeps):
        improved = False
        for i in range(flat.size):
            base = flat[i]
            best_g, best_v = base, cur
            for g in GRID:
                if g == base:
                    continue
                flat[i] = g
                v = scorer(x)
                if v < best_v - 1e-9:
                    best_g, best_v = g, v
            flat[i] = best_g
            if best_v < cur - 1e-9:
                cur, improved = best_v, True
        print(f"    {tag} sweep {s+1}: {cur:12.1f}", flush=True)
        if not improved:
            break
    return cur, x


def run_case(name, rain, recession, n_seg=4):
    print(f"\n=== {name} (recession {recession:.0f} h) ===", flush=True)
    unc = score_constant(np.ones(N), rain, recession)
    print(f"  uncontrolled          {unc:12.1f}", flush=True)

    best_u, best_s = np.inf, None
    for s in GRID:
        v = score_constant(np.full(N, s), rain, recession)
        if v < best_u:
            best_u, best_s = v, s
    print(f"  best uniform          {best_u:12.1f}  (setting {best_s:.2f})",
          flush=True)

    c_val, c_x = descend(np.full(N, best_s),
                         lambda x: score_constant(x, rain, recession),
                         tag="const")
    print(f"  best constant         {c_val:12.1f}", flush=True)

    X0 = np.tile(c_x, (n_seg, 1))
    p_val, p_X = descend(X0,
                         lambda X: score_piecewise(X, rain, recession),
                         tag="piecew")
    print(f"  best piecewise ({n_seg} seg) {p_val:12.1f}", flush=True)

    gap = (c_val - p_val) / c_val if c_val > 0 else 0.0
    print(f"  --> value of timing: {gap:.1%} of the constant-schedule cost",
          flush=True)
    return c_val, p_val, p_X


if __name__ == "__main__":
    single = design_storm(T=10.0, duration_hr=6.0, dt=60.0)
    b2b = back_to_back(T1=5.0, T2=10.0, duration_hr=6.0, gap_hr=6.0, dt=60.0)

    run_case("single event, long recession", single, 24.0)
    run_case("single event, short recession", single, 8.0)
    run_case("back-to-back, short recession", b2b, 8.0)
