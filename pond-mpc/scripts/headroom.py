"""Decompose the available headroom into spatial and temporal parts.

* uniform            -- one setting for the whole network
* per-basin constant -- coordinate descent over 11 static settings.
  This is spatial allocation with no timing whatsoever.

Whatever a planner wins *beyond* the per-basin constant is attributable to
timing, and timing is the only thing a travel-time model provides. If that
margin turns out to be small, the coupling model is not earning its place
and we want to know now rather than after building it.
"""
import numpy as np
from pondmpc import GammaLike, design_storm

R = design_storm(T=10.0, duration_hr=6.0, dt=60.0)
GRID = np.linspace(0.05, 1.0, 12)


def score(x):
    sc = GammaLike(rainfall=R, flow_threshold=1.0)
    sc.rollout(lambda s, x=x: x)
    return sc.performance()


best_u, best_s = np.inf, None
for s in GRID:
    v = score(np.full(11, s))
    if v < best_u:
        best_u, best_s = v, s
print("uniform best        {:>10.1f}  (setting {:.2f})".format(best_u, best_s))

x = np.full(11, best_s)
cur = best_u
for sweep in range(4):
    improved = False
    for i in range(11):
        for g in GRID:
            if g == x[i]:
                continue
            trial = x.copy(); trial[i] = g
            v = score(trial)
            if v < cur - 1e-9:
                cur, x, improved = v, trial, True
    print("  sweep {}: {:>10.1f}".format(sweep + 1, cur))
    if not improved:
        break

print("per-basin constant  {:>10.1f}".format(cur))
print("  settings:", dict(zip(GammaLike(rainfall=R).env.order,
                              np.round(x, 2))))
print("\nspatial gain over uniform: {:.1%}".format(1 - cur / best_u))
print("anything a planner wins below {:.0f} is attributable to timing.".format(cur))
