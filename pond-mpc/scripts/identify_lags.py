"""Are the travel times identifiable from data at all?

Cross-correlation between an upstream release and the downstream basin's
inflow, under valve excitation. This is the crudest possible estimator --
if it cannot see the lag, SINDy with a delay library will not either, and
the coupling half of the architecture is in trouble.

Two conditions are compared deliberately:

* storm-only  -- valves fixed open, response driven by rainfall alone
* excited     -- multi-level valve schedule on top of the same storm

The gap between them is the argument for the excitation campaign in P3.
"""
import numpy as np
from pondmpc import GammaLike, design_storm, gamma_like
from pondmpc.excitation import ScheduledValves, multilevel


def lag_by_xcorr(upstream_release, downstream_inflow, dt, max_lag_s=1800):
    """Lag maximizing cross-correlation of the two mean-removed signals."""
    x = np.asarray(upstream_release, float)
    y = np.asarray(downstream_inflow, float)
    x = x - x.mean()
    y = y - y.mean()
    max_k = int(max_lag_s / dt)
    scores = []
    for k in range(max_k + 1):
        a, b = (x[:-k], y[k:]) if k > 0 else (x, y)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        scores.append(float(a @ b / denom) if denom > 0 else -np.inf)
    return int(np.argmax(scores)) * dt, np.array(scores)


def collect(excited, seed=0):
    sc = GammaLike(rainfall=design_storm(T=10.0, duration_hr=6.0, dt=60.0),
                   flow_threshold=1.0)
    if excited:
        sched = multilevel(sc.n_steps, len(sc.env.order), dwell_steps=15,
                           seed=seed)
        sc.rollout(ScheduledValves(sched))
    else:
        sc.rollout(None)
    return sc


truth = gamma_like().travel_times()
dt = 60.0

print(f"{'reach':>10}{'length m':>10}{'true tau':>10}"
      f"{'storm-only':>12}{'excited':>10}{'err s':>8}")
rows = []
for excited in (False, True):
    sc = collect(excited)
    est = {}
    for (src, tgt), tau in truth.items():
        lag, _ = lag_by_xcorr(sc.data_log["flow"][src],
                              sc.data_log["inflow"][tgt], dt)
        est[(src, tgt)] = lag
    rows.append(est)

storm_only, exc = rows
errs_s, errs_e = [], []
for (src, tgt), tau in truth.items():
    length = next(l for s, t, l in
                  __import__("pondmpc").GAMMA_REACHES if s == src and t == tgt)
    e_s, e_e = storm_only[(src, tgt)] - tau, exc[(src, tgt)] - tau
    errs_s.append(abs(e_s)); errs_e.append(abs(e_e))
    print(f"{src+'->'+tgt:>10}{length:>10.0f}{tau:>10.0f}"
          f"{storm_only[(src,tgt)]:>12.0f}{exc[(src,tgt)]:>10.0f}{e_e:>8.0f}")

print(f"\nmean |error|:  storm-only {np.mean(errs_s):6.1f} s"
      f"   excited {np.mean(errs_e):6.1f} s")
