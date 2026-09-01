"""Find a storage/forcing scale where local control is NOT enough.

We want a regime in which throttling locally to the flow threshold overtops
the basin -- so the controller has to decide which basin holds water and
when to release it, which is the only setting where travel time matters.
"""
import numpy as np
import pondmpc.network as N
from pondmpc import GammaLike
from pondmpc.controllers import FixedValve, ThresholdHold, Uncontrolled

BASE = dict(N.GAMMA_BASINS)


def scaled(storage_scale, catch_scale, depth_scale):
    return {k: (v[0] * storage_scale, v[1] * depth_scale,
                v[2], v[3] * catch_scale) for k, v in BASE.items()}


def run(ctrl_name, thr, T):
    from pondmpc.storms import design_storm
    sc = GammaLike(rainfall=design_storm(T=T, duration_hr=6.0, dt=60.0),
                   flow_threshold=thr)
    if ctrl_name == "unc":
        ctrl = None
    elif ctrl_name == "local":
        ctrl = ThresholdHold([sc.env.basins[n] for n in sc.env.order], thr)
    else:
        ctrl = FixedValve(11, float(ctrl_name))
    sc.rollout(ctrl)
    return sc.summary()


print(f"{'storage':>8}{'catch':>7}{'depth':>7} | "
      f"{'unc perf':>10}{'unc pk':>7} | {'loc perf':>12}{'loc flood':>11}{'loc pk':>7}")
for ss, cs, ds in [(1.0, 1.0, 1.0), (0.9, 1.1, 0.95), (0.8, 1.2, 0.9),
                   (0.7, 1.25, 0.85), (0.6, 1.3, 0.8), (0.5, 1.4, 0.75)]:
    N.GAMMA_BASINS = scaled(ss, cs, ds)
    u = run("unc", 1.0, 10.0)
    l = run("local", 1.0, 10.0)
    print(f"{ss:>8.2f}{cs:>7.2f}{ds:>7.2f} | {u['performance']:>10.1f}"
          f"{u['peak_flow']:>7.2f} | {l['performance']:>12.1f}"
          f"{l['flood_volume_m3']:>11.1f}{l['peak_flow']:>7.2f}")
N.GAMMA_BASINS = BASE
