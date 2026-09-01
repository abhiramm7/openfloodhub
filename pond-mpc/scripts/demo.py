"""P0 summary: what the testbed looks like and why it is a control problem."""
import numpy as np
from pondmpc import GammaLike, SingleBasin, design_storm, gamma_like
from pondmpc.controllers import EqualFilling, FixedValve, ThresholdHold


def show(tag, scen):
    s = scen.summary()
    print("  {:<18} perf={:>10.1f}  peakQ={:>5.2f}  exceed={:>5d}  "
          "spill={:>5.0%}  flood={:>7.1f}  hT={:.2f}".format(
              tag, s["performance"], s["peak_flow"], s["exceedance_steps"],
              s["spill_fraction"], s["flood_volume_m3"], s["final_depth_max"]))


net = gamma_like()
print("gamma-like network: {} basins, {} reaches".format(
    len(net.order), len(net.reaches)))
print("longest path to outfall: {:.0f} s ({:.0f} min)  <- minimum useful "
      "planning horizon".format(net.longest_path_time(),
                                net.longest_path_time() / 60))
tt = net.travel_times()
print("reach travel times (s): min {:.0f}, max {:.0f}".format(
    min(tt.values()), max(tt.values())))

r = design_storm(T=10.0, duration_hr=6.0, dt=60.0)
print("\n10-yr 6-hr event, flow threshold 1.0 m3/s:")
for tag, mk in [
    ("uncontrolled", lambda s: None),
    ("all shut", lambda s: FixedValve(11, 0.0)),
    ("uniform 0.25", lambda s: FixedValve(11, 0.25)),
    ("equal filling", lambda s: EqualFilling(
        s.env.order, [s.env.basins[n].p.max_depth for n in s.env.order])),
    ("local threshold", ThresholdHold.for_scenario),
]:
    sc = GammaLike(rainfall=r, flow_threshold=1.0)
    sc.rollout(mk(sc))
    show(tag, sc)

print("\nThe local controller has full information about its own basin and")
print("still loses to a uniform setting. What it lacks is timing.")
