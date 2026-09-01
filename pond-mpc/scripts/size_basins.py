"""Derive the basin table from drainage area instead of guessing it.

With an orifice as the only outlet there is no relief path, so basin size
has to be justified: a basin must buffer the difference between what
arrives and what it is allowed to release. Sizing every basin off its
*cumulative* drainage area gives a network where that is true everywhere,
rather than one where the downstream basins are accidentally too small and
the whole scenario is infeasible.

Two dimensionless design choices:

* ``alpha``  storage as a fraction of the design event's runoff volume from
             the basin's cumulative area. Sets how much buffering exists.
* ``beta``   full-open discharge as a multiple of the flow threshold. Sets
             how much control authority the valve has: at beta = 3 a full
             basin must be roughly a third open to sit on the threshold.
"""
import numpy as np

from pondmpc.basin import G
from pondmpc.network import GAMMA_REACHES
from pondmpc.storms import depth_for_return_period

LOCAL_HA = {"1": 54.0, "2": 26.4, "3": 21.6, "4": 42.0, "5": 24.0, "6": 33.6,
            "7": 16.8, "8": 19.2, "9": 18.0, "10": 28.8, "11": 20.4}
RUNOFF_COEFF = 0.35
ORIFICE_COEFF = 0.65


def cumulative_areas():
    up = {}
    for s, t, _ in GAMMA_REACHES:
        up.setdefault(t, []).append(s)
    memo = {}

    def cum(n):
        if n not in memo:
            memo[n] = LOCAL_HA[n] + sum(cum(u) for u in up.get(n, []))
        return memo[n]

    return {n: cum(n) for n in LOCAL_HA}


def design_table(T=10.0, duration_hr=6.0, threshold=1.0, alpha=0.35, beta=3.0):
    depth_mm = depth_for_return_period(T, duration_hr)
    runoff_m = depth_mm / 1000.0 * RUNOFF_COEFF
    cum = cumulative_areas()
    v_ref = max(cum.values()) * 1.0e4 * runoff_m

    rows = {}
    for n, ha in cum.items():
        v_runoff = ha * 1.0e4 * runoff_m
        v_max = alpha * v_runoff
        # Deeper basins downstream, but only weakly: depth scales as the
        # cube root of volume so plan area still carries most of the growth.
        h_max = 2.5 + 3.0 * (v_max / (alpha * v_ref)) ** (1.0 / 3.0)
        b_s = 1.15
        k_s = v_max / h_max ** b_s
        # q_max = Cd * A0 * sqrt(2 g h_max) = beta * threshold
        a0 = beta * threshold / (ORIFICE_COEFF * np.sqrt(2.0 * G * h_max))
        rows[n] = dict(k_s=k_s, max_depth=h_max, orifice_area=a0,
                       catch_ha=LOCAL_HA[n], cum_ha=ha, v_max=v_max)
    return rows


if __name__ == "__main__":
    t = design_table()
    print("{:>4}{:>9}{:>9}{:>11}{:>9}{:>9}{:>9}".format(
        "id", "loc ha", "cum ha", "V_max m3", "h_max", "k_s", "A0"))
    for n in [str(i) for i in range(1, 12)]:
        r = t[n]
        print("{:>4}{:>9.1f}{:>9.1f}{:>11.0f}{:>9.2f}{:>9.0f}{:>9.3f}".format(
            n, r["catch_ha"], r["cum_ha"], r["v_max"], r["max_depth"],
            r["k_s"], r["orifice_area"]))
    print("\nGAMMA_BASINS = {")
    for n in [str(i) for i in range(1, 12)]:
        r = t[n]
        print('    "{}":{} ({:.1f}, {:.2f}, {:.3f}, {:.1f}),'.format(
            n, " " * (3 - len(n)), r["k_s"], r["max_depth"],
            r["orifice_area"], r["catch_ha"]))
    print("}")
