"""Physics-level checks on the pure-Python plant.

These exist because every later claim -- that a learned model is accurate,
that an identified lag is right -- is measured against this simulator. If it
does not conserve mass, nothing downstream means anything.
"""
import numpy as np
import pytest

from pondmpc import (Basin, BasinParams, GammaLike, Reach, SingleBasin,
                     design_storm, gamma_like, hyetograph)


def test_storage_curve_roundtrips():
    p = BasinParams("t", k_s=2000.0, b_s=1.15, max_depth=4.0)
    for d in (0.0, 0.01, 1.0, 2.5, 4.0):
        assert p.depth(p.volume(d)) == pytest.approx(d, abs=1e-9)


def test_single_basin_mass_balance():
    """Inflow must equal outflow plus flooding plus storage change."""
    b = Basin(BasinParams("t", k_s=2000.0, b_s=1.15, max_depth=4.0,
                          orifice_area=0.4))
    dt, v0 = 60.0, b.volume
    inflow, out_v, flood_v = 1.2, 0.0, 0.0
    for _ in range(500):
        out, flood = b.step(inflow, 0.5, dt)
        out_v += out * dt
        flood_v += flood * dt
    balance = inflow * 500 * dt - (out_v + flood_v + (b.volume - v0))
    assert abs(balance) < 1e-6 * inflow * 500 * dt


def test_network_mass_balance():
    sc = GammaLike(rainfall=design_storm(T=10.0, duration_hr=6.0, dt=60.0))
    sc.rollout(None)
    dt = sc.env.dt
    log = sc.data_log
    for n in sc.env.order:
        # Everything that entered a basin left it or is still in it.
        inflow_v = np.sum(log["inflow"][n]) * dt
        out_v = np.sum(log["flow"][n]) * dt
        flood_v = np.sum(log["flooding"][n]) * dt
        stored = sc.env.basins[n].volume
        assert inflow_v == pytest.approx(out_v + flood_v + stored,
                                         rel=1e-6, abs=1e-3)


def test_substep_convergence():
    """One RK4 substep at dt=60 s must match eight. If this fails, the
    default in Basin.step needs raising."""
    import pondmpc.basin as B

    def run(substeps):
        orig = B.Basin.step
        B.Basin.step = lambda self, i, v, dt, _s=substeps, _o=orig: _o(
            self, i, v, dt, _s)
        sc = GammaLike(rainfall=design_storm(T=10.0, duration_hr=6.0, dt=60.0))
        sc.rollout(None)
        B.Basin.step = orig
        return sc.performance()

    assert run(1) == pytest.approx(run(8), rel=1e-4)


def test_flooding_caps_volume():
    p = BasinParams("t", k_s=1000.0, b_s=1.0, max_depth=2.0,
                    orifice_area=0.01, weir_crest=99.0)
    b = Basin(p)
    for _ in range(200):
        b.step(5.0, 0.0, 60.0)
    assert b.volume <= p.max_volume + 1e-9
    assert b.flooding > 0.0


def test_reach_is_causal():
    """Nothing may arrive before the travel time has elapsed."""
    r = Reach("a", "b", length_m=900.0, celerity=1.5, k_attenuation=0.0)
    dt = 60.0
    r.reset(dt)
    arrivals = [r.step(1.0 if i == 0 else 0.0, dt) for i in range(30)]
    first = next(i for i, v in enumerate(arrivals) if v > 1e-12)
    assert first * dt >= r.travel_time - dt


def test_reach_conserves_volume():
    r = Reach("a", "b", length_m=600.0, celerity=1.5, k_attenuation=180.0)
    dt = 60.0
    r.reset(dt)
    sent = [2.0] * 50 + [0.0] * 400
    got = sum(r.step(q, dt) for q in sent) * dt
    assert got == pytest.approx(sum(sent) * dt, rel=1e-3)


def test_hyetograph_preserves_depth():
    dt = 60.0
    series = hyetograph(80.0, 6.0, dt, peak_position=0.4)
    depth = np.sum(series) * dt / 3600.0
    assert depth == pytest.approx(80.0, rel=1e-6)


def test_longest_path_matches_topology():
    net = gamma_like()
    # 9 -> 8 -> 6 -> 5 -> 4 -> 3 -> 2 -> 1 is the deepest branch.
    expected = sum(l for s, t, l in
                   [("9", "8", 268.65), ("8", "6", 624.11), ("6", "5", 1187.70),
                    ("5", "4", 184.86), ("4", "3", 931.25), ("3", "2", 182.46),
                    ("2", "1", 797.24)]) / 1.5
    assert net.longest_path_time() == pytest.approx(expected, rel=1e-9)
