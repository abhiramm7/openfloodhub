"""The scenario surface must match pystorms, so controllers port unchanged.

Note these assert the behaviour pystorms *documents*, not the behaviour it
currently has: upstream mishandles dict actions and has an off-by-one in its
timestep guard. Matching the bugs would be worse than matching the contract.
"""
import numpy as np
import pytest

from pondmpc import GammaLike, SingleBasin, design_storm, perf_metrics


def test_state_shape_and_config():
    sc = GammaLike()
    assert len(sc.state()) == 11
    assert len(sc.config["action_space"]) == 11
    assert sc.config["name"] == "gamma_like"
    assert all(len(t) == 2 for t in sc.config["performance_targets"])


def test_step_returns_done_and_terminates():
    sc = SingleBasin(rainfall=design_storm(T=2.0, duration_hr=2.0, dt=60.0))
    steps = 0
    done = False
    while not done:
        done = sc.step(np.ones(1))
        steps += 1
        assert steps <= sc.n_steps + 1
    assert steps == sc.n_steps
    assert sc.step(np.ones(1)) is True  # idempotent once finished


def test_dict_and_array_actions_agree():
    """pystorms' environment mishandles this; ours must not."""
    r = design_storm(T=5.0, duration_hr=3.0, dt=60.0)
    a = GammaLike(rainfall=r)
    a.rollout(lambda s: np.full(11, 0.4))

    b = GammaLike(rainfall=r)
    names = b.env.order
    b.rollout(lambda s: {n: 0.4 for n in names})

    assert a.performance() == pytest.approx(b.performance(), rel=1e-12)


def test_action_length_is_validated():
    sc = GammaLike()
    with pytest.raises(ValueError):
        sc.step(np.ones(5))


def test_perf_metrics_matches_pystorms_contract():
    v = [1.0, 2.0, 3.0, 4.0]
    assert perf_metrics(v, "cumulative") == 10.0
    assert perf_metrics(v, "mean") == 2.5
    assert perf_metrics(v, "median") == 2.5
    assert perf_metrics(v, "maximum") == 4.0
    assert perf_metrics(v, "minimum") == 1.0
    assert perf_metrics(v, "recent") == 4.0
    with pytest.raises(ValueError):
        perf_metrics([], "mean")
    with pytest.raises(ValueError):
        perf_metrics(v, "nonsense")


def test_closing_valves_floods():
    """With the orifice as the only outlet there is no relief path: holding
    water past capacity floods rather than spilling. Storage is genuinely
    scarce, which is what makes release timing consequential."""
    r = design_storm(T=10.0, duration_hr=6.0, dt=60.0)
    shut = GammaLike(rainfall=r); shut.rollout(lambda s: np.zeros(11))
    open_ = GammaLike(rainfall=r); open_.rollout(None)
    s_shut, s_open = shut.summary(), open_.summary()
    assert s_shut["flood_volume_m3"] > 0.0
    assert s_open["flood_volume_m3"] == 0.0
    assert s_shut["spill_fraction"] == 0.0  # no spillway configured
    assert shut.performance() > open_.performance()


def test_throttling_trades_peak_against_flooding():
    """The tension the scenario is built around: throttling cuts the peak
    release but fills the basins, and past a point the basins overtop. A
    controller that cannot time its releases has to pick a point on this
    curve; one that can should beat all of them."""
    r = design_storm(T=10.0, duration_hr=6.0, dt=60.0)
    out = {}
    for v in (1.0, 0.5, 0.25):
        sc = GammaLike(rainfall=r, flow_threshold=1.0)
        sc.rollout(lambda s, v=v: np.full(11, v))
        out[v] = sc.summary()
    assert out[1.0]["peak_flow"] > out[0.5]["peak_flow"] > out[0.25]["peak_flow"]
    assert out[1.0]["flood_volume_m3"] < out[0.5]["flood_volume_m3"] < out[0.25]["flood_volume_m3"]
    assert out[0.5]["performance"] < out[1.0]["performance"]
    assert out[0.5]["performance"] < out[0.25]["performance"]


def test_control_authority_exists():
    """A uniform throttle must beat doing nothing, or there is no problem.

    0.5 is near the best uniform setting; throttling much harder crosses the
    flooding cliff and is far worse than doing nothing, which is the point
    of test_throttling_trades_peak_against_flooding above."""
    r = design_storm(T=10.0, duration_hr=6.0, dt=60.0)
    unc = GammaLike(rainfall=r); unc.rollout(None)
    thr = GammaLike(rainfall=r); thr.rollout(lambda s: np.full(11, 0.5))
    assert thr.performance() < 0.5 * unc.performance()
