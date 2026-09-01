"""Snapshot/restore and the surrogate the planner rolls out."""
import numpy as np
import pytest

from pondmpc import GammaLike, design_storm, gamma_like
from pondmpc.identify import identify_network
from pondmpc.surrogate import surrogate_from_identification


def test_snapshot_restore_is_exact():
    """A planner takes thousands of rollouts per control step from the same
    state; if restore is lossy the plan is computed against drift."""
    net = gamma_like()
    net.reset()
    valves = {n: 0.5 for n in net.order}
    for _ in range(200):
        net.step(8.0, valves)

    snap = net.snapshot()
    ref = [net.step(4.0, valves) for _ in range(50)]

    net.restore(snap)
    again = [net.step(4.0, valves) for _ in range(50)]

    for a, b in zip(ref, again):
        for name in a:
            for key in ("outflow", "depth", "inflow", "flooding"):
                assert a[name][key] == pytest.approx(b[name][key], abs=1e-12)


def test_snapshot_captures_water_in_transit():
    """Reach buffers and catchment stores hold water that has left a basin
    but not arrived. Restoring without them would silently lose mass."""
    net = gamma_like()
    net.reset()
    for _ in range(100):
        net.step(10.0, {n: 1.0 for n in net.order})

    snap = net.snapshot()
    assert any(np.any(buf != 0.0) for buf, _ in snap[1])
    assert any(st != 0.0 for _, st in snap[1])
    assert any(np.any(store != 0.0) for store, _ in snap[2])


def test_surrogate_matches_truth_when_identification_is_exact():
    """With every lag recovered exactly, the surrogate should track the true
    network closely -- this is what separates model error from planner error
    in the results table."""
    from pondmpc import StormSampler
    from pondmpc.excitation import ScheduledValves, multilevel

    logs = []
    sampler = StormSampler(dt=60.0, seed=11)
    for i in range(3):
        ev = sampler.sample("train")
        sc = GammaLike(rainfall=ev["intensity"], flow_threshold=1.0)
        sc.rollout(ScheduledValves(
            multilevel(sc.n_steps, 11, dwell_steps=15, seed=i)))
        logs.append(sc.data_log)

    true_net = gamma_like()
    ident = identify_network(logs, true_net)
    surro = surrogate_from_identification(true_net, ident)

    for reach in surro.reaches:
        key = (reach.source, reach.target)
        assert reach.lag_steps(surro.dt) == int(
            round(ident[key]["travel_time"] / surro.dt))

    rain = design_storm(T=10.0, duration_hr=6.0, dt=60.0)
    true_net.reset()
    surro.reset()
    valves = {n: 0.5 for n in true_net.order}
    err, mag = 0.0, 0.0
    for r in rain:
        a = true_net.step(r, valves)
        b = surro.step(r, valves)
        for n in true_net.order:
            err += abs(a[n]["outflow"] - b[n]["outflow"])
            mag += abs(a[n]["outflow"])
    assert err / max(mag, 1e-9) < 0.05
