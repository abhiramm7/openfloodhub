"""Identifiability of the coupling -- the assumption the project rests on."""
import numpy as np
import pytest

from pondmpc import GammaLike, design_storm, gamma_like
from pondmpc.excitation import ScheduledValves, multilevel, prbs


def _xcorr_lag(x, y, dt, max_lag_s=1800):
    x = np.asarray(x, float) - np.mean(x)
    y = np.asarray(y, float) - np.mean(y)
    best, best_k = -np.inf, 0
    for k in range(int(max_lag_s / dt) + 1):
        a, b = (x[:-k], y[k:]) if k > 0 else (x, y)
        d = np.linalg.norm(a) * np.linalg.norm(b)
        if d > 0 and (a @ b) / d > best:
            best, best_k = (a @ b) / d, k
    return best_k * dt


def _run(excited, seed=0):
    sc = GammaLike(rainfall=design_storm(T=10.0, duration_hr=6.0, dt=60.0))
    if excited:
        sc.rollout(ScheduledValves(
            multilevel(sc.n_steps, 11, dwell_steps=15, seed=seed)))
    else:
        sc.rollout(None)
    return sc


def _lag_errors(sc):
    truth = gamma_like().travel_times()
    return np.array([
        _xcorr_lag(sc.data_log["flow"][s], sc.data_log["inflow"][t], 60.0) - tau
        for (s, t), tau in truth.items()])


def test_excitation_beats_storm_only():
    """The excitation campaign is load-bearing, not a nicety."""
    quiet = np.abs(_lag_errors(_run(False))).mean()
    loud = np.abs(_lag_errors(_run(True))).mean()
    assert loud < 0.6 * quiet


def test_xcorr_lag_is_biased_by_attenuation():
    """Cross-correlation recovers translation PLUS attenuation, so it cannot
    be used to read off travel time directly. This is the empirical reason
    the coupling model needs both a delayed term and a storage term."""
    errs = _lag_errors(_run(True))
    # Systematic, not noisy: the spread is small next to the offset.
    assert errs.mean() > 3.0 * errs.std()
    assert errs.mean() > 100.0  # ~ k_attenuation = 180 s


def test_excitation_schedules_are_in_range():
    for gen in (prbs, multilevel):
        s = gen(500, 11, seed=1)
        assert s.shape == (500, 11)
        assert s.min() >= 0.0 and s.max() <= 1.0


def test_excitation_is_reproducible():
    assert np.array_equal(multilevel(300, 11, seed=7),
                          multilevel(300, 11, seed=7))
    assert not np.array_equal(multilevel(300, 11, seed=7),
                              multilevel(300, 11, seed=8))


def _campaign(n=3, seed=11):
    from pondmpc import StormSampler
    logs = []
    sampler = StormSampler(dt=60.0, seed=seed)
    for i in range(n):
        ev = sampler.sample(split="train")
        sc = GammaLike(rainfall=ev["intensity"], flow_threshold=1.0)
        sc.rollout(ScheduledValves(
            multilevel(sc.n_steps, 11, dwell_steps=15, seed=i)))
        logs.append(sc.data_log)
    return logs


def test_travel_times_recovered_exactly():
    """Every reach lag, to the timestep, from three excited events.

    Scored against implemented_delays rather than travel_times: the delay
    line rounds to whole steps, and an identification method can only ever
    see what the discretized network actually does.
    """
    from pondmpc.identify import identify_network
    net = gamma_like()
    res = identify_network(_campaign(), net)
    impl = net.implemented_delays()
    errs = np.array([res[k]["travel_time"] - impl[k] for k in impl])
    assert np.all(errs == 0.0)


def test_discrete_form_recovers_the_storage_constant():
    """The continuous form gets the lag right but the storage constant
    badly wrong -- it has to finite-difference dy/dt and then invert a
    coefficient. The discrete form matches the data-generating process and
    lands within a few percent."""
    from pondmpc.identify import identify_network
    net = gamma_like()
    impl = net.implemented_delays()

    cont = identify_network(_campaign(), net, discrete=False)
    disc = identify_network(_campaign(), net, discrete=True)

    k_cont = np.nanmean([cont[k]["k_from_b"] for k in impl])
    k_disc = np.nanmean([disc[k]["k_from_b"] for k in impl])
    assert abs(k_disc - 180.0) / 180.0 < 0.10
    assert abs(k_cont - 180.0) > abs(k_disc - 180.0)


def test_discrete_form_is_robust_to_the_runoff_nuisance():
    """Local runoff arrives at a basin alongside the routed water and is
    never observed separately. Restricting the fit to dry weather removes
    it, at the cost of also removing the strongest excitation.

    Under the continuous form that trade is bad -- the lag estimates
    collapse. Under the discrete form it makes no difference at all: the
    lags come out exact either way, so the runoff term does not have to be
    masked out, and the full record can be used.
    """
    from pondmpc.identify import identify_network
    net = gamma_like()
    impl = net.implemented_delays()
    logs = _campaign()

    def dry(log, settle=120):
        rain = np.asarray(log["rainfall"])
        m = np.ones(len(rain), bool)
        for i in np.flatnonzero(rain > 1e-9):
            m[i:i + settle] = False
        return m

    n_exact = lambda r: sum(r[k]["travel_time"] == impl[k] for k in impl)

    # Discrete: masking is unnecessary, both are exact.
    assert n_exact(identify_network(logs, net)) == 10
    assert n_exact(identify_network(logs, net, mask_fn=dry)) == 10

    # Continuous: masking away the excitation is actively harmful.
    assert n_exact(identify_network(logs, net, discrete=False)) == 10
    assert n_exact(identify_network(logs, net, discrete=False, mask_fn=dry)) < 5
