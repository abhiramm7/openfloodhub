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
