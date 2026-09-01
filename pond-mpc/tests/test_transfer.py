"""The nondimensional collapse that a transferable basin model rests on."""
import numpy as np
import pytest

from pondmpc import back_to_back, storm_sequence
from pondmpc.basin import Basin
from pondmpc.randomize import (dimensionless_groups, random_basin_params,
                               scales)


def _knn_residual(X, y, k=8, n_train=1500, n_test=800, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx][:n_train + n_test], y[idx][:n_train + n_test]
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    Xtr, ytr, Xte, yte = Xs[:n_train], y[:n_train], Xs[n_train:], y[n_train:]
    pred = np.array([ytr[np.argpartition(np.sum((Xtr - x) ** 2, 1), k)[:k]].mean()
                     for x in Xte])
    return float(np.mean((pred - yte) ** 2) / (np.var(yte) + 1e-12))


def _sample(p, rng, n, sub_crest):
    b = Basin(p)
    q_max = scales(p)["q_max"]
    hi = p.weir_crest if sub_crest else p.max_depth
    rows = []
    for _ in range(n):
        h = rng.uniform(0.05 * p.max_depth, hi)
        u = rng.uniform(0.0, 1.0)
        rows.append((h / p.max_depth, u,
                     (b.orifice_flow(h, u) + b.weir_flow(h)) / q_max))
    return np.array(rows)


def test_subcrest_collapse_is_exact():
    """Below the spillway crest, every basin is the same function of
    (h/h_max, u). This is what makes one pretrained basin model plausible:
    pooling 30 basins costs nothing against fitting one.
    """
    rng = np.random.default_rng(0)
    basins = [random_basin_params(rng, name=str(i)) for i in range(30)]
    pooled = np.vstack([_sample(p, rng, 200, True) for p in basins])
    solo = np.vstack([_sample(basins[0], rng, 3000, True)])

    floor = _knn_residual(solo[:, :2], solo[:, 2])
    together = _knn_residual(pooled[:, :2], pooled[:, 2])
    assert floor < 0.01
    assert together < 5 * max(floor, 1e-4)


def test_spillway_breaks_the_collapse():
    """Above the crest it stops being one function -- the spillway-to-orifice
    capacity ratio spans two orders of magnitude across the prior. Recorded
    so that a later model is not credited with fixing something that was
    never broken."""
    rng = np.random.default_rng(0)
    basins = [random_basin_params(rng, name=str(i)) for i in range(30)]
    ratios = [dimensionless_groups(p)["weir_ratio"] for p in basins]
    assert max(ratios) / min(ratios) > 20

    pooled = np.vstack([_sample(p, rng, 200, False) for p in basins])
    solo = np.vstack([_sample(basins[0], rng, 3000, False)])
    # The separation is ~23x at the sample sizes in scripts/collapse_check2.py;
    # this test runs smaller and cheaper, where the single-basin floor is
    # itself inflated by the kink at the crest, so the margin is thinner.
    assert (_knn_residual(pooled[:, :2], pooled[:, 2])
            > 4 * _knn_residual(solo[:, :2], solo[:, 2]))


def test_scales_are_consistent():
    rng = np.random.default_rng(1)
    p = random_basin_params(rng)
    s = scales(p)
    assert s["v_max"] == pytest.approx(p.volume(p.max_depth))
    assert s["t_drain"] == pytest.approx(s["v_max"] / s["q_max"])
    b = Basin(p)
    assert b.orifice_flow(p.max_depth, 1.0) == pytest.approx(s["q_max"])


def test_storm_sequence_has_dry_gaps():
    dt = 60.0
    r = storm_sequence([(5.0, 3.0), (10.0, 3.0)], gap_hr=4.0, dt=dt)
    assert len(r) == int((3 + 4 + 3) * 3600 / dt)
    gap = r[int(3 * 3600 / dt):int(7 * 3600 / dt)]
    assert np.all(gap == 0.0)


def test_back_to_back_conserves_both_depths():
    dt = 60.0
    r = back_to_back(T1=5.0, T2=10.0, duration_hr=6.0, gap_hr=6.0, dt=dt)
    from pondmpc.storms import depth_for_return_period
    expected = (depth_for_return_period(5.0, 6.0)
                + depth_for_return_period(10.0, 6.0))
    assert np.sum(r) * dt / 3600.0 == pytest.approx(expected, rel=1e-6)
