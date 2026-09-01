"""Collapse check, with the controls the first pass was missing.

Two things confound a naive version:

* No noise floor. A k-NN residual of 0.15 across basins means nothing
  unless we know what the same estimator scores on ONE basin, where the
  collapse question does not arise. That is the floor.
* dh^/dt^ is singular. For a storage exponent b > 1 the surface area goes
  to zero at the invert, so the depth derivative blows up near empty and
  the variance is dominated by a handful of near-empty samples. The
  volumetric form dV^/dt^ = (q_in - q_out)/q_max has no such singularity.

The cleanest target of all is the constitutive relation itself: does
Q_out/q_max depend only on (h/h_max, u)?
"""
import numpy as np

from pondmpc.basin import G, Basin
from pondmpc.randomize import dimensionless_groups, random_basin_params, scales


def sample(p, rng, n=400, h_lo=0.05):
    """Analytic release. Recovering it by finite-differencing volume over a
    tiny step is catastrophic cancellation, which shows up as a spurious
    noise floor and swamps the effect being measured."""
    s = scales(p)
    b = Basin(p)
    out = []
    for _ in range(n):
        h = rng.uniform(h_lo * p.max_depth, p.max_depth)
        u = rng.uniform(0.0, 1.0)
        q_out = b.orifice_flow(h, u) + b.weir_flow(h)
        out.append((h / p.max_depth, u, q_out / s["q_max"]))
    return np.array(out)


def knn_residual(X, y, k=8, n_train=4000, n_test=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx][:n_train + n_test], y[idx][:n_train + n_test]
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    Xtr, ytr = Xs[:n_train], y[:n_train]
    Xte, yte = Xs[n_train:], y[n_train:]
    pred = np.array([ytr[np.argpartition(np.sum((Xtr - x) ** 2, 1), k)[:k]].mean()
                     for x in Xte])
    return float(np.mean((pred - yte) ** 2) / (np.var(yte) + 1e-12))


rng = np.random.default_rng(0)
basins = [random_basin_params(rng, name=str(i)) for i in range(60)]
data = [sample(p, rng) for p in basins]
groups = [dimensionless_groups(p) for p in basins]

# Noise floor: one basin, same estimator, same training size.
solo_p = basins[0]
solo = np.vstack([sample(solo_p, rng, n=6000)])
floor = knn_residual(solo[:, :2], solo[:, 2])

pooled = np.vstack(data)
nd = knn_residual(pooled[:, :2], pooled[:, 2])

grp = np.vstack([np.column_stack([
    d[:, 0], d[:, 1],
    np.full(len(d), g["weir_ratio"]), np.full(len(d), g["crest_ratio"]),
    d[:, 2]])
    for d, g in zip(data, groups)])
withg = knn_residual(grp[:, :4], grp[:, 4])

print("target: Q_out / q_max      (the constitutive relation)\n")
print("{:<40}{:>10}".format("", "residual"))
print("{:<40}{:>10.4f}   <- estimator noise floor".format(
    "single basin, (h^, u)", floor))
print("{:<40}{:>10.4f}".format("60 basins pooled, (h^, u)", nd))
print("{:<40}{:>10.4f}".format("  + weir_ratio, crest_ratio", withg))

excess = (nd - floor) / max(floor, 1e-9)
print("\npooled excess over floor: {:.0%}".format(excess))
print("groups recover {:.0%} of the gap".format(
    (nd - withg) / max(nd - floor, 1e-9)))
