"""Does the nondimensional collapse actually hold across basins?

For a foundation basin model to be worth building, most cross-basin
variation has to be removable by scaling. This measures how much.

Three coordinate systems are compared on the same randomized basins:

* dimensional        -- predict dV/dt from (h, u)
* nondimensional     -- predict dh^/dt^ from (h^, u)
* nondim + groups    -- predict dh^/dt^ from (h^, u, b_s, weir_ratio, crest_ratio)

Score is the fraction of variance left unexplained by a k-NN regressor fit
across ALL basins at once. Low residual means one model covers every basin;
high residual means per-basin fitting is unavoidable.
"""
import numpy as np

from pondmpc.basin import G, Basin
from pondmpc.randomize import dimensionless_groups, random_basin_params, scales


def sample_dynamics(p, rng, n=400):
    """(state, action, derivative) triples spread over the operating range."""
    s = scales(p)
    b = Basin(p)
    rows = []
    for _ in range(n):
        h = rng.uniform(0.0, p.max_depth)
        u = rng.uniform(0.0, 1.0)
        # Inflow scaled to the basin so the regime is comparable across basins.
        q_in_hat = rng.uniform(0.0, 1.5)
        q_in = q_in_hat * s["q_max"]

        b.volume = p.volume(h)
        dt = 0.02 * s["t_drain"]
        v0 = b.volume
        b.step(q_in, u, dt)
        dvdt = (b.volume - v0) / dt
        h1 = p.depth(b.volume)
        dhdt_hat = ((h1 - h) / p.max_depth) / (dt / s["t_drain"])

        rows.append((h, u, q_in, dvdt, h / p.max_depth, q_in_hat, dhdt_hat))
    return np.array(rows)


def knn_residual(X, y, k=8, n_train=4000, seed=0):
    """Fraction of variance in y left unexplained by k-NN on X (held out)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    tr, te = slice(0, n_train), slice(n_train, n_train + 2000)
    Xtr, ytr, Xte, yte = Xs[tr], y[tr], Xs[te], y[te]
    preds = np.empty(len(Xte))
    for i, x in enumerate(Xte):
        d = np.sum((Xtr - x) ** 2, axis=1)
        nn = np.argpartition(d, k)[:k]
        preds[i] = ytr[nn].mean()
    return float(np.mean((preds - yte) ** 2) / (np.var(yte) + 1e-12))


rng = np.random.default_rng(0)
N_BASINS = 60
dim_rows, nd_rows, grp_rows = [], [], []
for i in range(N_BASINS):
    p = random_basin_params(rng, name=str(i))
    g = dimensionless_groups(p)
    d = sample_dynamics(p, rng)
    dim_rows.append(np.column_stack([d[:, 0], d[:, 1], d[:, 2], d[:, 3]]))
    nd_rows.append(np.column_stack([d[:, 4], d[:, 1], d[:, 5], d[:, 6]]))
    grp_rows.append(np.column_stack([
        d[:, 4], d[:, 1], d[:, 5],
        np.full(len(d), g["b_s"]), np.full(len(d), g["weir_ratio"]),
        np.full(len(d), g["crest_ratio"]), d[:, 6]]))

dim = np.vstack(dim_rows); nd = np.vstack(nd_rows); grp = np.vstack(grp_rows)

print("{} random basins, {} samples each\n".format(N_BASINS, len(dim) // N_BASINS))
print("{:<34}{:>12}".format("coordinates", "residual"))
print("{:<34}{:>12.4f}".format("dimensional (h, u, q_in) -> dV/dt",
                               knn_residual(dim[:, :3], dim[:, 3])))
print("{:<34}{:>12.4f}".format("nondim (h^, u, q^) -> dh^/dt^",
                               knn_residual(nd[:, :3], nd[:, 3])))
print("{:<34}{:>12.4f}".format("  + 3 dimensionless groups",
                               knn_residual(grp[:, :6], grp[:, 6])))
print("\nlower is better; 1.0 = no better than predicting the mean")
