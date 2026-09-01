"""Randomized basins, for pretraining a transferable dynamics model.

The argument for a single basin model that transfers is that every basin
obeys the same equation and differs only in its parameters. If that is
true, a model pretrained over a wide prior on those parameters should
transfer to a basin it has never seen -- and the parameters can be folded
into dimensionless groups so the model does not have to learn the scaling
at all.

``dimensionless_groups`` enumerates what a basin model has to be
conditioned on if the nondimensionalization is going to carry the transfer.
"""
import numpy as np

from .basin import G, BasinParams

# Wide enough to bracket the gamma basins with room on either side; a prior
# that does not cover the target is the usual cause of negative transfer.
PRIORS = {
    "k_s": (800.0, 6000.0),        # plan-area coefficient, m^2
    "b_s": (1.0, 1.4),             # storage exponent
    "max_depth": (2.0, 7.0),       # m
    "orifice_area": (0.15, 1.20),  # m^2
    "weir_length": (2.0, 12.0),    # m
    "crest_ratio": (0.75, 0.95),   # weir crest as a fraction of max depth
}


def random_basin_params(rng, name="r", priors=None):
    p = dict(PRIORS if priors is None else priors)
    lo_hi = lambda k: rng.uniform(*p[k])
    max_depth = lo_hi("max_depth")
    return BasinParams(
        name,
        k_s=lo_hi("k_s"),
        b_s=lo_hi("b_s"),
        max_depth=max_depth,
        orifice_area=lo_hi("orifice_area"),
        weir_crest=lo_hi("crest_ratio") * max_depth,
        weir_length=lo_hi("weir_length"),
    )


def scales(p):
    """The three constants that nondimensionalize a basin.

    ``q_max`` is the full-open orifice discharge at maximum depth, ``v_max``
    the storage at maximum depth, and ``t_drain`` the time to empty a full
    basin at ``q_max`` -- the natural clock of the basin.
    """
    q_max = p.orifice_coeff * p.orifice_area * np.sqrt(2.0 * G * p.max_depth)
    v_max = p.volume(p.max_depth)
    return {"h_max": p.max_depth, "q_max": q_max, "v_max": v_max,
            "t_drain": v_max / q_max}


def dimensionless_groups(p):
    """The groups a nondimensional basin model still has to be told about.

    The orifice relation nondimensionalizes exactly -- Q/q_max = u*sqrt(h/h_max)
    for every basin, with no free parameters. What does *not* vanish:

    * ``b_s``          storage shape, which sets how depth responds to volume
    * ``weir_ratio``   spillway capacity at max depth relative to q_max
    * ``crest_ratio``  where the spillway starts, as a fraction of max depth

    So the prediction is that conditioning on three numbers is enough, and
    that a model given them transfers to an unseen basin without retraining.
    """
    s = scales(p)
    head = p.max_depth - p.weir_crest
    weir_cap = p.weir_coeff * p.weir_length * max(head, 0.0) ** 1.5
    return {"b_s": p.b_s,
            "weir_ratio": weir_cap / s["q_max"],
            "crest_ratio": p.weir_crest / p.max_depth}
