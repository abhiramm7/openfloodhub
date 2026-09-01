"""Build a network model from identified parameters.

The planner needs a model to roll forward. Rather than a separate
prediction code path, the surrogate is an ordinary :class:`Network` whose
reaches carry *identified* travel times and storage constants instead of
the true ones. Planning against the true network then means passing the
true network in -- so oracle MPC and model-based MPC differ only in which
object the planner is handed, and any gap between them is model error and
nothing else.
"""
import numpy as np

from .network import Network, gamma_like
from .routing import Reach


def surrogate_from_identification(template, identified, dt=None):
    """Clone ``template``'s structure, substituting identified routing.

    Basin geometry is taken from the template. With the outflow relation
    collapsing exactly onto ``Q/q_max = u*sqrt(h/h_max)``, basin dynamics
    are the part of the model that transfers and does not need refitting
    per site; the reaches are the part that does.
    """
    dt = template.dt if dt is None else dt
    net = gamma_like(dt=dt)

    for reach in net.reaches:
        est = identified.get((reach.source, reach.target))
        if est is None:
            continue
        tau = est["travel_time"]
        k = est.get("k", est.get("k_from_b", np.nan))
        # Hold length fixed and move celerity, so travel_time reports the
        # identified lag while lag_steps stays consistent with it.
        reach.celerity = reach.length_m / max(tau, 1e-6)
        reach.k_attenuation = float(k) if np.isfinite(k) and k > 0 else 180.0
    net.reset()
    return net


def perfect_model(template):
    """A surrogate with the true parameters -- the oracle-MPC upper bound."""
    net = gamma_like(dt=template.dt)
    net.reset()
    return net
