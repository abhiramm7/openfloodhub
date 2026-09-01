"""Pure-Python stormwater testbed for model-based distributed control.

P0 of the pond-mpc project: a fast, fully-known plant that mirrors the
pystorms API, so dynamics models and planners can be developed and
validated against ground truth before being run on SWMM.
"""
from .basin import Basin, BasinParams
from .routing import Reach
from .network import Network, gamma_like, single_basin, GAMMA_REACHES
from .scenarios import (GammaLike, PondScenario, SingleBasin, perf_metrics,
                        sampled_scenario, threshold)
from .storms import (Catchment, NashCascade, StormSampler, design_storm,
                     depth_for_return_period, hyetograph)

__all__ = [
    "Basin", "BasinParams", "Reach", "Network", "gamma_like", "single_basin",
    "GAMMA_REACHES", "PondScenario", "SingleBasin", "GammaLike",
    "sampled_scenario", "perf_metrics", "threshold", "Catchment",
    "NashCascade", "StormSampler", "design_storm", "depth_for_return_period",
    "hyetograph",
]
