"""pystorms-compatible scenario wrappers around the pure-Python network.

The API mirrors ``pystorms.scenarios.scenario`` exactly -- ``state()``,
``step(actions) -> done``, ``performance(metric)``, ``data_log`` -- so a
controller written against this testbed runs unchanged on SWMM later. That
is the only reason to bother matching an API with known rough edges.

The performance function follows gamma: a hinge penalty on every basin's
release above a threshold, a large penalty for flooding, and a terminal
penalty for water left in storage at the end of the event.
"""
import numpy as np

from .network import gamma_like, single_basin
from .storms import StormSampler, design_storm

# pystorms' gamma uses a flat 1e6 per flooding timestep and a flat 7e5
# terminal penalty. That makes the objective effectively lexicographic and
# almost flat everywhere else, which is hard for a planner to descend. The
# default objective here is volumetric instead -- every term is in m^3, so
# the weights are interpretable and the landscape has a gradient. Set
# ``gamma_compatible=True`` to reproduce gamma's exact penalty for the
# like-for-like comparison in P3.
GAMMA_FLOOD_PENALTY = 1.0e6
GAMMA_TERMINAL_PENALTY = 7.0e5


def perf_metrics(values, metric="cumulative"):
    """Same metric set as ``pystorms.utilities.perf_metrics``."""
    if len(values) < 1:
        raise ValueError("Run step; performance metrics have not been computed")
    if metric == "mean":
        return float(np.mean(values))
    if metric == "cumulative":
        return float(np.sum(values))
    if metric == "median":
        return float(np.median(values))
    if metric == "maximum":
        return float(np.max(values))
    if metric == "minimum":
        return float(np.min(values))
    if metric == "recent":
        return float(values[-1])
    raise ValueError(
        "mean, cumulative, median, maximum, minimum, and recent are the "
        "only valid metrics"
    )


def threshold(value, target=0.10, scaling=1.0):
    """Hinge penalty above ``target``; matches pystorms' utility."""
    return (value - target) * scaling if value > target else 0.0


class PondScenario:
    """Base scenario over a :class:`~pondmpc.network.Network`.

    Parameters
    ----------
    network
        The plant.
    rainfall
        Intensity series (mm/hr) at the network's timestep. The episode ends
        when the series is exhausted plus ``recession_hr`` of dry weather, so
        the controller is scored on draining down as well as on the peak.
    flow_threshold
        Release above this (m^3/s) is penalized at every basin.
    """

    def __init__(self, network, rainfall, flow_threshold=1.0,
                 recession_hr=24.0, terminal_depth=0.10,
                 flood_weight=100.0, terminal_weight=10.0,
                 gamma_compatible=False):
        self.env = network
        self.flow_threshold = float(flow_threshold)
        self.terminal_depth = float(terminal_depth)
        self.flood_weight = float(flood_weight)
        self.terminal_weight = float(terminal_weight)
        self.gamma_compatible = bool(gamma_compatible)

        dry = int(recession_hr * 3600.0 / network.dt)
        self.rainfall = np.concatenate([np.asarray(rainfall, dtype=float),
                                        np.zeros(dry)])
        self.n_steps = len(self.rainfall)
        self._i = 0

        names = self.env.order
        self.config = {
            "name": "pond",
            "states": [(n, "depthN") for n in names],
            "action_space": list(names),
            "performance_targets": (
                [(n, "flow") for n in names]
                + [(n, "flooding") for n in names]
                + [(n, "depthN") for n in names]
            ),
        }

        self.data_log = {
            "performance_measure": [],
            "simulation_time": [],
            "flow": {n: [] for n in names},
            "flooding": {n: [] for n in names},
            "depthN": {n: [] for n in names},
            "inflow": {n: [] for n in names},
            "spill": {n: [] for n in names},
            "valve": {n: [] for n in names},
            "rainfall": [],
        }
        self.env.reset()

    # -- pystorms surface -------------------------------------------------
    def state(self):
        return np.array([self.env.basins[n].depth for n in self.env.order])

    def performance(self, metric="cumulative"):
        return perf_metrics(self.data_log["performance_measure"], metric)

    def step(self, actions=None, log=True):
        if self._i >= self.n_steps:
            return True

        names = self.env.order
        if actions is None:
            valves = {n: 1.0 for n in names}
        elif isinstance(actions, dict):
            valves = {n: float(actions.get(n, 1.0)) for n in names}
        elif isinstance(actions, (list, tuple, np.ndarray)):
            arr = np.asarray(actions, dtype=float).ravel()
            if len(arr) != len(names):
                raise ValueError(
                    "expected {} actions, got {}".format(len(names), len(arr))
                )
            valves = dict(zip(names, arr))
        else:
            raise ValueError(
                "actions must be dict, list or np.ndarray; got {}".format(
                    type(actions)
                )
            )

        rain = self.rainfall[self._i]
        results = self.env.step(rain, valves)
        self._i += 1
        done = self._i >= self.n_steps

        if log:
            self.data_log["simulation_time"].append(self.env.t)
            self.data_log["rainfall"].append(float(rain))
            for n in names:
                r = results[n]
                self.data_log["flow"][n].append(r["outflow"])
                self.data_log["flooding"][n].append(r["flooding"])
                self.data_log["depthN"][n].append(r["depth"])
                self.data_log["inflow"][n].append(r["inflow"])
                self.data_log["spill"][n].append(r["spill"])
                self.data_log["valve"][n].append(r["valve"])

        self.data_log["performance_measure"].append(
            self._performance(results, done)
        )
        return done

    # -- objective --------------------------------------------------------
    def _performance(self, results, done):
        """Penalty for this timestep.

        Volumetric mode (default), all terms in m^3:

        * excess release  -- volume discharged above the threshold
        * flooding        -- volume lost out of the basin, weighted
        * terminal        -- volume left in storage at the end, weighted

        Weighting flooding rather than treating it as a hard failure keeps
        the objective differentiable-ish and lets a planner trade a small
        overtop against a large amount of downstream threshold violation,
        which is the tradeoff an operator actually faces.
        """
        dt = self.env.dt
        total = 0.0
        for name, r in results.items():
            if self.gamma_compatible:
                if r["flooding"] > 0.0:
                    total += GAMMA_FLOOD_PENALTY
                total += threshold(r["outflow"], self.flow_threshold,
                                   scaling=1.0)
                if done and r["depth"] > self.terminal_depth:
                    total += GAMMA_TERMINAL_PENALTY
                continue

            total += threshold(r["outflow"], self.flow_threshold,
                               scaling=1.0) * dt
            total += self.flood_weight * r["flooding"] * dt
            if done and r["depth"] > self.terminal_depth:
                basin = self.env.basins[name]
                held = basin.volume - basin.p.volume(self.terminal_depth)
                total += self.terminal_weight * max(held, 0.0)
        return total

    # -- convenience ------------------------------------------------------
    def rollout(self, controller=None):
        """Run to termination. ``controller`` maps state -> action array."""
        done = False
        while not done:
            actions = controller(self.state()) if controller else None
            done = self.step(actions)
        return self.performance()

    def summary(self):
        flows = np.array([self.data_log["flow"][n]
                          for n in self.env.order])
        floods = np.array([self.data_log["flooding"][n]
                           for n in self.env.order])
        spills = np.array([self.data_log["spill"][n]
                           for n in self.env.order])
        return {
            "performance": self.performance(),
            "peak_flow": float(flows.max()),
            "outfall_peak": float(np.max(self.data_log["flow"][self.env.outfall_from])),
            "exceedance_steps": int((flows > self.flow_threshold).sum()),
            "flood_volume_m3": float(floods.sum() * self.env.dt),
            "spill_volume_m3": float(spills.sum() * self.env.dt),
            "spill_fraction": float(
                spills.sum() / flows.sum()) if flows.sum() > 0 else 0.0,
            "final_depth_max": float(max(
                self.data_log["depthN"][n][-1] for n in self.env.order)),
        }


class SingleBasin(PondScenario):
    """One basin, one catchment -- the P1 testbed."""

    def __init__(self, rainfall=None, dt=60.0, flow_threshold=1.0, **kw):
        if rainfall is None:
            rainfall = design_storm(T=5.0, duration_hr=6.0, dt=dt)
        super().__init__(single_basin(dt=dt), rainfall,
                         flow_threshold=flow_threshold, **kw)
        self.config["name"] = "single_basin"


class GammaLike(PondScenario):
    """Eleven basins on gamma's topology, with known travel times."""

    def __init__(self, rainfall=None, dt=60.0, flow_threshold=1.0,
                 celerity=1.5, flow_dependent=False, **kw):
        if rainfall is None:
            rainfall = design_storm(T=10.0, duration_hr=6.0, dt=dt)
        net = gamma_like(dt=dt, celerity=celerity, flow_dependent=flow_dependent)
        super().__init__(net, rainfall, flow_threshold=flow_threshold, **kw)
        self.config["name"] = "gamma_like"


def sampled_scenario(cls, sampler=None, split="train", double_peak=False,
                     seed=0, **kw):
    """Build a scenario driven by a sampled storm rather than a design event."""
    sampler = sampler or StormSampler(dt=kw.get("dt", 60.0), seed=seed)
    event = sampler.sample(split=split, double_peak=double_peak)
    scen = cls(rainfall=event["intensity"], **kw)
    scen.event = event
    return scen
