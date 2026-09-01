"""Reference controllers. None of these learn anything.

They exist so that every later result has a floor (uncontrolled), a
credible operational baseline (equal-filling), and a sanity check that the
control authority in the plant is enough to matter at all.
"""
import numpy as np


class Uncontrolled:
    """All valves open. The do-nothing floor."""

    def __init__(self, n):
        self.n = n

    def __call__(self, state):
        return np.ones(self.n)


class FixedValve:
    """Every valve held at the same setting. Sweeping this is the cheapest
    way to check whether the scenario needs coordination or just throttling."""

    def __init__(self, n, setting=0.3):
        self.n = n
        self.setting = float(setting)

    def __call__(self, state):
        return np.full(self.n, self.setting)


class EqualFilling:
    """Release harder from the fullest basins, measured against the network
    mean depth. The standard rule-based comparison in the literature."""

    def __init__(self, names, max_depths, gain=1.5, floor=0.0):
        self.names = list(names)
        self.max_depths = np.asarray(max_depths, dtype=float)
        self.gain = float(gain)
        self.floor = float(floor)

    def __call__(self, state):
        fill = np.asarray(state, dtype=float) / self.max_depths
        mean_fill = fill.mean()
        actions = self.floor + self.gain * (fill - mean_fill) + 0.5
        return np.clip(actions, 0.0, 1.0)


class ThresholdHold:
    """Close each valve just enough that this basin's own release stays under
    the threshold. Purely local: no basin knows anything about any other.

    This is the baseline a coordinating controller has to beat, so it is
    written to read the depths out of the ``state`` vector it is handed
    rather than holding references to basin objects -- pointing it at a
    different scenario instance used to make it silently do nothing.
    """

    def __init__(self, params, threshold):
        self.params = list(params)
        self.threshold = float(threshold)

    @classmethod
    def for_scenario(cls, scenario):
        return cls([scenario.env.basins[n].p for n in scenario.env.order],
                   scenario.flow_threshold)

    def __call__(self, state):
        depths = np.asarray(state, dtype=float)
        actions = np.ones(len(self.params))
        for i, p in enumerate(self.params):
            d = max(float(depths[i]), 0.0)
            free_flow = p.orifice_coeff * p.orifice_area * np.sqrt(2.0 * 9.81 * d)
            if free_flow > self.threshold:
                actions[i] = self.threshold / free_flow
        return np.clip(actions, 0.0, 1.0)
