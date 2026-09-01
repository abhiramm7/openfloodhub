"""Reach routing: the part of the network a controller has to wait for.

A reach is a pure translation (travel time) followed by a linear reservoir
(attenuation). The travel time is *set here and known*, which is the whole
point of P0: SINDy has to recover it before we trust the identification on
SWMM, where no ground truth exists.

Optionally the celerity rises with flow, so travel time shortens as the
conduit fills. That is a real hydraulic effect and the target of one of the
paper's figures, but it is off by default -- a constant lag gives an
unambiguous number to recover first.
"""
import numpy as np


class Reach:
    """Delay line plus linear reservoir between two basins.

    Parameters
    ----------
    length_m
        Conduit length. Travel time is ``length / celerity``; keeping length
        explicit means the recovered lags can be plotted against a quantity
        that is known in the field.
    celerity
        Wave speed (m/s) at reference flow.
    k_attenuation
        Linear reservoir residence time (s). Spreads the wave without
        changing its arrival time much.
    flow_dependent
        If true, celerity scales with flow as
        ``c = celerity * (1 + alpha * Q / q_ref)``, capped at ``max_speedup``.
    """

    def __init__(self, source, target, length_m, celerity=1.5,
                 k_attenuation=180.0, flow_dependent=False, q_ref=2.0,
                 alpha=0.6, max_speedup=2.0):
        self.source = source
        self.target = target
        self.length_m = float(length_m)
        self.celerity = float(celerity)
        self.k_attenuation = float(k_attenuation)
        self.flow_dependent = bool(flow_dependent)
        self.q_ref = float(q_ref)
        self.alpha = float(alpha)
        self.max_speedup = float(max_speedup)
        self._buffer = None
        self._storage = 0.0

    @property
    def travel_time(self):
        """Nominal (reference-flow) travel time in seconds."""
        return self.length_m / self.celerity

    def lag_steps(self, dt, flow=0.0):
        c = self.celerity
        if self.flow_dependent and flow > 0.0:
            speedup = min(1.0 + self.alpha * flow / self.q_ref, self.max_speedup)
            c = self.celerity * speedup
        return int(round((self.length_m / c) / dt))

    def implemented_delay(self, dt):
        """Onset lag the discretized reach actually produces, in seconds.

        Differs from ``travel_time`` by the rounding of the delay line to a
        whole number of steps. This is what an identification method sees,
        so it is the honest target to score against.
        """
        return self.lag_steps(dt) * dt

    def reset(self, dt):
        n = max(self.lag_steps(dt), 0) + 2
        self._buffer = np.zeros(n)
        self._storage = 0.0

    def step(self, inflow, dt):
        """Push ``inflow`` (the upstream release) in; return what arrives now."""
        if self._buffer is None:
            self.reset(dt)

        self._buffer = np.roll(self._buffer, 1)
        self._buffer[0] = inflow

        lag = self.lag_steps(dt, flow=inflow)
        lag = min(max(lag, 0), len(self._buffer) - 1)
        delayed = self._buffer[lag]

        if self.k_attenuation <= 0.0:
            return delayed

        # Semi-implicit: solve S_new = S + dt*(delayed - S_new/k) rather than
        # taking the outflow from the storage *before* this step's arrival.
        # The explicit form costs a full extra timestep of lag, which shows
        # up as a one-step bias in anything that identifies the travel time
        # from data -- a discretization artifact masquerading as physics.
        self._storage = (self._storage + dt * delayed) / (1.0 + dt / self.k_attenuation)
        if self._storage < 0.0:
            self._storage = 0.0
        return self._storage / self.k_attenuation
