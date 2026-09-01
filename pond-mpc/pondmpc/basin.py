"""Single detention basin: storage, gated orifice, spillway, flooding.

The physics here is the *known* part of the problem. P0 uses it as ground
truth: the learned basin model (JEPA) and the identified routing model are
both validated against a simulator whose parameters we chose, before either
is trusted on SWMM.

SI units throughout -- metres, seconds, m^2, m^3, m^3/s.
"""
import numpy as np

G = 9.81


class BasinParams:
    """Geometry and hydraulics of one basin.

    Storage is a power law ``V = k_s * h**b_s``. ``b_s = 1`` is a prismatic
    basin (constant plan area ``k_s``); ``b_s > 1`` widens with depth, which
    is the usual shape of a real detention pond and the reason a controller
    tuned at low stage misbehaves at high stage.

    The outlet is a gated orifice. With no spillway the discharge relation
    is ``Q / q_max = u * sqrt(h / h_max)`` for every basin, with no free
    parameters -- which is what makes a single pretrained basin model
    possible.
    """

    def __init__(self, name, k_s=2000.0, b_s=1.15, max_depth=4.0,
                 orifice_area=0.5, orifice_coeff=0.65,
                 weir_crest=None, weir_length=0.0, weir_coeff=1.7):
        self.name = name
        self.k_s = float(k_s)
        self.b_s = float(b_s)
        self.max_depth = float(max_depth)
        self.orifice_area = float(orifice_area)
        self.orifice_coeff = float(orifice_coeff)
        # No spillway by default: the outlet is the gated orifice, and
        # water above max_depth is flooding. A weir is a second, ungated
        # discharge path whose capacity relative to the orifice varies over
        # two orders of magnitude across plausible geometries, and it is
        # the only thing that stops the outflow relation collapsing onto
        # one curve for every basin. Set weir_length > 0 to restore it and
        # measure what it costs.
        self.weir_crest = (float(weir_crest) if weir_crest is not None
                           else 0.9 * self.max_depth)
        self.weir_length = float(weir_length)
        self.weir_coeff = float(weir_coeff)

    @property
    def max_volume(self):
        return self.volume(self.max_depth)

    def volume(self, depth):
        d = max(float(depth), 0.0)
        return self.k_s * d ** self.b_s

    def depth(self, volume):
        v = max(float(volume), 0.0)
        return (v / self.k_s) ** (1.0 / self.b_s)

    def surface_area(self, depth):
        d = max(float(depth), 1e-6)
        return self.k_s * self.b_s * d ** (self.b_s - 1.0)


class Basin:
    """Stateful basin integrated with RK4 on volume.

    ``valve`` is the orifice setting in [0, 1]. Outflow is the orifice
    discharge plus any spillway flow; both continue downstream. Volume above
    ``max_volume`` is flooding and leaves the system.
    """

    def __init__(self, params, init_depth=0.0):
        self.p = params
        self.volume = self.p.volume(init_depth)
        self.valve = 1.0
        self.flooding = 0.0   # most recent flood rate, m^3/s
        self.outflow = 0.0    # most recent total release, m^3/s
        self.spill = 0.0      # of which passed over the spillway, m^3/s

    def reset(self, init_depth=0.0):
        self.volume = self.p.volume(init_depth)
        self.valve = 1.0
        self.flooding = 0.0
        self.outflow = 0.0
        self.spill = 0.0

    @property
    def depth(self):
        return self.p.depth(self.volume)

    def orifice_flow(self, depth, valve):
        if depth <= 0.0:
            return 0.0
        v = min(max(float(valve), 0.0), 1.0)
        return v * self.p.orifice_coeff * self.p.orifice_area * np.sqrt(2.0 * G * depth)

    def weir_flow(self, depth):
        """Spillway discharge. Zero unless a weir was configured, in which
        case outflow above the crest is orifice plus weir."""
        if self.p.weir_length <= 0.0:
            return 0.0
        head = depth - self.p.weir_crest
        if head <= 0.0:
            return 0.0
        return self.p.weir_coeff * self.p.weir_length * head ** 1.5

    def _release(self, volume, valve):
        d = self.p.depth(volume)
        return self.orifice_flow(d, valve) + self.weir_flow(d)

    def step(self, inflow, valve, dt, substeps=4):
        """Advance ``dt`` seconds with constant inflow and valve setting.

        Returns ``(outflow, flood_rate)`` averaged over the step, so that
        mass balance closes at the step level rather than only in the limit.

        Substeps matter only once flooding is active: capping the volume at
        ``max_volume`` is a discontinuity, so the flood term converges
        first-order in the substep size while everything else is fifth. Four
        substeps sits within 0.08% of the next refinement on the gamma-like
        network; one substep is 0.3% low. Raise it if dt is increased.
        """
        self.valve = min(max(float(valve), 0.0), 1.0)
        h = dt / substeps
        released = 0.0
        flooded = 0.0
        spilled = 0.0

        for _ in range(substeps):
            v = self.volume

            def deriv(vol):
                return inflow - self._release(max(vol, 0.0), self.valve)

            k1 = deriv(v)
            k2 = deriv(v + 0.5 * h * k1)
            k3 = deriv(v + 0.5 * h * k2)
            k4 = deriv(v + h * k3)
            v_new = v + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            if v_new < 0.0:
                v_new = 0.0
            # Release over the sub-step recovered from the volume change, so
            # the reported outflow is exactly what left the basin.
            out = inflow - (v_new - v) / h

            flood = 0.0
            if v_new > self.p.max_volume:
                flood = (v_new - self.p.max_volume) / h
                v_new = self.p.max_volume

            # Split the release into the part the valve controls and the
            # part that went over the spillway. A controller that "meets"
            # the threshold by spilling is not controlling anything, so the
            # two are tracked separately. Trapezoidal over the sub-step.
            spill_rate = 0.5 * (self.weir_flow(self.p.depth(v))
                                + self.weir_flow(self.p.depth(v_new)))

            self.volume = v_new
            released += max(out, 0.0) * h
            flooded += flood * h
            spilled += min(spill_rate, max(out, 0.0)) * h

        self.outflow = released / dt
        self.flooding = flooded / dt
        self.spill = spilled / dt
        return self.outflow, self.flooding
