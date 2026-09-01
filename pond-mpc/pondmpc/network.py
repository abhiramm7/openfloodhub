"""A network of basins wired together by reaches.

Step order within one timestep, all explicit:

1. rainfall -> per-basin catchment runoff
2. each reach advances, delivering an earlier upstream release
3. basin inflow = local runoff + arrivals
4. each basin integrates and produces a release

Because reaches carry the previous step's releases, a control action taken
now cannot affect a downstream basin until its travel time has elapsed --
which is the property the whole project is about.
"""
import numpy as np

from .basin import Basin, BasinParams
from .routing import Reach
from .storms import Catchment


class Network:
    def __init__(self, basins, reaches, catchments, outfall_from, dt=60.0):
        self.basins = {b.p.name: b for b in basins}
        self.order = [b.p.name for b in basins]
        self.reaches = list(reaches)
        self.catchments = dict(catchments)
        self.outfall_from = outfall_from
        self.dt = float(dt)
        self._pending = {name: 0.0 for name in self.order}
        self.t = 0.0
        self.outfall_flow = 0.0

    # -- topology helpers -------------------------------------------------
    def upstream_of(self, name):
        return [r.source for r in self.reaches if r.target == name]

    def downstream_of(self, name):
        for r in self.reaches:
            if r.source == name:
                return r.target
        return None

    def travel_times(self):
        """Ground-truth lags, in seconds, keyed by (source, target)."""
        return {(r.source, r.target): r.travel_time for r in self.reaches}

    def longest_path_time(self):
        """Travel time from the most distant basin to the outfall.

        Sets the lower bound on a useful planning horizon.
        """
        memo = {}

        def to_outfall(name):
            if name in memo:
                return memo[name]
            nxt = self.downstream_of(name)
            if nxt is None:
                memo[name] = 0.0
            else:
                reach = next(r for r in self.reaches if r.source == name)
                memo[name] = reach.travel_time + to_outfall(nxt)
            return memo[name]

        return max(to_outfall(n) for n in self.order)

    # -- simulation -------------------------------------------------------
    def reset(self, init_depth=0.0):
        for b in self.basins.values():
            b.reset(init_depth)
        for r in self.reaches:
            r.reset(self.dt)
        for c in self.catchments.values():
            c.reset()
        self._pending = {name: 0.0 for name in self.order}
        self.t = 0.0
        self.outfall_flow = 0.0

    def step(self, rainfall_mm_hr, valves):
        """Advance one timestep.

        ``rainfall_mm_hr`` is a scalar (uniform over the network) or a dict
        keyed by basin name. ``valves`` is a dict keyed by basin name.

        Returns a dict of per-basin (inflow, outflow, flooding, depth).
        """
        dt = self.dt

        if np.isscalar(rainfall_mm_hr):
            rain = {n: float(rainfall_mm_hr) for n in self.order}
        else:
            rain = dict(rainfall_mm_hr)

        # 1. local runoff
        inflow = {}
        for name in self.order:
            c = self.catchments.get(name)
            inflow[name] = c.step(rain.get(name, 0.0), dt) if c else 0.0

        # 2 + 3. routed arrivals from the previous step's releases
        for reach in self.reaches:
            arriving = reach.step(self._pending[reach.source], dt)
            inflow[reach.target] += arriving

        # 4. integrate each basin
        results = {}
        releases = {}
        for name in self.order:
            b = self.basins[name]
            out, flood = b.step(inflow[name], valves.get(name, 1.0), dt)
            releases[name] = out
            results[name] = {"inflow": inflow[name], "outflow": out,
                             "flooding": flood, "spill": b.spill,
                             "depth": b.depth, "volume": b.volume,
                             "valve": b.valve}

        self._pending = releases
        self.outfall_flow = releases[self.outfall_from]
        self.t += dt
        return results


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def single_basin(dt=60.0, threshold_ready=True):
    """One basin on one catchment. The P1 unit test."""
    p = BasinParams("1", k_s=2200.0, b_s=1.15, max_depth=4.0,
                    orifice_area=0.45)
    basin = Basin(p)
    catch = Catchment(area_m2=5.0e5, runoff_coeff=0.35, k_s=1200.0)
    return Network([basin], [], {"1": catch}, outfall_from="1", dt=dt)


# Reach lengths (m) taken from the conduits in pystorms' gamma network, so
# the pure-Python testbed has the same geometry -- and therefore the same
# spread of travel times -- as the SWMM scenario we ultimately target.
GAMMA_REACHES = [
    ("9", "8", 268.65),
    ("8", "6", 624.11),
    ("7", "6", 140.89),
    ("6", "5", 1187.70),
    ("5", "4", 184.86),
    ("11", "10", 1113.63),
    ("10", "4", 635.00),
    ("4", "3", 931.25),
    ("3", "2", 182.46),
    ("2", "1", 797.24),
]

# Basin plan-area coefficient (m^2), max depth (m), orifice area (m^2),
# local catchment area (ha).
#
# Derived, not guessed: see scripts/size_basins.py. Storage is 35% of the
# design event's runoff volume from each basin's CUMULATIVE drainage area,
# and every orifice is sized so that a full basin at full open discharges
# three times the flow threshold. With no spillway there is no relief path,
# so a basin that is undersized for what drains into it simply floods --
# sizing off local area alone (the earlier table) left the downstream
# basins far too small and made the scenario infeasible.
GAMMA_BASINS = {
    "1":   (4310.7, 5.50, 0.444, 54.0),
    "2":   (3692.4, 5.31, 0.452, 26.4),
    "3":   (3378.4, 5.21, 0.457, 21.6),
    "4":   (3115.0, 5.12, 0.461, 42.0),
    "5":   (1916.2, 4.65, 0.483, 24.0),
    "6":   (1568.6, 4.48, 0.492, 33.6),
    "7":   (381.7, 3.64, 0.546, 16.8),
    "8":   (761.4, 3.99, 0.522, 19.2),
    "9":   (405.6, 3.67, 0.544, 18.0),
    "10":  (966.4, 4.13, 0.513, 28.8),
    "11":  (452.6, 3.72, 0.540, 20.4),
}


def gamma_like(dt=60.0, celerity=1.5, flow_dependent=False, seed=0):
    """The gamma topology in pure Python, with known travel times.

    Same tree and same conduit lengths as pystorms' gamma scenario; basin
    geometry is our own, since gamma's storage curves are tabular and we
    want a testbed whose parameters we control.
    """
    basins, catchments = [], {}
    for name, (k_s, max_depth, orifice, catch_ha) in GAMMA_BASINS.items():
        p = BasinParams(name, k_s=k_s, b_s=1.15, max_depth=max_depth,
                        orifice_area=orifice)
        basins.append(Basin(p))
        catchments[name] = Catchment(area_m2=catch_ha * 1.0e4,
                                     runoff_coeff=0.35, k_s=1200.0)

    reaches = [Reach(s, t, length, celerity=celerity,
                     flow_dependent=flow_dependent)
               for s, t, length in GAMMA_REACHES]

    # Keep basin order upstream-first so a single explicit pass is sensible.
    order = ["9", "11", "7", "8", "10", "6", "5", "4", "3", "2", "1"]
    basins.sort(key=lambda b: order.index(b.p.name))
    return Network(basins, reaches, catchments, outfall_from="1", dt=dt)
