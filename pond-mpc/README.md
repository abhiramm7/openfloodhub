# pond-mpc

Model-based control of distributed stormwater networks: a **learned latent
model of what a basin does**, composed with a **sparsely-identified model of
how long water takes to get to the next one**.

The target testbed is `gamma` from [pystorms](https://github.com/kLabUM/pystorms)
— eleven storage basins on a tree, each with a gated bottom orifice, with the
objective penalizing release above a threshold at *every* basin. This repo
starts with a pure-Python replica of that plant so models and planners can be
developed against ground truth before touching SWMM.

## Status

**P0 (this repo, complete):** pure-Python plant, pystorms-compatible API,
storm generator, excitation signals, baseline controllers, test suite.

Next: P1 batched simulator + single-basin model comparison · P2 JEPA basin
model + SINDy routing on the pure-Python gamma topology · P3 pystorms gamma.

## Install

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest tests -q
```

## Use

The scenario surface mirrors `pystorms.scenarios.scenario`, so a controller
written here runs unchanged on SWMM later:

```python
from pondmpc import GammaLike
from pondmpc.controllers import ThresholdHold

scen = GammaLike(flow_threshold=1.0)
ctrl = ThresholdHold.for_scenario(scen)

done = False
while not done:
    done = scen.step(ctrl(scen.state()))

print(scen.performance(), scen.summary())
```

`scripts/demo.py` prints the baseline table, `scripts/headroom.py`
decomposes what is winnable, and `scripts/identify_lags.py` runs the
travel-time identifiability check.

## Layout

| module | role |
|---|---|
| `basin.py` | storage / orifice physics, RK4 on volume |
| `routing.py` | reach = translation (travel time) + linear reservoir |
| `network.py` | basins wired by reaches; `gamma_like()` preset |
| `storms.py` | hyetographs, Nash-cascade runoff, train/test storm splits |
| `scenarios.py` | pystorms-compatible wrappers and the objective |
| `controllers.py` | uncontrolled, fixed, equal-filling, local threshold |
| `excitation.py` | PRBS / multi-level / chirp valve schedules for system ID |
| `randomize.py` | randomized basins + dimensionless groups, for pretraining |

## What P0 established

**The plant is calibrated so local control is not enough.** Storage is scarce
enough that throttling each basin to the threshold on its own starts to
overtop. A purely local controller (37,550) loses to a naive uniform valve
setting (19,288) despite having complete information about its own basin.

**Cross-correlation cannot read off travel time.** Correlating an upstream
release against a downstream inflow recovers *translation plus attenuation*,
overestimating the lag by almost exactly the reach's linear-reservoir
constant:

| `k_attenuation` | mean bias | sd |
|---|---|---|
| 0 s | +58 s | 14 s |
| 90 s | +124 s | 22 s |
| 180 s | +184 s | 22 s |
| 360 s | +304 s | 44 s |

The bias is systematic, not noise. Separating the two requires a model
carrying both a delayed term and a storage term — which is the argument for
the SINDy formulation over any peak-matching heuristic, and it is now
empirical rather than asserted.

**Excitation is load-bearing.** Under rainfall alone the lag estimate is off
by 373 s on average; under a multi-level valve schedule, 184 s. Rainfall is
correlated across the whole network, so storm response alone cannot separate
routing from common forcing. The identification campaign has to move valves.

**Difficulty is set by the drain deadline, not the storm.** With 24 h of
recession, a per-basin *constant* valve schedule tuned offline scores 2.8 —
the timing problem disappears entirely. With 8 h, the same schedule scores
400,822, worse than doing nothing, because it holds water it cannot release
in time. Recession length is the knob that decides whether this is a
scheduling problem at all.

**The nondimensional collapse is exact.** The outlet is a gated orifice and
nothing else — above capacity a basin floods rather than spilling. Scaling
each basin by its own constants (depth by `h_max`, flow by the full-open
discharge `q_max = Cd*A0*sqrt(2*g*h_max)`, time by `V_max/q_max`) then makes
the discharge relation

    Q / q_max = u * sqrt(h / h_max)

identically, for every basin, with no free parameters. Verified to 2.2e-16
over 200 randomly drawn basins. Pooling costs nothing against fitting one
basin, and there is nothing left to condition on — which is what makes a
single pretrained basin model possible.

A spillway destroys this, which is why there is not one. Turning the weir
back on (`random_basin_params(..., spillway=True)`) adds a second, ungated
discharge path whose capacity relative to the orifice spans 180x across
plausible geometries, and the pooled k-NN residual goes from the
single-basin floor to roughly 20x it. The weir code is retained so that
cost can be measured, but it is off by default.

This also sharpens the architecture. Basin dynamics transfer, because every
basin is the same equation with different constants. Reach travel times do
not — they are this network's geometry. So one half of the model can be
pretrained once and reused, and the other half has to be identified per
site from a short excitation campaign. That asymmetry, rather than the
choice of JEPA or SINDy, is the reason for splitting the model in two.

**Basin sizes are derived, not chosen.** With no relief path, an undersized
basin simply floods, so storage is set to 35% of the design event's runoff
volume from each basin's *cumulative* drainage area, and every orifice is
sized so a full basin at full open discharges three times the flow
threshold (`scripts/size_basins.py`). Sizing off local area alone left the
downstream basins far too small and made the scenario infeasible — every
uniform valve setting was worse than leaving the valves open.

The resulting scenario has the tension it needs. Throttling cuts the peak
release but fills the basins, and past a point they overtop:

| uniform setting | performance | peak release | flooding m³ |
|---|---|---|---|
| open | 131,369 | 2.62 | 0 |
| 0.70 | 82,497 | 2.01 | 0 |
| **0.50** | **53,743** | 1.50 | 283 |
| 0.40 | 175,147 | 1.20 | 1,277 |
| 0.25 | 702,426 | 0.75 | 4,276 |

A controller that cannot time its releases has to pick a point on this
curve. One that can should beat all of them.

## Deliberate differences from pystorms

- **Volumetric objective.** gamma's flat 1e6-per-flooding-timestep penalty
  makes the objective near-lexicographic and flat elsewhere, which is hard to
  plan against. Every term here is in m³. Pass `gamma_compatible=True` for
  gamma's exact penalty.
- **Spillway release is tracked separately from metered release.** A
  controller that "meets" the threshold by spilling is not controlling
  anything; `summary()["spill_fraction"]` exposes that.
- **Dict and array actions agree**, actions are length-checked, and the
  timestep bookkeeping has no off-by-one. Upstream pystorms has bugs in all
  three; this matches the documented contract rather than the behaviour.
