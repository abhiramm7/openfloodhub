# pond-mpc — handoff brief

Read this before touching anything. It covers what the project is, what is
already established, the mistakes already made (do not repeat them), and
what to do next.

Rename or symlink this to `CLAUDE.md` if your agent auto-loads that.

---

## 1. What this is

Model-based control of a distributed stormwater network. The architecture
is deliberately split in two, because the physics is:

| part | transfers across sites? | how it is obtained |
|---|---|---|
| **basin dynamics** | yes — same equation, different constants | pretrain once, reuse |
| **reach travel times** | no — it is this network's geometry | identify per site from a short campaign |

That asymmetry is the thesis. One half is universal; the other is cheap to
identify. It is not "we used method X for one part and method Y for the
other."

The eventual target is the `gamma` scenario in
[pystorms](https://github.com/kLabUM/pystorms) — 11 storage basins on a
tree, each with a gated bottom orifice, objective penalizing release above
a threshold at *every* basin. This repo is a pure-NumPy replica of that
plant (same topology, same conduit lengths) so models and planners can be
built against a system whose parameters are known, before running on SWMM.

**Paper framing is positive, not anti-RL.** No RL baselines. The claim is
that a composed, identified model controls this network well; the
comparison set is uncontrolled, static schedules, and an oracle.

---

## 2. Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest tests -q          # 33 tests, ~50 s
```

`torch` is declared under the `learn` extra but **nothing imports it yet**.
Everything so far is NumPy. Install it when you start the JEPA arm.

Quick looks:

```bash
.venv/bin/python scripts/demo.py            # baseline table
.venv/bin/python scripts/identify_lags.py   # travel-time identifiability
.venv/bin/python scripts/results.py         # full sweep — 3.8 HOURS
```

`results.json` is committed, so the README numbers need no rerun.

---

## 3. Layout

| module | role |
|---|---|
| `basin.py` | storage / orifice physics, RK4 on volume |
| `routing.py` | reach = translation (travel time) + linear reservoir |
| `network.py` | basins wired by reaches; `gamma_like()` preset; snapshot/restore |
| `storms.py` | hyetographs, Nash-cascade runoff, train/test splits, back-to-back |
| `scenarios.py` | pystorms-compatible wrappers and the objective |
| `controllers.py` | uncontrolled, fixed, equal-filling, local threshold |
| `excitation.py` | PRBS / multi-level / chirp valve schedules for system ID |
| `randomize.py` | randomized basins + dimensionless groups, for pretraining |
| `identify.py` | sparse identification of travel times and storage constants |
| `surrogate.py` | a `Network` carrying identified parameters, for planning |
| `planner.py` | CEM over valve trajectories |

The scenario API mirrors `pystorms.scenarios.scenario` (`state()`,
`step(actions) -> done`, `performance(metric)`, `data_log`) so controllers
port to SWMM unchanged. **It matches what pystorms documents, not what it
does** — upstream mishandles dict actions, has an off-by-one in its
timestep guard, and omits `pystorms.scenarios` from its install list.

---

## 4. What is established

**Basin collapse is exact.** With a gated orifice as the only outlet,
`Q/q_max = u*sqrt(h/h_max)` holds identically for every basin — verified
to 2.2e-16 over 200 random draws. No conditioning, no dimensionless groups.
This is what makes one pretrained basin model possible.

**All ten travel times recovered exactly**, to the timestep, from three
excited events. Storage constant within ~4%.

| method | travel time | k (true 180 s) |
|---|---|---|
| cross-correlation | 184 s mean error | not identifiable |
| sparse fit, continuous form | 10/10 exact | 311 ± 35 |
| sparse fit, discrete form | **10/10 exact** | **187 ± 7** |

**Control results** (static policies tuned on train storms T=1–5, scored on
held-out T=25–100 plus double-peak and back-to-back):

| controller | train mean | test mean |
|---|---|---|
| uncontrolled | 81,767 | 634,326 |
| uniform (train-tuned) | 8,286 | 1,441,728 |
| per-basin static (train-tuned) | **0** | 1,544,365 |
| local threshold | 45,627 | 1,861,085 |
| per-basin static (hindsight — saw the test storm) | 0 | 885,707 |
| **CEM + identified model** | 176 | **242,657** |
| CEM + true model (oracle) | 259 | 244,304 |

Read those two columns together. In distribution a static lookup table is
*perfect* and the planner is pointless. Out of distribution every static
policy is worse than doing nothing, because a throttle tuned on a 5-year
event fills the basins on a 100-year one and they overtop. And a static
schedule fitted to the very storm it is scored on still loses by 3.6x — so
the gain is feedback, not tuning.

Identified model ≈ true model (0.7% apart, ordering flips between storms).
The residual gap to optimal is planning error, not model error.

---

## 5. Gotchas

- **No spillway, by design.** The outlet is an orifice; above capacity a
  basin floods. A weir is a second ungated path whose capacity relative to
  the orifice spans 180x across plausible geometries, and it is the only
  thing that breaks the collapse. `random_basin_params(..., spillway=True)`
  turns it back on to measure that cost. Do not re-enable it by default.

- **Basin sizes are derived, not chosen** (`scripts/size_basins.py`):
  storage is 35% of the design event's runoff volume from each basin's
  *cumulative* drainage area, orifices sized so a full basin at full open
  passes 3x the threshold. Sizing off local area alone makes the scenario
  infeasible — every uniform valve setting was worse than leaving the
  valves open. Re-run that script if you change the topology or forcing.

- **Score identification against `implemented_delays()`, not
  `travel_times()`.** The delay line rounds to whole steps. Scoring against
  the nominal length/celerity shows a spurious sub-step bias.

- **Only the flood term is substep-sensitive.** Clipping volume at capacity
  is a discontinuity, so it converges first-order while everything else is
  fifth. Default is 4 substeps; 1 is 0.3% low. Raise it if `dt` grows.

- **The objective is volumetric** (every term in m³), unlike gamma's flat
  1e6-per-flooding-timestep, which is near-lexicographic and hard to plan
  against. `gamma_compatible=True` reproduces gamma's exact penalty for the
  like-for-like comparison.

- **`sync()` corrects basin volumes only** — not reach buffers or catchment
  stores. That is correct (the model carries the unobservable state), but
  it means **you must not reuse one model object across planner
  instances**. Doing so made one trajectory score 4,318 / 5,534 / 5,516 on
  what should have been a single state. Build a fresh model per planner.

- **Difficulty is set by the drain deadline**, not storm size. With 24 h of
  recession a constant valve schedule is optimal and the timing problem
  disappears; with 8 h the same schedule is worse than doing nothing.

---

## 6. Dead ends — do not redo these

- **All candidate lags in one sparse-regression library.** Adjacent lags of
  a smooth signal are near-collinear; least squares spreads weight across a
  band with alternating signs and the storage term gets swamped. Scan one
  lag at a time.

- **Continuous-derivative form for identification.** Gets the lag right but
  k 73% high. The reach *is* a discrete linear reservoir — fit
  `y[t+1] = c0*y[t] + c*Q_up[t-L]` and read k off the pole.

- **Restricting identification to dry weather** to remove the runoff
  nuisance. Also removes the strongest excitation. Under the discrete form
  it makes no difference; under the continuous form it collapses the
  estimates. No reason to do it.

- **MPPI's softmax update.** With a few dozen samples in a few hundred
  dimensions the weights are diffuse, and averaging trajectories that
  differ in *which* basin they throttle yields one that throttles none
  properly — it returned a plan costing 8,642 from a nominal costing 6,288.
  Replaced with elite selection plus a guard that never returns worse than
  the best trajectory evaluated.

- **i.i.d. per-block action noise.** Good policies here are nearly constant
  in time. Sampling spends the whole budget on jagged trajectories. Use a
  per-basin constant offset plus AR(1)-smoothed variation.

- **Under-budgeting the search.** Convergence against iterations on one
  state: 6,288 (1), 5,335 (3), 4,799 (5), 4,351 (8), vs 4,318 for the
  static optimum. And replanning frequency matters more than sample count:
  64 samples / 12-4 iters / replan 60 scored 400 where 48 / 10-3 / replan
  90 scored 6,504 on the same storm. Do not cheapen the planner to save
  wall-clock and then read the result as a property of the method.

---

## 7. Next steps, in priority order

**1. Time-varying terminal weight (highest value, small change).**
The planner's residual cost is almost entirely the terminal penalty for
undrained storage — flooding is 0–6% of it, threshold violation is small.
It finishes the big test storms with 2.1–2.4 m still impounded. It charges
0.08 per m³ at its horizon where the scenario charges 10 per m³ at episode
end, 125x more. That constant was swept on a train storm where holding
water is free. Make it rise as the drain deadline approaches; a proper
cost-to-go is: free if there is time to release below threshold, full
penalty if not. **The published test numbers are a floor because of this.**

**2. Batched/vectorized simulator.** `Network.step` is ~515 µs (Python
loop), which is why the sweep takes 3.8 h and why the planner had to be
cheapened. Vectorize across rollouts (arrays of shape `(B, 11)`) and keep
the readable implementation as the reference the batched one is tested
against. This unblocks everything else.

**3. JEPA basin arm.** Currently the basin model is the analytic form. The
honest experiment is whether a learned latent model matches it — and it
should be run under *partial observability* (say 40% of nodes gauged),
which is where a latent model can beat a fitted physical one. Pretrain on
randomized basins (`randomize.py`) in nondimensional coordinates; evaluate
zero-shot on held-out basins. The error-vs-context-length curve ("how long
until a new asset is controllable?") is the figure worth having.

**4. Port to pystorms `gamma`.** Benchmark SWMM throughput first — it
shapes the data budget. Vendor or fix the upstream bugs listed in §3 and
pin `pyswmm<2.0.0`. Report sim-to-sim transfer from this testbed as a
bonus result.

**5. Decentralized coordination.** Each basin plans locally and broadcasts
its predicted release schedule; downstream neighbors receive it
time-shifted by the identified travel time. If decentralized matches
centralized, the result scales past 11 basins. This is the most novel piece
and is not started.

---

## 8. Standing rules

- Do not tune the testbed until the method looks good. The calibration
  history is in the git log; changing basin sizes or the drain deadline
  changes what the results mean, so say so explicitly if you do.
- Every claim in the README is backed by a test or a committed script.
  Keep it that way — several "findings" here turned out to be measurement
  bugs (catastrophic cancellation in a finite difference, a discretization
  artifact in the reach, leaked state across planners), and each was caught
  by having a control to compare against. Add the control before believing
  the number.
