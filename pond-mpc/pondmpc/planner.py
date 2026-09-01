"""Model-predictive control by MPPI over valve trajectories.

The planner holds its own copy of the network and advances it alongside the
plant, correcting basin volumes from measured depths at each control step.
Everything it cannot measure -- water in transit down a reach, water still
in a catchment's response -- is carried by the model. That is precisely
what the routing identification buys: without a travel-time model the
in-transit water is invisible, and a controller can only react to what has
already arrived.

Sampling is over open-loop valve trajectories; the cost mirrors the
scenario objective (release above threshold, flooding) plus a terminal term
on stored volume, without which the planner hoards water it will not be
charged for until after the horizon ends.
"""
import numpy as np


class MPPI:
    def __init__(self, model, order, flow_threshold=1.0, n_blocks=36,
                 block=10, n_samples=48, sigma=0.25, temperature=None,
                 flood_weight=100.0, terminal_weight=0.35,
                 replan_every=30, seed=0, substeps=1, n_iters=2):
        self.model = model
        self.order = list(order)
        self.n = len(self.order)
        self.flow_threshold = float(flow_threshold)
        # Actions are planned in blocks: one setting held for `block` steps.
        # The episode is a 30-hour drawdown while a reach travels in under
        # 15 minutes, so a horizon fine enough to resolve routing is far too
        # short to see the drain deadline. Blocking buys both -- n_blocks *
        # block steps of lookahead at a fraction of the rollout cost.
        self.n_blocks = int(n_blocks)
        self.block = int(block)
        self.horizon = self.n_blocks * self.block
        self.n_samples = int(n_samples)
        self.n_iters = int(n_iters)
        self.sigma = float(sigma)
        self.flood_weight = float(flood_weight)
        self.terminal_weight = float(terminal_weight)
        self.replan_every = int(replan_every)
        self.substeps = int(substeps)
        self.rng = np.random.default_rng(seed)

        self.nominal = np.full((self.n_blocks, self.n), 0.5)
        self._since_replan = self.replan_every
        self._last_action = np.full(self.n, 0.5)
        self._k = 0
        # Scale for the softmax over trajectory costs. Set from the spread of
        # the sampled costs each step when not given, which keeps the weights
        # informative regardless of the objective's magnitude.
        self.temperature = temperature

    # -- forecast ---------------------------------------------------------
    def set_forecast(self, rainfall):
        """Rainfall the planner is allowed to see, indexed by episode step."""
        self.rainfall = np.asarray(rainfall, dtype=float)

    def _shift(self):
        """Warm start: drop the blocks already executed, extend with the last."""
        shift = max(self.replan_every // self.block, 1)
        self.nominal = np.vstack([self.nominal[shift:],
                                  np.repeat(self.nominal[-1:], shift, axis=0)])

    def _forecast(self, k):
        idx = np.arange(k, k + self.horizon)
        idx = np.clip(idx, 0, len(self.rainfall) - 1)
        f = self.rainfall[idx]
        if k + self.horizon > len(self.rainfall):
            f[len(self.rainfall) - k:] = 0.0
        return f

    # -- state estimation -------------------------------------------------
    def sync(self, depths):
        """Correct the model's basin volumes to the measured depths."""
        for name, d in zip(self.order, depths):
            b = self.model.basins[name]
            b.volume = b.p.volume(max(float(d), 0.0))

    def advance(self, rain, action):
        """Advance the model one step with the action actually applied."""
        self.model.step(rain, dict(zip(self.order, action)))

    # -- planning ---------------------------------------------------------
    def _rollout_cost(self, blocks, forecast, snap):
        self.model.restore(snap)
        dt = self.model.dt
        cost = 0.0
        t = 0
        for bi in range(self.n_blocks):
            act = dict(zip(self.order, blocks[bi]))
            for _ in range(self.block):
                res = self.model.step(forecast[t], act)
                t += 1
                for r in res.values():
                    over = r["outflow"] - self.flow_threshold
                    if over > 0.0:
                        cost += over * dt
                    if r["flooding"] > 0.0:
                        cost += self.flood_weight * r["flooding"] * dt
        # Water left in storage is a liability the horizon does not see.
        cost += self.terminal_weight * sum(
            self.model.basins[n].volume for n in self.order)
        return cost

    def plan(self, depths, k):
        self.sync(depths)
        self._shift()
        snap = self.model.snapshot()
        forecast = self._forecast(k)

        for _ in range(self.n_iters):
            noise = self.rng.normal(
                0.0, self.sigma,
                size=(self.n_samples, self.n_blocks, self.n))
            samples = np.clip(self.nominal[None] + noise, 0.0, 1.0)
            costs = np.array([self._rollout_cost(s, forecast, snap)
                              for s in samples])
            lam = self.temperature
            if lam is None:
                lam = max(costs.std(), 1e-6)
            w = np.exp(-(costs - costs.min()) / lam)
            w /= w.sum()
            self.nominal = np.clip(
                np.einsum("s,sbh->bh", w, samples), 0.0, 1.0)
        self.model.restore(snap)
        return self.nominal[0].copy()

    # -- controller interface --------------------------------------------
    def __call__(self, depths):
        if self._since_replan >= self.replan_every:
            self._last_action = self.plan(depths, self._k)
            self._since_replan = 0
        else:
            # Between replans, follow the plan already computed.
            bi = min(self._since_replan // self.block, self.n_blocks - 1)
            self._last_action = self.nominal[bi].copy()
        self._since_replan += 1
        self._k += 1
        return self._last_action

    def after_step(self, rain, action):
        self.advance(rain, action)
