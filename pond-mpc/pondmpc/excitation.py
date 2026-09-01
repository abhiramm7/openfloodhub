"""Excitation signals for system identification.

Travel time cannot be identified from storm response alone: rainfall is
correlated across the whole network, so an upstream release and a
downstream inflow rise together for reasons that have nothing to do with
routing. The lag is only identifiable if upstream valves move in ways the
rainfall does not explain.

These generators produce those movements. Getting this wrong is the single
most likely way for the identification step to fail, so it is a first-class
part of the testbed rather than an afterthought in a training script.
"""
import numpy as np


def prbs(n_steps, n_channels, dwell_steps=20, low=0.1, high=1.0, seed=0):
    """Pseudo-random binary valve schedule, independent per channel.

    ``dwell_steps`` sets how long a valve holds a setting. It must be short
    relative to the storm but long relative to the travel times you want to
    resolve, or the delayed signals all look alike.
    """
    rng = np.random.default_rng(seed)
    n_holds = int(np.ceil(n_steps / dwell_steps))
    holds = rng.choice([low, high], size=(n_holds, n_channels))
    return np.repeat(holds, dwell_steps, axis=0)[:n_steps]


def multilevel(n_steps, n_channels, dwell_steps=20, levels=(0.1, 0.35, 0.65, 1.0),
               seed=0):
    """Multi-level variant. Excites the nonlinear part of the orifice curve,
    which a two-level PRBS leaves poorly identified."""
    rng = np.random.default_rng(seed)
    n_holds = int(np.ceil(n_steps / dwell_steps))
    holds = rng.choice(np.asarray(levels), size=(n_holds, n_channels))
    return np.repeat(holds, dwell_steps, axis=0)[:n_steps]


def chirp(n_steps, n_channels, f_lo=1.0 / 600, f_hi=1.0 / 60, seed=0,
          low=0.1, high=1.0):
    """Frequency sweep per channel, phase-randomized across channels.

    Sweeping covers a band of timescales in one event, which is efficient
    when simulator time is expensive.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)
    out = np.empty((n_steps, n_channels))
    for c in range(n_channels):
        f = f_lo * (f_hi / f_lo) ** (t / max(n_steps - 1, 1))
        phase = rng.uniform(0, 2 * np.pi)
        wave = np.sin(2 * np.pi * np.cumsum(f) + phase)
        out[:, c] = low + (high - low) * (wave + 1.0) / 2.0
    return out


class ScheduledValves:
    """Replay a precomputed schedule as a controller callable."""

    def __init__(self, schedule):
        self.schedule = np.asarray(schedule, dtype=float)
        self.i = 0

    def __call__(self, state):
        row = self.schedule[min(self.i, len(self.schedule) - 1)]
        self.i += 1
        return row


def excited_rollout(scenario, kind="multilevel", dwell_steps=20, seed=0,
                    **kw):
    """Run a scenario under an excitation schedule and return its data log."""
    n = scenario.n_steps
    m = len(scenario.env.order)
    gen = {"prbs": prbs, "multilevel": multilevel, "chirp": chirp}[kind]
    schedule = gen(n, m, seed=seed, **(
        {"dwell_steps": dwell_steps} if kind != "chirp" else {}), **kw)
    scenario.rollout(ScheduledValves(schedule))
    return scenario.data_log
