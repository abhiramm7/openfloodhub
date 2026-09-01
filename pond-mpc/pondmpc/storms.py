"""Synthetic storm forcing and rainfall-runoff transformation.

Everything here is deterministic given a seed, and cheap enough to generate
tens of thousands of events for training a dynamics model.

Units are SI throughout: depths in mm, intensities in mm/hr, flows in m^3/s,
areas in m^2, time in seconds.
"""
import numpy as np

# 6-hour point rainfall depths (mm) by return period, loosely NOAA Atlas 14
# for the mid-Atlantic. Used only to give the synthetic events a realistic
# magnitude ladder; nothing downstream depends on these being exact.
_IDF_6HR = {2: 55.0, 5: 70.0, 10: 82.0, 25: 98.0, 50: 112.0, 100: 127.0}


def depth_for_return_period(T, duration_hr=6.0):
    """Interpolate a design storm depth (mm) log-linearly in return period."""
    periods = np.array(sorted(_IDF_6HR))
    depths = np.array([_IDF_6HR[p] for p in periods])
    d6 = np.interp(np.log(T), np.log(periods), depths)
    # Duration scaling: depth grows sublinearly with duration (Bell-type).
    return float(d6 * (duration_hr / 6.0) ** 0.45)


def hyetograph(total_depth_mm, duration_hr, dt, peak_position=0.4, shape=4.0,
               second_peak=None):
    """Build a rainfall intensity series (mm/hr) of length ceil(duration/dt).

    The single-peak shape is a gamma-like pulse; ``peak_position`` in (0, 1)
    slides the peak through the event, which is the main knob that separates
    a front-loaded storm (hard to control, the pond fills before you can
    react) from a back-loaded one.

    ``second_peak`` is ``(position, fraction)`` and splits the depth into two
    pulses -- these are the events that break controllers tuned on design
    storms, so they are held out rather than trained on.
    """
    n = int(np.ceil(duration_hr * 3600.0 / dt))
    t = (np.arange(n) + 0.5) / n  # normalized time within the event

    def pulse(centre):
        # Gamma-shaped pulse centred (in the mode sense) on `centre`.
        k = shape
        theta = centre / max(k - 1.0, 1e-6)
        x = np.clip(t, 1e-9, None)
        y = x ** (k - 1.0) * np.exp(-x / theta)
        return y / y.sum() if y.sum() > 0 else np.ones(n) / n

    if second_peak is None:
        w = pulse(peak_position)
    else:
        pos2, frac = second_peak
        w = (1.0 - frac) * pulse(peak_position) + frac * pulse(pos2)

    # w sums to 1 over the event; convert the depth split to an intensity.
    return total_depth_mm * w / (dt / 3600.0)


class NashCascade:
    """n linear reservoirs in series -- the catchment's response function.

    This is the *known* part of the rainfall-runoff path. It gives the inflow
    hydrograph a physically sensible lag and attenuation without needing a
    full hydrologic model, and its state is small enough to be part of the
    environment's observable history.
    """

    def __init__(self, n=3, k_s=1200.0):
        self.n = int(n)
        self.k_s = float(k_s)  # residence time of each reservoir, seconds
        self.storage = np.zeros(self.n)

    def reset(self):
        self.storage[:] = 0.0

    def step(self, inflow, dt):
        """Advance one step, returning the outflow (same units as inflow)."""
        q = inflow
        for i in range(self.n):
            out = self.storage[i] / self.k_s
            self.storage[i] += dt * (q - out)
            if self.storage[i] < 0.0:
                self.storage[i] = 0.0
            q = out
        return q


class Catchment:
    """Rainfall (mm/hr) -> runoff (m^3/s) via losses plus a Nash cascade."""

    def __init__(self, area_m2=5.0e5, runoff_coeff=0.35,
                 initial_abstraction_mm=5.0, n_reservoirs=3, k_s=1200.0):
        self.area_m2 = float(area_m2)
        self.runoff_coeff = float(runoff_coeff)
        self.initial_abstraction_mm = float(initial_abstraction_mm)
        self.cascade = NashCascade(n_reservoirs, k_s)
        self._abstracted_mm = 0.0

    def reset(self):
        self.cascade.reset()
        self._abstracted_mm = 0.0

    def step(self, intensity_mm_hr, dt):
        depth_mm = intensity_mm_hr * dt / 3600.0
        # Soak up the initial abstraction before any runoff is generated.
        remaining = max(self.initial_abstraction_mm - self._abstracted_mm, 0.0)
        soaked = min(depth_mm, remaining)
        self._abstracted_mm += soaked
        effective_mm = (depth_mm - soaked) * self.runoff_coeff
        # mm over the catchment -> m^3/s
        gross = effective_mm / 1000.0 * self.area_m2 / dt
        return self.cascade.step(gross, dt)


class StormSampler:
    """Draws storm events. Splitting train/test on this object is the whole
    distribution-shift experiment, so keep the two sets defined here."""

    TRAIN_RETURN_PERIODS = (1.0, 2.0, 5.0)
    TEST_RETURN_PERIODS = (10.0, 25.0, 50.0, 100.0)

    def __init__(self, dt=60.0, seed=0):
        self.dt = float(dt)
        self.rng = np.random.default_rng(seed)

    def sample(self, split="train", double_peak=False):
        periods = (self.TRAIN_RETURN_PERIODS if split == "train"
                   else self.TEST_RETURN_PERIODS)
        T = float(self.rng.choice(periods))
        duration_hr = float(self.rng.uniform(3.0, 12.0))
        depth = depth_for_return_period(T, duration_hr)
        depth *= float(self.rng.uniform(0.85, 1.15))  # sampling variability
        peak_position = float(self.rng.uniform(0.25, 0.65))
        second = None
        if double_peak:
            second = (float(self.rng.uniform(0.6, 0.9)),
                      float(self.rng.uniform(0.3, 0.5)))
        series = hyetograph(depth, duration_hr, self.dt,
                            peak_position=peak_position, second_peak=second)
        return {"intensity": series, "return_period": T,
                "duration_hr": duration_hr, "depth_mm": depth,
                "peak_position": peak_position, "double_peak": second}


def design_storm(T=5.0, duration_hr=6.0, dt=60.0, peak_position=0.4):
    """A single reproducible event, for demos and regression tests."""
    depth = depth_for_return_period(T, duration_hr)
    return hyetograph(depth, duration_hr, dt, peak_position=peak_position)


def storm_sequence(specs, gap_hr=6.0, dt=60.0, peak_position=0.4):
    """Concatenate several events separated by dry gaps.

    Back-to-back storms are what turn a drawdown into a deadline: storage
    has to be recovered before the next peak arrives, and how fast you can
    recover it is limited by the release threshold. With a single event and
    a long recession there is no deadline, and release timing stops
    mattering -- a constant valve schedule is then optimal.

    ``specs`` is a list of ``(return_period, duration_hr)`` or
    ``(return_period, duration_hr, peak_position)``.
    """
    pieces = []
    gap = np.zeros(int(round(gap_hr * 3600.0 / dt)))
    for i, spec in enumerate(specs):
        if len(spec) == 3:
            T, dur, pos = spec
        else:
            (T, dur), pos = spec, peak_position
        depth = depth_for_return_period(T, dur)
        pieces.append(hyetograph(depth, dur, dt, peak_position=pos))
        if i < len(specs) - 1:
            pieces.append(gap)
    return np.concatenate(pieces)


def back_to_back(T1=5.0, T2=10.0, duration_hr=6.0, gap_hr=6.0, dt=60.0):
    """The standard two-event forcing used to create a drawdown deadline."""
    return storm_sequence([(T1, duration_hr), (T2, duration_hr)],
                          gap_hr=gap_hr, dt=dt)
