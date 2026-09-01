"""Results for the composed-model controller on the gamma-like network.

Protocol
--------
Static policies are tuned on the TRAIN storms and evaluated on the TEST
storms. Tuning a static schedule on the storm it is then scored on is not a
baseline, it is an oracle -- so that variant is reported separately, as an
upper bound on what any policy without feedback could achieve if it knew
the event in advance.

Controllers
-----------
uncontrolled        valves open
uniform (train)     one setting for the network, tuned on train storms
per-basin (train)   one setting per basin, coordinate descent on train
local threshold     each basin throttles to its own threshold, no coupling
MPPI + identified   plans with reaches identified from an excitation campaign
MPPI + true model   plans with the true reaches (oracle: isolates model error)
per-basin (oracle)  static schedule tuned on the storm it is scored on
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "scripts")
from run_controller import run_mppi

from pondmpc import GammaLike, StormSampler, back_to_back, gamma_like
from pondmpc.controllers import ThresholdHold
from pondmpc.excitation import ScheduledValves, multilevel
from pondmpc.identify import identify_network
from pondmpc.surrogate import surrogate_from_identification

THRESHOLD = 1.0
MPPI_KW = dict(n_blocks=36, block=10, n_samples=32, replan_every=60,
               n_iters=1, terminal_weight=0.35, sigma=0.25)


def make_storms():
    s = StormSampler(dt=60.0, seed=101)
    train = [("train T=%.0f D=%.1fh" % (e["return_period"], e["duration_hr"]),
              e["intensity"]) for e in (s.sample("train") for _ in range(3))]
    test = [("test T=%.0f D=%.1fh" % (e["return_period"], e["duration_hr"]),
             e["intensity"]) for e in (s.sample("test") for _ in range(3))]
    e = s.sample("test", double_peak=True)
    test.append(("test double-peak", e["intensity"]))
    test.append(("test back-to-back", back_to_back(T1=5.0, T2=10.0, dt=60.0)))
    return train, test


def score(rain, ctrl):
    sc = GammaLike(rainfall=rain, flow_threshold=THRESHOLD)
    sc.rollout(ctrl)
    return sc


def mean_perf(storms, ctrl_fn):
    return float(np.mean([score(r, ctrl_fn(r)).performance()
                          for _, r in storms]))


def tune_uniform(storms, grid):
    best = min(grid, key=lambda v: mean_perf(storms, lambda r, v=v:
                                             (lambda s: np.full(11, v))))
    return best


def tune_per_basin(storms, grid, x0, sweeps=2):
    x = np.full(11, x0)
    cur = mean_perf(storms, lambda r: (lambda s: x))
    for _ in range(sweeps):
        improved = False
        for i in range(11):
            base = x[i]
            best_g, best_v = base, cur
            for g in grid:
                if g == base:
                    continue
                x[i] = g
                v = mean_perf(storms, lambda r: (lambda s, xx=x.copy(): xx))
                if v < best_v - 1e-9:
                    best_g, best_v = g, v
            x[i] = best_g
            if best_v < cur - 1e-9:
                cur, improved = best_v, True
        if not improved:
            break
    return x, cur


def run_identification():
    logs = []
    s = StormSampler(dt=60.0, seed=11)
    for i in range(3):
        ev = s.sample("train")
        sc = GammaLike(rainfall=ev["intensity"], flow_threshold=THRESHOLD)
        sc.rollout(ScheduledValves(
            multilevel(sc.n_steps, 11, dwell_steps=15, seed=i)))
        logs.append(sc.data_log)
    net = gamma_like()
    return identify_network(logs, net), net


def summarize(sc):
    s = sc.summary()
    return {k: round(float(v), 3) for k, v in s.items()}


def main():
    t_start = time.time()
    train, test = make_storms()
    grid = list(np.round(np.linspace(0.1, 1.0, 8), 3))

    print("identifying reach routing from an excitation campaign...", flush=True)
    ident, true_net = run_identification()
    impl = true_net.implemented_delays()
    n_exact = sum(ident[k]["travel_time"] == impl[k] for k in impl)
    print("  travel times recovered exactly: %d/%d" % (n_exact, len(impl)),
          flush=True)
    for k in impl:
        print("    %-8s true %4.0f s   est %4.0f s   k %5.1f s"
              % ("->".join(k), impl[k], ident[k]["travel_time"],
                 ident[k]["k_from_b"]), flush=True)

    print("\ntuning static baselines on train storms...", flush=True)
    u_best = tune_uniform(train, grid)
    print("  best uniform: %.2f" % u_best, flush=True)
    x_best, x_val = tune_per_basin(train, grid, u_best)
    print("  best per-basin: %s (train mean %.1f)"
          % (np.round(x_best, 2).tolist(), x_val), flush=True)

    def mk_true():
        n = gamma_like(); n.reset(); return n

    def mk_ident():
        n = surrogate_from_identification(true_net, ident); return n

    controllers = {
        "uncontrolled": ("static", lambda r: None),
        "uniform (train)": ("static", lambda r: (lambda s: np.full(11, u_best))),
        "per-basin (train)": ("static", lambda r: (lambda s: x_best)),
        "local threshold": ("static", lambda r: ThresholdHold(
            [gamma_like().basins[n].p for n in gamma_like().order], THRESHOLD)),
        "MPPI + identified": ("mppi", mk_ident),
        "MPPI + true model": ("mppi", mk_true),
    }

    rows = {}
    for split, storms in (("train", train), ("test", test)):
        for name, (kind, fn) in controllers.items():
            for label, rain in storms:
                t0 = time.time()
                if kind == "static":
                    sc = score(rain, fn(rain))
                else:
                    sc = run_mppi(rain, fn, flow_threshold=THRESHOLD,
                                  substeps=1, seed=0, **MPPI_KW)
                rows.setdefault(name, {})[label] = summarize(sc)
                print("  %-20s %-22s perf=%10.1f  (%.0fs)"
                      % (name, label, sc.performance(), time.time() - t0),
                      flush=True)

    print("\ncomputing the hindsight-static upper bound...", flush=True)
    for label, rain in train + test:
        x, v = tune_per_basin([(label, rain)], grid, u_best, sweeps=1)
        sc = score(rain, lambda s, xx=x: xx)
        rows.setdefault("per-basin (oracle)", {})[label] = summarize(sc)
        print("  %-22s perf=%10.1f" % (label, sc.performance()), flush=True)

    out = {"identification": {"->".join(k): {"true_s": impl[k],
                                             "est_s": ident[k]["travel_time"],
                                             "k_s": ident[k]["k_from_b"]}
                              for k in impl},
           "n_exact": n_exact,
           "uniform_setting": float(u_best),
           "per_basin_setting": np.round(x_best, 3).tolist(),
           "results": rows,
           "mppi": MPPI_KW}
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 78)
    order = ["uncontrolled", "uniform (train)", "per-basin (train)",
             "local threshold", "MPPI + identified", "MPPI + true model",
             "per-basin (oracle)"]
    for split, storms in (("TRAIN", train), ("TEST", test)):
        print("\n%s" % split)
        header = "%-20s" % "controller" + "".join(
            "%14s" % l.replace("train ", "").replace("test ", "")
            for l, _ in storms) + "%14s" % "mean"
        print(header)
        for name in order:
            vals = [rows[name][l]["performance"] for l, _ in storms]
            print("%-20s" % name + "".join("%14.1f" % v for v in vals)
                  + "%14.1f" % np.mean(vals))
    print("\ntotal %.0f s" % (time.time() - t_start))


if __name__ == "__main__":
    main()
