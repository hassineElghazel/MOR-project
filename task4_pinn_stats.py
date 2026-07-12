#!/usr/bin/env python3
"""
Task 4 (PINN) summary statistics -> data/pinn_summary_stats.csv

Two blocks, both on the SAME 150 held-out test parameters / same FE error
machinery as ROM/POD-NN:
  * milestones   -- the key single-number results of the PINN study
  * distribution -- min/q1/median/mean/q3/max/std of the final hybrid PINN's
                    accuracy, plus its online/offline cost and speed-up.

Run with:  python task4_pinn_stats.py
"""
from __future__ import annotations
import csv
import numpy as np
from src import Config

STATS = ["min", "q1", "median", "mean", "q3", "max", "std"]


def dist(a):
    a = np.atleast_1d(np.asarray(a, float))
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return [a.min(), q1, med, a.mean(), q3, a.max(), a.std()]


def main() -> None:
    c = Config()
    err = np.load(c.data_dir / "pinn_test_errors.npz")     # final hybrid, 150 pts
    tim = np.load(c.data_dir / "pinn_timing.npz")
    fom_mean = float(tim["fom_mean"]); pred = tim["predict_times"]
    train_time = float(tim["train_time"]); speedup = float(tim["speedup"])

    rows = []
    # --- milestones -------------------------------------------------------
    rows.append(["milestone", "physics-only PINN (single mu)", "rel_l2_u_mean", 0.384])
    rows.append(["milestone", "physics-only PINN (parametric, 150)", "rel_l2_u_mean", 1.00])
    rows.append(["milestone", "supervised fit (network capacity floor)", "rel_l2_u", 0.008])
    rows.append(["milestone", "hybrid single-mu (in-sample, 250 data pts)", "rel_l2_u", 0.0004])
    rows.append(["milestone", "data-only PINN best (N=150)", "rel_l2_u_mean_150", 0.418])
    rows.append(["milestone", "final hybrid PINN (parametric, 150)", "rel_l2_u_mean_150",
                 float(err["rel_l2_u"].mean())])

    # --- final hybrid accuracy distribution (150 test pts) ----------------
    acc_rows = []
    for key, label in [("rel_l2_u", "rel L2(u)"), ("rel_l2_p", "rel L2(p)"),
                       ("rel_h1_u", "rel H1(u)")]:
        acc_rows.append(["accuracy", label] + dist(err[key]))

    # --- cost -------------------------------------------------------------
    cost_rows = [
        ["cost", "online predict [ms]"] + [1e3 * v for v in dist(pred)],
        ["cost", "offline train [s]"] + [train_time] * 7,
        ["cost", "FOM online [ms] / speedup",
         "", "", "", 1e3 * fom_mean, "", "", f"{speedup:.1f}x"],
    ]

    out = c.data_dir / "pinn_summary_stats.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block", "quantity", "metric/unit"] + STATS)
        for r in rows:                       # milestones: value in 'mean' col
            w.writerow([r[0], r[1], r[2], "", "", "", r[3], "", "", ""])
        w.writerow([])
        w.writerow(["block", "quantity/metric"] + STATS + [""])
        for r in acc_rows + cost_rows:
            w.writerow(r)
    print(f"Saved {out}")
    # echo a readable summary
    print("\nFinal HYBRID PINN on 150 disjoint test params (same FE metrics):")
    print(f"  rel L2(u):  mean={err['rel_l2_u'].mean():.4f}  median={np.median(err['rel_l2_u']):.4f}  max={err['rel_l2_u'].max():.4f}")
    print(f"  rel L2(p):  mean={err['rel_l2_p'].mean():.4f}   rel H1(u): mean={err['rel_h1_u'].mean():.4f}")
    print(f"  online: {1e3*pred.mean():.3f} ms  (FOM {1e3*fom_mean:.1f} ms -> {speedup:.0f}x speedup)")
    print(f"  offline (training): {train_time:.0f} s")


if __name__ == "__main__":
    main()
