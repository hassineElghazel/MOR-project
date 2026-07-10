#!/usr/bin/env python3
"""
Task 3 extension: full statistical summary (min/Q1/median/mean/Q3/max/std)
of accuracy and computational cost for FOM, ROM, and PODNN, plus a 3D
error-surface plot over the (mu0, mu1) test-parameter plane.

This script performs no new solves -- like task3_comparison.py, it only
loads artefacts already saved by Task 1 (`test_errors.npz`,
`timing_data.npz`) and Task 2 (`podnn_test_errors.npz`, `podnn_timing.npz`),
all of which share identical held-out test parameters, identical FOM ground
truth, and identical error machinery (see task3_comparison.py's header for
the full alignment argument). It only differs from task3_comparison.py in
reporting: instead of just the mean/max, it reports the full 7-statistic
suite, and instead of a 2D scatter it plots 3D error surfaces.

FOM cost convention (user-confirmed): FOM's *online* cost is its
per-solve time distribution (150 held-out solves, same as everyone else's
test set); FOM's *offline* cost is exactly 0 -- the FOM has no reusable
precomputation, every solve is a complete standalone computation, so
reporting anything else would misrepresent it as having an offline
investment it doesn't have.

Single-scalar costs (FOM offline=0, ROM offline, PODNN offline) are
reported plainly as degenerate distributions (all 7 stats collapse to that
one value, std=0) rather than reshaped to look like real distributions.

Run with:
    python task3_stats_summary.py
"""

from __future__ import annotations

import csv

import numpy as np

from src import Config, NavierStokesProblem, Visualizer

STATS = ["min", "q1", "median", "mean", "q3", "max", "std"]


def compute_stats(arr: np.ndarray) -> dict:
    arr = np.atleast_1d(np.asarray(arr, dtype=float))
    if arr.size == 1:
        v = float(arr[0])
        return {"min": v, "q1": v, "median": v, "mean": v, "q3": v, "max": v, "std": 0.0}
    q1, median, q3 = np.percentile(arr, [25, 50, 75])
    return {
        "min": float(arr.min()), "q1": float(q1), "median": float(median),
        "mean": float(arr.mean()), "q3": float(q3), "max": float(arr.max()),
        "std": float(arr.std()),
    }


def print_table(title: str, row_stats: dict, col_width: int = 11) -> None:
    print("\n" + "-" * 64)
    print(title)
    print("-" * 64)
    header = f"{'':<16}" + "".join(f"{s:>{col_width}}" for s in STATS)
    print(header)
    for row_label, stats in row_stats.items():
        line = f"{row_label:<16}"
        for s in STATS:
            v = stats[s]
            line += f"{v:>{col_width}.3e}"
        print(line)


def save_csv(path, row_stats: dict) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row"] + STATS)
        for row_label, stats in row_stats.items():
            w.writerow([row_label] + [stats[s] for s in STATS])


def main() -> None:
    print("=" * 64)
    print("Task 3 — Full statistics summary (accuracy + cost)")
    print("=" * 64)

    config = Config()
    problem = NavierStokesProblem(config)
    vis = Visualizer(problem)

    # --- 1. Load artefacts ------------------------------------------------
    rom_err = np.load(config.data_dir / "test_errors.npz")
    podnn_err = np.load(config.data_dir / "podnn_test_errors.npz")
    rom_timing = np.load(config.data_dir / "timing_data.npz")
    podnn_timing = np.load(config.data_dir / "podnn_timing.npz")

    test_params = rom_err["test_params"]
    assert np.allclose(test_params, podnn_err["test_params"]), \
        "Task 1 / Task 2 test_params mismatch -- comparison would be invalid"
    M_test = test_params.shape[0]

    # --- 2. Accuracy stats table -------------------------------------------
    metrics = [("rel_l2_u", "rel L2(u)"), ("rel_l2_p", "rel L2(p)"), ("rel_h1_u", "rel H1(u)")]
    accuracy_rows = {}
    accuracy_rows["FOM (ref)"] = {m: compute_stats(np.zeros(M_test)) for m, _ in metrics}
    accuracy_rows["ROM"] = {m: compute_stats(rom_err[m]) for m, _ in metrics}
    accuracy_rows["PODNN"] = {m: compute_stats(podnn_err[m]) for m, _ in metrics}

    for metric_key, metric_label in metrics:
        row_stats = {method: accuracy_rows[method][metric_key] for method in accuracy_rows}
        print_table(f"ACCURACY -- {metric_label} (over {M_test} held-out test points)", row_stats)

    accuracy_csv = config.data_dir / "stats_summary_accuracy.csv"
    with open(accuracy_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "metric"] + STATS)
        for method, metric_dict in accuracy_rows.items():
            for metric_key, metric_label in metrics:
                stats = metric_dict[metric_key]
                w.writerow([method, metric_label] + [stats[s] for s in STATS])
    print(f"\nAccuracy stats saved to: {accuracy_csv}")

    # --- 3. Cost stats table -------------------------------------------------
    fom_online = rom_timing["fom_times_test"]          # (150,) real distribution
    rom_online = rom_timing["rom_times_test"]           # (150,) real distribution
    podnn_online = podnn_timing["predict_times"]        # (150,) real distribution

    fom_offline = np.array([0.0])                       # exact: FOM has no offline cost
    rom_offline = np.array([float(rom_timing["offline_total"])])       # single value
    podnn_offline = np.array([float(podnn_timing["podnn_offline_total"])])  # single value

    cost_rows = {
        "FOM online": compute_stats(fom_online),
        "FOM offline": compute_stats(fom_offline),
        "ROM online": compute_stats(rom_online),
        "ROM offline": compute_stats(rom_offline),
        "PODNN online": compute_stats(podnn_online),
        "PODNN offline": compute_stats(podnn_offline),
    }
    print_table("COMPUTATIONAL COST (seconds)", cost_rows)

    cost_csv = config.data_dir / "stats_summary_cost.csv"
    save_csv(cost_csv, cost_rows)
    print(f"\nCost stats saved to: {cost_csv}")

    # --- 4. 3D error surfaces ------------------------------------------------
    print("\nGenerating 3D error-surface plot...")
    out = vis.plot_error_surface_3d(test_params, rom_err["rel_l2_u"], podnn_err["rel_l2_u"])
    print(f"Saved: {out}")

    print("\n" + "=" * 64)
    print("Done.")
    print("=" * 64)


if __name__ == "__main__":
    main()
