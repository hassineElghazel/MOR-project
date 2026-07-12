#!/usr/bin/env python3
"""
Task 4 (PINN) figures -- rendered through the project Visualizer so they share
the style of every other plot.

  * plot 26: physics-only residual-vs-error divergence
             (data/pinn_lossvserr_trace.npz)
  * plot 27: data ablation, data-only vs hybrid PINN
             (data/pinn_ablation.csv)

Run with:
    python task4_pinn_plots.py
"""
from __future__ import annotations
import csv
import numpy as np
from src import Config, NavierStokesProblem, Visualizer

# ROM / POD-NN reference errors (mean rel L2(u) on the 150 held-out test params)
ROM_REF, PODNN_REF = 0.0119, 0.0468


def main() -> None:
    config = Config()
    problem = NavierStokesProblem(config)
    vis = Visualizer(problem)

    # --- plot 26: residual-vs-error divergence -----------------------------
    tr = np.load(config.data_dir / "pinn_lossvserr_trace.npz")
    sw = tr["switch_iter"]
    out26 = vis.plot_pinn_loss_vs_error(
        tr["iters"], tr["loss"], tr["err"],
        switch_iter=(int(sw) if sw.size and sw != None else None))  # noqa: E711
    print(f"Saved: {out26}")

    # --- plot 27: data ablation -------------------------------------------
    N, d, h = [], [], []
    with open(config.data_dir / "pinn_ablation.csv") as f:
        for row in csv.DictReader(f):
            N.append(int(row["n_snapshots"]))
            d.append(float(row["dataonly_relL2u"]))
            h.append(float(row["hybrid_relL2u"]))
    out27 = vis.plot_pinn_ablation(np.array(N), np.array(d), np.array(h),
                                   rom_ref=ROM_REF, podnn_ref=PODNN_REF)
    print(f"Saved: {out27}  ({len(N)} ablation points: {N})")


if __name__ == "__main__":
    main()
