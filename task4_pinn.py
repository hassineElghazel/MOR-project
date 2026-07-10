#!/usr/bin/env python3
"""
Task 4 driver: Physics-Informed Neural Network (PINN) for steady Navier-Stokes.

The PINN learns  (x, mu) -> (u1, u2, p)  directly from the PDE residual.

== DOF reconstruction ==
FEniCS P2 VectorElement stores DOFs in interleaved order (u1/u2 alternating).
Confirmed by V.sub(0).dofmap().dofs() returning only even global indices.
We use parent_dofs_0/parent_dofs_1 to scatter predictions into correct slots.

== Loss (exactly the project spec) ==
    MSE = MSE_b + lambda * MSE_p
MSE_b = boundary term (no-slip u=0 on dOmega, + pressure pin p(0,0)=0);
MSE_p = mean squared steady-NS residual over Omega x P, normalised by a FIXED
        reference forcing scale (a constant folded into lambda -- see
        PINNModel.f_ref2), so MSE_p is O(1) and lambda stays a single fixed
        number as the spec intends.

== Training strategy ==
Two-phase curriculum, expressed through the spec's single lambda:
  Phase 1 (warmup):  lambda = 0            -> boundary-only, drive MSE_b -> ~0
  Phase 2 (physics): lambda = LAMBDA_PHYS  -> add physics while the boundary
           term (weight 1) keeps the no-slip BC enforced.

== Spectral-bias fix ==
The spatial input x is embedded through random Fourier features
gamma(x)=[x, sin(2*pi*Bx), cos(2*pi*Bx)] (see src/pinn.py). A plain tanh MLP
cannot represent the solution's high-frequency structure (forcing ~
cos(mu1^2*pi*x), ~4.5 oscillations at mu1=3) and collapses to ~0 (98.5%
error); the embedding is a pure input transform that leaves the spec loss and
residual unchanged.

Run with:
    python task4_pinn.py
"""

from __future__ import annotations

import csv
import time

import numpy as np

from src import Config, NavierStokesProblem, Visualizer, PINNModel
from src.pinn import sample_collocation_points
from src.analysis import ErrorAnalyzer

# -----------------------------------------------------------------------
# Hyper-parameters
# -----------------------------------------------------------------------
HIDDEN_SIZES = [96, 96, 96]

N_FOURIER     = 32         # random Fourier features for the spatial input
FOURIER_SIGMA = 4.0        # freq spread; covers forcing freq up to ~mu1^2=9

WARMUP_EPOCHS  = 1_500     # Phase 1: boundary only (lambda = 0)
PHYSICS_EPOCHS = 6_000     # Phase 2: full PINN  (lambda = LAMBDA_PHYS)
TOTAL_EPOCHS   = WARMUP_EPOCHS + PHYSICS_EPOCHS

LR_WARMUP  = 5e-4
LR_PHYSICS = 1e-3
# Spec loss MSE = MSE_b + lambda*MSE_p. Boundary term keeps weight 1; the
# physics weight lambda is < 1 so the no-slip BC is enforced strongly enough
# to avoid the trivial u=0 attractor while the (fixed-normalised) physics
# residual is minimised.
LAMBDA_PHYS = 1.0e-4   # physics weight; smaller because MSE_p uses per-sample
                       # relative-residual normalisation (larger absolute scale)
H_FD        = 1e-3

N_INTERIOR = 2_000
N_BOUNDARY = 800
N_PIN      = 300
LOG_EVERY  = 500

SEED = 42


def _get_parent_dof_indices(problem: NavierStokesProblem):
    V = problem.V
    V0 = V.sub(0).collapse()
    parent_dofs_0 = np.array(V.sub(0).dofmap().dofs())
    parent_dofs_1 = np.array(V.sub(1).dofmap().dofs())
    dof_coords_u  = V0.tabulate_dof_coordinates()
    dof_coords_p  = problem.Q.tabulate_dof_coordinates()
    return parent_dofs_0, parent_dofs_1, dof_coords_u, dof_coords_p


def _pinn_dof_vectors(model, mu, parent_dofs_0, parent_dofs_1,
                      dof_coords_u, dof_coords_p, N_u, N_p):
    n_sc = dof_coords_u.shape[0]
    out_u = model.predict(dof_coords_u, np.tile(mu, (n_sc, 1)))
    out_p = model.predict(dof_coords_p, np.tile(mu, (N_p, 1)))
    U = np.zeros(N_u)
    U[parent_dofs_0] = out_u[:, 0]
    U[parent_dofs_1] = out_u[:, 1]
    return U, out_p[:, 2]


def main() -> None:
    print("=" * 64)
    print("Task 4 — Physics-Informed Neural Network (PINN)")
    print("=" * 64)

    config = Config()
    problem = NavierStokesProblem(config)
    N_u, N_p = problem.N_u, problem.N_p
    print(f"Mesh: {config.mesh_n}x{config.mesh_n}  |  N_u={N_u}, N_p={N_p}")

    fom_test = np.load(config.data_dir / "fom_solutions_test.npz")
    u_fom_test  = fom_test["u_fom_test"]
    p_fom_test  = fom_test["p_fom_test"]
    test_params = fom_test["test_params"]
    M_test      = test_params.shape[0]

    timing_data    = np.load(config.data_dir / "timing_data.npz")
    fom_times_test = timing_data["fom_times_test"]
    fom_mean       = float(fom_times_test.mean())

    try:
        rom_err      = np.load(config.data_dir / "test_errors.npz")
        podnn_err    = np.load(config.data_dir / "podnn_test_errors.npz")
        rom_timing   = np.load(config.data_dir / "timing_data.npz")
        podnn_timing = np.load(config.data_dir / "podnn_timing.npz")
        has_task13   = True
    except FileNotFoundError:
        print("  [warn] Task 1/2/3 artefacts not found -- skipping plot 24.")
        has_task13 = False

    parent_dofs_0, parent_dofs_1, dof_coords_u, dof_coords_p = \
        _get_parent_dof_indices(problem)
    vert_coords = problem.mesh.coordinates().copy()
    N_vert = vert_coords.shape[0]

    assert np.all(parent_dofs_0 % 2 == 0), "Expected u0 DOFs at even indices"
    assert np.all(parent_dofs_1 % 2 == 1), "Expected u1 DOFs at odd indices"
    print(f"DOF layout: interleaved (u1=even, u2=odd) ✓  n_scalar={dof_coords_u.shape[0]}")
    print(f"Test parameters: {M_test}   Mesh vertices: {N_vert}")
    _in_dim = 2 + 2 * N_FOURIER + 2
    print(f"PINN: {[_in_dim] + HIDDEN_SIZES + [3]}  (Fourier features: {N_FOURIER}, sigma={FOURIER_SIGMA})")
    print(f"Training: {WARMUP_EPOCHS} warmup + {PHYSICS_EPOCHS} physics epochs")
    print(f"  loss = MSE_b + lambda*MSE_p,  lambda={LAMBDA_PHYS}  (MSE_p fixed-normalised to O(1))")
    print(f"  N_int={N_INTERIOR}, N_bnd={N_BOUNDARY}, N_pin={N_PIN}")

    # ---------------------------------------------------------------
    # Train: Phase 1 (boundary warmup)
    # ---------------------------------------------------------------
    model = PINNModel(mu0_range=config.mu0_range, mu1_range=config.mu1_range,
                      hidden_sizes=HIDDEN_SIZES, seed=SEED,
                      n_fourier=N_FOURIER, fourier_sigma=FOURIER_SIGMA)
    rng_train = np.random.default_rng(SEED)
    hist_total = np.zeros(TOTAL_EPOCHS)
    hist_b     = np.zeros(TOTAL_EPOCHS)
    hist_p     = np.zeros(TOTAL_EPOCHS)

    print("\n[Phase 1] Boundary warmup")
    t0 = time.time()
    for ep in range(WARMUP_EPOCHS):
        coll = sample_collocation_points(N_INTERIOR, N_BOUNDARY, N_PIN,
                                         config.mu0_range, config.mu1_range, rng_train)
        total, lb, lp = model.train_step(coll, LR_WARMUP, lambda_p=0.0, h=H_FD,
                                          lambda_b=1.0)   # lambda = 0 (spec)
        hist_total[ep] = lb; hist_b[ep] = lb; hist_p[ep] = 0.0
        if ep % LOG_EVERY == 0:
            print(f"  epoch {ep:5d}  MSE_b {lb:.3e}")
    t_warmup = time.time() - t0
    print(f"Phase 1 done in {t_warmup:.1f}s  (final MSE_b={hist_b[WARMUP_EPOCHS-1]:.3e})")

    # ---------------------------------------------------------------
    # Train: Phase 2 (full PINN, lambda_b=100 >> lambda_p=1)
    # ---------------------------------------------------------------
    print(f"\n[Phase 2] Full PINN  (MSE = MSE_b + lambda*MSE_p, lambda={LAMBDA_PHYS})")
    err_probe = ErrorAnalyzer(problem)   # for live rel-error monitoring
    t1 = time.time()
    for ep in range(PHYSICS_EPOCHS):
        coll = sample_collocation_points(N_INTERIOR, N_BOUNDARY, N_PIN,
                                         config.mu0_range, config.mu1_range, rng_train)
        total, lb, lp = model.train_step(coll, LR_PHYSICS, lambda_p=LAMBDA_PHYS,
                                          h=H_FD, lambda_b=1.0)
        idx = WARMUP_EPOCHS + ep
        hist_total[idx] = total; hist_b[idx] = lb; hist_p[idx] = lp
        if ep % LOG_EVERY == 0:
            # live rel-error on a small fixed subset (10 test pts) so we can
            # see actual accuracy during training, not just the loss.
            eu = []
            for j in range(0, M_test, max(1, M_test // 10)):
                Uj, Pj = _pinn_dof_vectors(model, test_params[j], parent_dofs_0,
                                           parent_dofs_1, dof_coords_u, dof_coords_p, N_u, N_p)
                e, _, _ = err_probe.relative_errors(u_fom_test[:, j], p_fom_test[:, j], Uj, Pj)
                eu.append(e)
            print(f"  epoch {ep:5d}  total {total:.3e}  MSE_b {lb:.3e}  "
                  f"MSE_p {lp:.3e}  ~relL2(u) {np.mean(eu):.3f}")
    t_physics = time.time() - t1
    pinn_train_time = t_warmup + t_physics
    print(f"Phase 2 done in {t_physics:.1f}s")
    print(f"Total training: {pinn_train_time:.1f}s  "
          f"(final total={hist_total[-1]:.3e}, "
          f"MSE_b={hist_b[-1]:.3e}, MSE_p(norm)={hist_p[-1]:.3e})")

    model.save(config.data_dir / "pinn_model.npz")
    print(f"Model saved: {config.data_dir / 'pinn_model.npz'}")

    # ---------------------------------------------------------------
    # Evaluate on 15 test parameters
    # ---------------------------------------------------------------
    print("\nEvaluating on test set...")
    err_analyzer     = ErrorAnalyzer(problem)
    pinn_u_mag       = np.zeros((N_vert, M_test))
    U_pinn           = np.zeros((N_u, M_test))
    P_pinn           = np.zeros((N_p, M_test))
    pinn_times_test  = np.zeros(M_test)

    for i in range(M_test):
        mu = test_params[i]
        t_start = time.time()
        U_pinn[:, i], P_pinn[:, i] = _pinn_dof_vectors(
            model, mu, parent_dofs_0, parent_dofs_1,
            dof_coords_u, dof_coords_p, N_u, N_p)
        out_v = model.predict(vert_coords, np.tile(mu, (N_vert, 1)))
        pinn_u_mag[:, i] = np.sqrt(out_v[:, 0]**2 + out_v[:, 1]**2)
        pinn_times_test[i] = time.time() - t_start
        print(f"  test {i+1:2d}/{M_test}: mu=({mu[0]:.2f},{mu[1]:.2f})  "
              f"t={pinn_times_test[i]:.3f}s")

    report            = err_analyzer.batch_report(u_fom_test, p_fom_test, U_pinn, P_pinn)
    pinn_summary      = report.summary()
    pinn_predict_mean = float(pinn_times_test.mean())
    speedup           = fom_mean / max(pinn_predict_mean, 1e-12)

    print("\nPINN errors on test set:")
    print(f"  mean rel L2(u) = {pinn_summary['mean_l2_u']:.3e}   "
          f"max = {pinn_summary['max_l2_u']:.3e}")
    print(f"  mean rel L2(p) = {pinn_summary['mean_l2_p']:.3e}   "
          f"max = {pinn_summary['max_l2_p']:.3e}")
    print(f"  mean rel H1(u) = {pinn_summary['mean_h1_u']:.3e}   "
          f"max = {pinn_summary['max_h1_u']:.3e}")
    print(f"  FOM={fom_mean:.4f}s  PINN={pinn_predict_mean:.4f}s  "
          f"speedup={speedup:.1f}x")

    # ---------------------------------------------------------------
    # Persist artefacts
    # ---------------------------------------------------------------
    np.savez_compressed(config.data_dir / "pinn_training_history.npz",
                        hist_total=hist_total, hist_b=hist_b, hist_p=hist_p)
    np.savez_compressed(config.data_dir / "pinn_test_errors.npz",
                        rel_l2_u=report.rel_l2_u, rel_l2_p=report.rel_l2_p,
                        rel_h1_u=report.rel_h1_u, test_params=test_params)
    np.savez_compressed(config.data_dir / "pinn_timing.npz",
                        train_time=np.array(pinn_train_time),
                        predict_mean=np.array(pinn_predict_mean),
                        predict_times=pinn_times_test,
                        fom_mean=np.array(fom_mean), speedup=np.array(speedup))
    np.savez_compressed(config.data_dir / "pinn_vertex_umag.npz", pinn_u_mag=pinn_u_mag)

    table_path = config.data_dir / "pinn_comparison_table.csv"
    with open(table_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "mu0", "mu1", "fom_time_s",
                                           "pinn_time_s", "pinn_rel_l2_u",
                                           "pinn_rel_l2_p", "pinn_rel_h1_u"])
        w.writeheader()
        for i in range(M_test):
            w.writerow({"idx": i, "mu0": test_params[i,0], "mu1": test_params[i,1],
                         "fom_time_s": fom_times_test[i], "pinn_time_s": pinn_times_test[i],
                         "pinn_rel_l2_u": report.rel_l2_u[i],
                         "pinn_rel_l2_p": report.rel_l2_p[i],
                         "pinn_rel_h1_u": report.rel_h1_u[i]})
    print(f"CSV: {table_path}")

    # ---------------------------------------------------------------
    # Plots 21-24
    # ---------------------------------------------------------------
    print("\nGenerating plots...")
    vis = Visualizer(problem)
    vis.plot_pinn_training_curve(hist_total, hist_b, hist_p)
    vis.plot_pinn_vs_fom(test_params, u_fom_test, pinn_u_mag)
    vis.plot_pinn_error_parameter_space(test_params, report.rel_l2_u)
    n_plots = 3
    if has_task13:
        def _s(e):
            return {"mean_l2_u": float(e["rel_l2_u"].mean()),
                    "max_l2_u": float(e["rel_l2_u"].max()),
                    "mean_l2_p": float(e["rel_l2_p"].mean()),
                    "max_l2_p": float(e["rel_l2_p"].max()),
                    "mean_h1_u": float(e["rel_h1_u"].mean()),
                    "max_h1_u": float(e["rel_h1_u"].max())}
        vis.plot_pinn_summary_comparison(
            rom_summary=_s(rom_err), podnn_summary=_s(podnn_err),
            pinn_summary=pinn_summary, fom_mean=fom_mean,
            rom_mean=float(rom_timing["rom_times_test"].mean()),
            podnn_mean=float(podnn_timing["predict_mean"]),
            pinn_mean=pinn_predict_mean)
        n_plots = 4
    print(f"Saved {n_plots} plots to: {config.plots_dir}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 64)
    print("RESULT SUMMARY")
    print("=" * 64)
    print(f"PINN:                         {model.layer_sizes}  (Fourier {N_FOURIER}/sigma {FOURIER_SIGMA})")
    print(f"Training:                     {WARMUP_EPOCHS} warmup + {PHYSICS_EPOCHS} physics")
    print(f"loss:                         MSE_b + lambda*MSE_p,  lambda={LAMBDA_PHYS}")
    print(f"DOF layout:                   interleaved ✓")
    print(f"Mean rel. L2(u):              {pinn_summary['mean_l2_u']:.3e}")
    print(f"Max  rel. L2(u):              {pinn_summary['max_l2_u']:.3e}")
    print(f"Mean rel. L2(p):              {pinn_summary['mean_l2_p']:.3e}")
    print(f"Mean rel. H1(u):              {pinn_summary['mean_h1_u']:.3e}")
    print(f"Mean FOM time:                {fom_mean:.4f} s")
    print(f"Mean PINN predict time:       {pinn_predict_mean:.4f} s  ({speedup:.1f}x speedup)")
    print(f"PINN training time:           {pinn_train_time:.2f} s")
    print(f"\nArtefacts: {config.data_dir}")
    print(f"Plots:     {config.plots_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
