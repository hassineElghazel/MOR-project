#!/usr/bin/env python3
"""
Parametric PINN for the steady Navier-Stokes problem, in the style of the course
PINN labs (``professor's lab/PINN*.ipynb``) and the project spec
(``project2026.pdf``):

  * a fully-connected MLP with **tanh** activations -- the activation of the
    professor's Burgers ``PhysicsInformedNN``/``DNN`` (his data+physics model),
    with depth/width in the same spirit;
  * the network is **parametric**, w(x, mu): (x0, x1, mu0, mu1) -> (u1, u2, p);
  * spatial derivatives by ``torch.autograd`` (no finite differences);
  * **soft** boundary conditions: a boundary MSE penalises u on dOmega (u = 0)
    plus the pressure pin p(0,0) = 0 -- the ``mse_u`` / ``MSE_b`` term;
  * physics MSE = mean |R(steady NS)|^2 (spec ``MSE_p``);
  * optional **FOM-data term** (``--data``), i.e. the Burgers-style
    physics+data hybrid the professor uses for his hard example;
  * optional **random Fourier features** on x (``--fourier``): the plain net
    has a severe spectral bias against the ~mu1^2*pi forcing and cannot
    represent the field without them (verified: supervised fit stalls ~98%);
  * total loss  MSE = MSE_b + lambda*MSE_p + lam_d*MSE_data;
  * Adam warm-up -> L-BFGS with a strong-Wolfe line search.

Runs in the isolated torch env; the FE mass/stiffness matrices and DOF
coordinates come from ``data/pinn_fe_export.npz`` (no FEniCS needed here).

Usage:
    # pure physics (faithful spec):
    python pinn_torch.py --mode parametric --lam 1e-3
    # Burgers-style physics+data hybrid with Fourier features:
    python pinn_torch.py --mode parametric --data 400 --fourier 32 --lam 1e-6 --lam_d 100
"""
import argparse
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)
DATA = Path(__file__).resolve().parent / "data"
PI = np.pi


# ==========================================================================
# Reusable infrastructure (FE export, error metrics, forcing, autograd)
# ==========================================================================
def load_fe():
    d = np.load(DATA / "pinn_fe_export.npz")

    def spmat(k):
        return sp.coo_matrix((d[f"{k}_data"], (d[f"{k}_row"], d[f"{k}_col"])),
                             shape=tuple(d[f"{k}_shape"])).tocsr()

    return dict(N_u=int(d["N_u"]), N_p=int(d["N_p"]),
                pd0=d["parent_dofs_0"], pd1=d["parent_dofs_1"],
                xu=d["dof_coords_u"], xp=d["dof_coords_p"],
                Mu=spmat("Mu"), Mp=spmat("Mp"), Ku=spmat("Ku"),
                mu0_range=d["mu0_range"], mu1_range=d["mu1_range"])


def rel_errors(fe, U, P, u_true, p_true):
    """M-weighted relative errors for one sample (numpy vectors)."""
    Mu, Mp, Ku = fe["Mu"], fe["Mp"], fe["Ku"]
    du, dp = U - u_true, P - p_true
    l2u = np.sqrt(max(du @ (Mu @ du), 0) / (u_true @ (Mu @ u_true)))
    l2p = np.sqrt(max(dp @ (Mp @ dp), 0) / (p_true @ (Mp @ p_true)))
    Hu = Mu + Ku
    h1u = np.sqrt(max(du @ (Hu @ du), 0) / (u_true @ (Hu @ u_true)))
    return l2u, l2p, h1u


def forcing(x0, x1, mu1):
    """Analytic source term f = (f1, f2) exactly as in the project spec."""
    f1 = (-(mu1**3 * PI**2 * torch.cos(mu1**2 * PI * x0) - mu1**2 * PI**2)
          * torch.sin(mu1 * PI * x1) * torch.cos(mu1 * PI * x1)
          + mu1 * PI * torch.cos(mu1 * PI * x0) * torch.cos(mu1 * PI * x1))
    f2 = (-(-mu1**3 * PI**2 * torch.cos(mu1**2 * PI * x1) + mu1**2 * PI**2)
          * torch.sin(mu1 * PI * x0) * torch.cos(mu1 * PI * x0)
          - mu1 * PI * torch.sin(mu1 * PI * x0) * torch.sin(mu1 * PI * x1))
    return f1, f2


def grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


# ==========================================================================
# Network: plain sigmoid MLP, parametric, no Fourier / no hard BC
# ==========================================================================
class PINN(nn.Module):
    def __init__(self, mu0_range, mu1_range, hidden=(20,) * 8,
                 n_fourier=0, sigma=2.0, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        # Optional random Fourier features on x to overcome the spectral bias of
        # the plain MLP -- the forcing has spatial frequency ~mu1^2*pi and the
        # bare sigmoid net cannot represent it (verified: supervised fit stalls
        # at ~98% without this). Everything else stays lab-style.
        self.n_fourier = n_fourier
        if n_fourier > 0:
            self.register_buffer("B", torch.randn(2, n_fourier, generator=g) * sigma)
            in_dim = 2 + 2 * n_fourier + 2
        else:
            self.B = None
            in_dim = 4
        self.register_buffer("mu_lb", torch.tensor([float(mu0_range[0]), float(mu1_range[0])]))
        self.register_buffer("mu_ub", torch.tensor([float(mu0_range[1]), float(mu1_range[1])]))
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]   # professor's Burgers DNN activation
            d = h
        layers += [nn.Linear(d, 3)]           # (u1, u2, p), linear output
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _feat(self, x0, x1, mu0, mu1):
        mun = 2.0 * (torch.cat([mu0, mu1], dim=1) - self.mu_lb) / (self.mu_ub - self.mu_lb) - 1.0
        xn = 2.0 * torch.cat([x0, x1], dim=1) - 1.0        # x in [0,1] -> [-1,1]
        if self.B is None:
            return torch.cat([xn, mun], dim=1)
        proj = 2.0 * PI * (torch.cat([x0, x1], dim=1) @ self.B)
        return torch.cat([xn, torch.sin(proj), torch.cos(proj), mun], dim=1)

    def fields(self, x0, x1, mu0, mu1):
        out = self.net(self._feat(x0, x1, mu0, mu1))
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]


# ==========================================================================
# Residual, boundary/pin penalties, samplers
# ==========================================================================
def residual_terms(model, x0, x1, mu0, mu1):
    u1, u2, p = model.fields(x0, x1, mu0, mu1)
    u1_x0 = grad(u1, x0); u1_x1 = grad(u1, x1)
    u2_x0 = grad(u2, x0); u2_x1 = grad(u2, x1)
    p_x0 = grad(p, x0);   p_x1 = grad(p, x1)
    lap_u1 = grad(u1_x0, x0) + grad(u1_x1, x1)
    lap_u2 = grad(u2_x0, x0) + grad(u2_x1, x1)
    f1, f2 = forcing(x0, x1, mu1)
    # steady NS momentum (both components) + continuity
    R1 = -mu0 * lap_u1 + (u1 * u1_x0 + u2 * u1_x1) + p_x0 - f1
    R2 = -mu0 * lap_u2 + (u1 * u2_x0 + u2 * u2_x1) + p_x1 - f2
    R3 = u1_x0 + u2_x1
    return R1, R2, R3


def mse_physics(model, x0, x1, mu0, mu1):
    """MSE_p = mean |R|^2 over interior collocation points (spec, unscaled)."""
    R1, R2, R3 = residual_terms(model, x0, x1, mu0, mu1)
    return (R1**2).mean() + (R2**2).mean() + (R3**2).mean()


def mse_boundary(model, xb0, xb1, mb0, mb1, model_pin_z, mp0, mp1):
    """MSE_b: no-slip u=0 on dOmega  +  pressure pin p(0,0)=0."""
    u1, u2, _ = model.fields(xb0, xb1, mb0, mb1)
    bc = (u1**2).mean() + (u2**2).mean()
    _, _, p0 = model.fields(model_pin_z, model_pin_z, mp0, mp1)
    return bc + (p0**2).mean()


def sample_interior(n, mu0_range, mu1_range, fixed_mu=None, dev="cpu"):
    x0 = torch.rand(n, 1, device=dev, requires_grad=True)
    x1 = torch.rand(n, 1, device=dev, requires_grad=True)
    if fixed_mu is None:
        mu0 = torch.rand(n, 1, device=dev) * (mu0_range[1] - mu0_range[0]) + mu0_range[0]
        mu1 = torch.rand(n, 1, device=dev) * (mu1_range[1] - mu1_range[0]) + mu1_range[0]
    else:
        mu0 = torch.full((n, 1), fixed_mu[0], device=dev)
        mu1 = torch.full((n, 1), fixed_mu[1], device=dev)
    return x0, x1, mu0, mu1


def sample_boundary(n, mu0_range, mu1_range, fixed_mu=None, dev="cpu"):
    """n points spread over the four edges of the unit square."""
    m = max(1, n // 4)
    t = torch.rand(m, 1, device=dev)
    z = torch.zeros(m, 1, device=dev)
    o = torch.ones(m, 1, device=dev)
    x0 = torch.cat([t, t, z, o], dim=0)        # bottom, top, left, right
    x1 = torch.cat([z, o, t, t], dim=0)
    nb = x0.shape[0]
    if fixed_mu is None:
        mu0 = torch.rand(nb, 1, device=dev) * (mu0_range[1] - mu0_range[0]) + mu0_range[0]
        mu1 = torch.rand(nb, 1, device=dev) * (mu1_range[1] - mu1_range[0]) + mu1_range[0]
    else:
        mu0 = torch.full((nb, 1), fixed_mu[0], device=dev)
        mu1 = torch.full((nb, 1), fixed_mu[1], device=dev)
    return x0, x1, mu0, mu1


def sample_pin(n, mu0_range, mu1_range, fixed_mu=None, dev="cpu"):
    z = torch.zeros(n, 1, device=dev)
    if fixed_mu is None:
        mu0 = torch.rand(n, 1, device=dev) * (mu0_range[1] - mu0_range[0]) + mu0_range[0]
        mu1 = torch.rand(n, 1, device=dev) * (mu1_range[1] - mu1_range[0]) + mu1_range[0]
    else:
        mu0 = torch.full((n, 1), fixed_mu[0], device=dev)
        mu1 = torch.full((n, 1), fixed_mu[1], device=dev)
    return z, mu0, mu1


# ==========================================================================
# Evaluation on the FE mesh (same 150 test params / same error machinery)
# ==========================================================================
@torch.no_grad()
def reconstruct(model, fe, mu):
    xu = torch.tensor(fe["xu"]); xp = torch.tensor(fe["xp"])
    m0 = torch.full((xu.shape[0], 1), mu[0]); m1 = torch.full((xu.shape[0], 1), mu[1])
    u1, u2, _ = model.fields(xu[:, 0:1], xu[:, 1:2], m0, m1)
    m0p = torch.full((xp.shape[0], 1), mu[0]); m1p = torch.full((xp.shape[0], 1), mu[1])
    _, _, p = model.fields(xp[:, 0:1], xp[:, 1:2], m0p, m1p)
    z = torch.zeros(1, 1)
    _, _, p0 = model.fields(z, z, torch.tensor([[mu[0]]]), torch.tensor([[mu[1]]]))
    U = np.zeros(fe["N_u"]); U[fe["pd0"]] = u1[:, 0].numpy(); U[fe["pd1"]] = u2[:, 0].numpy()
    P = (p[:, 0] - p0[0, 0]).numpy()
    return U, P


def full_report(model, fe, test_params, u_fom, p_fom):
    M = test_params.shape[0]
    L2U = np.zeros(M); L2P = np.zeros(M); H1U = np.zeros(M); pt = np.zeros(M)
    for i in range(M):
        t = time.time()
        U, P = reconstruct(model, fe, test_params[i])
        pt[i] = time.time() - t
        L2U[i], L2P[i], H1U[i] = rel_errors(fe, U, P, u_fom[:, i], p_fom[:, i])
    return L2U, L2P, H1U, pt


def probe_mean_l2u(model, fe, test_params, u_fom, p_fom, n=15):
    idx = range(0, test_params.shape[0], max(1, test_params.shape[0] // n))
    e = []
    for i in idx:
        U, P = reconstruct(model, fe, test_params[i])
        e.append(rel_errors(fe, U, P, u_fom[:, i], p_fom[:, i])[0])
    return float(np.mean(e))


# ==========================================================================
# Training:  loss = MSE_b + lambda * MSE_p ,  Adam -> L-BFGS(strong Wolfe)
# ==========================================================================
def build_data(fe, n_data, seed):
    """FOM-snapshot data pool for the Burgers-style physics+data hybrid.

    Mirrors the professor's PhysicsInformedNN (Burgers) which trains on data
    (X_u, u_train) alongside the physics residual. Returns torch tensors of
    (coords, mu, u1, u2) for velocity and (coords, mu, p) for pressure.
    """
    snap = np.load(DATA / "snapshot_data.npz")
    S_u, S_p, train_params = snap["S_u"], snap["S_p"], snap["params"]
    rng = np.random.default_rng(seed)
    sel = rng.choice(train_params.shape[0],
                     min(n_data, train_params.shape[0]), replace=False)
    xu, xp, pd0, pd1 = fe["xu"], fe["xp"], fe["pd0"], fe["pd1"]
    Nu, Np = xu.shape[0], xp.shape[0]
    Xv, Mv, U1, U2, Xp, Mp, Pp = [], [], [], [], [], [], []
    for k in sel:
        mu = train_params[k]
        Xv.append(xu); Mv.append(np.tile(mu, (Nu, 1)))
        U1.append(S_u[pd0, k]); U2.append(S_u[pd1, k])
        Xp.append(xp); Mp.append(np.tile(mu, (Np, 1))); Pp.append(S_p[:, k])
    return dict(
        Xv=torch.tensor(np.vstack(Xv)), Mv=torch.tensor(np.vstack(Mv)),
        U1=torch.tensor(np.concatenate(U1))[:, None],
        U2=torch.tensor(np.concatenate(U2))[:, None],
        Xp=torch.tensor(np.vstack(Xp)), Mp=torch.tensor(np.vstack(Mp)),
        Pp=torch.tensor(np.concatenate(Pp))[:, None], n=len(sel))


def train(mode, mu0, mu1, adam_iters, lbfgs_iters, n_coll, n_bc, lam,
          hidden, seed, eval_every, n_data=0, lam_d=1.0, n_fourier=0, sigma=2.0,
          w_p=1.0):
    fe = load_fe()
    fom = np.load(DATA / "fom_solutions_test.npz")
    test_params = fom["test_params"]; u_fom = fom["u_fom_test"]; p_fom = fom["p_fom_test"]
    fom_mean = float(np.load(DATA / "timing_data.npz")["fom_times_test"].mean())
    mu0_range = tuple(fe["mu0_range"]); mu1_range = tuple(fe["mu1_range"])
    fixed = (mu0, mu1) if mode == "single" else None

    model = PINN(mu0_range, mu1_range, hidden=hidden, n_fourier=n_fourier,
                 sigma=sigma, seed=seed)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"mode={mode} fixed_mu={fixed} hidden={hidden} fourier={n_fourier} "
          f"sigma={sigma} lambda={lam} n_data={n_data} lam_d={lam_d} "
          f"params={nparam}", flush=True)

    dp = build_data(fe, n_data, seed) if n_data > 0 else None
    if dp is not None:
        print(f"data term: {dp['n']} snapshots -> {dp['Xv'].shape[0]} vel + "
              f"{dp['Xp'].shape[0]} pres points", flush=True)

    def mse_data(iv, ip):
        # RELATIVE data misfit: normalise each field by its own magnitude so
        # velocity (|u|~0.06) and pressure (|p|~O(1)) are fitted in comparable
        # relative terms and neither scale dominates. w_p weights pressure.
        pu1, pu2, _ = model.fields(dp["Xv"][iv, 0:1], dp["Xv"][iv, 1:2],
                                   dp["Mv"][iv, 0:1], dp["Mv"][iv, 1:2])
        su = (dp["U1"][iv]**2).mean() + (dp["U2"][iv]**2).mean() + 1e-12
        ld = (((pu1 - dp["U1"][iv])**2).mean() + ((pu2 - dp["U2"][iv])**2).mean()) / su
        _, _, pp = model.fields(dp["Xp"][ip, 0:1], dp["Xp"][ip, 1:2],
                                dp["Mp"][ip, 0:1], dp["Mp"][ip, 1:2])
        sp = (dp["Pp"][ip]**2).mean() + 1e-12
        return ld + w_p * ((pp - dp["Pp"][ip])**2).mean() / sp

    hist = {"iters": [], "total": [], "mse_b": [], "mse_p": [], "err": []}

    def probe():
        if mode == "single":
            U, P = reconstruct(model, fe, (mu0, mu1))
            j = int(np.argmin(np.sum((test_params - np.array([mu0, mu1]))**2, axis=1)))
            return rel_errors(fe, U, P, u_fom[:, j], p_fom[:, j])[0]
        return probe_mean_l2u(model, fe, test_params, u_fom, p_fom, n=15)

    t0 = time.time()
    step = [0]

    def record(mb, mp, total):
        hist["iters"].append(step[0]); hist["total"].append(total)
        hist["mse_b"].append(mb); hist["mse_p"].append(mp)
        e = probe(); hist["err"].append(e)
        print(f"  {step[0]:6d}  loss {total:.3e}  MSE_b {mb:.3e}  "
              f"MSE_p {mp:.3e}  relL2(u) {e:.3f}  [{time.time()-t0:.0f}s]", flush=True)

    # ---- Adam warm-up (collocation resampled each epoch, lab style) --------
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for it in range(adam_iters):
        opt.zero_grad()
        xb0, xb1, mb0, mb1 = sample_boundary(n_bc, mu0_range, mu1_range, fixed)
        zp, mp0, mp1 = sample_pin(max(64, n_bc // 8), mu0_range, mu1_range, fixed)
        xc0, xc1, mc0, mc1 = sample_interior(n_coll, mu0_range, mu1_range, fixed)
        mb = mse_boundary(model, xb0, xb1, mb0, mb1, zp, mp0, mp1)
        mp = mse_physics(model, xc0, xc1, mc0, mc1)
        loss = mb + lam * mp
        if dp is not None:
            iv = torch.randint(0, dp["Xv"].shape[0], (min(4000, dp["Xv"].shape[0]),))
            ip = torch.randint(0, dp["Xp"].shape[0], (min(1000, dp["Xp"].shape[0]),))
            loss = loss + lam_d * mse_data(iv, ip)
        loss.backward(); opt.step()
        step[0] = it
        if it % eval_every == 0:
            record(mb.item(), mp.item(), loss.item())

    switch_iter = adam_iters

    # ---- L-BFGS (strong Wolfe) on a fixed large batch ----------------------
    xb0, xb1, mb0, mb1 = sample_boundary(4 * n_bc, mu0_range, mu1_range, fixed)
    zp, mp0, mp1 = sample_pin(max(256, n_bc // 2), mu0_range, mu1_range, fixed)
    xc0, xc1, mc0, mc1 = sample_interior(4 * n_coll, mu0_range, mu1_range, fixed)
    if dp is not None:                       # fixed data batch for L-BFGS
        ivb = torch.randint(0, dp["Xv"].shape[0], (min(40000, dp["Xv"].shape[0]),))
        ipb = torch.randint(0, dp["Xp"].shape[0], (min(10000, dp["Xp"].shape[0]),))
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=lbfgs_iters,
                            max_eval=lbfgs_iters, history_size=50,
                            tolerance_grad=1e-9, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")
    lit = [0]

    def closure():
        opt.zero_grad()
        mb = mse_boundary(model, xb0, xb1, mb0, mb1, zp, mp0, mp1)
        mp = mse_physics(model, xc0, xc1, mc0, mc1)
        loss = mb + lam * mp
        if dp is not None:
            loss = loss + lam_d * mse_data(ivb, ipb)
        loss.backward()
        lit[0] += 1
        if lit[0] % 50 == 0:
            step[0] = adam_iters + lit[0]
            record(mb.item(), mp.item(), loss.item())
        return loss

    opt.step(closure)

    # ---- final evaluation on all 150 test params + per-point timing --------
    L2U, L2P, H1U, pt = full_report(model, fe, test_params, u_fom, p_fom)
    predict_mean = float(pt.mean()); speedup = fom_mean / max(predict_mean, 1e-12)
    train_time = time.time() - t0
    print(f"\nFINAL  relL2(u) mean={L2U.mean():.4f} median={np.median(L2U):.4f} "
          f"max={L2U.max():.4f} | relL2(p) mean={L2P.mean():.4f} | "
          f"relH1(u) mean={H1U.mean():.4f}", flush=True)
    print(f"       online {predict_mean*1e3:.3f} ms  speedup {speedup:.1f}x  "
          f"train {train_time:.0f}s", flush=True)

    # ---- persist (Task-3 contract + training trace) ------------------------
    np.savez_compressed(DATA / "pinn_test_errors.npz",
                        rel_l2_u=L2U, rel_l2_p=L2P, rel_h1_u=H1U,
                        test_params=test_params)
    np.savez_compressed(DATA / "pinn_timing.npz",
                        train_time=np.array(train_time),
                        predict_mean=np.array(predict_mean),
                        predict_times=pt, fom_mean=np.array(fom_mean),
                        speedup=np.array(speedup))
    np.savez_compressed(DATA / "pinn_training_history.npz",
                        iters=np.array(hist["iters"]), total=np.array(hist["total"]),
                        mse_b=np.array(hist["mse_b"]), mse_p=np.array(hist["mse_p"]),
                        err=np.array(hist["err"]))
    np.savez_compressed(DATA / "pinn_lossvserr_trace.npz",
                        iters=np.array(hist["iters"]), loss=np.array(hist["total"]),
                        err=np.array(hist["err"]), switch_iter=np.array(switch_iter))
    torch.save(model.state_dict(), DATA / "pinn_torch_model.pt")
    print("saved pinn_test_errors.npz, pinn_timing.npz, pinn_training_history.npz, "
          "pinn_lossvserr_trace.npz", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "parametric"], default="parametric")
    ap.add_argument("--mu0", type=float, default=1.0)
    ap.add_argument("--mu1", type=float, default=1.5)
    ap.add_argument("--adam", type=int, default=5000)
    ap.add_argument("--lbfgs", type=int, default=2000)
    ap.add_argument("--coll", type=int, default=4000)
    ap.add_argument("--nb", type=int, default=1000)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--data", type=int, default=0,
                    help="number of FOM snapshots for the physics+data hybrid (0=pure physics)")
    ap.add_argument("--lam_d", type=float, default=1.0, help="data-term weight")
    ap.add_argument("--wp", type=float, default=1.0, help="pressure weight in the (relative) data loss")
    ap.add_argument("--fourier", type=int, default=0, help="# random Fourier features on x (0=off)")
    ap.add_argument("--sigma", type=float, default=2.0, help="Fourier feature bandwidth")
    ap.add_argument("--width", type=int, default=20)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    train(a.mode, a.mu0, a.mu1, a.adam, a.lbfgs, a.coll, a.nb, a.lam,
          tuple([a.width] * a.depth), a.seed, a.eval_every,
          n_data=a.data, lam_d=a.lam_d, n_fourier=a.fourier, sigma=a.sigma,
          w_p=a.wp)
