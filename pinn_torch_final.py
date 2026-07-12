#!/usr/bin/env python3
"""
Final HYBRID PINN (physics + data) for the parametric problem, evaluated on the
SAME 150 held-out test parameters and the SAME FE error machinery as ROM/POD-NN.

Trains on FOM training snapshots (data) + PDE residual (physics), autograd,
hard BC, Adam -> L-BFGS. Saves, in the schema the Task-3 comparison expects:
  data/pinn_test_errors.npz  (rel_l2_u, rel_l2_p, rel_h1_u, test_params)
  data/pinn_timing.npz       (train_time, predict_mean, predict_times,
                              fom_mean, speedup)
so task3_comparison.py / task3_stats_summary.py pick it up automatically.

Usage:  python pinn_torch_final.py --nsnap 400
"""
import argparse, time, os, sys
import numpy as np
PROJ = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, PROJ); os.chdir(PROJ)
import torch
from pinn_torch import PINN, load_fe, rel_errors, residual, sample
torch.set_default_dtype(torch.float64)
DATA = "data"


def main(nsnap, adam, lbfgs, ncoll, lam_d, seed):
    fe = load_fe()
    snap = np.load(f"{DATA}/snapshot_data.npz"); S_u, S_p, train_params = snap["S_u"], snap["S_p"], snap["params"]
    fom = np.load(f"{DATA}/fom_solutions_test.npz")
    test_params, u_fom, p_fom = fom["test_params"], fom["u_fom_test"], fom["p_fom_test"]
    timing = np.load(f"{DATA}/timing_data.npz"); fom_mean = float(timing["fom_times_test"].mean())
    mu0r, mu1r = tuple(fe["mu0_range"]), tuple(fe["mu1_range"])
    xu = torch.tensor(fe["xu"]); xp = torch.tensor(fe["xp"]); pd0, pd1 = fe["pd0"], fe["pd1"]
    Nu_dof, Np_dof = xu.shape[0], xp.shape[0]; M_test = test_params.shape[0]

    rng = np.random.default_rng(seed)
    idx = rng.choice(train_params.shape[0], min(nsnap, train_params.shape[0]), replace=False)
    # build data pool
    Xv=[]; Mv=[]; U1=[]; U2=[]; Xpp=[]; Mp=[]; Pp=[]
    for k in idx:
        mu = train_params[k]
        Xv.append(fe["xu"]); Mv.append(np.tile(mu,(Nu_dof,1))); U1.append(S_u[pd0,k]); U2.append(S_u[pd1,k])
        Xpp.append(fe["xp"]); Mp.append(np.tile(mu,(Np_dof,1))); Pp.append(S_p[:,k])
    Xv=torch.tensor(np.vstack(Xv)); Mv=torch.tensor(np.vstack(Mv))
    U1=torch.tensor(np.concatenate(U1))[:,None]; U2=torch.tensor(np.concatenate(U2))[:,None]
    Xpp=torch.tensor(np.vstack(Xpp)); Mp=torch.tensor(np.vstack(Mp)); Pp=torch.tensor(np.concatenate(Pp))[:,None]
    print(f"nsnap={len(idx)}  data pool: {Xv.shape[0]} vel + {Xpp.shape[0]} pres points")

    m = PINN(mu0r, mu1r, hidden=(64,64,64,64), n_fourier=24, sigma=2.0, seed=0)

    def data_loss(nb=4000):
        iv = torch.randint(0, Xv.shape[0], (min(nb,Xv.shape[0]),))
        pu1,pu2,_ = m.fields(Xv[iv,0:1],Xv[iv,1:2],Mv[iv,0:1],Mv[iv,1:2])
        ld = ((pu1-U1[iv])**2).mean() + ((pu2-U2[iv])**2).mean()
        ip = torch.randint(0, Xpp.shape[0], (min(nb//4,Xpp.shape[0]),))
        _,_,pp = m.fields(Xpp[ip,0:1],Xpp[ip,1:2],Mp[ip,0:1],Mp[ip,1:2])
        return ld + 0.01*((pp-Pp[ip])**2).mean()

    t0 = time.time()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for it in range(adam):
        opt.zero_grad()
        x0,x1,mm0,mm1 = sample(ncoll, mu0r, mu1r, None)
        loss = residual(m, x0,x1,mm0,mm1) + lam_d*data_loss()
        loss.backward(); opt.step()
        if it % 500 == 0: print(f"  adam {it} loss {loss.item():.3e} [{time.time()-t0:.0f}s]", flush=True)
    ivb = torch.randint(0, Xv.shape[0], (min(16000,Xv.shape[0]),))
    xb0,xb1,mb0,mb1 = sample(6000, mu0r, mu1r, None)
    opt = torch.optim.LBFGS(m.parameters(), lr=1.0, max_iter=lbfgs, history_size=50, line_search_fn="strong_wolfe")
    def cl():
        opt.zero_grad()
        pu1,pu2,_ = m.fields(Xv[ivb,0:1],Xv[ivb,1:2],Mv[ivb,0:1],Mv[ivb,1:2])
        loss = residual(m, xb0,xb1,mb0,mb1) + lam_d*(((pu1-U1[ivb])**2).mean()+((pu2-U2[ivb])**2).mean())
        loss.backward(); return loss
    opt.step(cl)
    train_time = time.time() - t0
    print(f"trained in {train_time:.0f}s")

    # ---- evaluate on the SAME 150 test params + per-point predict timing ----
    L2U=np.zeros(M_test); L2P=np.zeros(M_test); H1U=np.zeros(M_test); pt=np.zeros(M_test)
    for i in range(M_test):
        mu = test_params[i]
        t=time.time()
        with torch.no_grad():
            m0=torch.full((Nu_dof,1),mu[0]); m1=torch.full((Nu_dof,1),mu[1])
            pu1,pu2,_=m.fields(xu[:,0:1],xu[:,1:2],m0,m1)
            m0p=torch.full((Np_dof,1),mu[0]); m1p=torch.full((Np_dof,1),mu[1])
            _,_,pp=m.fields(xp[:,0:1],xp[:,1:2],m0p,m1p)
            z=torch.zeros(1,1); _,_,p0=m.fields(z,z,torch.tensor([[mu[0]]]),torch.tensor([[mu[1]]]))
        U=np.zeros(fe["N_u"]); U[pd0]=pu1[:,0].numpy(); U[pd1]=pu2[:,0].numpy()
        P=(pp[:,0]-p0[0,0]).numpy()
        pt[i]=time.time()-t
        L2U[i],L2P[i],H1U[i]=rel_errors(fe,U,P,u_fom[:,i],p_fom[:,i])
    predict_mean=float(pt.mean()); speedup=fom_mean/max(predict_mean,1e-12)
    print(f"TEST(150): relL2(u) mean={L2U.mean():.4f} median={np.median(L2U):.4f} max={L2U.max():.4f}")
    print(f"           relL2(p) mean={L2P.mean():.4f}  relH1(u) mean={H1U.mean():.4f}")
    print(f"           predict_mean={predict_mean*1000:.3f} ms  speedup={speedup:.1f}x  train={train_time:.0f}s")
    np.savez_compressed(f"{DATA}/pinn_test_errors.npz", rel_l2_u=L2U, rel_l2_p=L2P, rel_h1_u=H1U, test_params=test_params)
    np.savez_compressed(f"{DATA}/pinn_timing.npz", train_time=np.array(train_time),
                        predict_mean=np.array(predict_mean), predict_times=pt,
                        fom_mean=np.array(fom_mean), speedup=np.array(speedup))
    print("saved pinn_test_errors.npz + pinn_timing.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsnap", type=int, default=400)
    ap.add_argument("--adam", type=int, default=3000)
    ap.add_argument("--lbfgs", type=int, default=600)
    ap.add_argument("--ncoll", type=int, default=3000)
    ap.add_argument("--lam_d", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.nsnap, a.adam, a.lbfgs, a.ncoll, a.lam_d, a.seed)
