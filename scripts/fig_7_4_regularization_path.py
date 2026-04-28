"""Figure 7.4: Regularization path.

Sweeps λ from 1e-1 down to 1e-8 for Z ∈ {1, 10, 100} on the UEG test
problem. Plots the data-fit residual ‖Ax* − b‖₂ as a function of λ
(decreasing left to right), illustrating the bias–regularization tradeoff.

Saves `regularization_path_rel.pdf` — the relative-residual form
shown in the paper.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from selfscaling import generate_ueg_problem, selfscaling_solve
from selfscaling.figstyle import setup as figsetup, loggrid, TEXTWIDTH

figsetup()

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


# --- Config ---
Zs = [1, 10, 100]
LAMBDAS = np.logspace(-1, -8, 20)
NOISE_LEVEL = 1e-4
N_SAMPLES = 1000
TOL = 1e-8
MAX_ITERS = 300
TAU0 = 1.0
MU_FLOOR = 1e-16

colors = {1: "C0", 10: "C1", 100: "C2"}

# --- Precompute floored prior (shared across Z) ---
base = generate_ueg_problem(Z=1.0, noise_level=NOISE_LEVEL,
                            n_samples=N_SAMPLES, seed=43)
mu_raw = base["mu"].astype(np.float64)
mu_floor = np.maximum(mu_raw, MU_FLOOR)
mu_floor = mu_floor / mu_floor.sum()

# --- Sweep ---
results = {}
for Z in Zs:
    prob = generate_ueg_problem(Z=Z, noise_level=NOISE_LEVEL,
                                n_samples=N_SAMPLES, seed=42 + Z)
    A = prob["A"]
    b = prob["b"]
    n = A.shape[1]
    c = np.zeros(n)

    b_norm = float(np.linalg.norm(b))
    residuals = []
    rho_finals = []
    iter_counts = []
    tau_finals = []
    converged_flags = []

    for lam in LAMBDAS:
        res = selfscaling_solve(
            A, b, c, mu_floor, lam=lam, tau0=TAU0,
            tol=TOL, max_iters=MAX_ITERS, verbose=False,
        )
        x = res["x"]
        resid = float(np.linalg.norm(A @ x - b))
        residuals.append(resid)
        rho_finals.append(float(res["rho"]))
        iter_counts.append(len(res["history"]["iter"]))
        tau_finals.append(float(res["tau"]))
        converged_flags.append(res["rho"] < TOL * 10)

    results[Z] = dict(
        lambdas=LAMBDAS,
        residuals=np.array(residuals),
        rel_residuals=np.array(residuals) / b_norm,
        rho_finals=np.array(rho_finals),
        iters=np.array(iter_counts),
        taus=np.array(tau_finals),
        converged=converged_flags,
    )

# --- Diagnostics ---
print("=== Regularization path results ===")
for Z in Zs:
    r = results[Z]
    print(f"\nZ = {Z}:")
    print(f"  {'lambda':>12} {'||Ax-b||':>12} {'rho':>12} {'iters':>6} {'tau':>12} {'conv':>5}")
    print(f"  {'-'*65}")
    for i, lam in enumerate(r["lambdas"]):
        print(f"  {lam:>12.2e} {r['residuals'][i]:>12.4e} {r['rho_finals'][i]:>12.4e} "
              f"{r['iters'][i]:>6} {r['taus'][i]:>12.4e} {str(r['converged'][i]):>5}")

# --- Plot (normalized) ---
fig, ax = plt.subplots(figsize=(0.7 * TEXTWIDTH, 2.0))
for Z in Zs:
    r = results[Z]
    ax.loglog(r["lambdas"], r["rel_residuals"], "-o", color=colors[Z],
              label=fr"$Z = 10^{{{int(np.log10(Z))}}}$")
ax.invert_xaxis()
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"$\|Ax^\ast - b\| / \|b\|$")
ax.legend()
loggrid(ax)
fig.tight_layout()
out_path_rel = FIG_DIR / "regularization_path_rel.pdf"
fig.savefig(out_path_rel, bbox_inches="tight")
print(f"Saved {out_path_rel}")
