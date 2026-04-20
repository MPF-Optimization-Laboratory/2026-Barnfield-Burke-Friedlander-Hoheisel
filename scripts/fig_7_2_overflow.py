"""Figure 7.2: Overflow resilience across Z on UEG data.

Demonstrates that the classical dual
    ψ_d(y) = ⟨b, y⟩ − (λ/2)‖y‖² − Σ μ·exp(Aᵀy − 1 − c)
requires evaluation of exp(Aᵀy − 1 − c), which overflows whenever any
coordinate of Aᵀy − 1 − c exceeds ~709 (the float64 overflow threshold).
Algorithm 1 evaluates exponentials only through the shifted log-sum-exp
identity, whose exponents are ≤ 0 by construction; overflow is
structurally impossible.

Runs classical dual Newton (with Armijo backtracking that includes an
overflow-detection fallback) and Algorithm 1 on the UEG problem of
Chuna et al. (2025) for a range of target scales Z. For each Z we record:
  (a) the exponent argument of the unsafeguarded full Newton step,
  (b) classical-dual grad-norm history,
  (c) Algorithm 1 merit-function history.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

from selfscaling import generate_ueg_problem, naive_dual_newton, selfscaling_solve
from selfscaling.figstyle import setup as figsetup, loggrid, TEXTWIDTH

figsetup()

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# --- Config ---
Zs = [2**4, 2**6, 2**8, 2**10]  # 16, 64, 256, 1024
NOISE_LEVEL = 1e-4
N_SAMPLES = 1000
LAM = 1e-5
TOL = 1e-8
MAX_ITERS_CLASSICAL = 50
MAX_ITERS_ALG1 = 200
TAU0 = 1.0
MU_FLOOR = 1e-16
OVERFLOW_LOG_THRESHOLD = np.log(np.finfo(np.float64).max)  # ~709.78


# --- Base problem: A, mu, grids are Z-independent ---
base = generate_ueg_problem(Z=1, noise_level=NOISE_LEVEL, n_samples=N_SAMPLES, seed=43)
A = base["A"]
mu_raw = base["mu"].astype(np.float64)
m, n = A.shape

mu_floor = np.maximum(mu_raw, MU_FLOOR)
mu_floor = mu_floor / mu_floor.sum()

print(f"=== Problem: m = {m}, n = {n} ===")
print(f"mu_raw: min(positive) = {mu_raw[mu_raw > 0].min():.4e}, "
      f"entries < 1e-10: {int(np.sum(mu_raw < 1e-10))} / {n}")
print(f"Overflow threshold (log): {OVERFLOW_LOG_THRESHOLD:.2f}")
print()


# --- Sweep ---
results_classical = {}
results_alg1 = {}

for Z in Zs:
    prob = generate_ueg_problem(Z=Z, noise_level=NOISE_LEVEL,
                                n_samples=N_SAMPLES, seed=42 + Z)
    b_Z = prob["b"]
    c = np.zeros(n)

    res_c = naive_dual_newton(
        A, b_Z, c, mu_floor, LAM, max_iters=MAX_ITERS_CLASSICAL, tol=TOL
    )
    results_classical[Z] = res_c

    res_a = selfscaling_solve(
        A, b_Z, c, mu_floor, lam=LAM, tau0=TAU0,
        tol=TOL, max_iters=MAX_ITERS_ALG1, verbose=False,
    )
    results_alg1[Z] = res_a


# --- Summary ---
def classical_status(res):
    hist = res["history"]
    status = res["status"]
    iters_done = len(hist["iter"])
    first_ov = hist.get("first_overflow_iter")
    ov_suffix = f" (ov@{first_ov})" if first_ov is not None else ""
    if status == "converged":
        tag = "converged" + ov_suffix
    elif status == "armijo_fatal":
        tag = f"armijo fatal @ k={iters_done - 1}"
    elif status == "overflow_fatal":
        tag = f"overflow fatal @ k={iters_done - 1}"
    else:
        tag = "max_iters" + ov_suffix
    final_grad = hist["grad_norm"][-1] if hist["grad_norm"] else float("nan")
    max_full_arg = (max(hist["full_step_max_exp_arg"])
                    if hist["full_step_max_exp_arg"] else float("nan"))
    return iters_done, tag, final_grad, max_full_arg


print("=== Per-Z results ===")
header = (
    f"{'Z':>6} {'classical iters':>16} {'classical tag':>16} "
    f"{'grad_final':>12} {'max full-step arg':>20} | "
    f"{'Alg 1 iters':>12} {'Alg 1 rho':>14} {'Alg 1 tau':>14}"
)
print(header)
print("-" * len(header))
for Z in Zs:
    iters_c, tag_c, grad_f, max_arg = classical_status(results_classical[Z])
    ra = results_alg1[Z]
    print(
        f"{Z:>6} {iters_c:>16} {tag_c:>16} {grad_f:>12.4e} {max_arg:>20.4e} | "
        f"{len(ra['history']['iter']):>12} {ra['rho']:>14.4e} {ra['tau']:>14.4e}"
    )
print()


# --- Termination marker styles ---
TERM_STYLE = {
    "converged":      dict(marker="o", color="green",  s=50, edgecolors="black", linewidths=0.8),
    "armijo_fatal":   dict(marker="v", color="red",    s=50, edgecolors="black", linewidths=0.8),
    "overflow_fatal": dict(marker="X", color="red",    s=50, edgecolors="black", linewidths=0.8),
    "max_iters":      dict(marker="s", color="white",  s=50, edgecolors="gray",  linewidths=1.2),
}

def endpoint_marker(ax, iters, values, status):
    style = TERM_STYLE.get(status, TERM_STYLE["max_iters"])
    ax.scatter([iters[-1]], [values[-1]], zorder=10, clip_on=False, **style)


# --- Plot ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.5))
colors = {16: "C0", 64: "C1", 256: "C2", 1024: "C3"}

# Panel (a): max exp argument of the UNSAFEGUARDED full Newton step.
for Z in Zs:
    hist = results_classical[Z]["history"]
    arr = np.array(hist["full_step_max_exp_arg"], dtype=float)
    if arr.size == 0:
        continue
    arr_clipped = np.maximum(arr, 1.0)
    iters = np.arange(arr.size)
    ax1.semilogy(iters, arr_clipped, marker="o", color=colors[Z])
    endpoint_marker(ax1, iters, arr_clipped, results_classical[Z]["status"])
ax1.axhline(OVERFLOW_LOG_THRESHOLD, color="k", linestyle="--", linewidth=0.8)
trans = mtransforms.blended_transform_factory(ax1.transAxes, ax1.transData)
ax1.text(0.97, OVERFLOW_LOG_THRESHOLD * 1.15, "overflow threshold",
         transform=trans, ha="right", va="bottom", fontsize=7, color="0.3")
ax1.set_xlabel("Iteration $k$")
ax1.set_ylabel(r"$\max_i\, [A^\top\!(y^k\!+\!d^k) - \mathbf{1} - c]_i$")
loggrid(ax1)

# Panel (b): classical dual Newton gradient norm
for Z in Zs:
    hist = results_classical[Z]["history"]
    if hist["iter"]:
        ax2.semilogy(hist["iter"], hist["grad_norm"], marker="o", color=colors[Z])
        endpoint_marker(ax2, hist["iter"], hist["grad_norm"],
                        results_classical[Z]["status"])
ax2.set_xlabel("Iteration $k$")
ax2.set_ylabel(r"$\|\nabla \psi_d(y^k)\|$")
loggrid(ax2)

# Panel (c): Alg 1 merit
for Z in Zs:
    ra = results_alg1[Z]
    hist = ra["history"]
    alg1_status = "converged" if ra["rho"] < TOL else "max_iters"
    ax3.semilogy(hist["iter"], hist["rho"], marker="o", color=colors[Z])
    endpoint_marker(ax3, hist["iter"], hist["rho"], alg1_status)
ax3.set_xlabel("Iteration $k$")
ax3.set_ylabel(r"$\rho(z^k) = \|F(z^k)\|$")
loggrid(ax3)

# Figure-level Z color legend (shared across panels)
z_handles = [mlines.Line2D([], [], color=colors[Z], marker="o", markersize=3,
             label=f"$Z = 2^{{{int(np.log2(Z))}}}$") for Z in Zs]
fig.legend(handles=z_handles, loc="upper center", bbox_to_anchor=(0.5, 1.05),
           ncol=len(Zs), columnspacing=1.5, handletextpad=0.3, frameon=False)

fig.tight_layout()
out_path = FIG_DIR / "overflow_sweep.pdf"
fig.savefig(out_path, bbox_inches="tight")
print(f"Saved {out_path}")
