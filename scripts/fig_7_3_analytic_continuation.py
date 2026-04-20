"""Figure 7.3: UEG analytic continuation across scales Z.

Sweeps Z ∈ {1, 10, 100} on Chuna's UEG test problem (completed Mermin DSF
ground truth, RPA prior, two-sided periodic Laplace kernel, PIMC-style
multiplicative noise). Demonstrates self-scaling: τ* = Z is recovered from
τ₀ = 1 across two orders of magnitude.

RPA prior has exponentially small tails; solver requires μᵢ > 0, so we
floor μ ← max(μ, 1e-16) and renormalize. Quantitative distortion metrics
are printed for transcription into §7.4 prose.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from selfscaling import generate_ueg_problem, selfscaling_solve
from selfscaling.figstyle import setup as figsetup, loggrid, TEXTWIDTH

figsetup()

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# --- Config ---
Zs = [1, 10, 100]
NOISE_LEVEL = 1e-4
N_SAMPLES = 1000
LAM = 1e-5
TOL = 1e-10
MAX_ITERS = 300
TAU0 = 1.0
MU_FLOOR = 1e-16
SUPP_THRESHOLD = 1e-10  # threshold defining "effective support" of μ


# --- Generate Z=1 to pull shared base quantities ---
base = generate_ueg_problem(Z=Zs[0], noise_level=NOISE_LEVEL,
                            n_samples=N_SAMPLES, seed=42 + Zs[0])
A = base["A"]
omegas = base["omegas"]
taus = base["taus"]
beta = float(base["beta"])
mu_raw = base["mu"].astype(np.float64)
x_base = base["x_true"].astype(np.float64) / Zs[0]  # at Z=1, x_true = x_base

m, n = A.shape

# --- μ distortion report ---
n_below = int(np.sum(mu_raw < SUPP_THRESHOLD))
mu_pos = mu_raw[mu_raw > 0]
mu_min_nonzero = float(mu_pos.min()) if mu_pos.size else float("nan")
mu_floor = np.maximum(mu_raw, MU_FLOOR)
added_mass = float(np.sum(mu_floor - mu_raw))
supp_mask = mu_raw > SUPP_THRESHOLD
if supp_mask.any():
    max_rel_shift = float(
        np.max(np.abs(mu_floor[supp_mask] - mu_raw[supp_mask]) / mu_raw[supp_mask])
    )
else:
    max_rel_shift = 0.0
mu_floor = mu_floor / mu_floor.sum()

print("=== Problem dimensions ===")
print(f"m = {m}, n = {n}")
print(f"beta = {beta:.4f}, Δτ = {float(taus[1] - taus[0]):.4e}")
print(f"omega range: [{float(omegas.min()):.4e}, {float(omegas.max()):.4e}]")
print()
print("=== μ flooring report ===")
print(f"  entries < {SUPP_THRESHOLD:g}: {n_below} / {n}")
print(f"  min nonzero μ (raw): {mu_min_nonzero:.4e}")
print(f"  total added mass (before renorm): {added_mass:.4e}")
print(f"  max relative shift on supp(μ > {SUPP_THRESHOLD:g}): {max_rel_shift:.4e}")
print()


# --- Solve for each Z ---
results = {}
for Z in Zs:
    prob = generate_ueg_problem(Z=Z, noise_level=NOISE_LEVEL,
                                n_samples=N_SAMPLES, seed=42 + Z)
    b_Z = prob["b"]
    x_true_Z = prob["x_true"]

    c = np.zeros(n)
    res = selfscaling_solve(
        prob["A"], b_Z, c, mu_floor,
        lam=LAM, tau0=TAU0, tol=TOL, max_iters=MAX_ITERS, verbose=False,
    )

    converged = res["rho"] < TOL * 10
    x_rec = res["x"]
    primal_resid_rel = float(np.linalg.norm(prob["A"] @ x_rec - b_Z) / np.linalg.norm(b_Z))
    x_norm = x_rec / Z
    tv_dist = 0.5 * float(np.sum(np.abs(x_norm - x_base)))
    peak_idx_rec = int(np.argmax(x_norm))
    peak_idx_true = int(np.argmax(x_base))
    peak_omega_err = float(omegas[peak_idx_rec] - omegas[peak_idx_true])
    peak_amp_ratio = float(x_norm[peak_idx_rec] / x_base[peak_idx_true])
    iters = len(res["history"]["iter"])

    results[Z] = dict(
        converged=converged,
        x=x_rec,
        tau=float(res["tau"]),
        iters=iters,
        rho_final=float(res["rho"]),
        primal_resid_rel=primal_resid_rel,
        tv_dist=tv_dist,
        peak_omega_err=peak_omega_err,
        peak_amp_ratio=peak_amp_ratio,
        history=res["history"],
        b=b_Z,
        x_true=x_true_Z,
    )


# --- Per-Z summary table ---
print("=== Per-Z results ===")
header = (
    f"{'Z':>5} {'τ*':>12} {'|τ*-Z|/Z':>12} {'iters':>6} {'ρ_final':>12} "
    f"{'‖Ax-b‖/‖b‖':>14} {'TV':>10} {'Δω_peak':>10} {'amp_rec/true':>14} {'conv':>5}"
)
print(header)
print("-" * len(header))
for Z in Zs:
    r = results[Z]
    print(
        f"{Z:>5} {r['tau']:>12.4e} {abs(r['tau'] - Z) / Z:>12.4e} {r['iters']:>6} "
        f"{r['rho_final']:>12.4e} {r['primal_resid_rel']:>14.4e} {r['tv_dist']:>10.4e} "
        f"{r['peak_omega_err']:>10.3e} {r['peak_amp_ratio']:>14.4f} {str(r['converged']):>5}"
    )

for Z in Zs:
    if not results[Z]["converged"]:
        print(f"\n[NON-CONVERGED] Z={Z}: ρ={results[Z]['rho_final']:.4e} > 10·tol. Skipping in plots.")


# --- Plot ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.5))
colors = {1: "C0", 10: "C1", 100: "C2"}

# Panel (a): normalized spectral recovery
ax1.plot(omegas, x_base, "k--", linewidth=1.5, label=r"truth $x_{\rm base}$")
for Z in Zs:
    r = results[Z]
    if not r["converged"]:
        continue
    ax1.plot(omegas, r["x"] / Z, color=colors[Z], label=f"$Z={Z}$")
ax1.set_xlabel(r"$\omega$")
ax1.set_ylabel(r"$x^\ast / Z$")
ax1.set_title("(a) Spectral recovery (normalized)")
truth_handle = mlines.Line2D([], [], color="k", linestyle="--",
                             linewidth=1.5, label=r"truth $x_{\rm base}$")
ax1.legend(handles=[truth_handle], fontsize=7, loc="upper right")

# Panel (b): τ trajectories
for Z in Zs:
    r = results[Z]
    if not r["converged"]:
        continue
    h = r["history"]
    ax2.semilogy(h["iter"], h["tau"], color=colors[Z], marker="o", label=f"$Z={Z}$")
    ax2.axhline(Z, color=colors[Z], linestyle="--", linewidth=0.7, alpha=0.5)
ax2.set_xlabel("Iteration $k$")
ax2.set_ylabel(r"$\tau_k$")
ax2.set_title(r"(b) Scale trajectory $\tau_k$")
loggrid(ax2)

# Panel (c): merit function
for Z in Zs:
    r = results[Z]
    if not r["converged"]:
        continue
    h = r["history"]
    ax3.semilogy(h["iter"], h["rho"], color=colors[Z], marker="o", label=f"$Z={Z}$")
ax3.set_xlabel("Iteration $k$")
ax3.set_ylabel(r"$\rho(z^k) = \|F(z^k)\|$")
ax3.set_title("(c) Merit function")
loggrid(ax3)

# Figure-level Z color legend (shared across panels)
z_handles = [mlines.Line2D([], [], color=colors[Z], marker="o", markersize=3,
             label=f"$Z = {Z}$") for Z in Zs]
fig.legend(handles=z_handles, loc="upper center", bbox_to_anchor=(0.5, 1.05),
           ncol=len(Zs), columnspacing=1.5, handletextpad=0.3, frameon=False)

fig.tight_layout()
out_path = FIG_DIR / "analytic_continuation.pdf"
fig.savefig(out_path, bbox_inches="tight")
print(f"\nSaved {out_path}")
