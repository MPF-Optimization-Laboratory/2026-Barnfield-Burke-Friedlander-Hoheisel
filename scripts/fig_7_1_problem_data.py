"""Figure 7.1: UEG problem data — singular values of A and RPA prior q.

Two-panel figure showing (a) the singular-value decay of the periodic
Laplace kernel and (b) the RPA prior and ground-truth spectral function
on a semilog scale.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from selfscaling import generate_ueg_problem
from selfscaling.figstyle import setup as figsetup, loggrid, TEXTWIDTH

figsetup()

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Load problem data ──
data = generate_ueg_problem(Z=1.0, noise_level=1e-4, n_samples=1000, seed=42)
A = data["A"]
mu = data["mu"]
x_true = data["x_true"]
omegas = data["omegas"]

# ── SVD ──
s = np.linalg.svd(A, compute_uv=False)
nrank = int(np.sum(s > 1e-10))

# ── Figure ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 0.38 * TEXTWIDTH))

# Panel (a): singular values
m = A.shape[0]
ax1.semilogy(range(1, m + 1), s[:m], "o", color="C0", markersize=2.5)
ax1.axhline(1e-10, color="0.4", linestyle="--", linewidth=0.8)
eps_floor = s[0] * np.finfo(float).eps
ax1.axhspan(s[:m].min() / 3, eps_floor, color="0.85", zorder=0)
ax1.text(105, eps_floor / 8, r"$\sigma_1 \cdot \varepsilon_{\mathrm{mach}}$",
         fontsize=7, color="0.4", ha="center")
ax1.annotate(
    rf"rank $= {nrank}$",
    xy=(nrank, s[nrank - 1]),
    xytext=(nrank + 25, 1e-3),
    fontsize=7.5,
    arrowprops=dict(arrowstyle="->", color="0.4", linewidth=0.7),
    color="0.4",
)
ax1.set_ylim(bottom=s[:m].min() / 3, top=s[0] * 3)
ax1.set_xlabel(r"Index $i$")
ax1.set_ylabel(r"$\sigma_i(A)$")
ax1.set_title("(a)")
loggrid(ax1)

# Panel (b): prior q and ground truth x*
ax2.semilogy(omegas, x_true, color="C0", label=r"Truth $x^\ast$ (Mermin)")
ax2.semilogy(omegas, mu, color="C1", linestyle="--", label=r"Prior $q$ (RPA)")
ax2.axhline(1e-16, color="0.4", linestyle=":", linewidth=0.8, label=r"Floor $10^{-16}$")
ax2.set_xlabel(r"$\omega$")
ax2.set_ylabel("Magnitude")
ax2.set_title("(b)")
ax2.legend(fontsize=7, loc="lower left")
loggrid(ax2)

# Inset: zoom on the plasmon peak
axins = ax2.inset_axes([0.42, 0.54, 0.52, 0.40])
mask = (omegas >= 0.0) & (omegas <= 1.8)
axins.plot(omegas[mask], x_true[mask], color="C0")
axins.plot(omegas[mask], mu[mask], color="C1", linestyle="--")
axins.set_xlim(0.0, 1.8)
axins.tick_params(labelsize=5.5)
axins.yaxis.set_label_position("right")
axins.yaxis.tick_right()

fig.tight_layout()
out_path = FIG_DIR / "problem_data.pdf"
fig.savefig(out_path, bbox_inches="tight")
print(f"Saved {out_path}")
