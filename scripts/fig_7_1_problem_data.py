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

from scaleshape import generate_ueg_problem
from scaleshape.figstyle import (
    setup as figsetup, TEXTWIDTH, OKABE_ITO, ANNOT_FONTSIZE, REF_GRAY,
    log_ticks, panel_caption,
)

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

# ── Figure (drawn at its printed size, width = \linewidth) ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.9))

# Panel (a): singular values
m = A.shape[0]
ax1.semilogy(range(1, m + 1), s[:m], "o", color="black", markersize=1.6,
             markeredgewidth=0, zorder=3)
ax1.axhline(1e-10, color=REF_GRAY, linestyle=(0, (4, 2)), linewidth=0.7)
eps_floor = s[0] * np.finfo(float).eps
ax1.axhspan(s[:m].min() / 3, eps_floor, color="0.9", linewidth=0, zorder=0)
# Label the shaded band in its empty lower-left corner, clear of the data.
ax1.text(0.05, 0.06, r"below $\sigma_1 \varepsilon_{\mathrm{mach}}$",
         transform=ax1.transAxes, fontsize=ANNOT_FONTSIZE, color="0.3",
         ha="left", va="bottom")
ax1.annotate(
    rf"rank $= {nrank}$",
    xy=(nrank, s[nrank - 1]),
    xytext=(nrank + 25, 1e-3),
    fontsize=ANNOT_FONTSIZE,
    arrowprops=dict(arrowstyle="-", color="black", linewidth=0.6,
                    shrinkA=1, shrinkB=2),
    va="center",
)
ax1.set_ylim(bottom=s[:m].min() / 3, top=s[0] * 3)
log_ticks(ax1, every=6)
ax1.set_xlabel(r"index $i$")
ax1.set_ylabel(r"$\sigma_i(A)$")

# Panel (b): prior q and ground truth x*
ax2.semilogy(omegas, x_true, color="black", label=r"truth $x^\natural$")
ax2.semilogy(omegas, mu, color=OKABE_ITO["orange"], linestyle=(0, (5, 2)),
             label=r"prior $q$")
ax2.axhline(1e-16, color=REF_GRAY, linestyle=(0, (1.5, 1)), linewidth=0.8,
            label=r"floor $10^{-16}$")
log_ticks(ax2, every=10)
ax2.set_xlabel(r"$\omega$")
ax2.set_ylabel(r"density $x^\natural_j$, $q_j$")
ax2.legend(loc="lower left")

panel_caption(ax1, r"(a) Singular values of $A$")
panel_caption(ax2, r"(b) Truth and prior")

out_path = FIG_DIR / "problem_data.pdf"
fig.savefig(out_path)
print(f"Saved {out_path}")
