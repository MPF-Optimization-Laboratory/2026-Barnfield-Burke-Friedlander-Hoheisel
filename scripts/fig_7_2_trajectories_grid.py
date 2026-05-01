"""Section 7.2 trajectory grid (prototype: single combined PDF).

Replaces the 8-PDF (6 panels + 2 legend strips) layout of
``fig_7_iteration_trajectories.py`` with one figure assembled internally
via gridspec. The two row-legends sit in dedicated narrow rows above
their panels; panel labels (a)-(f) live as in-axes titles, so the
LaTeX subfigure captions vanish.

Output:
  figures/trajectory_grid.pdf   single 2x3 panel figure
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

from scaleshape import generate_ueg_problem, naive_dual_newton, scaleshape_solve
from scaleshape.figstyle import setup as figsetup, loggrid, TEXTWIDTH

figsetup()

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

NOISE_LEVEL = 1e-4
N_SAMPLES = 1000
LAM = 1e-5
MU_FLOOR = 1e-16
OVERFLOW_LOG_THRESHOLD = np.log(np.finfo(np.float64).max)

ALG1_TOL = 1e-8
ALG1_MAX_ITERS = 300

ROW1_Zs = [2**4, 2**6, 2**8, 2**10]
ROW1_MAX_CLASSICAL = 50
ROW1_MAX_ALG1 = ALG1_MAX_ITERS
ROW1_TAU0 = 1.0

ROW2_Zs = [1, 10, 100]
ROW2_TAU0 = 1.0

# --- Compute (identical to fig_7_iteration_trajectories.py) ---
base = generate_ueg_problem(Z=1, noise_level=NOISE_LEVEL, n_samples=N_SAMPLES, seed=43)
A = base["A"]
omegas = base["omegas"]
mu_raw = base["mu"].astype(np.float64)
m, n = A.shape

mu_floor = np.maximum(mu_raw, MU_FLOOR)
mu_floor = mu_floor / mu_floor.sum()

base_r2 = generate_ueg_problem(
    Z=ROW2_Zs[0], noise_level=NOISE_LEVEL, n_samples=N_SAMPLES,
    seed=42 + ROW2_Zs[0],
)
x_base = base_r2["x_true"].astype(np.float64) / ROW2_Zs[0]

row1_classical = {}
row1_alg1 = {}
for Z in ROW1_Zs:
    prob = generate_ueg_problem(
        Z=Z, noise_level=NOISE_LEVEL, n_samples=N_SAMPLES, seed=42 + Z,
    )
    b_Z = prob["b"]
    c = np.zeros(n)
    row1_classical[Z] = naive_dual_newton(
        A, b_Z, c, mu_floor, LAM,
        max_iters=ROW1_MAX_CLASSICAL, tol=ALG1_TOL,
    )
    row1_alg1[Z] = scaleshape_solve(
        A, b_Z, c, mu_floor, lam=LAM, tau0=ROW1_TAU0,
        tol=ALG1_TOL, max_iters=ROW1_MAX_ALG1, verbose=False,
    )

row2_results = {}
for Z in ROW2_Zs:
    prob = generate_ueg_problem(
        Z=Z, noise_level=NOISE_LEVEL, n_samples=N_SAMPLES, seed=42 + Z,
    )
    b_Z = prob["b"]
    c = np.zeros(n)
    res = scaleshape_solve(
        prob["A"], b_Z, c, mu_floor,
        lam=LAM, tau0=ROW2_TAU0, tol=ALG1_TOL,
        max_iters=ALG1_MAX_ITERS, verbose=False,
    )
    row2_results[Z] = dict(
        converged=(res["rho"] < ALG1_TOL * 10),
        x=res["x"], tau=float(res["tau"]),
        history=res["history"],
    )

TERM_STYLE = {
    "converged":      dict(marker="o", color="green", s=50, edgecolors="black", linewidths=0.8),
    "armijo_fatal":   dict(marker="v", color="red",   s=50, edgecolors="black", linewidths=0.8),
    "overflow_fatal": dict(marker="X", color="red",   s=50, edgecolors="black", linewidths=0.8),
    "max_iters":      dict(marker="s", color="white", s=50, edgecolors="gray",  linewidths=1.2),
}


def endpoint_marker(ax, iters, values, status):
    style = TERM_STYLE.get(status, TERM_STYLE["max_iters"])
    ax.scatter([iters[-1]], [values[-1]], zorder=10, clip_on=False, **style)


row1_colors = {16: "C0", 64: "C1", 256: "C2", 1024: "C3"}
row2_colors = {1: "C0", 10: "C1", 100: "C2"}
MERIT_YLIM = (1e-14, 1e5)


# --- Combined figure ---
# 4-row gridspec: thin legend row, panel row, thin legend row, panel row.
# height_ratios chosen so each panel ends with ~3:5 aspect (h:w),
# tighter than the 4:3 of the per-panel PDFs.
fig = plt.figure(figsize=(TEXTWIDTH, 3.1), layout="constrained")
gs = fig.add_gridspec(
    nrows=4, ncols=3,
    height_ratios=[0.10, 1.0, 0.10, 1.0],
    hspace=0.05, wspace=0.05,
)

ax_leg1 = fig.add_subplot(gs[0, :])
ax_a = fig.add_subplot(gs[1, 0])
ax_b = fig.add_subplot(gs[1, 1])
ax_c = fig.add_subplot(gs[1, 2])
ax_leg2 = fig.add_subplot(gs[2, :])
ax_d = fig.add_subplot(gs[3, 0])
ax_e = fig.add_subplot(gs[3, 1])
ax_f = fig.add_subplot(gs[3, 2])

for spacer in (ax_leg1, ax_leg2):
    spacer.axis("off")

# (a) max-exponent
for Z in ROW1_Zs:
    hist = row1_classical[Z]["history"]
    arr = np.array(hist["full_step_max_exp_arg"], dtype=float)
    if arr.size == 0:
        continue
    arr_clipped = np.maximum(arr, 1.0)
    iters = np.arange(arr.size)
    ax_a.semilogy(iters, arr_clipped, marker="o", color=row1_colors[Z])
    endpoint_marker(ax_a, iters, arr_clipped, row1_classical[Z]["status"])
ax_a.axhline(OVERFLOW_LOG_THRESHOLD, color="k", linestyle="--", linewidth=0.8)
trans_a = mtransforms.blended_transform_factory(ax_a.transAxes, ax_a.transData)
ax_a.text(0.97, OVERFLOW_LOG_THRESHOLD * 1.15, "overflow threshold",
          transform=trans_a, ha="right", va="bottom", fontsize=7, color="0.3")
ax_a.set_ylabel(r"$\max_i\, [A^\top\!(y+d) - \mathbf{1} - c]_i$")

# (b)
for Z in ROW1_Zs:
    hist = row1_classical[Z]["history"]
    if hist["iter"]:
        ax_b.semilogy(hist["iter"], hist["grad_norm"], marker="o",
                      color=row1_colors[Z])
        endpoint_marker(ax_b, hist["iter"], hist["grad_norm"],
                        row1_classical[Z]["status"])
ax_b.set_ylabel(r"$\|\nabla \psi_d(y^k)\|$")

# (c)
for Z in ROW1_Zs:
    ra = row1_alg1[Z]
    hist = ra["history"]
    status = "converged" if ra["rho"] < ALG1_TOL else "max_iters"
    ax_c.semilogy(hist["iter"], hist["rho"], marker="o", color=row1_colors[Z])
    endpoint_marker(ax_c, hist["iter"], hist["rho"], status)
ax_c.set_ylabel(r"$\rho(z^k) = \|F(z^k)\|$")
ax_c.set_ylim(*MERIT_YLIM)

for ax in (ax_a, ax_b, ax_c):
    ax.set_xlabel("Iteration $k$", labelpad=2)
    loggrid(ax)

# (d) shape recovery
ax_d.plot(omegas, x_base, "k--", linewidth=1.2)
for Z in ROW2_Zs:
    r = row2_results[Z]
    if not r["converged"]:
        continue
    ax_d.plot(omegas, r["x"] / Z, color=row2_colors[Z])
ax_d.set_xlabel(r"$\omega$", labelpad=2)
ax_d.set_ylabel(r"$\hat{x} / Z$")
truth_handle = mlines.Line2D([], [], color="k", linestyle="--",
                             linewidth=1.2, label=r"truth $x_{\rm base}$")
ax_d.legend(handles=[truth_handle], fontsize=7, loc="upper right")
loggrid(ax_d)

# (e)
for Z in ROW2_Zs:
    r = row2_results[Z]
    if not r["converged"]:
        continue
    h = r["history"]
    ax_e.semilogy(h["iter"], h["tau"], color=row2_colors[Z], marker="o")
    ax_e.axhline(Z, color=row2_colors[Z], linestyle="--",
                 linewidth=0.7, alpha=0.5)
ax_e.set_xlabel("Iteration $k$", labelpad=2)
ax_e.set_ylabel(r"$\tau_k$")
loggrid(ax_e)

# (f)
for Z in ROW2_Zs:
    r = row2_results[Z]
    if not r["converged"]:
        continue
    h = r["history"]
    ax_f.semilogy(h["iter"], h["rho"], color=row2_colors[Z], marker="o")
ax_f.set_xlabel("Iteration $k$", labelpad=2)
ax_f.set_ylabel(r"$\rho(z^k) = \|F(z^k)\|$")
ax_f.set_ylim(*MERIT_YLIM)
loggrid(ax_f)

# Per-panel tags above the axes; descriptive names live in the LaTeX caption.
PANELS = [(ax_a, "(a)"), (ax_b, "(b)"), (ax_c, "(c)"),
          (ax_d, "(d)"), (ax_e, "(e)"), (ax_f, "(f)")]
for ax, tag in PANELS:
    ax.set_title(tag, loc="left", fontsize=8, pad=2)

# Row legends
row1_handles = [
    mlines.Line2D([], [], color=row1_colors[Z], marker="o", markersize=3,
                  label=fr"$Z = 2^{{{int(np.log2(Z))}}}$")
    for Z in ROW1_Zs
]
row2_handles = [
    mlines.Line2D([], [], color=row2_colors[Z], marker="o", markersize=3,
                  label=fr"$Z = 10^{{{int(np.log10(Z))}}}$")
    for Z in ROW2_Zs
]
ax_leg1.legend(handles=row1_handles, loc="center", ncol=4,
               columnspacing=2.0, handletextpad=0.4, frameon=False,
               fontsize=8)
ax_leg2.legend(handles=row2_handles, loc="center", ncol=3,
               columnspacing=2.0, handletextpad=0.4, frameon=False,
               fontsize=8)

out = FIG_DIR / "trajectory_grid.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
