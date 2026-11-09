"""Section 7 iteration-trajectory panels (six panels + two legend strips).

Legacy layout: the paper now includes the single-PDF version written by
``fig_7_2_trajectories_grid.py`` (same data, same styling). This script
produces eight individual PDFs that LaTeX can arrange into a 2 x 3
subcaption grid, with labels (a)-(f) typeset in LaTeX.

Row 1 (Overflow resilience, § 7.1; Z in {2^4, 2^6, 2^8, 2^10}):
  trajectory_a.pdf  unsafeguarded full-Newton-step max exponent
  trajectory_b.pdf  classical dual Newton gradient norm
  trajectory_c.pdf  Algorithm 1 merit
  trajectory_legend_row1.pdf   Z colour key

Row 2 (Scale recovery, § 7.2; Z in {1, 10, 100}):
  trajectory_d.pdf  normalized spectral recovery x*/Z
  trajectory_e.pdf  scale trajectory tau_k
  trajectory_f.pdf  Algorithm 1 merit
  trajectory_legend_row2.pdf   Z colour key

Panels (c) and (f) share a log-y range so the merit comparison is visual.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

from scaleshape import generate_ueg_problem, naive_dual_newton, scaleshape_solve
from scaleshape.figstyle import (
    setup as figsetup, TEXTWIDTH, ANNOT_FONTSIZE, LEVEL_STYLE,
    series, log_ticks, endpoint_marker,
)

figsetup()

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Each panel is drawn at its printed size in the subfigure layout,
# width = 0.325\linewidth, so LaTeX includes it without scaling.
PANEL_SIZE = (0.325 * TEXTWIDTH, 1.25)

# --- Shared config ---
NOISE_LEVEL = 1e-4
N_SAMPLES = 1000
LAM = 1e-5
MU_FLOOR = 1e-16
OVERFLOW_LOG_THRESHOLD = np.log(np.finfo(np.float64).max)  # ~709.78

# --- Unified Algorithm 1 settings (ε, K_max) = (1e-8, 300) ---
ALG1_TOL = 1e-8
ALG1_MAX_ITERS = 300

# --- Row 1 (overflow) config ---
ROW1_Zs = [2**4, 2**6, 2**8, 2**10]
ROW1_TOL = ALG1_TOL
ROW1_MAX_CLASSICAL = 50
ROW1_MAX_ALG1 = ALG1_MAX_ITERS
ROW1_TAU0 = 1.0

# --- Row 2 (scale recovery) config ---
ROW2_Zs = [1, 10, 100]
ROW2_TOL = ALG1_TOL
ROW2_MAX_ITERS = ALG1_MAX_ITERS
ROW2_TAU0 = 1.0


# --- Shared A, mu from a Z = 1 instance (A is Z-independent) ---
base = generate_ueg_problem(Z=1, noise_level=NOISE_LEVEL, n_samples=N_SAMPLES, seed=43)
A = base["A"]
omegas = base["omegas"]
mu_raw = base["mu"].astype(np.float64)
m, n = A.shape

mu_floor = np.maximum(mu_raw, MU_FLOOR)
mu_floor = mu_floor / mu_floor.sum()

# Truth shape at Z = 1 for row 2 overlay
base_r2 = generate_ueg_problem(Z=ROW2_Zs[0], noise_level=NOISE_LEVEL,
                               n_samples=N_SAMPLES, seed=42 + ROW2_Zs[0])
x_base = base_r2["x_true"].astype(np.float64) / ROW2_Zs[0]


# --- Row 1 sweep (overflow) ---
row1_classical = {}
row1_alg1 = {}
for Z in ROW1_Zs:
    prob = generate_ueg_problem(Z=Z, noise_level=NOISE_LEVEL,
                                n_samples=N_SAMPLES, seed=42 + Z)
    b_Z = prob["b"]
    c = np.zeros(n)
    row1_classical[Z] = naive_dual_newton(
        A, b_Z, c, mu_floor, LAM,
        max_iters=ROW1_MAX_CLASSICAL, tol=ROW1_TOL,
    )
    row1_alg1[Z] = scaleshape_solve(
        A, b_Z, c, mu_floor, lam=LAM, tau0=ROW1_TAU0,
        tol=ROW1_TOL, max_iters=ROW1_MAX_ALG1, verbose=False,
    )


# --- Row 2 sweep (scale recovery) ---
row2_results = {}
for Z in ROW2_Zs:
    prob = generate_ueg_problem(Z=Z, noise_level=NOISE_LEVEL,
                                n_samples=N_SAMPLES, seed=42 + Z)
    b_Z = prob["b"]
    c = np.zeros(n)
    res = scaleshape_solve(
        prob["A"], b_Z, c, mu_floor,
        lam=LAM, tau0=ROW2_TAU0, tol=ROW2_TOL,
        max_iters=ROW2_MAX_ITERS, verbose=False,
    )
    row2_results[Z] = dict(
        converged=(res["rho"] < ROW2_TOL * 10),
        x=res["x"], tau=float(res["tau"]),
        rho_final=float(res["rho"]),
        history=res["history"],
    )

# --- Presentation (shared with fig_7_2_trajectories_grid.py) ---
row1_style = {Z: series(i) for i, Z in enumerate(ROW1_Zs)}
row2_style = {Z: series(i) for i, Z in enumerate(ROW2_Zs)}

# Shared log-y extent and stopping tolerance on convergence panels (b), (c), (f).
MERIT_YLIM = (1e-14, 1e5)
TRUTH_STYLE = dict(color="0.78", linewidth=2.0, solid_capstyle="butt")
OMEGA_XLIM = (0.0, 1.5)   # x_base is < 1e-3 max(x_base) for omega > 1.37


def save_panel(fig, stem):
    # No bbox_inches="tight": preserve the figsize so every panel PDF has
    # identical outer dimensions, keeping the LaTeX 2x3 grid aligned.
    out = FIG_DIR / f"{stem}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# === (a) Unsafeguarded max-exponent ===
fig, ax = plt.subplots(figsize=PANEL_SIZE)
for Z in ROW1_Zs:
    hist = row1_classical[Z]["history"]
    arr = np.array(hist["full_step_max_exp_arg"], dtype=float)
    if arr.size == 0:
        continue
    arr_clipped = np.maximum(arr, 1.0)
    iters = np.arange(arr.size)
    ax.semilogy(iters, arr_clipped, **row1_style[Z])
    endpoint_marker(ax, iters[-1], arr_clipped[-1],
                    row1_classical[Z]["status"], row1_style[Z]["color"])
ax.axhline(OVERFLOW_LOG_THRESHOLD, color="black", linestyle=(0, (4, 2)),
           linewidth=0.7, zorder=1)
trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
ax.text(0.98, OVERFLOW_LOG_THRESHOLD * 1.3, "overflow threshold",
        transform=trans, ha="right", va="bottom", fontsize=ANNOT_FONTSIZE)
ax.set_xlim(-1, 51)
ax.set_ylim(1, 1e5)
log_ticks(ax, every=2)
ax.set_xlabel("iteration $k$")
ax.set_ylabel(r"exponent $\xi_k$")
save_panel(fig, "trajectory_a")

# === (b) Classical dual Newton gradient norm ===
fig, ax = plt.subplots(figsize=PANEL_SIZE)
for Z in ROW1_Zs:
    hist = row1_classical[Z]["history"]
    if hist["iter"]:
        ax.semilogy(hist["iter"], hist["grad_norm"], **row1_style[Z])
        endpoint_marker(ax, hist["iter"][-1], hist["grad_norm"][-1],
                        row1_classical[Z]["status"], row1_style[Z]["color"])
ax.axhline(ROW1_TOL, **LEVEL_STYLE)
ax.set_xlim(-1, 51)
ax.set_ylim(*MERIT_YLIM)
log_ticks(ax, every=6)
ax.set_xlabel("iteration $k$")
ax.set_ylabel(r"$\|\nabla \psi_d(y^k)\|$")
save_panel(fig, "trajectory_b")

# === (c) Algorithm 1 merit (row 1) ===
fig, ax = plt.subplots(figsize=PANEL_SIZE)
for Z in ROW1_Zs:
    ra = row1_alg1[Z]
    hist = ra["history"]
    status = "converged" if ra["rho"] < ROW1_TOL else "max_iters"
    ax.semilogy(hist["iter"], hist["rho"], **row1_style[Z])
    endpoint_marker(ax, hist["iter"][-1], hist["rho"][-1], status,
                    row1_style[Z]["color"])
ax.axhline(ROW1_TOL, **LEVEL_STYLE)
ax.set_xlim(-1, 51)
ax.set_ylim(*MERIT_YLIM)
log_ticks(ax, every=6)
ax.set_xlabel("iteration $k$")
ax.set_ylabel(r"$\rho(z^k)$")
save_panel(fig, "trajectory_c")

# === (d) Normalized spectral recovery ===
fig, ax = plt.subplots(figsize=PANEL_SIZE)
ax.plot(omegas, x_base, **TRUTH_STYLE, zorder=1)
for Z in ROW2_Zs:
    r = row2_results[Z]
    if not r["converged"]:
        continue
    ax.plot(omegas, r["x"] / Z, **row2_style[Z], zorder=2)
ax.set_xlim(*OMEGA_XLIM)
ax.set_ylim(bottom=0)
ax.set_xlabel(r"$\omega$")
ax.set_ylabel(r"$\hat{x} / Z$")
save_panel(fig, "trajectory_d")

# === (e) Scale trajectory tau_k ===
fig, ax = plt.subplots(figsize=PANEL_SIZE)
for Z in ROW2_Zs:
    ax.axhline(Z, **LEVEL_STYLE)
for Z in ROW2_Zs:
    r = row2_results[Z]
    if not r["converged"]:
        continue
    h = r["history"]
    ax.semilogy(h["iter"], h["tau"], **row2_style[Z], zorder=2)
ax.set_xlim(-0.5, 13)
ax.set_xticks([0, 4, 8, 12])
ax.set_ylim(0.6, 200)
log_ticks(ax, every=1)
ax.set_xlabel("iteration $k$")
ax.set_ylabel(r"$\tau_k$")
save_panel(fig, "trajectory_e")

# === (f) Algorithm 1 merit (row 2) ===
fig, ax = plt.subplots(figsize=PANEL_SIZE)
for Z in ROW2_Zs:
    r = row2_results[Z]
    if not r["converged"]:
        continue
    h = r["history"]
    ax.semilogy(h["iter"], h["rho"], **row2_style[Z])
    endpoint_marker(ax, h["iter"][-1], h["rho"][-1], "converged",
                    row2_style[Z]["color"])
ax.axhline(ROW2_TOL, **LEVEL_STYLE)
ax.set_xlim(-0.5, 13)
ax.set_xticks([0, 4, 8, 12])
ax.set_ylim(*MERIT_YLIM)
log_ticks(ax, every=6)
ax.set_xlabel("iteration $k$")
ax.set_ylabel(r"$\rho(z^k)$")
save_panel(fig, "trajectory_f")


# === Legend strips (one per row) ===
def save_legend_strip(handles, stem, figsize=(TEXTWIDTH, 0.2)):
    fig = plt.figure(figsize=figsize)
    fig.legend(handles=handles, loc="center", ncol=len(handles))
    out = FIG_DIR / f"{stem}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


row1_handles = [
    mlines.Line2D([], [], **row1_style[Z],
                  label=fr"$Z = 2^{{{int(np.log2(Z))}}}$")
    for Z in ROW1_Zs
]
row2_handles = [
    mlines.Line2D([], [], **row2_style[Z],
                  label=fr"$Z = 10^{{{int(np.log10(Z))}}}$")
    for Z in ROW2_Zs
] + [mlines.Line2D([], [], **TRUTH_STYLE, label=r"truth $p^\natural$")]
save_legend_strip(row1_handles, "trajectory_legend_row1")
save_legend_strip(row2_handles, "trajectory_legend_row2")
