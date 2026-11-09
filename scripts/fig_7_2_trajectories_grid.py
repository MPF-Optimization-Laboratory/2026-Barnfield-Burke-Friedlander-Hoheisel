"""Section 7.2 trajectory grid (Figure 7.2 of the paper; single PDF).

Replaces the 8-PDF (6 panels + 2 legend strips) layout of
``fig_7_iteration_trajectories.py`` with one figure drawn at its printed
size (``TEXTWIDTH`` = siamart \\textwidth). Each row has a legend for its Z
sweep above it; each panel has an "(a) ..." caption below it.

Output:
  figures/trajectory_grid.pdf   single 2x3 panel figure

Options (presentation only):
  --log-d       panel (d) on a log y-axis
  --out PATH    write to PATH instead of figures/trajectory_grid.pdf
"""

import sys
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
    series, log_ticks, panel_caption, endpoint_marker, align_columns,
)

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


# --- Presentation ---
# Panel (d) is linear by default. ``--log-d`` draws it on a log axis instead
# (as Figure 7.1(b)); ``--out PATH`` writes to PATH instead of figures/.
PANEL_D_LOG = "--log-d" in sys.argv
OUT = (Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv
       else FIG_DIR / "trajectory_grid.pdf")

row1_style = {Z: series(i) for i, Z in enumerate(ROW1_Zs)}
row2_style = {Z: series(i) for i, Z in enumerate(ROW2_Zs)}
MERIT_YLIM = (1e-14, 1e5)   # shared by (b), (c), (f)
ITER_XLIM_ROW1 = (-1, 51)   # shared by (a), (b), (c)
ITER_XLIM_ROW2 = (-0.5, 13)
TRUTH_STYLE = dict(color="0.78", linewidth=2.0, solid_capstyle="butt")
OMEGA_XLIM = (0.0, 1.5)   # x_base is < 1e-3 max(x_base) for omega > 1.37

# Two subfigures (one per row); each carries its Z legend above its panels.
fig = plt.figure(figsize=(TEXTWIDTH, 2.65))
sf1, sf2 = fig.subfigures(2, 1, hspace=0.07)
ax_a, ax_b, ax_c = sf1.subplots(1, 3)
ax_d, ax_e, ax_f = sf2.subplots(1, 3)

# (a) max-exponent xi_k of the unsafeguarded full Newton step
for Z in ROW1_Zs:
    hist = row1_classical[Z]["history"]
    arr = np.array(hist["full_step_max_exp_arg"], dtype=float)
    if arr.size == 0:
        continue
    arr_clipped = np.maximum(arr, 1.0)
    iters = np.arange(arr.size)
    ax_a.semilogy(iters, arr_clipped, **row1_style[Z])
    endpoint_marker(ax_a, iters[-1], arr_clipped[-1],
                    row1_classical[Z]["status"], row1_style[Z]["color"])
ax_a.axhline(OVERFLOW_LOG_THRESHOLD, color="black", linestyle=(0, (4, 2)),
             linewidth=0.7, zorder=1)
trans_a = mtransforms.blended_transform_factory(ax_a.transAxes, ax_a.transData)
ax_a.text(0.98, OVERFLOW_LOG_THRESHOLD * 1.3, "overflow threshold",
          transform=trans_a, ha="right", va="bottom",
          fontsize=ANNOT_FONTSIZE, color="black")
ax_a.set_ylim(1, 1e5)
ax_a.set_ylabel(r"exponent $\xi_k$")

# (b) classical dual Newton gradient norm
for Z in ROW1_Zs:
    hist = row1_classical[Z]["history"]
    if hist["iter"]:
        ax_b.semilogy(hist["iter"], hist["grad_norm"], **row1_style[Z])
        endpoint_marker(ax_b, hist["iter"][-1], hist["grad_norm"][-1],
                        row1_classical[Z]["status"], row1_style[Z]["color"])
ax_b.set_ylabel(r"$\|\nabla \psi_d(y^k)\|$")

# (c) Algorithm 1 merit
for Z in ROW1_Zs:
    ra = row1_alg1[Z]
    hist = ra["history"]
    status = "converged" if ra["rho"] < ALG1_TOL else "max_iters"
    ax_c.semilogy(hist["iter"], hist["rho"], **row1_style[Z])
    endpoint_marker(ax_c, hist["iter"][-1], hist["rho"][-1], status,
                    row1_style[Z]["color"])
ax_c.set_ylabel(r"$\rho(z^k)$")

for ax in (ax_a, ax_b, ax_c):
    ax.set_xlabel("iteration $k$")
    ax.set_xlim(*ITER_XLIM_ROW1)

# (d) shape recovery: truth as a gray band, recoveries on top
ax_d.plot(omegas, x_base, **TRUTH_STYLE, zorder=1)
for Z in ROW2_Zs:
    r = row2_results[Z]
    if not r["converged"]:
        continue
    ax_d.plot(omegas, r["x"] / Z, **row2_style[Z], zorder=2)
ax_d.set_xlim(*OMEGA_XLIM)
if PANEL_D_LOG:
    ax_d.set_yscale("log")
    ax_d.set_ylim(1e-6, 0.3)
    log_ticks(ax_d, every=2)
else:
    ax_d.set_ylim(bottom=0)
ax_d.set_xlabel(r"$\omega$")
ax_d.set_ylabel(r"$\hat{x} / Z$")

# (e) scale trajectory with targets tau* = Z
for Z in ROW2_Zs:
    ax_e.axhline(Z, **LEVEL_STYLE)
for Z in ROW2_Zs:
    r = row2_results[Z]
    if not r["converged"]:
        continue
    h = r["history"]
    ax_e.semilogy(h["iter"], h["tau"], **row2_style[Z], zorder=2)
ax_e.set_ylim(0.6, 200)
ax_e.set_xlabel("iteration $k$")
ax_e.set_ylabel(r"$\tau_k$")

# (f) Algorithm 1 merit
for Z in ROW2_Zs:
    r = row2_results[Z]
    if not r["converged"]:
        continue
    h = r["history"]
    ax_f.semilogy(h["iter"], h["rho"], **row2_style[Z])
    endpoint_marker(ax_f, h["iter"][-1], h["rho"][-1], "converged",
                    row2_style[Z]["color"])
ax_f.set_xlabel("iteration $k$")
ax_f.set_ylabel(r"$\rho(z^k)$")

for ax in (ax_e, ax_f):
    ax.set_xlim(*ITER_XLIM_ROW2)
    ax.set_xticks([0, 4, 8, 12])

# Common scale and stopping tolerance on the three convergence panels.
for ax in (ax_b, ax_c, ax_f):
    ax.set_ylim(*MERIT_YLIM)
    log_ticks(ax, every=6)
    ax.axhline(ALG1_TOL, **LEVEL_STYLE)
log_ticks(ax_a, every=2)
log_ticks(ax_e, every=1)

# Panel captions: method and quantity; the figure caption in the paper
# describes each panel in full. Kept short so that no caption is wider than
# its panel, which would widen that column in constrained layout.
CAPTIONS = [
    (ax_a, "(a) Standard, exponent"), (ax_b, "(b) Standard, gradient"),
    (ax_c, "(c) Scale-shape, merit"), (ax_d, "(d) Computed solutions"),
    (ax_e, "(e) Scale trajectory"), (ax_f, "(f) Merit convergence"),
]
for ax, text in CAPTIONS:
    panel_caption(ax, text)

# Row legends above each row
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
sf1.legend(handles=row1_handles, loc="outside upper center",
           ncol=len(row1_handles))
sf2.legend(handles=row2_handles, loc="outside upper center",
           ncol=len(row2_handles))

for sf in (sf1, sf2):
    sf.align_xlabels()

# Lay out once, then give each column of panels a common width.
align_columns(fig, [(ax_a, ax_d), (ax_b, ax_e), (ax_c, ax_f)])

fig.savefig(OUT)
plt.close(fig)
print(f"Saved {OUT}")
