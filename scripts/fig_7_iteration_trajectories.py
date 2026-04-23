"""Section 7 iteration-trajectory panels (six panels + two legend strips).

Produces eight individual PDFs that LaTeX arranges into a 2 x 3 subcaption
grid. Keeping labels (a)-(f) in LaTeX rather than burnt into the figure
makes panel labels trivial to edit without regenerating.

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

from selfscaling import generate_ueg_problem, naive_dual_newton, selfscaling_solve
from selfscaling.figstyle import setup as figsetup, loggrid, TEXTWIDTH

figsetup()

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Each panel renders at ~1/3 of TEXTWIDTH (= 6 in) so LaTeX can include at
# width = 0.32\linewidth without further scaling distortions.
PANEL_SIZE = (TEXTWIDTH / 3.0, 1.5)

# --- Shared config ---
NOISE_LEVEL = 1e-4
N_SAMPLES = 1000
LAM = 1e-5
MU_FLOOR = 1e-16
OVERFLOW_LOG_THRESHOLD = np.log(np.finfo(np.float64).max)  # ~709.78

# --- Row 1 (overflow) config ---
ROW1_Zs = [2**4, 2**6, 2**8, 2**10]
ROW1_TOL = 1e-8
ROW1_MAX_CLASSICAL = 50
ROW1_MAX_ALG1 = 200
ROW1_TAU0 = 1.0

# --- Row 2 (scale recovery) config ---
ROW2_Zs = [1, 10, 100]
ROW2_TOL = 1e-10
ROW2_MAX_ITERS = 300
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
    row1_alg1[Z] = selfscaling_solve(
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
    res = selfscaling_solve(
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


# --- Termination marker styles (row 1) ---
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

# Shared log-y extent on merit panels (c) and (f).
MERIT_YLIM = (1e-14, 1e5)


def save_panel(fig, stem):
    # No bbox_inches="tight": preserve the figsize so every panel PDF has
    # identical outer dimensions, keeping the LaTeX 2x3 grid aligned.
    out = FIG_DIR / f"{stem}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# === (a) Unsafeguarded max-exponent ===
fig, ax = plt.subplots(figsize=PANEL_SIZE, layout="constrained")
for Z in ROW1_Zs:
    hist = row1_classical[Z]["history"]
    arr = np.array(hist["full_step_max_exp_arg"], dtype=float)
    if arr.size == 0:
        continue
    arr_clipped = np.maximum(arr, 1.0)
    iters = np.arange(arr.size)
    ax.semilogy(iters, arr_clipped, marker="o", color=row1_colors[Z])
    endpoint_marker(ax, iters, arr_clipped, row1_classical[Z]["status"])
ax.axhline(OVERFLOW_LOG_THRESHOLD, color="k", linestyle="--", linewidth=0.8)
trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
ax.text(0.97, OVERFLOW_LOG_THRESHOLD * 1.15, "overflow threshold",
        transform=trans, ha="right", va="bottom", fontsize=7, color="0.3")
ax.set_xlabel("Iteration $k$")
ax.set_ylabel(r"$\max_i\, [A^\top\!(y+d) - \mathbf{1} - c]_i$")
loggrid(ax)
save_panel(fig, "trajectory_a")

# === (b) Classical dual Newton gradient norm ===
fig, ax = plt.subplots(figsize=PANEL_SIZE, layout="constrained")
for Z in ROW1_Zs:
    hist = row1_classical[Z]["history"]
    if hist["iter"]:
        ax.semilogy(hist["iter"], hist["grad_norm"], marker="o",
                    color=row1_colors[Z])
        endpoint_marker(ax, hist["iter"], hist["grad_norm"],
                        row1_classical[Z]["status"])
ax.set_xlabel("Iteration $k$")
ax.set_ylabel(r"$\|\nabla \psi_d(y^k)\|$")
loggrid(ax)
save_panel(fig, "trajectory_b")

# === (c) Algorithm 1 merit (row 1) ===
fig, ax = plt.subplots(figsize=PANEL_SIZE, layout="constrained")
for Z in ROW1_Zs:
    ra = row1_alg1[Z]
    hist = ra["history"]
    status = "converged" if ra["rho"] < ROW1_TOL else "max_iters"
    ax.semilogy(hist["iter"], hist["rho"], marker="o", color=row1_colors[Z])
    endpoint_marker(ax, hist["iter"], hist["rho"], status)
ax.set_xlabel("Iteration $k$")
ax.set_ylabel(r"$\rho(z^k) = \|F(z^k)\|$")
ax.set_ylim(*MERIT_YLIM)
loggrid(ax)
save_panel(fig, "trajectory_c")

# === (d) Normalized spectral recovery ===
fig, ax = plt.subplots(figsize=PANEL_SIZE, layout="constrained")
ax.plot(omegas, x_base, "k--", linewidth=1.2)
for Z in ROW2_Zs:
    r = row2_results[Z]
    if not r["converged"]:
        continue
    ax.plot(omegas, r["x"] / Z, color=row2_colors[Z])
ax.set_xlabel(r"$\omega$")
ax.set_ylabel(r"$\hat{x} / Z$")
truth_handle = mlines.Line2D([], [], color="k", linestyle="--",
                             linewidth=1.2, label=r"truth $x_{\rm base}$")
ax.legend(handles=[truth_handle], fontsize=7, loc="upper right")
loggrid(ax)
save_panel(fig, "trajectory_d")

# === (e) Scale trajectory tau_k ===
fig, ax = plt.subplots(figsize=PANEL_SIZE, layout="constrained")
for Z in ROW2_Zs:
    r = row2_results[Z]
    if not r["converged"]:
        continue
    h = r["history"]
    ax.semilogy(h["iter"], h["tau"], color=row2_colors[Z], marker="o")
    ax.axhline(Z, color=row2_colors[Z], linestyle="--",
               linewidth=0.7, alpha=0.5)
ax.set_xlabel("Iteration $k$")
ax.set_ylabel(r"$\tau_k$")
loggrid(ax)
save_panel(fig, "trajectory_e")

# === (f) Algorithm 1 merit (row 2) ===
fig, ax = plt.subplots(figsize=PANEL_SIZE, layout="constrained")
for Z in ROW2_Zs:
    r = row2_results[Z]
    if not r["converged"]:
        continue
    h = r["history"]
    ax.semilogy(h["iter"], h["rho"], color=row2_colors[Z], marker="o")
ax.set_xlabel("Iteration $k$")
ax.set_ylabel(r"$\rho(z^k) = \|F(z^k)\|$")
ax.set_ylim(*MERIT_YLIM)
loggrid(ax)
save_panel(fig, "trajectory_f")


# === Legend strips (one per row) ===
def save_legend_strip(handles, stem, figsize=(TEXTWIDTH, 0.28)):
    fig = plt.figure(figsize=figsize)
    fig.legend(handles=handles, loc="center",
               ncol=len(handles), columnspacing=1.8,
               handletextpad=0.4, frameon=False)
    out = FIG_DIR / f"{stem}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


row1_handles = [
    mlines.Line2D([], [], color=row1_colors[Z], marker="o", markersize=3,
                  label=fr"$Z = 2^{{{int(np.log2(Z))}}}$")
    for Z in ROW1_Zs
]
row2_handles = [
    mlines.Line2D([], [], color=row2_colors[Z], marker="o", markersize=3,
                  label=fr"$Z = {Z}$")
    for Z in ROW2_Zs
]
save_legend_strip(row1_handles, "trajectory_legend_row1")
save_legend_strip(row2_handles, "trajectory_legend_row2")
