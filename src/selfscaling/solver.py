# TODO: support zero entries in mu (currently rejected by softmax_mu / g_mu).

"""

########## To run the solver, call `selfscaling_solve(A, b, c, mu, lam, ...)`. ########

Details of the Self-Scaling algorithm for Entropy-regularized least squares

Problem setup:
  Primal objective:
      ψ_p(x) = (1/(2λ)) ||A x - b||^2 + <c, x> + g_μ(x),
      where g_μ(x) = sum_i x_i * log(x_i / μ_i) (with the usual 0·log 0 := 0 convention).

  Dual objective:
      ψ_d(y) = <b, y> - (λ/2) ||y||^2 - Σ_i μ_i * exp(A_i^T y - 1 - c_i).

  Tau-parametrized dual objective:
      ϕ_d(y, τ) = <b, y> - (λ/2) ||y||^2 - τ log( Σ_i μ_i * exp(A_i^T y - c_i) ) + τ log τ.

  Optimality system:
      F(y, τ) = [ F1(y, τ); F2(y, τ) ] with
        F1(y, τ) = -λ y + b - τ A · softmax_μ(A^T y - c),
        F2(y, τ) = -log(Σ_i μ_i exp(A_i^T y - c_i)) + log τ + 1.

  Jacobian of F:
      J(y, τ) = [ H   g ] with
                [ g^T 1/τ]
      where  H = -λ I - τ A S(y) A^T,
             g = -A softmax_μ(A^T y - c),
             S(y) = diag(x) - x x^T,
             x = softmax_μ(A^T y - c) ∈ Δ_n.

 Self-Scaling algorithm:
  - Find direction d_k = [Δy_k; Δτ_k] such that ||F(z_k) + J(z_k) d_k|| ≤ η_k ||F(z_k)||,
    with η_k ∈ [0, η̄).
  - Safeguard τ positivity & level set constraint via ᾱ backtracking with factor γ1.
  - Merit-function backtracking for α with factor γ2 using
        ||F(z_k + α d_k)|| ≤ ||F(z_k)|| + c α (η_k − 1) ||F(z_k)||.


"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import math
import numpy as np
from scipy.linalg import solve as dense_solve
from scipy.sparse.linalg import minres


# Floor for the τ safeguard when the Lambert-W bound underflows float64.
_TAU_FLOOR_MIN = 1e-16
# Cap on the inexactness parameter η_k (Algorithm 1 requires η_k < 1).
_ETA_MAX = 1.0 - 1e-6


# ============================
# Utility math & model pieces
# ============================

def softmax_mu(z: np.ndarray, mu: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Stable μ-weighted softmax and log-sum-exp with μ.

    Args:
        z: (n,) array of logits.
        mu: (n,) strictly positive weights.

    Returns:
        x: (n,) vector in simplex, x_j = μ_j e^{z_j} / Σ_i μ_i e^{z_i}.
        sum_w: Σ_i μ_i e^{z_i - max(z)}  (scaled sum used internally)
        log_sum: log Σ_i μ_i e^{z_i}
    """
    z = np.asarray(z, dtype=float)
    mu = np.asarray(mu, dtype=float)
    assert z.ndim == 1 and mu.ndim == 1 and z.shape == mu.shape
    if not np.all(mu > 0):
        raise ValueError("All entries of mu must be strictly positive.")

    m = float(np.max(z))
    ez = np.exp(z - m)
    w = mu * ez
    sum_w = float(np.sum(w))
    if not np.isfinite(sum_w) or sum_w <= 0:
        raise FloatingPointError("softmax_mu encountered non-finite sum; check scaling.")
    x = w / sum_w
    log_sum = math.log(sum_w) + m
    return x, sum_w, log_sum


def g_mu(x: np.ndarray, mu: np.ndarray) -> float:
    """Entropy regularizer g_μ(x) = Σ x_i log(x_i / μ_i) with standard conventions.

    Assumes μ_i > 0. Terms with x_i = 0 contribute 0.
    """
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if not np.all(mu > 0):
        raise ValueError("All entries of mu must be strictly positive.")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(x > 0, x / mu, 1.0)  # safe: value won't be used when x==0
        vals = np.where(x > 0, x * np.log(ratio), 0.0)
    s = float(np.sum(vals))
    if not np.isfinite(s):
        raise FloatingPointError("g_mu produced a non-finite value.")
    return s


@dataclass
class Model:
    """Container for problem data.

    Attributes:
        A: (m, n) matrix.
        b: (m,) vector.
        c: (n,) vector.
        mu: (n,) vector with strictly positive entries.
        lam: positive scalar λ.
    """
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    mu: np.ndarray
    lam: float

    def __post_init__(self) -> None:
        self.A = np.asarray(self.A, dtype=float)
        self.b = np.asarray(self.b, dtype=float).reshape(-1)
        self.c = np.asarray(self.c, dtype=float).reshape(-1)
        self.mu = np.asarray(self.mu, dtype=float).reshape(-1)
        m, n = self.A.shape
        if self.b.shape != (m,) or self.c.shape != (n,) or self.mu.shape != (n,):
            raise ValueError("Dimension mismatch among A, b, c, mu.")
        if not (self.lam > 0):
            raise ValueError("λ must be positive.")
        if not np.all(self.mu > 0):
            raise ValueError("All entries of μ must be strictly positive.")

    # ---- primal/dual objectives ----
    def primal_obj(self, x: np.ndarray) -> float:
        Ax_minus_b = self.A @ x - self.b
        quad = 0.5 / self.lam * float(np.dot(Ax_minus_b, Ax_minus_b))
        lin = float(np.dot(self.c, x))
        ent = g_mu(x, self.mu)
        return quad + lin + ent

    def dual_obj_psi(self, y: np.ndarray) -> float:
        # ψ_d(y) = <b,y> - (λ/2)||y||^2 - Σ μ_i exp(A_i^T y - 1 - c_i)
        # The sum term can overflow far from optimum; clamp to -inf in that regime
        # so finalization never crashes on non-converged iterates.
        z = self.A.T @ y - self.c
        _, _, log_sum = softmax_mu(z, self.mu)
        lin = float(np.dot(self.b, y)) - 0.5 * self.lam * float(np.dot(y, y))
        if log_sum - 1.0 > 700.0:  # math.exp would overflow float64
            return float("-inf")
        return lin - math.exp(log_sum - 1.0)

    def dual_obj_phi(self, y: np.ndarray, tau: float) -> float:
        # ϕ_d(y, τ) = <b,y> - (λ/2)||y||^2 - τ log Σ μ_i exp(A_i^T y - c_i) + τ log τ
        z = self.A.T @ y - self.c
        _, _, log_sum = softmax_mu(z, self.mu)
        return float(np.dot(self.b, y)) - 0.5 * self.lam * float(np.dot(y, y)) - tau * log_sum + tau * math.log(tau)

    # ---- F, J, and related mappings ----
    def primal_dual_map(self, y: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """x(y) = softmax_μ(A^T y - c). Returns (x(y), sum_w, log_sum)."""
        z = self.A.T @ y - self.c
        return softmax_mu(z, self.mu)

    def F(self, y: np.ndarray, tau: float) -> np.ndarray:
        if not (tau > 0):
            raise ValueError("τ must be positive.")
        x, _, log_sum = self.primal_dual_map(y)
        F1 = -self.lam * y + self.b - tau * (self.A @ x)
        F2 = -log_sum + math.log(tau) + 1.0
        return np.concatenate([F1, np.array([F2])])

    def J(self, y: np.ndarray, tau: float) -> np.ndarray:
        if not (tau > 0):
            raise ValueError("τ must be positive.")
        x, _, _ = self.primal_dual_map(y)
        # S = diag(x) - x x^T  ⇒  A S A^T = A diag(x) A^T - (A x)(A x)^T
        Ax = self.A @ x  # (m,)
        A_diagx_AT = (self.A * x) @ self.A.T  # multiply each column j of A by x_j
        ASAT = A_diagx_AT - np.outer(Ax, Ax)
        H = -self.lam * np.eye(self.A.shape[0]) - tau * ASAT
        g = -(Ax)
        # Assemble J
        m = self.A.shape[0]
        J = np.empty((m + 1, m + 1), dtype=float)
        J[:m, :m] = H
        J[:m, m] = g
        J[m, :m] = g
        J[m, m] = 1.0 / tau
        return J

    def x_from(self, y: np.ndarray, tau: float) -> np.ndarray:
        x, _, _ = self.primal_dual_map(y)
        return tau * x


# =====================================
# Linear solver for the inexact step
# =====================================


def compute_direction(
    model,
    y: np.ndarray,
    tau: float,
    eta_k: float,
    max_lin_iters: int = 200,
    solver: str = "auto",
    rtol: float = 1e-16,
) -> Tuple[np.ndarray, float]:
    """Compute an inexact Newton step d solving ||F + J d||_2 ≤ η_k ||F||_2.

    Methods:
      - 'exact'  : symmetric–indefinite direct solve (dense LAPACK), residual ~ 0.
      - 'minres' : MINRES on J d = -F (symmetric, possibly indefinite J), stops at max(η_k, rtol).
      - 'auto'   : 'exact' if dim F ≤ 256, else 'minres'.

    Returns:
        d: step vector of shape (m+1,)
        res_norm: achieved residual ||F + J d||_2
    """
    # Pull model quantities
    A, lam = model.A, model.lam
    m = A.shape[0]

    # Ensure 1-D vectors (NumPy linalg & MINRES expect shape (M,), not (M,1))
    Fk = model.F(y, tau)
    Jk = model.J(y, tau) # dense symmetric matrix

    M = Fk.size
    F_norm = float(np.linalg.norm(Fk))

    if solver == "auto":
        solver = "exact" if M <= 500 else "minres"

    if solver == "exact":
        # Dense symmetric–indefinite direct solve (Bunch–Kaufman under the hood)
        d = dense_solve(Jk, -Fk, assume_a='sym', check_finite=False)
        res = Fk + Jk @ d
        return d, float(np.linalg.norm(res))

    if solver == "minres":
        # MINRES minimizes ||F + J d||_2 for symmetric (possibly indefinite) J.
        # Scipy's built-in stopping test is ||r||/(||A||·||x||) ≤ rtol, which
        # does NOT enforce the inexact Newton condition ||F + J d|| ≤ η_k ||F||.
        # We set rtol to near machine epsilon and use a callback to enforce (3.14).
        target = float(max(eta_k, rtol)) * F_norm

        class _InexactNewtonConverged(Exception):
            def __init__(self, d): self.d = d

        def _monitor(d_current):
            res_norm = float(np.linalg.norm(Fk + Jk @ d_current))
            if res_norm <= target:
                raise _InexactNewtonConverged(d_current.copy())

        try:
            d, _info = minres(Jk, -Fk, rtol=1e-15, maxiter=max_lin_iters,
                              callback=_monitor)
        except _InexactNewtonConverged as e:
            d = e.d

        d = np.asarray(d)
        res_norm = float(np.linalg.norm(Fk + Jk @ d))
        # Fallback: if MINRES did not meet the inexact Newton condition
        # ||F + J d|| ≤ η_k ||F|| within max_lin_iters, drop to a direct
        # symmetric-indefinite solve. Mirrors Algorithm 1's requirement that
        # the computed direction satisfy the inexactness tolerance.
        if eta_k > 0 and res_norm > eta_k * F_norm:
            d = dense_solve(Jk, -Fk, assume_a='sym', check_finite=False)
            res_norm = float(np.linalg.norm(Fk + Jk @ d))
        return d, res_norm

    raise ValueError(f"Unknown solver option: {solver}")



# =====================================
# Main solver (Algorithm 1)
# =====================================

def selfscaling_solve(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    mu: np.ndarray,
    lam: float = 1e-4,
    y0: Optional[np.ndarray] = None,
    tau0: float = 1.0,
    eta: float = 0.0,
    eta_scale: float = 1.0,
    eta_fn: Optional[Callable[[int, float], float]] = None,
    c_backtrack: float = 0.49,
    gamma: float = 0.5,
    tol: float = 1e-8,
    max_iters: int = 200,
    max_lin_iters: int = 200,
    lin_solver: str = "auto",
    verbose: bool = True,
) -> Dict[str, object]:
    """Run Self-Scaling Algorithm.

    Args:
        A, b, c, mu, lam: problem data; see class `Model`. Default λ=1e-4.
        y0: initial dual iterate (default zeros).
        tau0: initial τ > 0. Default 1.0.
        eta: inexactness parameter in [0, 1). Default 0 (exact Newton step).
        eta_scale: if in (0,1), uses η_k = η * (eta_scale ** k). Default 1 (no decay).
        c_backtrack: Armijo-like parameter μ ∈ (0, 1/2) for the merit rule (Remark 3.3).
                     Iter count at large Z / far-from-optimal τ₀ is a strong function of μ;
                     default chosen near the upper end of the admissible interval. Default 0.49.
        gamma: Armijo backtracking factor γ ∈ (0, 1). Default 0.5.
        tol: stopping tolerance on ρ(z) = ||F(z)||. Default 1e-8.
        max_iters: max outer iterations. Default 200.
        max_lin_iters: max inner linear iterations. Default 200.
        lin_solver: 'auto' | 'exact' | 'minres'
        verbose: if True, prints iteration diagnostics.

    Returns:
        {
          "x": x, "y": y, "tau": tau,
          "primal_obj": pobj,
          "dual_obj_phi": dphi,
          "dual_obj_psi": dpsi,
          "rho": rho,
          "history": history
        }
        where `history` is a dict with arrays/lists for keys:
          iter, tau, alpha, x, y, rho
    """
    if not (0 <= eta < 1):
        raise ValueError("eta must satisfy 0 ≤ eta < 1")
    if not (eta_scale > 0):
        raise ValueError("eta_scale must be positive.")
    if not (tau0 > 0):
        raise ValueError("Initial tau0 must be positive.")
    if not (0 < c_backtrack < 0.5):
        raise ValueError("c_backtrack (Armijo μ) must lie in (0, 1/2); see Remark 3.3.")
    if not (0 < gamma < 1):
        raise ValueError("gamma must lie in (0, 1).")

    model = Model(A=np.asarray(A), b=np.asarray(b), c=np.asarray(c), mu=np.asarray(mu), lam=float(lam))
    m = model.A.shape[0]

    ##################### Step 1: initialization #####################
    y = np.zeros(m) if y0 is None else np.asarray(y0, dtype=float).reshape(m)
    tau = float(tau0)

    # Initial merit value & level-set radius β
    F0 = model.F(y, tau)
    rho = float(np.linalg.norm(F0))
    beta = 1.5 * rho  # any β > ρ(z⁰) works; mild cushion

    # Algorithm 1, Step 1: precompute tau_min_hat via the Lambert-W bound of
    # Lemma 3.1(ii). tau_floor = 0.5 * tau_min_hat is the safeguard Step 3 uses
    # to keep tau_k + alpha_bar * dtau_k uniformly bounded away from 0.
    # When A_max/lam is large, the exponential in zeta can underflow to 0 in
    # float64, yielding a computed tau_min_hat of 0. In that regime any small
    # positive value is a valid lower estimate; we fall back to _TAU_FLOOR_MIN.
    # Local import: utils.py imports from solver.py, so a module-top import
    # would create a circular dependency.
    from .utils import lambert_w_bounds
    tau_min_hat, _ = lambert_w_bounds(model.A, model.b, model.c, model.mu,
                                      model.lam, beta)
    tau_floor = max(0.5 * tau_min_hat, _TAU_FLOOR_MIN)
    if tau < tau_floor:
        raise ValueError(
            f"tau0={tau:.3e} < tau_floor={tau_floor:.3e}; Algorithm 1 "
            "requires z0 in lev_rho(beta).")

    # History container
    history: Dict[str, List] = {"iter": [], "tau": [], "alpha": [], "x": [], "y": [], "rho": [],
                                    "eta_k": [], "res_ratio": []}

    if verbose:
        print("\n=== Self-Scaling Algorithm on F(y, τ) = 0 ===")
        print(f"eta={eta} (scale={eta_scale}), c={c_backtrack}, gamma={gamma}")
        print(f"init: rho={rho:.6e}, tau={tau0:.6e}")
        print("Iter |   rho(F)        |    ϕ_d(y,τ)       ψ_d(y)         ψ_p(x)        |  α (step)   |  τ ")
        print("-----+-----------------+------------------------------------------------+-------------+----")

    for k in range(max_iters):
        # Save history. eta_k / res_ratio are NaN placeholders here and get
        # patched below once the Newton step runs; this keeps every history
        # array the same length even when the loop exits at convergence.
        x_curr = model.x_from(y, tau)
        history["iter"].append(k)
        history["tau"].append(float(tau))
        history["x"].append(x_curr.copy())
        history["y"].append(y.copy())
        history["alpha"].append(float(0.0)) # no step taken
        history["rho"].append(float(rho))
        history["eta_k"].append(float("nan"))
        history["res_ratio"].append(float("nan"))

        ############### Step 2: Convergence check and Newton step ################

        if rho <= tol: # converged
            if verbose:
                print(f"Terminated: rho={rho:.3e} ≤ tol={tol:.1e}")
            break

        # Inexactness schedule
        if eta_fn is not None:
            eta_k = float(min(eta_fn(k, rho), _ETA_MAX))
        elif eta > 0:
            eta_k = min(eta * (eta_scale ** k), _ETA_MAX)
        else:
            eta_k = 0.0

        # Compute direction
        d, res_norm = compute_direction(model, y, tau, eta_k, max_lin_iters=max_lin_iters, solver=lin_solver)
        history["eta_k"][-1] = float(eta_k)
        history["res_ratio"][-1] = float(res_norm / rho) if rho > 0 else 0.0
        dy = d[:m]
        dtau = float(d[m])

        ############### Step 3: safeguard τ via ᾱ backtracking if dtau < 0 ###############

        # Keep tau_k + alpha_bar * dtau_k >= 0.5 * tau_min_hat > 0 (Algorithm 1).
        if tau + dtau >= tau_floor:
            alpha_bar = 1.0
        else:
            # dtau < 0 here; else tau + dtau >= tau >= tau_floor would hold.
            alpha_bar = float(np.clip((tau_floor - tau) / dtau, 0.0, 1.0))

        ############### Step 4: Armijo-like backtracking on merit function rho ###############
        # Step 3's closed-form alpha_bar guarantees tau + a * dtau >= tau_floor > 0
        # for all a in [0, alpha_bar], so no positivity guard is needed here.
        alpha = alpha_bar
        rhs_coeff = c_backtrack * (eta_k - 1.0) * rho  # <= 0
        for _ in range(60):
            y_trial = y + alpha * dy
            tau_trial = tau + alpha * dtau
            F_trial = model.F(y_trial, tau_trial)
            lhs = float(np.linalg.norm(F_trial))
            rhs = rho + alpha * rhs_coeff
            if lhs <= rhs:
                break
            alpha *= gamma
        else:
            raise RuntimeError("Armijo backtracking failed to find acceptable step.")

        # Accept step; reuse the last trial evaluation to avoid recomputing F.
        y = y_trial
        tau = tau_trial
        rho = lhs
        history["alpha"][-1] = float(alpha)

        if verbose:
            dphi = model.dual_obj_phi(y, tau)
            dpsi = model.dual_obj_psi(y)
            xk = model.x_from(y, tau)
            pobj = model.primal_obj(xk)
            print(f"{k:4d} | {rho: .6e} | {dphi: .6e}  {dpsi: .6e}  {pobj: .6e} | {alpha: .3e} | {tau: .6e}")

    # Finalization
    x = model.x_from(y, tau)
    pobj = model.primal_obj(x)
    dphi = model.dual_obj_phi(y, tau)
    dpsi = model.dual_obj_psi(y)

    if verbose:
        print("\n--- Termination ---")
        print(f"optimal tau ~ {tau:.12g}")
        print(f"final rho(F) = {rho:.6e}")
        print(f"dual (phi_d) = {dphi:.12g}")
        print(f"dual (psi_d) = {dpsi:.12g}")
        print(f"primal obj   = {pobj:.12g}")

    result = {
        "x": x,
        "y": y,
        "tau": float(tau),
        "primal_obj": float(pobj),
        "dual_obj_phi": float(dphi),
        "dual_obj_psi": float(dpsi),
        "rho": float(rho),
        "history": history,
    }
    return result


# Simple plotting utility

def plot_history(history: Dict[str, List], keys: Iterable[str] = ("rho", "tau")) -> None:
    """Quick plot of selected scalar sequences in `history` vs iteration.

    Usage:
        out = selfscaling_solve(..., verbose=False)
        plot_history(out["history"], keys=("rho", "tau"))

    Args:
        history: dict returned under result["history"] by `selfscaling_solve`.
        keys: which scalar sequences to plot (each must be in `history`).
    """
    import matplotlib.pyplot as plt

    iters = history.get("iter", list(range(len(next(iter(history.values()))))))
    for key in keys:
        if key not in history:
            raise KeyError(f"history does not contain key '{key}'")
        vals = history[key]
        # accept arrays/lists of scalars
        arr = np.asarray(vals, dtype=float)
        plt.figure()
        if key == "alpha":
            plt.yscale("log")
        plt.plot(iters[: len(arr)], arr, marker="o", linewidth=1.5)
        plt.xlabel("Iteration k")
        if key == "alpha":
            plt.ylabel("log(alpha)")
            plt.title("log(alpha) vs iteration")
        else:
            plt.ylabel(key)
            plt.title(f"{key} vs iteration")
        plt.grid(True, linestyle=":", linewidth=0.7)
        plt.tight_layout()
        plt.show()

