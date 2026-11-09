"""Shared utilities for numerical experiments.

Provides:
- `naive_dual_newton`: Newton's method on the classical (scale-free) dual ψ_d(y),
  used as the Section 7.1 overflow-comparison baseline.
- `lambert_w_bounds`: Lambert-W bounds on τ from Lemma 3.1.
- `make_random_problem`: small synthetic problem for smoke tests.
"""

import numpy as np
from numpy.linalg import norm
from scipy.special import lambertw, logsumexp

from .solver import Model, scaleshape_solve  # re-exported for convenience

__all__ = [
    "naive_dual_newton",
    "lambert_w_bounds",
    "make_random_problem",
    "scaleshape_solve",
    "Model",
]


def naive_dual_newton(A, b, c, mu, lam, y0=None, max_iters=100, tol=1e-10,
                      c_armijo=1e-4, verbose=False):
    """Newton's method on the classical dual ψ_d(y).

    Returns dict with keys:
        y, converged, overflow_iter, history,
        status ∈ {"converged", "overflow_fatal", "armijo_fatal", "max_iters"}.
    """
    m, n = A.shape
    y = np.zeros(m) if y0 is None else np.array(y0, dtype=float)
    history = {"iter": [], "y": [], "grad_norm": [], "objective": [],
               "alpha": [], "full_step_max_exp_arg": [],
               "first_overflow_iter": None}

    def _result(status):
        return {"y": y, "converged": status == "converged",
                "overflow_iter": history.get("_fatal_iter"),
                "status": status, "history": history}

    for k in range(max_iters):
        z = A.T @ y - np.ones(n) - c
        exp_z = mu * np.exp(z)

        if not np.all(np.isfinite(exp_z)):
            history["_fatal_iter"] = k
            if verbose:
                print(f"  Overflow at iteration {k}")
            return _result("overflow_fatal")

        psi = float(b @ y - 0.5 * lam * norm(y)**2 - np.sum(exp_z))
        grad = b - lam * y - A @ exp_z
        grad_norm = float(norm(grad))
        history["iter"].append(k)
        history["y"].append(y.copy())
        history["grad_norm"].append(grad_norm)
        history["objective"].append(psi)

        if grad_norm < tol:
            history["alpha"].append(0.0)
            history["full_step_max_exp_arg"].append(float(z.max()))
            return _result("converged")

        H = -lam * np.eye(m) - (A * exp_z) @ A.T
        try:
            dy = np.linalg.solve(H, -grad)
        except np.linalg.LinAlgError:
            history["_fatal_iter"] = k
            history["alpha"].append(0.0)
            history["full_step_max_exp_arg"].append(float("nan"))
            return _result("overflow_fatal")

        full_step_exp_arg = float((A.T @ (y + dy) - np.ones(n) - c).max())
        history["full_step_max_exp_arg"].append(full_step_exp_arg)

        slope = float(grad @ dy)
        alpha = 1.0
        overflow_fails = 0
        armijo_fails = 0
        for _ in range(30):
            y_trial = y + alpha * dy
            z_trial = A.T @ y_trial - np.ones(n) - c
            exp_trial = mu * np.exp(z_trial)
            if not np.all(np.isfinite(exp_trial)):
                if history["first_overflow_iter"] is None:
                    history["first_overflow_iter"] = k
                overflow_fails += 1
                alpha *= 0.5
                continue
            psi_trial = float(b @ y_trial - 0.5 * lam * norm(y_trial)**2 - np.sum(exp_trial))
            if psi_trial >= psi + c_armijo * alpha * slope:
                break
            armijo_fails += 1
            alpha *= 0.5
        else:
            status = "armijo_fatal" if armijo_fails > 0 else "overflow_fatal"
            history["_fatal_iter"] = k
            history["alpha"].append(0.0)
            return _result(status)

        history["alpha"].append(float(alpha))
        y = y_trial

    return _result("max_iters")


def lambert_w_bounds(A, b, c, mu, lam, beta):
    """Lambert-W bounds on τ from Lemma 3.1 of the paper.

    Args:
        A, b, c, mu, lam: problem data
        beta: level-set radius (must satisfy β > ρ(z⁰))

    Returns:
        (tau_min, tau_max)
    """
    b_norm = float(norm(b))
    q_sum = float(np.sum(mu))
    c_min = float(np.min(c))
    c_max = float(np.max(c))
    # A_max := max_i ||A_i|| over columns A_i of A (paper eq (3.2)).
    A_max = float(np.max(norm(A, axis=0)))

    # Upper bound: τ_max ≤ B / W(B·exp(θ))
    B = (beta + b_norm) ** 2 / (4 * lam)
    theta = 1 - beta - np.log(q_sum) + c_min
    arg_upper = B * np.exp(theta)
    w_upper = float(np.real(lambertw(arg_upper)))
    tau_max = float(B / w_upper) if w_upper > 0 else np.inf

    # Lower bound (Lemma 3.1(ii)): τ_min ≥ C̃ · exp(−W(ζ̃)), with
    #   Σ₀ = ⟨q, exp(Aᵀb/λ − c)⟩,  g₀ = A p(b/λ),  κ₀ = max_i ⟨g₀, A_i⟩,
    #   C̃ = Σ₀ exp(−1 − (β/λ)·sqrt(λ² + ‖g₀‖²)),  ζ̃ = κ₀ C̃ / λ.
    # Σ₀ overflows for small λ, so everything is evaluated in the log domain.
    log_tau_min = _log_tau_min_bound(A, b, c, mu, lam, beta)
    tau_min = float(np.exp(log_tau_min))  # may underflow to 0.0

    return tau_min, tau_max


def _log_lambertw_of_log(s):
    """log W(x) given s = log x ≥ −∞, robust for any real s."""
    if s == -np.inf:
        return -np.inf
    if s < -30:
        return s  # W(x) = x − x² + …, relative error < e^{−30}
    if s < 20:
        return float(np.log(np.real(lambertw(np.exp(s)))))
    w = s - np.log(s)  # W(x) solves w + log w = s; Newton from the asymptote
    for _ in range(50):
        step = (w + np.log(w) - s) / (1 + 1 / w)
        w -= step
        if abs(step) <= 1e-15 * w:
            break
    return float(np.log(w))


def _log_tau_min_bound(A, b, c, mu, lam, beta):
    """log of the Lemma 3.1(ii) lower bound C̃ e^{−W(ζ̃)} on τ_{β,min}."""
    A = np.asarray(A, dtype=float)
    mu = np.asarray(mu, dtype=float)
    u = A.T @ (np.asarray(b, dtype=float) / lam) - np.asarray(c, dtype=float)
    log_Sigma0 = float(logsumexp(u, b=mu))
    with np.errstate(divide="ignore"):
        p0 = np.exp(u + np.log(mu) - log_Sigma0)
    g0 = A @ p0
    kappa0 = float(np.max(A.T @ g0))
    log_C = log_Sigma0 - 1 - (beta / lam) * np.sqrt(lam**2 + float(g0 @ g0))
    if kappa0 <= 0:
        return log_C  # W(0) = 0
    log_k_over_lam = np.log(kappa0) - np.log(lam)
    # C̃ e^{−W(ζ̃)} = (λ/κ₀) W(ζ̃),  log ζ̃ = log κ₀ + log C̃ − log λ
    return float(_log_lambertw_of_log(log_k_over_lam + log_C) - log_k_over_lam)


def make_random_problem(m, n, seed=42):
    """Generate a small random problem instance (for smoke tests)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    c = np.zeros(n)
    mu = np.ones(n)
    return A, b, c, mu
