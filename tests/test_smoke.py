"""Smoke tests for the self-scaling solver.

These run in under a minute on modest hardware and verify:

1. The solver converges on a small random instance.
2. τ* is recovered from τ₀ = 1 on a UEG instance with Z = 10.
3. `naive_dual_newton` overflows (as expected) on a large-Z UEG instance.
"""

import numpy as np
import pytest

from selfscaling import (
    generate_ueg_problem,
    make_random_problem,
    naive_dual_newton,
    selfscaling_solve,
)


def test_solver_converges_on_random_problem():
    """Solver converges on a small random instance."""
    A, b, c, mu = make_random_problem(m=20, n=50, seed=0)
    result = selfscaling_solve(A, b, c, mu, lam=1e-3, tau0=1.0,
                               tol=1e-10, max_iters=100, verbose=False)
    assert result["rho"] < 1e-8, f"rho={result['rho']:.2e}"
    assert np.isfinite(result["tau"]) and result["tau"] > 0
    # Primal objective should exceed dual (weak duality); gap should be tiny at convergence
    gap = result["primal_obj"] - result["dual_obj_phi"]
    assert gap >= -1e-6, f"primal-dual gap {gap:.3e} went negative"


def test_tau_recovery_on_ueg_Z10():
    """On a UEG problem with Z=10, τ* should be recovered within 1%."""
    prob = generate_ueg_problem(Z=10.0, noise_level=1e-4, n_samples=1000, seed=52)
    A, b, mu = prob["A"], prob["b"], prob["mu"]
    mu = np.maximum(mu, 1e-16)
    mu = mu / mu.sum()
    c = np.zeros(A.shape[1])

    result = selfscaling_solve(A, b, c, mu, lam=1e-5, tau0=1.0,
                               tol=1e-10, max_iters=300, verbose=False)
    assert result["rho"] < 1e-9
    assert abs(result["tau"] - 10.0) / 10.0 < 1e-2, \
        f"tau = {result['tau']:.6f}, expected ≈ 10"


def test_naive_dual_overflows_at_large_Z():
    """Classical dual Newton should overflow on Z=2^10 UEG problem.

    This is the structural phenomenon demonstrated in Figure 7.2:
    the classical dual ψ_d(y) = ⟨b,y⟩ − (λ/2)‖y‖² − Σμᵢexp(Aᵢᵀy − 1 − cᵢ)
    cannot be evaluated at y = 0 when Σμᵢ is small (it can) but the
    unsafeguarded Newton step produces Aᵀ(y+d) values far exceeding
    log(FLT_MAX) ≈ 709 when Z is large. Our baseline's Armijo backtracking
    includes overflow detection, and we expect `status != "converged"`.
    """
    Z = 2**10  # 1024
    prob = generate_ueg_problem(Z=Z, noise_level=1e-4, n_samples=1000, seed=42 + Z)
    A, b, mu = prob["A"], prob["b"], prob["mu"]
    mu = np.maximum(mu, 1e-16)
    mu = mu / mu.sum()
    c = np.zeros(A.shape[1])

    res = naive_dual_newton(A, b, c, mu, lam=1e-5, max_iters=50, tol=1e-8)
    assert res["status"] != "converged", \
        "classical dual Newton was expected to fail at Z=1024 (overflow/armijo)"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
