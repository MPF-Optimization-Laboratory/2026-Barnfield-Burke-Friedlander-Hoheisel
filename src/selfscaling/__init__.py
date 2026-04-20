"""Self-scaling inexact Newton method for entropy-regularized least squares.

Companion code for Section 7 of:

  N. Barnfield, J. Burke, M. P. Friedlander, T. Hoheisel,
  "Self-scaling inexact Newton methods for entropy-regularized
  least-squares problems" (2026).

Public API
----------
- Model, selfscaling_solve, softmax_mu, g_mu   (core algorithm)
- naive_dual_newton, lambert_w_bounds          (comparisons and bounds)
- generate_ueg_problem                          (UEG analytic-continuation problem)
"""

from .solver import Model, selfscaling_solve, softmax_mu, g_mu, compute_direction
from .utils import naive_dual_newton, lambert_w_bounds, make_random_problem
from .ueg import generate_ueg_problem

__version__ = "1.0.0"
__all__ = [
    "Model",
    "selfscaling_solve",
    "softmax_mu",
    "g_mu",
    "compute_direction",
    "naive_dual_newton",
    "lambert_w_bounds",
    "make_random_problem",
    "generate_ueg_problem",
]
