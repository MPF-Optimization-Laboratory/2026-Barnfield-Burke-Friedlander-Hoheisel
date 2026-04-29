"""A scale-shape dual Newton method for entropic least squares.

Companion code for Section 7 of:

  N. Barnfield, J. Burke, M. P. Friedlander, T. Hoheisel,
  "A scale-shape dual Newton method for entropic least squares" (2026).

Public API
----------
- Model, scaleshape_solve, softmax_mu, g_mu   (core algorithm)
- naive_dual_newton, lambert_w_bounds          (comparisons and bounds)
- generate_ueg_problem                          (UEG analytic-continuation problem)
"""

from .solver import Model, scaleshape_solve, softmax_mu, g_mu, compute_direction
from .utils import naive_dual_newton, lambert_w_bounds, make_random_problem
from .ueg import generate_ueg_problem

__version__ = "1.0.0"
__all__ = [
    "Model",
    "scaleshape_solve",
    "softmax_mu",
    "g_mu",
    "compute_direction",
    "naive_dual_newton",
    "lambert_w_bounds",
    "make_random_problem",
    "generate_ueg_problem",
]
