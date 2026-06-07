# ============================================================
# TOPIC: Numerical Integration — step size h / number n for error tolerance
# DESCRIPTION:
# Determines the maximum h (and hence minimum n) so that the error of the
# summed rectangle, trapezoidal, or Simpson rule falls below a given
# tolerance. Uses the error estimates from Theorem 7.1 with numerically
# determined max|f''| and max|f^(4)| on [a, b].
# USE WHEN:
# It is to be decided in advance how many subintervals are needed to
# guarantee a certain accuracy.
# EXAMPLE:
# How many subintervals are needed for int_1^2 ln(x^2) dx with
# error < 1e-5 for the rectangle, trapezoidal, and Simpson rules?
# ============================================================

import numpy as np
import sympy as sp

# ============================================================
# PART 1 — Inputs
# ============================================================
x_sym = sp.Symbol('x')
f_sym = sp.ln(x_sym**2)        # function to integrate (symbolic)

a   = 1.0
b   = 2.0
tol = 1e-5

# ============================================================
# PART 2 — Method selection
# ============================================================
# Options:
#   "rectangle"   -> |error| <= h^2 / 24 * (b-a) * max|f''|
#   "trapezoidal" -> |error| <= h^2 / 12 * (b-a) * max|f''|
#   "simpson"     -> |error| <= h^4 / 2880 * (b-a) * max|f^(4)|
#   "all"         -> all three
method = "all"

# ============================================================
# PART 3 — Implementation
# ============================================================
def estimate_required_step_size(f_sym, x_sym, a, b, tol, method):
    f2_lam = sp.lambdify(x_sym, sp.diff(f_sym, x_sym, 2), "numpy")
    f4_lam = sp.lambdify(x_sym, sp.diff(f_sym, x_sym, 4), "numpy")

    xs = np.linspace(a, b, 10000)
    max_f2 = np.max(np.abs(f2_lam(xs)))
    max_f4 = np.max(np.abs(f4_lam(xs)))
    print(f"max|f''(x)|   on [{a},{b}] = {max_f2}")
    print(f"max|f^(4)(x)| on [{a},{b}] = {max_f4}\n")

    def report(name, h, n):
        print(f"{name:<20}  h <= {h:.6e}   n >= {int(np.ceil(n))}")

    if method in ("rectangle", "all"):
        h = np.sqrt(24 * tol / ((b - a) * max_f2))
        report("Rectangle rule", h, (b - a) / h)
    if method in ("trapezoidal", "all"):
        h = np.sqrt(12 * tol / ((b - a) * max_f2))
        report("Trapezoidal rule", h, (b - a) / h)
    if method in ("simpson", "all"):
        h = (2880 * tol / ((b - a) * max_f4))**(1/4)
        report("Simpson rule", h, (b - a) / h)

# ============================================================
# PART 4 — Call
# ============================================================
estimate_required_step_size(f_sym, x_sym, a, b, tol, method)
