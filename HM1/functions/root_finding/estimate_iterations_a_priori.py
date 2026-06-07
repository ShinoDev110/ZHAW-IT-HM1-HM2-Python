# ============================================================
# TOPIC: Fixed-point iteration — a-priori iteration estimate
# DESCRIPTION:
# Estimates per Banach the maximum number of iterations needed so that
# |x_n - x̄| < tolerance, from the Lipschitz constant alpha and the
# first step size |x_1 - x_0|.
# USE WHEN:
# When before starting a fixed-point iteration it should be estimated
# how many steps are needed for a desired accuracy.
# EXAMPLE:
# F(x) = exp(x) - exp(1), interval [-3, -2], initial value x0 = -2.5,
# tolerance 1e-5.
# ============================================================

from math import ceil
import numpy as np
from sympy import diff, log, sympify

# ============================================================
# PART 1 — Inputs
# ============================================================
function  = "exp(x) - exp(1)"  # F(x) of the fixed-point iteration
x_0       = {"x": -2.5}        # initial value x0
interval  = [-3, -2]           # interval over which alpha is estimated
tolerance = 1e-5               # desired tolerance

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here. alpha is always estimated as max|F'(x)| over the interval
# by discrete approximation; n follows from n >= log(tol·(1-alpha)/|x1-x0|) / log(alpha).

# ============================================================
# PART 3 — Implementation
# ============================================================
def _get_alpha(function, interval):
    f = sympify(function)
    symbols = list(f.free_symbols)
    if not symbols:
        raise ValueError("No unknown found in function.")
    if len(interval) != 2:
        raise ValueError("Interval must consist of two values.")
    s = symbols[0]
    values = []
    for t in np.linspace(interval[0], interval[1], 100):
        values.append(diff(f, s).subs(s, t).evalf())
    return np.max(np.abs(values))

def _a_priori(alpha, x_0, x_1, tol):
    return log((tol * (1 - alpha)) / np.abs(x_1 - x_0)) / log(alpha)

def estimate_iterations_a_priori(function, x_0, interval, tolerance):
    alpha = _get_alpha(function, interval)
    print(f"alpha (Lipschitz) ~= {alpha}")

    x0_val = x_0[list(x_0.keys())[0]]
    x1_val = sympify(function).subs(x_0).evalf()

    n_real = _a_priori(alpha, x0_val, x1_val, tolerance)
    print(f"Number of iteration steps: {n_real} i.e. {ceil(n_real)}")
    return n_real, ceil(n_real)

# ============================================================
# PART 4 — Call
# ============================================================
estimate_iterations_a_priori(function, x_0, interval, tolerance)
