# ============================================================
# TOPIC: Functions — condition number of a real function
# DESCRIPTION:
# Computes κ_f(x) = | x · f'(x) / f(x) | using either the analytic
# derivative, a central numeric derivative, or as a sweep over an
# interval (grid) to locate poorly conditioned points.
# USE WHEN:
# When assessing how strongly relative input errors in x
# are amplified into relative output errors in f(x).
# EXAMPLE:
# f(x) = exp(x) - 3x at x0 = 2 with analytic derivative;
# additionally a sweep over [0.1, 5.0].
# ============================================================

import math
import numpy as np

# ============================================================
# PART 1 — Inputs
# ============================================================
x0 = 2.0  # evaluation point for the pointwise condition number
h  = 1e-6 # step size for the numeric derivative (method "numeric" only)

def f(x: float) -> float:
    # example function (adjust as needed)
    return math.exp(x) - 3.0 * x

def df(x: float) -> float:
    # analytic derivative (for method "analytic")
    return math.exp(x) - 3.0

x_grid = np.linspace(0.1, 5.0, 20)  # grid for method "grid"

# ============================================================
# PART 2 — Method selection
# ============================================================
# method:
#   "analytic" -> κ_f(x0) using the given derivative df
#   "numeric"  -> κ_f(x0) using central numeric derivative (no df needed)
#   "grid"     -> evaluate κ_f on x_grid, find critical points
method = "grid"

# ============================================================
# PART 3 — Implementation
# ============================================================
def _kappa_analytic(f, df, x):
    fx = float(f(x))
    if fx == 0.0:
        return math.inf
    return abs(x * float(df(x)) / fx)

def _central_difference(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2.0 * h)

def _kappa_numeric(f, x, h=1e-6):
    fx = float(f(x))
    if fx == 0.0:
        return math.inf
    return abs(x * float(_central_difference(f, x, h)) / fx)

def _print_report(x, fx, dfx, kappa):
    print("============================================================")
    print("Condition number of a real function")
    print("============================================================")
    print(f"x      = {x}")
    print(f"f(x)   = {fx}")
    print(f"f'(x)  = {dfx}")
    print(f"kappa  = |x f'(x) / f(x)| = {kappa}")
    print()

def compute_condition_number_of_function(method, f, df, x0, h, x_grid):
    if method == "analytic":
        fx, dfx = f(x0), df(x0)
        _print_report(x0, fx, dfx, _kappa_analytic(f, df, x0))
    elif method == "numeric":
        fx, dfx = f(x0), _central_difference(f, x0, h)
        _print_report(x0, fx, dfx, _kappa_numeric(f, x0, h))
    elif method == "grid":
        print("x\tkappa")
        print("-" * 30)
        for x in x_grid:
            k = _kappa_analytic(f, df, float(x))
            print(f"{x:.3f}\t{k:.6g}")
    else:
        raise ValueError(f"Unknown method: {method!r}")

# ============================================================
# PART 4 — Call
# ============================================================
compute_condition_number_of_function(method, f, df, x0, h, x_grid)
