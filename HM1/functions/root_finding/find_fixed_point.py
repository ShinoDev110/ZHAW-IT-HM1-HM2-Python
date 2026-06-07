# ============================================================
# TOPIC: Root-finding methods — fixed-point iteration x_{k+1} = F(x_k)
# DESCRIPTION:
# General fixed-point iteration of a symbolically given function F(x).
# Choice between stopping after tolerance |x_{k+1} - x_k| < tol or a fixed
# number of iterations.
# USE WHEN:
# When a fixed point x̄ of an iteration rule F(x) is sought
# and F is contracting in the search region (|F'(x)| < 1).
# EXAMPLE:
# F(x) = (230x^4 + 18x^3 + 9x^2 - 9) / 221, initial value x0 = 0.
# ============================================================

import numpy as np
from sympy import sympify

# ============================================================
# PART 1 — Inputs
# ============================================================
function    = "(230 * x**4 + 18 * x**3 + 9 * x**2 - 9) / 221"  # F(x), already solved for x
x_0         = {"x": 0}    # initial value x0
tolerance   = 1e-6        # tolerance for stopping criterion
iterations  = 10          # max. iterations (for method "iters")

# ============================================================
# PART 2 — Method selection
# ============================================================
# method:
#   "tol"   -> stop when |x_{k+1} - x_k| < tolerance
#   "iters" -> always execute iterations steps
method = "tol"

# ============================================================
# PART 3 — Implementation
# ============================================================
def _fixed_point_step(fx, value):
    return fx.subs(value).evalf()

def find_fixed_point(method, function, x_0, tolerance, iterations):
    fx = sympify(function)
    symbols = list(fx.free_symbols)
    if len(symbols) == 0:
        raise ValueError("No unknown found in function.")
    s = symbols[0]
    previous_value = x_0[str(s)]

    if method == "tol":
        next_x_value = -100
        count = 0
        while np.abs(next_x_value - previous_value) > tolerance:
            if count > 0:
                previous_value = next_x_value
            next_x_value = _fixed_point_step(fx, {s: previous_value})
            count += 1
            print(f"{s}_{count} = {next_x_value}")
        return next_x_value
    elif method == "iters":
        next_x_value = -100
        for i in range(iterations):
            if i > 0:
                previous_value = next_x_value
            next_x_value = _fixed_point_step(fx, {s: previous_value})
            print(f"{s}_{i + 1} = {next_x_value}")
        return next_x_value
    else:
        raise ValueError(f"Unknown method: {method!r}")

# ============================================================
# PART 4 — Call
# ============================================================
find_fixed_point(method, function, x_0, tolerance, iterations)
