# ============================================================
# TOPIC: Root-finding methods — bisection (interval halving)
# DESCRIPTION:
# Classic interval halving for f(x) = 0. Prerequisite: f continuous
# on [a, b] and f(a)·f(b) < 0 (sign change). Returns the midpoint
# of the last interval as approximation and the iteration count.
# USE WHEN:
# When a robust (but linearly convergent) root solver without
# derivative information is needed.
# EXAMPLE:
# f(x) = e^(x^2) + x^-3 - 10 on [0.5, 2.0], tol = 1e-8.
# ============================================================

import math

# ============================================================
# PART 1 — Inputs
# ============================================================
def f(x):
    return math.exp(x * x) + x ** -3 - 10.0

interval_left  = 0.5
interval_right = 2.0
tolerance      = 1e-8
max_iterations = 100

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here.

# ============================================================
# PART 3 — Implementation
# ============================================================
def find_root_with_bisection(f, a, b, tol=1e-8, max_iter=100):
    a, b = float(a), float(b)
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("Bisection: no sign change in the starting interval.")

    iterations = 0
    while (b - a) / 2.0 > tol and iterations < max_iter:
        midpoint = 0.5 * (a + b)
        fm = f(midpoint)
        if fm == 0.0:
            print(f"Exact hit at x = {midpoint}, iterations = {iterations + 1}")
            return midpoint, iterations + 1
        if fa * fm < 0:
            b, fb = midpoint, fm
        else:
            a, fa = midpoint, fm
        iterations += 1

    x_approximation = 0.5 * (a + b)
    print(f"Root approximation: x ~= {x_approximation}")
    print(f"Iterations: {iterations}")
    print(f"f(x) ~= {f(x_approximation)}")
    return x_approximation, iterations

# ============================================================
# PART 4 — Call
# ============================================================
find_root_with_bisection(f, interval_left, interval_right, tolerance, max_iterations)
