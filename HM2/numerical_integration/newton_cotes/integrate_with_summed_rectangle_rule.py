# ============================================================
# TOPIC: Numerical Integration — summed rectangle/midpoint rule
# DESCRIPTION:
# Approximates int_a^b f(x) dx with Rf(h) = h * sum_{i=0}^{n-1} f(x_i + h/2),
# i.e. n equidistant subintervals with evaluation at the midpoint.
# USE WHEN:
# A definite integral is to be approximated with the summed midpoint rule
# (Newton-Cotes order 0).
# EXAMPLE:
# Compute int_2^4 1/x dx approximately with n = 4.
# ============================================================

import numpy as np

# ============================================================
# PART 1 — Inputs
# ============================================================
def f(x):
    return 1 / x

a = 2.0          # lower bound
b = 4.0          # upper bound
n = 4            # number of subintervals

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: summed midpoint rule.

# ============================================================
# PART 3 — Implementation
# ============================================================
def integrate_summed_rectangle(f, a, b, n):
    h = (b - a) / n
    midpoints = np.array([a + (i + 0.5) * h for i in range(n)])
    Rf = h * np.sum(f(midpoints))
    print(f"a = {a}, b = {b}, n = {n}, h = {h}")
    print(f"R(f) = {Rf}")
    return Rf

# ============================================================
# PART 4 — Call
# ============================================================
integrate_summed_rectangle(f, a, b, n)
