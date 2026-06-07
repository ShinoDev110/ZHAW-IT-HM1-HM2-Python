# ============================================================
# TOPIC: Numerical Integration — summed Simpson rule
# DESCRIPTION:
# Approximates int_a^b f(x) dx with Sf(h) = h/3 * (f(a)/2 +
# sum_{i=1}^{n-1} f(x_i) + 2 * sum_{i=1}^{n} f((x_{i-1}+x_i)/2) + f(b)/2)
# per script formula (Newton-Cotes order 2).
# USE WHEN:
# A definite integral is to be approximated with the summed Simpson rule.
# EXAMPLE:
# Compute int_2^4 1/x dx with n = 4.
# ============================================================

import numpy as np

# ============================================================
# PART 1 — Inputs
# ============================================================
def f(x):
    return 1 / x

a = 2.0
b = 4.0
n = 4

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: summed Simpson rule per skript formula.

# ============================================================
# PART 3 — Implementation
# ============================================================
def integrate_summed_simpson(f, a, b, n):
    h = (b - a) / n
    x = np.array([a + i * h for i in range(n + 1)])

    sum_endpoint = 0.5 * (f(x[0]) + f(x[-1]))
    sum_inner    = np.sum(f(x[1:-1]))
    midpoints    = (x[:-1] + x[1:]) / 2.0
    sum_mid      = 2 * np.sum(f(midpoints))

    Sf = h / 3 * (sum_endpoint + sum_inner + sum_mid)
    print(f"a = {a}, b = {b}, n = {n}, h = {h}")
    print(f"S(f) = {Sf}")
    return Sf

# ============================================================
# PART 4 — Call
# ============================================================
integrate_summed_simpson(f, a, b, n)
