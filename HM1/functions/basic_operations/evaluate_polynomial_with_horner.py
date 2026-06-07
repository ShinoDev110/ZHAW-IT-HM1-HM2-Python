# ============================================================
# TOPIC: Functions — polynomial p(x), p'(x) and antiderivative via Horner scheme
# DESCRIPTION:
# Evaluates a polynomial p(x) = a_0 + a_1 x + ... + a_n x^n on a
# grid, additionally computes p'(x) and the antiderivative P(x)
# with P(0) = 0. All evaluations are done consistently using the
# Horner scheme.
# USE WHEN:
# When a polynomial including its derivative and integral is needed
# numerically on a grid (e.g. for plotting).
# EXAMPLE:
# p(x) = 1 + 2x + 3x^2 - x^3 on [-2, 2] with 1000 support points.
# ============================================================

import numpy as np

# ============================================================
# PART 1 — Inputs
# ============================================================
coefficients         = [1.0, 2.0, 3.0, -1.0]   # [a0, a1, a2, a3]
x_min                = -2.0
x_max                = 2.0
num_support_points   = 1000

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here (Horner). p, p' and P are always computed.

# ============================================================
# PART 3 — Implementation
# ============================================================
def _horner(coeff, x_array):
    result = np.zeros_like(x_array, dtype=float)
    for a in coeff[::-1]:
        result = result * x_array + a
    return result

def evaluate_polynomial_with_horner(coefficients, x_min, x_max, num_support_points):
    coeff = np.array(coefficients, dtype=float)
    if coeff.ndim != 1 or coeff.size == 0:
        raise ValueError("coefficients must be a non-empty 1D vector.")
    if x_min >= x_max:
        raise ValueError("x_min must be less than x_max.")

    degree = coeff.size - 1
    xs     = np.linspace(x_min, x_max, num_support_points)

    p_vals = _horner(coeff, xs)

    if degree == 0:
        dp_coeff = np.array([0.0])
    else:
        dp_coeff = np.array([k * coeff[k] for k in range(1, degree + 1)], dtype=float)
    dp_vals = _horner(dp_coeff, xs)

    P_coeff = np.zeros(degree + 2, dtype=float)
    for k in range(degree + 1):
        P_coeff[k + 1] = coeff[k] / (k + 1)
    P_vals = _horner(P_coeff, xs)

    print(f"p(x) at x_min={x_min}: {p_vals[0]}")
    print(f"p(x) at x_max={x_max}: {p_vals[-1]}")
    print(f"p'(x) at x_min={x_min}: {dp_vals[0]}")
    print(f"P(x) at x_max={x_max} (with P(0)=0): {P_vals[-1]}")
    return xs, p_vals, dp_vals, P_vals

# ============================================================
# PART 4 — Call
# ============================================================
evaluate_polynomial_with_horner(coefficients, x_min, x_max, num_support_points)
