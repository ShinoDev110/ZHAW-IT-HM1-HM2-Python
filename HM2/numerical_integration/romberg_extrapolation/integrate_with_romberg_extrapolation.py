# ============================================================
# TOPIC: Numerical Integration — Romberg extrapolation
# DESCRIPTION:
# Computes the Romberg scheme T_{j,k}: first column = summed trapezoidal rule
# at step sizes h_j = (b-a)/2^j, further columns via recursion
# T_{j,k} = (4^k * T_{j+1,k-1} - T_{j,k-1}) / (4^k - 1). Returns the
# most accurate value T_{0,m}.
# USE WHEN:
# A highly accurate integral value is to be computed from a few trapezoidal
# approximations via extrapolation.
# EXAMPLE:
# Compute int_2^4 1/x dx with Romberg extrapolation, m = 3.
# ============================================================

import numpy as np

# ============================================================
# PART 1 — Inputs
# ============================================================
def f(x):
    return 1 / x

a = 2.0
b = 4.0
m = 3       # number of step size halvings (first column has m+1 entries)

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: Romberg extrapolation per skript recursion.

# ============================================================
# PART 3 — Implementation
# ============================================================
def integrate_romberg(f, a, b, m):
    T = np.zeros((m + 1, m + 1))

    # First column: summed trapezoidal rule for h_j = (b-a) / 2^j
    for j in range(m + 1):
        n_j = 2**j
        h_j = (b - a) / n_j
        x   = np.array([a + i * h_j for i in range(n_j + 1)])
        y   = f(x)
        T[j, 0] = h_j * ((y[0] + y[-1]) / 2 + np.sum(y[1:-1]))

    # Extrapolation columns
    for k in range(1, m + 1):
        for j in range(m + 1 - k):
            T[j, k] = (4**k * T[j + 1, k - 1] - T[j, k - 1]) / (4**k - 1)

    print("Romberg scheme (row j, column k):")
    for j in range(m + 1):
        row = []
        for k in range(m + 1 - j):
            row.append(f"{T[j, k]:.10f}")
        print("  ".join(row))

    best = T[0, m]
    print(f"\nMost accurate value: T(0, {m}) = {best}")
    return best

# ============================================================
# PART 4 — Call
# ============================================================
integrate_romberg(f, a, b, m)
