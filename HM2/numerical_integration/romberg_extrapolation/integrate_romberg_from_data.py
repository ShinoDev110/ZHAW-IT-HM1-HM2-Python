# ============================================================
# TOPIC: Numerical Integration — Romberg extrapolation from tabulated data
# DESCRIPTION:
# Like Romberg extrapolation, but WITHOUT a callable function f: the first
# column (summed trapezoidal rule at h_j = (b-a)/2^j) is built directly from
# a table of equally spaced node values by using every 2^(m-j)-th point.
# The number of intervals must be a power of two.
# USE WHEN:
# Only tabulated values (no formula) are available, but trapezoidal +
# Romberg is still required.
# EXAMPLE:
# x = [0,0.5,1,1.5,2], y = [2, sqrt(15)/2, sqrt(3), sqrt(7)/2, 0]
# (= sqrt(4-x^2)) -> integral ~ 3.09 (exact pi).
# ============================================================

import numpy as np

# ============================================================
# PART 1 — Inputs
# ============================================================
x_data = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
y_data = np.array([2.0, np.sqrt(15)/2, np.sqrt(3), np.sqrt(7)/2, 0.0])

exact     = np.pi   # exact value for comparison (or None)
use_exact = True

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: Romberg extrapolation from tabulated values.

# ============================================================
# PART 3 — Implementation
# ============================================================
def _trapez(ys, h):
    return h * ((ys[0] + ys[-1]) / 2.0 + np.sum(ys[1:-1]))

def integrate_romberg_from_data(x_data, y_data, exact=None, use_exact=False):
    x = np.asarray(x_data, dtype=float)
    y = np.asarray(y_data, dtype=float)
    n_int = len(x) - 1
    m = int(round(np.log2(n_int)))
    if 2**m != n_int:
        raise ValueError(f"Number of intervals ({n_int}) must be a power of two.")
    if not np.allclose(np.diff(x), x[1] - x[0]):
        raise ValueError("Nodes must be equally spaced.")

    a, b = x[0], x[-1]
    T = np.zeros((m + 1, m + 1))
    print("============================================================")
    print("Romberg extrapolation from tabulated data")
    print("============================================================")
    for j in range(m + 1):
        stride = 2**(m - j)
        ys = y[::stride]
        h_j = (b - a) / 2**j
        T[j, 0] = _trapez(ys, h_j)
        print(f"  T({j},0): h={h_j:<5g} with {len(ys)} points -> {T[j,0]:.8f}")

    for k in range(1, m + 1):
        for j in range(m + 1 - k):
            T[j, k] = (4**k * T[j + 1, k - 1] - T[j, k - 1]) / (4**k - 1)

    print("\nRomberg scheme (row j, column k):")
    for j in range(m + 1):
        row = [f"{T[j, k]:.8f}" for k in range(m + 1 - j)]
        print("  " + "   ".join(row))

    best = T[0, m]
    print(f"\nMost accurate value T(0,{m}) = {best:.8f}")
    if use_exact and exact is not None:
        print(f"Exact = {exact:.8f}, |error| = {abs(best - exact):.3e}")
    return best

# ============================================================
# PART 4 — Call
# ============================================================
integrate_romberg_from_data(x_data, y_data, exact, use_exact)
