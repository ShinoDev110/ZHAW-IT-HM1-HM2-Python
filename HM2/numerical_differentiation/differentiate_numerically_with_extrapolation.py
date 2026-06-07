# ============================================================
# TOPIC: Numerical Differentiation — difference quotient + h-extrapolation
# DESCRIPTION:
# Approximates f'(x0) with the forward, backward, or central difference
# quotient for a sequence of halved step sizes h_j = h0/2^j and improves
# the result with Richardson/Romberg extrapolation
# ("h-algorithm"): T_{j,k} = (r^k T_{j+1,k-1} - T_{j,k-1})/(r^k - 1) with
# r = 2 (forward/backward, error O(h)) or r = 4 (central, error O(h^2)).
# USE WHEN:
# A derivative is to be determined numerically and accurately from function
# values (typical task: "forward difference and extrapolation for h = 2, 1, 0.5").
# EXAMPLE:
# v(t) = 2000·ln(10000/(10000-100t)) - 9.8t, a(30) = v'(30): h0 = 2, 2 levels
# -> T(0,2) ~= 18.7714.
# ============================================================

import numpy as np

# ============================================================
# PART 1 — Inputs
# ============================================================
def f(t):
    return 2000.0 * np.log(10000.0 / (10000.0 - 100.0 * t)) - 9.8 * t   # v(t)

x0 = 30.0      # point at which f'(x0) is sought
h0 = 2.0       # largest step size h_0
m  = 2         # number of extrapolation levels -> h_j = h0/2^j, j = 0..m

def df_exact(t):
    return 2000.0 * 100.0 / (10000.0 - 100.0 * t) - 9.8   # v'(t), for comparison only

use_exact = True   # use df_exact for error output?

# ============================================================
# PART 2 — Method selection
# ============================================================
# kind:
#   "forward"  -> (f(x+h) - f(x)) / h          (error O(h),   r = 2)
#   "backward" -> (f(x) - f(x-h)) / h          (error O(h),   r = 2)
#   "central"  -> (f(x+h) - f(x-h)) / (2h)     (error O(h^2), r = 4)
kind = "forward"

# ============================================================
# PART 3 — Implementation
# ============================================================
def _diff_quotient(f, x0, h, kind):
    if kind == "forward":
        return (f(x0 + h) - f(x0)) / h
    if kind == "backward":
        return (f(x0) - f(x0 - h)) / h
    if kind == "central":
        return (f(x0 + h) - f(x0 - h)) / (2.0 * h)
    raise ValueError(f"Unknown kind: {kind!r}")

def differentiate_numerically_with_extrapolation(f, x0, h0, m, kind,
                                                 df_exact=None, use_exact=False):
    r = 4 if kind == "central" else 2
    T = np.zeros((m + 1, m + 1))

    for j in range(m + 1):
        h_j = h0 / 2**j
        T[j, 0] = _diff_quotient(f, x0, h_j, kind)

    for k in range(1, m + 1):
        for j in range(m + 1 - k):
            T[j, k] = (r**k * T[j + 1, k - 1] - T[j, k - 1]) / (r**k - 1)

    print("============================================================")
    print(f"Numerical differentiation — {kind} + extrapolation (r = {r})")
    print("============================================================")
    print(f"f'({x0}) sought, h_j = {h0}/2^j\n")
    print("Extrapolation scheme T_{j,k} (row j = step size, column k = level):")
    for j in range(m + 1):
        row = [f"{T[j, k]:.4f}" for k in range(m + 1 - j)]
        print(f"  h={h0/2**j:<6g} " + "   ".join(row))

    best = T[0, m]
    print(f"\nBest value T(0,{m}) = {best:.6f}")
    if use_exact and df_exact is not None:
        exact = df_exact(x0)
        print(f"Exact    f'({x0}) = {exact:.6f}")
        print(f"|Error| = {abs(best - exact):.3e}")
    return best

# ============================================================
# PART 4 — Call
# ============================================================
differentiate_numerically_with_extrapolation(f, x0, h0, m, kind, df_exact, use_exact)
