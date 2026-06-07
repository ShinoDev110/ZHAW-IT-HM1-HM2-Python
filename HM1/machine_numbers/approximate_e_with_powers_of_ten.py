# ============================================================
# TOPIC: Machine numbers — approximation of e via (1 + 1/10^n)^(10^n)
# DESCRIPTION:
# Computes the classical approximation e ~= (1 + 1/10^n)^(10^n) for
# increasing n and compares with math.e (absolute and relative
# deviation). Demonstrates that the approximation degrades again beyond
# a certain n due to rounding errors.
# USE WHEN:
# When the effect of large exponents and small increments on
# floating-point numbers should be made visible.
# EXAMPLE:
# n = 1, 2, ..., 16 -> approach to e and from which n it drifts again.
# ============================================================

import math

# ============================================================
# PART 1 — Inputs
# ============================================================
n_max = 16   # exponent range n = 1..n_max

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here. A sweep from n=1 to n_max is always executed.

# ============================================================
# PART 3 — Implementation
# ============================================================
def _approximate_e(n):
    denominator = 10 ** n
    return (1.0 + 1.0 / denominator) ** denominator

def _abs_dev(x, ref):
    return abs(x - ref)

def _rel_dev(x, ref):
    return abs(x - ref) / abs(ref)

def approximate_e_with_powers_of_ten(n_max):
    print(f"Reference value math.e = {math.e}")
    print(f"{'n':>3} | {'(1+1/10^n)^(10^n)':>26} | {'abs dev':>14} | {'rel dev':>14}")
    print("-" * 70)
    for n in range(1, n_max + 1):
        e_approx = _approximate_e(n)
        print(f"{n:>3} | {e_approx:>26.16f} | {_abs_dev(e_approx, math.e):>14.4e} | {_rel_dev(e_approx, math.e):>14.4e}")

# ============================================================
# PART 4 — Call
# ============================================================
approximate_e_with_powers_of_ten(n_max)
