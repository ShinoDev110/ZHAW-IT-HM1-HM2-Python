# ============================================================
# TOPIC: Root-finding methods — error bound via sign change
# DESCRIPTION:
# Checks whether in the interval [x* - r, x* + r] around an approximation x*
# a sign change of f exists. If so, (under continuity)
# |x_true - x*| <= r holds.
# USE WHEN:
# When after Newton/secant/etc. a hard error bound without
# derivative information is needed.
# EXAMPLE:
# f(x) = e^(x^2) + x^-3 - 10, x* ~= 1.65, r = 0.01.
# ============================================================

import math

# ============================================================
# PART 1 — Inputs
# ============================================================
def f(x):
    return math.exp(x * x) + x ** -3 - 10.0

x_approx = 1.65
radius   = 0.01

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here.

# ============================================================
# PART 3 — Implementation
# ============================================================
def verify_root_with_sign_change(f, x_approx, radius):
    left  = x_approx - radius
    right = x_approx + radius
    f_l   = f(left)
    f_r   = f(right)
    ok    = f_l * f_r <= 0.0
    print(f"f({left}) = {f_l}")
    print(f"f({right}) = {f_r}")
    if ok:
        print(f"-> Sign change: |x_true - {x_approx}| <= {radius}")
    else:
        print(f"-> NO sign change in interval [{left}, {right}]")
    return ok

# ============================================================
# PART 4 — Call
# ============================================================
verify_root_with_sign_change(f, x_approx, radius)
