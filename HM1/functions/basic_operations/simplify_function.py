# ============================================================
# TOPIC: Functions — symbolic simplification
# DESCRIPTION:
# Simplifies a symbolically given expression using sympy.simplify.
# USE WHEN:
# When a manually rearranged expression should be brought into a
# compact form (e.g. after deriving the Newton iteration formula).
# EXAMPLE:
# f(x) = x - ((x^2 - c) / (2x))  ->  simplified form.
# ============================================================

from sympy import simplify, sympify

# ============================================================
# PART 1 — Inputs
# ============================================================
fx = "x - ((x**2 - c) / (2*x))"   # expression to simplify

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here.

# ============================================================
# PART 3 — Implementation
# ============================================================
def simplify_function(fx):
    expr = sympify(fx)
    simplified = simplify(expr)
    print(f"Original:           {expr}")
    print(f"Simplified form:    {simplified}")
    return simplified

# ============================================================
# PART 4 — Call
# ============================================================
simplify_function(fx)
