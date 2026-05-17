# ============================================================
# TOPIC: Funktionen — symbolisches Vereinfachen
# DESCRIPTION:
# Vereinfacht einen symbolisch gegebenen Ausdruck mit sympy.simplify.
# USE WHEN:
# Wenn ein per Hand umgeformter Ausdruck in eine kompakte Form gebracht
# werden soll (z.B. nach Newton-Iterationsformel).
# EXAMPLE:
# f(x) = x - ((x^2 - c) / (2x))  ->  vereinfachte Form.
# ============================================================

from sympy import simplify, sympify

# ============================================================
# PART 1 — Inputs
# ============================================================
fx = "x - ((x**2 - c) / (2*x))"   # zu vereinfachender Ausdruck

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here.

# ============================================================
# PART 3 — Implementation
# ============================================================
def simplify_function(fx):
    expr = sympify(fx)
    vereinfacht = simplify(expr)
    print(f"Original:           {expr}")
    print(f"Vereinfachte Form:  {vereinfacht}")
    return vereinfacht

# ============================================================
# PART 4 — Call
# ============================================================
simplify_function(fx)
