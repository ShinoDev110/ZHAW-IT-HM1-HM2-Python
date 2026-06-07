# ============================================================
# TOPIC: Functions — symbolic differentiation and integration
# DESCRIPTION:
# Computes the derivative and the antiderivative of a symbolically
# defined function and optionally evaluates them at a given point.
# USE WHEN:
# When, for a given function f(x), f'(x) or ∫f(x) dx is needed quickly
# (e.g. as preparation for Newton's method).
# EXAMPLE:
# f(x) = sqrt(1 - x), derivative and integral, evaluated at x = 0.8.
# ============================================================

from sympy import diff, integrate, sympify

# ============================================================
# PART 1 — Inputs
# ============================================================
fx     = "sqrt(1 - x)"          # function as string
symbol = "x"                    # variable with respect to which to differentiate/integrate
values  = {"x": 0.8, "y": 0.6}   # values for evaluation

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here. Derivative AND integral are always printed.

# ============================================================
# PART 3 — Implementation
# ============================================================
def differentiate_and_integrate_symbolic(fx, symbol, values):
    fx_sym = sympify(fx)
    sym = sympify(symbol)

    derivative   = diff(fx_sym, sym)
    antiderivative  = integrate(fx_sym, sym)

    print(f"Function:           {fx_sym}")
    print(f"Derivative:         {derivative}")
    print(f"Antiderivative:     {antiderivative}")
    print()
    print(f"f(x)  evaluated:    {fx_sym.subs(values).evalf()}")
    print(f"f'(x) evaluated:    {derivative.subs(values).evalf()}")
    return derivative, antiderivative

# ============================================================
# PART 4 — Call
# ============================================================
differentiate_and_integrate_symbolic(fx, symbol, values)
