# ============================================================
# TOPIC: Simplified Newton method for systems (linearly convergent)
# DESCRIPTION:
# Solves a nonlinear system of equations with the simplified Newton method.
# The Jacobian matrix is evaluated only once at the initial vector and then kept fixed.
# USE WHEN:
# When the simplified Newton method is explicitly required.
# EXAMPLE:
# System with f1=2x1+4x2 and f2=4x1+8x2^3.
# ============================================================

import numpy as np
import sympy as sp

# ============================================================
# PART 1 — Inputs
# ============================================================
x1, x2 = sp.symbols('x1 x2')    # symbolic variables
X = sp.Matrix([x1, x2])         # vector of unknowns

f_sym = sp.Matrix([
    2*x1 + 4*x2,                  # first equation
    4*x1 + 8*x2**3                # second equation
])

x0       = np.array([4.0, 2.0], dtype=float)          # initial vector
tol      = 1e-5                                       # stop threshold for ||f(x)||
max_iter = 200                                        # maximum number of iterations

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: Df is evaluated ONCE at x^(0) and reused for all iterations.

# ============================================================
# PART 3 — Implementation
# ============================================================
def simplified_newton_method_systems(f_sym, X, x0, tol, max_iter):
    Df_sym = f_sym.jacobian(X)
    syms   = list(X)
    f_lam  = sp.lambdify(syms, f_sym,  "numpy")
    Df_lam = sp.lambdify(syms, Df_sym, "numpy")

    n_vars = len(syms)
    def f_eval(x_vec):
        return np.array(f_lam(*x_vec), dtype=float).reshape(-1)
    def Df_eval(x_vec):
        return np.array(Df_lam(*x_vec), dtype=float).reshape(n_vars, n_vars)

    x      = np.array(x0, dtype=float).flatten()
    Df_x0  = Df_eval(x)                            # frozen Jacobian
    err    = np.linalg.norm(f_eval(x), 2)
    n      = 0
    print(f"n = 0   x = {x}   ||f(x)|| = {err:.6e}")

    while err > tol and n < max_iter:
        delta = np.linalg.solve(Df_x0, -f_eval(x))
        x     = x + delta
        err   = np.linalg.norm(f_eval(x), 2)
        n    += 1
        print(f"n = {n}   x = {x}   ||f(x)|| = {err:.6e}")

    if err <= tol:
        print(f"\nConverged after {n} iterations.")
    else:
        print(f"\nMax. iterations reached ({max_iter}). Last error: {err}")
    print(f"Approximate solution: x = {x}")
    return x, n

# ============================================================
# PART 4 — Call
# ============================================================
simplified_newton_method_systems(f_sym, X, x0, tol, max_iter)
