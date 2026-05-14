# ============================================================
# TOPIC: Newton-Verfahren — mit ||f(x)||_2 und ||x^(k) - x^(k-1)||_2 pro Iteration
# ============================================================

import numpy as np
import sympy as sp

# ============================================================
# PART 1 — Inputs
# ============================================================
x1, x2 = sp.symbols('x1 x2')
X = sp.Matrix([x1, x2])

f_sym = sp.Matrix([
    20 - 18*x1 - 2*x2**2,
    -4*x2*(x1 - x2**2)
])

x0       = np.array([1.1, 0.9], dtype=float)
tol      = 1e-5
max_iter = 100
n_steps  = 2          # exact number of steps for method "fixed_steps"

# ============================================================
# PART 2 — Method selection
# ============================================================
# Options:
#   "tolerance"   -> iterate until ||f(x)||_2 < tol or max_iter reached
#   "fixed_steps" -> iterate exactly n_steps times (ignores tol)
method = "fixed_steps"

# ============================================================
# PART 3 — Implementation
# ============================================================
def newton_with_norms(f_sym, X, x0, tol, max_iter, n_steps, method):
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
    x_prev = x.copy()
    err_f  = np.linalg.norm(f_eval(x), 2)
    k      = 0
    print(f"k = 0   x = {x}   ||f(x)|| = {err_f:.6e}   ||x-x_prev|| = ---")

    if method == "tolerance":
        keep_going = lambda: err_f > tol and k < max_iter
    elif method == "fixed_steps":
        keep_going = lambda: k < n_steps
    else:
        raise ValueError("method must be 'tolerance' or 'fixed_steps'")

    while keep_going():
        delta  = np.linalg.solve(Df_eval(x), -f_eval(x))
        x_prev = x.copy()
        x      = x + delta
        err_f  = np.linalg.norm(f_eval(x), 2)
        err_x  = np.linalg.norm(x - x_prev, 2)
        k     += 1
        print(f"k = {k}   x = {x}   ||f(x)|| = {err_f:.6e}   ||x-x_prev|| = {err_x:.6e}")

    print(f"\nNäherungslösung nach {k} Schritten: x = {x}")
    return x, k

# ============================================================
# PART 4 — Call
# ============================================================
newton_with_norms(f_sym, X, x0, tol, max_iter, n_steps, method)
