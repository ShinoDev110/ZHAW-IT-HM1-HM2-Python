# ============================================================
# TOPIC: Newton-Verfahren — alle Lösungen aus einer Liste von Startvektoren finden
# DESCRIPTION:
# Wendet das Standard-Newton-Verfahren auf eine Liste von Startvektoren an und
# gibt alle gefundenen Lösungen tabellarisch aus.
# USE WHEN:
# Wenn ein System mehrere Nullstellen besitzt und alle gefunden werden sollen.
# EXAMPLE:
# Mit Startvektoren aus einer grafischen Voranalyse die vier Lösungen bestimmen.
# ============================================================

import numpy as np
import sympy as sp

# ============================================================
# PART 1 — Inputs
# ============================================================
x, y = sp.symbols('x y')
X = sp.Matrix([x, y])

f_sym = sp.Matrix([
    x**2 / 186**2 - y**2 / (300**2 - 186**2) - 1,
    (y - 500)**2 / 279**2 - (x - 300)**2 / (500**2 - 279**2) - 1
])

# Approximations from the implicit plot (one per expected intersection point)
start_vectors = [
    np.array([-200,  200], dtype=float),
    np.array([-300, -800], dtype=float),
    np.array([ 200, -800], dtype=float),
    np.array([ 700,  700], dtype=float),
]

tol      = 1e-5
max_iter = 100

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: run standard Newton from each start vector separately.

# ============================================================
# PART 3 — Implementation
# ============================================================
def newton_find_all_solutions(f_sym, X, start_vectors, tol, max_iter):
    Df_sym = f_sym.jacobian(X)
    syms   = list(X)
    f_lam  = sp.lambdify(syms, f_sym,  "numpy")
    Df_lam = sp.lambdify(syms, Df_sym, "numpy")

    n_vars = len(syms)
    def f_eval(x_vec):
        return np.array(f_lam(*x_vec), dtype=float).reshape(-1)
    def Df_eval(x_vec):
        return np.array(Df_lam(*x_vec), dtype=float).reshape(n_vars, n_vars)

    print(f"{'Start':<25} {'Lösung':<30} {'Iter':<6} {'||f(x)||':<12}")
    print("-" * 75)
    solutions = []
    for x0 in start_vectors:
        x   = np.array(x0, dtype=float).flatten()
        err = np.linalg.norm(f_eval(x), 2)
        n   = 0
        while err > tol and n < max_iter:
            delta = np.linalg.solve(Df_eval(x), -f_eval(x))
            x     = x + delta
            err   = np.linalg.norm(f_eval(x), 2)
            n    += 1
        flag = "" if err <= tol else "  (DIVERGENT)"
        print(f"{str(x0):<25} {str(x):<30} {n:<6} {err:.6e}{flag}")
        solutions.append((x0, x, n, err))
    return solutions

# ============================================================
# PART 4 — Call
# ============================================================
newton_find_all_solutions(f_sym, X, start_vectors, tol, max_iter)
