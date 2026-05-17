# ============================================================
# TOPIC: Vergleich — ungedämpftes vs. gedämpftes Gauss-Newton-Verfahren
# DESCRIPTION:
# Wendet beide Gauss-Newton-Varianten auf dasselbe nichtlineare
# Ausgleichsproblem an und vergleicht Iterationszahl und Konvergenz.
# Endet auch wenn das ungedämpfte Verfahren divergiert (max_iter).
# USE WHEN:
# Wenn eine Aufgabe nach dem Vergleich beider Gauss-Newton-Varianten fragt
# (Konvergenzbereich, Iterationszahl).
# EXAMPLE:
# Exponentialer Fit f(x) = a·exp(b·x) mit zwei verschiedenen Startvektoren
# (1, -1.5) und (2, 2) — letzterer divergiert beim ungedämpften Verfahren.
# ============================================================

import numpy as np
import sympy as sp

# ============================================================
# PART 1 — Inputs
# ============================================================
x_data = np.array([0, 1, 2, 3, 4], dtype=float)
y_data = np.array([3, 1, 0.5, 0.2, 0.05], dtype=float)

a, b   = sp.symbols('a b')
params = sp.Matrix([a, b])

def model(p, xv):
    return p[0] * sp.exp(p[1] * xv)

start_vectors = [
    np.array([1.0, -1.5]),
    np.array([2.0,  2.0]),
]
tol      = 1e-5
max_iter = 100
p_max    = 10

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: run undamped and damped Gauss-Newton from each start.

# ============================================================
# PART 3 — Implementation
# ============================================================
def compare_gauss_newton_methods(x_data, y_data, params, model, start_vectors, tol, max_iter, p_max):
    n_data, n_params = len(x_data), len(params)
    g_sym  = sp.Matrix([y_data[i] - model(params, x_data[i]) for i in range(n_data)])
    Dg_sym = g_sym.jacobian(params)
    syms   = list(params)
    g_lam  = sp.lambdify(syms, g_sym,  "numpy")
    Dg_lam = sp.lambdify(syms, Dg_sym, "numpy")

    def g_eval(p):  return np.array(g_lam(*p),  dtype=float).reshape(-1)
    def Dg_eval(p): return np.array(Dg_lam(*p), dtype=float).reshape(n_data, n_params)

    def run_undamped(p0):
        p = p0.copy().astype(float); k = 0
        while np.linalg.norm(g_eval(p), 2) > tol and k < max_iter:
            Q, R = np.linalg.qr(Dg_eval(p))
            p = p + np.linalg.solve(R, -Q.T @ g_eval(p))
            k += 1
        return p, k, np.linalg.norm(g_eval(p), 2)

    def run_damped(p0):
        p = p0.copy().astype(float); k = 0
        while np.linalg.norm(g_eval(p), 2) > tol and k < max_iter:
            Q, R = np.linalg.qr(Dg_eval(p))
            delta = np.linalg.solve(R, -Q.T @ g_eval(p))
            err_curr = np.linalg.norm(g_eval(p), 2)
            pf = None
            for pe in range(p_max + 1):
                if np.linalg.norm(g_eval(p + delta/(2**pe)), 2) < err_curr:
                    pf = pe; break
            if pf is None: pf = 0
            p = p + delta/(2**pf)
            k += 1
        return p, k, np.linalg.norm(g_eval(p), 2)

    print(f"{'Startvektor':<22} {'Verfahren':<22} {'Iter':<6} {'lambda':<35} {'||g||':<12}")
    print("-" * 100)
    for p0 in start_vectors:
        for name, runner in [("ungedämpft", run_undamped), ("gedämpft", run_damped)]:
            p_sol, k, err = runner(p0)
            flag = "" if err <= tol else "  (DIV)"
            print(f"{str(p0):<22} {name:<22} {k:<6} {str(p_sol):<35} {err:.6e}{flag}")

# ============================================================
# PART 4 — Call
# ============================================================
compare_gauss_newton_methods(x_data, y_data, params, model, start_vectors, tol, max_iter, p_max)
