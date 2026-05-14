# ============================================================
# TOPIC: Vergleich — Standard / Vereinfachtes / Gedämpftes Newton-Verfahren
# DESCRIPTION:
# Führt Standard-, vereinfachtes und gedämpftes Newton-Verfahren mit demselben
# Startvektor aus und vergleicht Iterationszahl und Restfehler tabellarisch.
# USE WHEN:
# Wenn ausdrücklich ein Vergleich der Newton-Varianten verlangt wird.
# EXAMPLE:
# Vergleich der drei Verfahren für das gegebene System bei x^(0)=(1,1)^T.
# ============================================================

import numpy as np
import sympy as sp

# ============================================================
# PART 1 — Inputs
# ============================================================
x1, x2 = sp.symbols('x1 x2')
X = sp.Matrix([x1, x2])

f_sym = sp.Matrix([
    5*x1**2 - x2**2,
    x2 - 0.25*(sp.sin(x1) + sp.cos(x2))
])

x0       = np.array([1.0, 1.0], dtype=float)
tol      = 1e-5
max_iter = 100
k_max    = 4

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: run all three Newton variants with the same x0 and compare.

# ============================================================
# PART 3 — Implementation
# ============================================================
def compare_newton_methods_systems(f_sym, X, x0, tol, max_iter, k_max):
    Df_sym = f_sym.jacobian(X)
    syms   = list(X)
    f_lam  = sp.lambdify(syms, f_sym,  "numpy")
    Df_lam = sp.lambdify(syms, Df_sym, "numpy")

    n_vars = len(syms)
    def f_eval(x_vec):
        return np.array(f_lam(*x_vec), dtype=float).reshape(-1)
    def Df_eval(x_vec):
        return np.array(Df_lam(*x_vec), dtype=float).reshape(n_vars, n_vars)

    def run_standard(x0):
        x = np.array(x0, dtype=float).flatten()
        n = 0
        while np.linalg.norm(f_eval(x), 2) > tol and n < max_iter:
            delta = np.linalg.solve(Df_eval(x), -f_eval(x))
            x     = x + delta
            n    += 1
        return x, n, np.linalg.norm(f_eval(x), 2)

    def run_simplified(x0):
        x     = np.array(x0, dtype=float).flatten()
        Df_x0 = Df_eval(x)
        n     = 0
        while np.linalg.norm(f_eval(x), 2) > tol and n < max_iter:
            delta = np.linalg.solve(Df_x0, -f_eval(x))
            x     = x + delta
            n    += 1
        return x, n, np.linalg.norm(f_eval(x), 2)

    def run_damped(x0):
        x = np.array(x0, dtype=float).flatten()
        n = 0
        while np.linalg.norm(f_eval(x), 2) > tol and n < max_iter:
            delta    = np.linalg.solve(Df_eval(x), -f_eval(x))
            err_curr = np.linalg.norm(f_eval(x), 2)
            k_found  = None
            for k in range(k_max + 1):
                if np.linalg.norm(f_eval(x + delta/(2**k)), 2) < err_curr:
                    k_found = k
                    break
            if k_found is None:
                k_found = 0
            x  = x + delta/(2**k_found)
            n += 1
        return x, n, np.linalg.norm(f_eval(x), 2)

    print(f"Startvektor x^(0) = {x0}")
    print(f"Toleranz = {tol}\n")
    print(f"{'Methode':<22} {'Iter':<6} {'x':<35} {'||f(x)||':<12}")
    print("-" * 82)
    for name, runner in [("Standard Newton",      run_standard),
                         ("Vereinfachtes Newton", run_simplified),
                         ("Gedämpftes Newton",    run_damped)]:
        x_sol, n_iter, err = runner(x0)
        print(f"{name:<22} {n_iter:<6} {str(x_sol):<35} {err:.6e}")

# ============================================================
# PART 4 — Call
# ============================================================
compare_newton_methods_systems(f_sym, X, x0, tol, max_iter, k_max)
