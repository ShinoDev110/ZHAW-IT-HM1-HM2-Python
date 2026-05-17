# ============================================================
# TOPIC: Newton-Verfahren für Systeme — Standard (quadratisch konvergent)
# DESCRIPTION:
# Löst ein nichtlineares Gleichungssystem f(x)=0 mit dem Standard-Newton-Verfahren.
# Verwenden, wenn eine quadratisch konvergente Nullstellensuche mit gutem
# Startvektor benötigt wird.
# USE WHEN:
# Wenn das Newton-Verfahren für ein System mit einer Näherungslösung gesucht ist.
# EXAMPLE:
# System mit f1=5x1^2-x2^2 und f2=x2-0.25(sin(x1)+cos(x2)).
# ============================================================

import numpy as np
import sympy as sp

# ============================================================
# PART 1 — Inputs
# ============================================================
x1, x2 = sp.symbols('x1 x2')    # symbolische Variablen
X = sp.Matrix([x1, x2])         # Vektor der Unbekannten

f_sym = sp.Matrix([
    5*x1**2 - x2**2,                                  # erste Gleichung
    x2 - 0.25*(sp.sin(x1) + sp.cos(x2))               # zweite Gleichung
])

x0       = np.array([0.25, 0.25], dtype=float)        # Startvektor
tol      = 1e-5                                       # Abbruchschwelle für ||f(x)||
max_iter = 100                                        # maximale Iterationszahl

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: solve Df(x^(n)) · δ = -f(x^(n))  and update x^(n+1) = x^(n) + δ.

# ============================================================
# PART 3 — Implementation
# ============================================================
def newton_method_systems(f_sym, X, x0, tol, max_iter):
    Df_sym = f_sym.jacobian(X)
    syms   = list(X)                                   # unpacked symbols
    f_lam  = sp.lambdify(syms, f_sym,  "numpy")
    Df_lam = sp.lambdify(syms, Df_sym, "numpy")

    n_vars = len(syms)
    def f_eval(x_vec):
        return np.array(f_lam(*x_vec), dtype=float).reshape(-1)
    def Df_eval(x_vec):
        return np.array(Df_lam(*x_vec), dtype=float).reshape(n_vars, n_vars)

    x   = np.array(x0, dtype=float).flatten()
    err = np.linalg.norm(f_eval(x), 2)
    n   = 0
    print(f"n = 0   x = {x}   ||f(x)|| = {err:.6e}")

    while err > tol and n < max_iter:
        delta = np.linalg.solve(Df_eval(x), -f_eval(x))
        x     = x + delta
        err   = np.linalg.norm(f_eval(x), 2)
        n    += 1
        print(f"n = {n}   x = {x}   ||f(x)|| = {err:.6e}")

    if err <= tol:
        print(f"\nKonvergiert nach {n} Iterationen.")
    else:
        print(f"\nMax. Iterationen erreicht ({max_iter}). Letzter Fehler: {err}")
    print(f"Näherungslösung: x = {x}")
    return x, n

# ============================================================
# PART 4 — Call
# ============================================================
newton_method_systems(f_sym, X, x0, tol, max_iter)
