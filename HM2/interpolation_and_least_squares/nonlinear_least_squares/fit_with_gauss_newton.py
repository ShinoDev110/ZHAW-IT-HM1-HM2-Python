# ============================================================
# TOPIC: Gauss-Newton-Verfahren (ungedämpft) für nichtlineare Ausgleichsprobleme
# DESCRIPTION:
# Iteriert lambda^(k+1) = lambda^(k) + delta^(k), wobei delta aus dem
# linearen Ausgleichsproblem min ||g(lambda) + Dg(lambda)·delta||^2 stammt
# (gelöst via QR-Zerlegung). g(lambda) = y - f(lambda).
# USE WHEN:
# Wenn die Parameter einer nichtlinearen Ansatzfunktion f(lambda, x) an
# Daten angepasst werden sollen und ein guter Startvektor verfügbar ist.
# EXAMPLE:
# Fit f(x) = a·exp(b·x) an Datenpunkte mit Startvektor (a0, b0) = (1, -1.5).
# ============================================================

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# ============================================================
# PART 1 — Inputs
# ============================================================
x_data = np.array([0, 1, 2, 3, 4], dtype=float)
y_data = np.array([3, 1, 0.5, 0.2, 0.05], dtype=float)

a, b   = sp.symbols('a b')
params = sp.Matrix([a, b])

# Symbolic model: returns sympy expression in (params, x_value)
def model(p, xv):
    return p[0] * sp.exp(p[1] * xv)

lambda_0 = np.array([1.0, -1.5])
tol      = 1e-5
max_iter = 100

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: undamped Gauss-Newton, linear LS via QR each step.

# ============================================================
# PART 3 — Implementation
# ============================================================
def gauss_newton_fit(x_data, y_data, params, model, lambda_0, tol, max_iter):
    n_data   = len(x_data)
    n_params = len(params)

    g_sym  = sp.Matrix([y_data[i] - model(params, x_data[i]) for i in range(n_data)])
    Dg_sym = g_sym.jacobian(params)

    syms   = list(params)
    g_lam  = sp.lambdify(syms, g_sym,  "numpy")
    Dg_lam = sp.lambdify(syms, Dg_sym, "numpy")

    def g_eval(p):
        return np.array(g_lam(*p), dtype=float).reshape(-1)
    def Dg_eval(p):
        return np.array(Dg_lam(*p), dtype=float).reshape(n_data, n_params)

    p   = np.array(lambda_0, dtype=float).flatten()
    err = np.linalg.norm(g_eval(p), 2)
    k   = 0
    print(f"k = 0   lambda = {p}   ||g|| = {err:.6e}")

    while err > tol and k < max_iter:
        Q, R  = np.linalg.qr(Dg_eval(p))
        delta = np.linalg.solve(R, -Q.T @ g_eval(p))
        p     = p + delta
        err   = np.linalg.norm(g_eval(p), 2)
        k    += 1
        print(f"k = {k}   lambda = {p}   ||g|| = {err:.6e}")

    if err <= tol:
        print(f"\nKonvergiert nach {k} Iterationen.")
    else:
        print(f"\nNicht konvergiert nach {max_iter} Iterationen.")
    print(f"Lösung: lambda = {p}")

    # Plot
    f_num = sp.lambdify((syms, sp.Symbol('x_var')), model(params, sp.Symbol('x_var')), "numpy")
    xs = np.linspace(x_data.min(), x_data.max(), 300)
    plt.figure(figsize=(9, 6))
    plt.plot(x_data, y_data, 'ko', markersize=8, label='Datenpunkte')
    plt.plot(xs, f_num(p, xs), 'b-', label='Gauss-Newton-Fit')
    plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.grid(True)
    plt.title('Gauss-Newton-Fit (ungedämpft)')
    plt.show()
    return p, k

# ============================================================
# PART 4 — Call
# ============================================================
gauss_newton_fit(x_data, y_data, params, model, lambda_0, tol, max_iter)
