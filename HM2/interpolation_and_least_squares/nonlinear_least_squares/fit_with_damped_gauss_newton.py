# ============================================================
# TOPIC: Damped Gauss-Newton method for nonlinear least-squares problems
# DESCRIPTION:
# Like Gauss-Newton, but each iteration searches for the minimum p in {0,...,p_max}
# such that ||g(lambda + delta/2^p)|| < ||g(lambda)||. If none
# exists, p = 0. More robust with poor initial vectors.
# USE WHEN:
# When the undamped Gauss-Newton method diverges or oscillates,
# or when the initial vector may be far from the optimum.
# EXAMPLE:
# Fit f(x) = λ0 + λ1·10^(λ2+λ3·x) / (1 + 10^(λ2+λ3·x)) with initial vector
# (100, 120, 3, -1).
# ============================================================

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# ============================================================
# PART 1 — Inputs
# ============================================================
x_data = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5])
y_data = np.array([159.57209984, 159.8851819, 159.89378952, 160.30305273, 160.84630757,
                   160.94703969, 161.56961845, 162.31468058, 162.32140561, 162.88880047,
                   163.53234609, 163.85817086, 163.55339958, 163.86393263, 163.90535931,
                   163.44385491])

l0, l1, l2, l3 = sp.symbols('l0 l1 l2 l3')
params = sp.Matrix([l0, l1, l2, l3])

def model(p, xv):
    return p[0] + p[1] * 10**(p[2] + p[3] * xv) / (1 + 10**(p[2] + p[3] * xv))

lambda_0 = np.array([100.0, 120.0, 3.0, -1.0])
tol      = 1e-5
max_iter = 100
p_max    = 10

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: damped Gauss-Newton with QR for the linear LS step.

# ============================================================
# PART 3 — Implementation
# ============================================================
def damped_gauss_newton_fit(x_data, y_data, params, model, lambda_0, tol, max_iter, p_max):
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

        p_found = None
        for pe in range(p_max + 1):
            if np.linalg.norm(g_eval(p + delta / (2**pe)), 2) < err:
                p_found = pe
                break
        if p_found is None:
            p_found = 0

        p   = p + delta / (2**p_found)
        err = np.linalg.norm(g_eval(p), 2)
        k  += 1
        print(f"k = {k}   p_damp = {p_found}   lambda = {p}   ||g|| = {err:.6e}")

    print(f"\nSolution: lambda = {p}")

    f_num = sp.lambdify((syms, sp.Symbol('x_var')), model(params, sp.Symbol('x_var')), "numpy")
    xs = np.linspace(x_data.min(), x_data.max(), 300)
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, 'ko', markersize=8, label='Data points')
    plt.plot(xs, f_num(p, xs), 'b-', label='Damped Gauss-Newton fit')
    plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.grid(True)
    plt.title('Damped Gauss-Newton Method')
    plt.show()
    return p, k

# ============================================================
# PART 4 — Call
# ============================================================
damped_gauss_newton_fit(x_data, y_data, params, model, lambda_0, tol, max_iter, p_max)
