# ============================================================
# TOPIC: Newton method for systems — Minimize a function g(x,y,...)
# DESCRIPTION:
# Determines a minimizer of a scalar function of several variables by solving
# the nonlinear system of equations grad g = 0 with the Newton method for
# systems. The Jacobian matrix of grad g is the Hessian matrix of g.
# Iterates until ||grad g||_inf < tol and checks via the Hessian eigenvalues
# whether a minimum actually exists.
# USE WHEN:
# When the minimum (or a stationary point) of a differentiable
# function g(x,y,...) is sought (optimization via root of the gradient).
# EXAMPLE:
# g(x,y) = x^4 - 2x^2 y + 10x^2 - 8.4x + y^2 + 5.764, start (0.2, 0.8)
# -> minimum at (0.42, 0.1764), g = 4.0.
# ============================================================

import numpy as np
import sympy as sp

# ============================================================
# PART 1 — Inputs
# ============================================================
x, y = sp.symbols('x y')          # variables of g
syms = [x, y]                      # order of unknowns

g_sym = x**4 - 2*x**2*y + 10*x**2 - 8.4*x + y**2 + 5.764   # function to minimize

x0       = np.array([0.2, 0.8], dtype=float)   # initial vector
tol      = 1e-8                                 # stop: ||grad g||_inf < tol
max_iter = 100                                  # maximum number of iterations

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: Newton without damping on grad g = 0
# (H(x) · delta = -grad g(x),  x <- x + delta).

# ============================================================
# PART 3 — Implementation
# ============================================================
def minimize_function_with_newton(g_sym, syms, x0, tol, max_iter):
    grad = sp.Matrix([sp.diff(g_sym, s) for s in syms])
    H    = grad.jacobian(sp.Matrix(syms))
    print("============================================================")
    print("Minimization via Newton method (grad g = 0)")
    print("============================================================")
    print("grad g =")
    sp.pprint(grad)
    print("\nHessian matrix H = Jacobian(grad g) =")
    sp.pprint(H)

    grad_l = sp.lambdify(syms, grad, "numpy")
    H_l    = sp.lambdify(syms, H, "numpy")
    g_l    = sp.lambdify(syms, g_sym, "numpy")
    nv = len(syms)

    def grad_eval(v): return np.array(grad_l(*v), dtype=float).reshape(-1)
    def H_eval(v):    return np.array(H_l(*v), dtype=float).reshape(nv, nv)

    v = np.array(x0, dtype=float)
    err = np.linalg.norm(grad_eval(v), np.inf)
    print(f"\nn = 0   x = {v}   ||grad g||_inf = {err:.6e}")

    n = 0
    while err > tol and n < max_iter:
        delta = np.linalg.solve(H_eval(v), -grad_eval(v))
        v = v + delta
        err = np.linalg.norm(grad_eval(v), np.inf)
        n += 1
        print(f"n = {n}   x = {v}   ||grad g||_inf = {err:.6e}")

    g_min = float(g_l(*v))
    eig = np.linalg.eigvals(H_eval(v))
    point_type = "minimum" if np.all(eig > 0) else ("maximum" if np.all(eig < 0) else "saddle point")
    print(f"\nStationary point: x = {v}")
    print(f"g(x) = {g_min:.10g}")
    print(f"Hessian eigenvalues = {eig}  ->  {point_type}")
    return v, g_min

# ============================================================
# PART 4 — Call
# ============================================================
minimize_function_with_newton(g_sym, syms, x0, tol, max_iter)
