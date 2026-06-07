# ============================================================
# TOPIC: Root-finding methods — secant method (1D)
# DESCRIPTION:
# Secant method x_{k+1} = x_k - f(x_k)(x_k - x_{k-1}) / (f(x_k) - f(x_{k-1}))
# for a symbolically given function. Approximates f' by the
# secant; requires two initial values. Stopping by tolerance or iteration count.
# USE WHEN:
# When a root is sought but the derivative is not analytically
# available or too expensive to evaluate.
# EXAMPLE:
# f(x) = (x^2 + 1)^2 - 10 - 5/((x-1)^2 + 1), initial values x0=1.6, x1=1.7.
# ============================================================

import sympy as sp
from sympy import sympify

# ============================================================
# PART 1 — Inputs
# ============================================================
x = sp.Symbol("x")
function_1 = (x**2 + 1)**2 - 10
function_2 = 5 / ((x - 1)**2 + 1)
function   = function_1 - function_2   # f(x) whose root is sought

x_0         = {"x": 1.6}   # first initial value
x_1         = {"x": 1.7}   # second initial value
tolerance   = 1e-6         # tolerance for stopping criterion
iterations  = 2            # max. iterations (for method "iters")
debug       = True         # show intermediate values

# ============================================================
# PART 2 — Method selection
# ============================================================
# method:
#   "tol"   -> stop when |x_{k+1} - x_k| < tolerance
#   "iters" -> always execute iterations steps
method = "tol"

# ============================================================
# PART 3 — Implementation
# ============================================================
def _secant_tol(fx, x0, x1, tol, debug=False, max_iter=1000):
    fx = sympify(fx)
    symbols = sorted(fx.free_symbols, key=lambda s: s.name)
    if not symbols:
        raise ValueError("No unknown found in function.")
    s = symbols[0]
    x_nm1 = float(x0[str(s)])
    x_n   = float(x1[str(s)])
    for k in range(max_iter):
        f_nm1 = float(fx.subs({s: x_nm1}).evalf())
        f_n   = float(fx.subs({s: x_n}).evalf())
        denom = f_n - f_nm1
        if denom == 0:
            raise ZeroDivisionError("Secant method: f(x_n) - f(x_{n-1}) = 0.")
        x_np1 = x_n - f_n * (x_n - x_nm1) / denom
        if debug:
            print(f"---- Iteration {k + 1}")
            print(f"{s}_{k} = {x_nm1}")
            print(f"{s}_{k+1} = {x_n}")
            print(f"f({s}_{k}) = {f_nm1}")
            print(f"f({s}_{k+1}) = {f_n}")
            print(f"{s}_{k+2} = {x_np1}\n")
        if abs(x_np1 - x_n) <= tol:
            return x_np1
        x_nm1, x_n = x_n, x_np1
    raise RuntimeError("Maximum iterations reached.")

def _secant_iters(fx, x0, x1, iters, debug=False):
    fx = sympify(fx)
    symbols = sorted(fx.free_symbols, key=lambda s: s.name)
    if not symbols:
        raise ValueError("No unknown found in function.")
    s = symbols[0]
    x_nm1 = float(x0[str(s)])
    x_n   = float(x1[str(s)])
    for k in range(iters):
        f_nm1 = float(fx.subs({s: x_nm1}).evalf())
        f_n   = float(fx.subs({s: x_n}).evalf())
        denom = f_n - f_nm1
        if denom == 0:
            raise ZeroDivisionError("Secant method: f(x_n) - f(x_{n-1}) = 0.")
        x_np1 = x_n - f_n * (x_n - x_nm1) / denom
        if debug:
            print(f"---- Iteration {k + 1}")
            print(f"{s}_{k} = {x_nm1}")
            print(f"{s}_{k+1} = {x_n}")
            print(f"f({s}_{k}) = {f_nm1}")
            print(f"f({s}_{k+1}) = {f_n}")
            print(f"{s}_{k+2} = {x_np1}\n")
        x_nm1, x_n = x_n, x_np1
    return x_n

def find_root_with_secant(method, function, x_0, x_1, tolerance, iterations, debug=False):
    if method == "tol":
        return _secant_tol(function, x_0, x_1, tolerance, debug)
    if method == "iters":
        return _secant_iters(function, x_0, x_1, iterations, debug)
    raise ValueError(f"Unknown method: {method!r}")

# ============================================================
# PART 4 — Call
# ============================================================
find_root_with_secant(method, function, x_0, x_1, tolerance, iterations, debug)
