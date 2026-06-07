# ============================================================
# TOPIC: Root-finding methods — Newton method (1D)
# DESCRIPTION:
# Classic Newton method x_{k+1} = x_k - f(x_k)/f'(x_k) for a
# symbolically given function. Stopping either by tolerance or
# fixed iteration count, with detailed debug output.
# USE WHEN:
# When a root of a differentiable function f(x) is sought
# and a good initial value is available (quadratic convergence).
# EXAMPLE:
# f(x) = exp(x) - (sqrt(x) + 2), initial value x0 = 0.5.
# ============================================================

from sympy import diff, sympify

# ============================================================
# PART 1 — Inputs
# ============================================================
function    = "exp(x) - (sqrt(x) + 2)"  # f(x); for fixed point: "F(x) - x"
x_0         = {"x": 0.5}                # initial value x0
tolerance   = 1e-7                      # tolerance for stopping criterion
iterations  = 5                         # max. iterations (for method "iters")
debug       = True                      # show intermediate values

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
def _newton_tol(fx, x0, tol, debug=False):
    fx = sympify(fx)
    symbols = list(fx.free_symbols)
    if not symbols:
        raise ValueError("No unknown found in function.")
    s = symbols[0]
    derivative = diff(fx)
    xn = x0[str(s)]
    i = 0
    while True:
        x_prev = xn
        xn = x_prev - (fx.subs({str(s): x_prev}).evalf() /
                       derivative.subs({str(s): x_prev}).evalf())
        if debug:
            print(f"---- Iteration {i + 1}")
            print(f"f({s}_{i}) = {fx.subs({str(s): x_prev}).evalf()}")
            print(f"f'({s}_{i}) = {derivative.subs({str(s): x_prev}).evalf()}")
            print(f"{s}_{i + 1} = {xn}\n")
        if abs(xn - x_prev) < tol:
            break
        i += 1
    return xn

def _newton_iters(fx, x0, iters, debug=False):
    fx = sympify(fx)
    symbols = list(fx.free_symbols)
    if not symbols:
        raise ValueError("No unknown found in function.")
    s = symbols[0]
    derivative = diff(fx)
    state = dict(x0)
    for i in range(iters):
        xk = state[str(s)] - (fx.subs(state).evalf() / derivative.subs(state).evalf())
        if debug:
            print(f"---- Iteration {i + 1}")
            print(f"f({s}_{i}) = {fx.subs(state).evalf()}")
            print(f"f'({s}_{i}) = {derivative.subs(state).evalf()}")
            print(f"{s}_{i + 1} = {xk}\n")
        state[str(s)] = xk
    return state[str(s)]

def find_root_with_newton(method, function, x_0, tolerance, iterations, debug=False):
    if method == "tol":
        return _newton_tol(function, x_0, tolerance, debug)
    if method == "iters":
        return _newton_iters(function, x_0, iterations, debug)
    raise ValueError(f"Unknown method: {method!r}")

# ============================================================
# PART 4 — Call
# ============================================================
find_root_with_newton(method, function, x_0, tolerance, iterations, debug)
