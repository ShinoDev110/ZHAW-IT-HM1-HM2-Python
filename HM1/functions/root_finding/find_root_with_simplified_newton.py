# ============================================================
# TOPIC: Root-finding methods — simplified Newton method
# DESCRIPTION:
# Simplified Newton: x_{k+1} = x_k - f(x_k) / f'(x_0). The derivative
# is evaluated only once at the initial value x0 and then treated as constant.
# Saves computational effort but converges only linearly.
# USE WHEN:
# When f'(x_0) is known and evaluating f'(x_k) is expensive, and the
# initial value is close enough to the root.
# EXAMPLE:
# f(x) = e^(x^2) + x^-3 - 10, f'(x) = 2x e^(x^2) - 3 x^-4, x0 = 1.5.
# ============================================================

import math

# ============================================================
# PART 1 — Inputs
# ============================================================
def f(x):
    return math.exp(x * x) + x ** -3 - 10.0

def df(x):
    return 2.0 * x * math.exp(x * x) - 3.0 * x ** -4

x0            = 1.5
tolerance     = 1e-8
max_iterations = 50

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here.

# ============================================================
# PART 3 — Implementation
# ============================================================
def find_root_with_simplified_newton(f, df, x0, tol=1e-8, max_iter=50):
    df0 = float(df(x0))
    if df0 == 0.0:
        raise ValueError("f'(x0) = 0, simplified Newton not applicable.")

    x_old = float(x0)
    sequence = [x_old]
    iterations = 0

    while iterations < max_iter:
        fx = float(f(x_old))
        x_new = x_old - fx / df0
        sequence.append(x_new)
        iterations += 1
        print(f"k={iterations} | x_k = {x_new} | f(x_k) = {f(x_new)}")
        if abs(x_new - x_old) < tol:
            break
        x_old = x_new

    print(f"\nApproximate solution: x ~= {x_new}")
    print(f"Iterations:           {iterations}")
    return x_new, iterations, sequence

# ============================================================
# PART 4 — Call
# ============================================================
find_root_with_simplified_newton(f, df, x0, tolerance, max_iterations)
