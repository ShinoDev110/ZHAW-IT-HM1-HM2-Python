# ============================================================
# TOPIC: ODE — midpoint method for y'(x) = f(x, y)
# DESCRIPTION:
# Half Euler step to (x + h/2, y + h/2 * f(x, y)), evaluate slope
# there, full step with that slope. Convergence order p = 2.
# USE WHEN:
# When a first-order ODE needs to be solved with a second-order method,
# without the effort of Runge-Kutta.
# EXAMPLE:
# Solve y' = x^2 / y with y(0) = 2 on [0, 1.4] with h = 0.7.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PART 1 — Inputs
# ============================================================
def f(x, y):
    return x**2 / y

a, b = 0.0, 1.4
y0   = 2.0
n    = 2

def y_exact(x):
    return np.sqrt(2 * x**3 / 3 + 4)

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: midpoint (modified midpoint / RK2 midpoint variant).

# ============================================================
# PART 3 — Implementation
# ============================================================
def solve_ode_midpoint(f, a, b, y0, n, y_exact=None):
    h = (b - a) / n
    x = np.array([a + i * h for i in range(n + 1)])
    y = np.zeros(n + 1)
    y[0] = y0

    print(f"{'i':<3} {'x_i':<10} {'y_i':<14}")
    print(f"{0:<3} {x[0]:<10.4f} {y[0]:<14.6f}")
    for i in range(n):
        x_h = x[i] + h / 2
        y_h = y[i] + (h / 2) * f(x[i], y[i])
        y[i + 1] = y[i] + h * f(x_h, y_h)
        print(f"{i+1:<3} {x[i+1]:<10.4f} {y[i+1]:<14.6f}")

    plt.figure(figsize=(9, 6))
    plt.plot(x, y, 'b-+', label='Midpoint')
    if y_exact is not None:
        xs = np.linspace(a, b, 300)
        plt.plot(xs, y_exact(xs), 'r-', label='Exact')
        err = np.abs(y_exact(x) - y)
        print("\nAbsolute error:")
        for i in range(n + 1):
            print(f"  i = {i}, x = {x[i]:.4f}, err = {err[i]:.6e}")
    plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.grid(True)
    plt.title("Midpoint Method")
    plt.show()
    return x, y

# ============================================================
# PART 4 — Call
# ============================================================
solve_ode_midpoint(f, a, b, y0, n, y_exact)
