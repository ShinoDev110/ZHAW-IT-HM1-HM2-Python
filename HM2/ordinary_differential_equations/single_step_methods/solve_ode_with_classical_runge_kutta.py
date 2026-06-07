# ============================================================
# TOPIC: ODE — classical four-stage Runge-Kutta method
# DESCRIPTION:
# Standard RK4: weighted average of slopes k1, k2, k3, k4 per
# y_{i+1} = y_i + h/6 * (k1 + 2k2 + 2k3 + k4). Convergence order p = 4.
# USE WHEN:
# When a first-order ODE needs to be solved with high accuracy, without
# step-size control or implicit methods.
# EXAMPLE:
# Solve y' = 1 - y/t with y(1) = 5 on [1, 6] with h = 0.01.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PART 1 — Inputs
# ============================================================
def f(t, y):
    return 1 - y / t

a, b = 1.0, 6.0
y0   = 5.0
n    = 500

def y_exact(t):
    return t / 2 + 9 / (2 * t)

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: classical RK4 with Butcher tableau (0, 1/2, 1/2, 1).

# ============================================================
# PART 3 — Implementation
# ============================================================
def solve_ode_classical_rk4(f, a, b, y0, n, y_exact=None):
    h = (b - a) / n
    x = np.array([a + i * h for i in range(n + 1)])
    y = np.zeros(n + 1)
    y[0] = y0

    for i in range(n):
        k1 = f(x[i],         y[i])
        k2 = f(x[i] + h / 2, y[i] + h / 2 * k1)
        k3 = f(x[i] + h / 2, y[i] + h / 2 * k2)
        k4 = f(x[i] + h,     y[i] + h * k3)
        y[i + 1] = y[i] + h / 6 * (k1 + 2*k2 + 2*k3 + k4)

    print(f"Final value: y({x[-1]}) = {y[-1]:.10f}")
    if y_exact is not None:
        print(f"Exact:       y({x[-1]}) = {y_exact(x[-1]):.10f}")
        print(f"|Error| at endpoint = {abs(y_exact(x[-1]) - y[-1]):.6e}")

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'g-', label='RK4')
    if y_exact is not None:
        xs = np.linspace(a, b, 500)
        plt.plot(xs, y_exact(xs), 'r--', label='Exact')
    plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.grid(True)
    plt.title("Classical Four-Stage Runge-Kutta")
    plt.show()
    return x, y

# ============================================================
# PART 4 — Call
# ============================================================
solve_ode_classical_rk4(f, a, b, y0, n, y_exact)
