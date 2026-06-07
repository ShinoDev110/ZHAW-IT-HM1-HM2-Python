# ============================================================
# TOPIC: ODE — custom s-stage Runge-Kutta method via Butcher tableau
# DESCRIPTION:
# Implements the general explicit s-stage Runge-Kutta method
# based on coefficients c, A, b from a Butcher tableau. Allows
# comparison with classical RK4 or custom designs.
# USE WHEN:
# When a self-designed or special RK method (e.g. Heun,
# Ralston, Kutta 3/8) should be applied.
# EXAMPLE:
# Design a new 4-stage RK method with custom c, A, b and solve
# y' = 1 - y/t with y(1) = 5 on [1, 6].
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PART 1 — Inputs
# ============================================================
def f(t, y):
    return 1 - y / t

a, b   = 1.0, 6.0
y0     = 5.0
n_step = 500

# Butcher tableau (custom example — sum(b) must equal 1!)
c = np.array([0.0, 0.25, 0.5, 0.75])
A = np.array([
    [0.0,  0.0,  0.0, 0.0],
    [0.5,  0.0,  0.0, 0.0],
    [0.0,  0.75, 0.0, 0.0],
    [0.0,  0.0,  1.0, 0.0],
])
b_vec = np.array([0.1, 0.3, 0.4, 0.2])

def y_exact(t):
    return t / 2 + 9 / (2 * t)

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: generic explicit s-stage Runge-Kutta from c, A, b.

# ============================================================
# PART 3 — Implementation
# ============================================================
def solve_ode_custom_runge_kutta(f, a, b_end, y0, n_step, c, A, b_vec, y_exact=None):
    assert abs(np.sum(b_vec) - 1.0) < 1e-12, "Sum of b coefficients must equal 1"
    s = len(b_vec)
    h = (b_end - a) / n_step
    x = np.array([a + i * h for i in range(n_step + 1)])
    y = np.zeros(n_step + 1)
    y[0] = y0

    for i in range(n_step):
        k = np.zeros(s)
        for j in range(s):
            y_arg = y[i] + h * sum(A[j, m] * k[m] for m in range(j))
            k[j]  = f(x[i] + c[j] * h, y_arg)
        y[i + 1] = y[i] + h * np.sum(b_vec * k)

    print(f"Final value: y({x[-1]}) = {y[-1]:.10f}")
    if y_exact is not None:
        print(f"Exact:       y({x[-1]}) = {y_exact(x[-1]):.10f}")
        print(f"|Error| at endpoint = {abs(y_exact(x[-1]) - y[-1]):.6e}")

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', label='Custom RK')
    if y_exact is not None:
        xs = np.linspace(a, b_end, 500)
        plt.plot(xs, y_exact(xs), 'r--', label='Exact')
    plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.grid(True)
    plt.title("Custom Runge-Kutta Method")
    plt.show()
    return x, y

# ============================================================
# PART 4 — Call
# ============================================================
solve_ode_custom_runge_kutta(f, a, b, y0, n_step, c, A, b_vec, y_exact)
