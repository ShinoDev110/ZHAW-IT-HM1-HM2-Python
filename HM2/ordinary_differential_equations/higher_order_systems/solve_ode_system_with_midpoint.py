# ============================================================
# TOPIC: ODE — Solve a first-order system with the midpoint method
# DESCRIPTION:
# Solves a vector-valued initial value problem z'(t) = f(t, z), z(t0) = z0
# with the midpoint method component-wise. Also suitable for higher-order
# ODEs that have been previously converted to a first-order system.
# USE WHEN:
# When a second-order (or higher-order) ODE is to be solved numerically.
# EXAMPLE:
# Boeing landing: m * x'' = -5 * x'^2 - 570000, x(0) = 0, x'(0) = 100,
# rewritten as z1' = z2, z2' = (-5*z2^2 - 570000) / m.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PART 1 — Inputs
# ============================================================
m = 97000.0

# z = (x, v); z' = (v, (-5 * v^2 - 570000) / m)
def f(t, z):
    x, v = z
    return np.array([v, (-5 * v**2 - 570000) / m])

t0, t_end = 0.0, 20.0
z0        = np.array([0.0, 100.0])
n         = 200

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: midpoint rule applied componentwise.

# ============================================================
# PART 3 — Implementation
# ============================================================
def solve_ode_system_midpoint(f, t0, t_end, z0, n):
    h = (t_end - t0) / n
    t = np.array([t0 + i * h for i in range(n + 1)])
    Z = np.zeros((n + 1, len(z0)))
    Z[0] = z0

    for i in range(n):
        t_h = t[i] + h / 2
        z_h = Z[i] + (h / 2) * f(t[i], Z[i])
        Z[i + 1] = Z[i] + h * f(t_h, z_h)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(t, Z[:, 0], 'b-', label='x(t) [m]')
    ax1.set_xlabel('t [s]'); ax1.set_ylabel('x(t) [m]', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(t, Z[:, 1], 'r-', label='v(t) [m/s]')
    ax2.set_ylabel('v(t) [m/s]', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title("First-order system — midpoint method")
    fig.tight_layout()
    plt.grid(True); plt.show()
    return t, Z

# ============================================================
# PART 4 — Call
# ============================================================
solve_ode_system_midpoint(f, t0, t_end, z0, n)
