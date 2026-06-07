# ============================================================
# TOPIC: Nonlinear system visualization — zeros of f: R^2 -> R^2
# DESCRIPTION:
# Visualizes the zeros of a 2D system f: R^2 -> R^2 as contour lines
# for f1=0 and f2=0 in a shared plot.
# USE WHEN:
# When the intersection points of two contour lines need to be determined
# graphically as solutions.
# EXAMPLE:
# First determine approximate solutions graphically for the system 2x1+4x2=0 and 4x1+8x2^3=0.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# ============================================================
# PART 1 — Inputs
# ============================================================
x1, x2 = sp.symbols('x1 x2')    # symbolic variables
X = sp.Matrix([x1, x2])         # vector of unknowns

f_sym = sp.Matrix([
    2*x1 + 4*x2,                  # first equation
    4*x1 + 8*x2**3                # second equation
])

x_range  = (-3, 3)              # x-axis range
y_range  = (-3, 3)              # y-axis range
n_points = 200                  # grid resolution

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: contour level 0 of each component f_i.
# Intersections = solutions of f(x) = 0.

# ============================================================
# PART 3 — Implementation
# ============================================================
def visualize_nonlinear_system_zeros(f_sym, X, x_range, y_range, n_points):
    f1 = sp.lambdify((X[0], X[1]), f_sym[0], "numpy")
    f2 = sp.lambdify((X[0], X[1]), f_sym[1], "numpy")

    xs = np.linspace(x_range[0], x_range[1], n_points)
    ys = np.linspace(y_range[0], y_range[1], n_points)
    XX, YY = np.meshgrid(xs, ys)
    Z1 = f1(XX, YY)
    Z2 = f2(XX, YY)

    fig, ax = plt.subplots(figsize=(8, 7))
    c1 = ax.contour(XX, YY, Z1, levels=[0], colors='red')
    c2 = ax.contour(XX, YY, Z2, levels=[0], colors='blue')

    h1, _ = c1.legend_elements()
    h2, _ = c2.legend_elements()
    ax.legend([h1[0], h2[0]], ['f_1(x_1, x_2) = 0', 'f_2(x_1, x_2) = 0'])

    ax.set_xlabel("x_1")
    ax.set_ylabel("x_2")
    ax.set_title("Zeros of the nonlinear system\n(intersections = solutions)")
    ax.grid(True)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    plt.tight_layout()
    plt.show()

# ============================================================
# PART 4 — Call
# ============================================================
visualize_nonlinear_system_zeros(f_sym, X, x_range, y_range, n_points)
