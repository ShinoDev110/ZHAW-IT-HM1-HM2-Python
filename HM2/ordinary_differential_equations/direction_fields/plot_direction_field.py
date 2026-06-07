# ============================================================
# TOPIC: ODE — Plot direction field of a first-order ODE y'(x) = f(x, y)
# DESCRIPTION:
# Draws the direction field of a first-order ODE on a grid with
# step sizes hx, hy via np.meshgrid and plt.quiver. Arrows show the
# slope f(x, y) at each grid point.
# USE WHEN:
# When the direction field of an ODE is to be visualised, often as a
# basis for subsequently drawing in numerical solutions.
# EXAMPLE:
# Direction field for y' = x^2 / y on [0, 1.4] x [0, 5] with hx = hy = 0.2.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PART 1 — Inputs
# ============================================================
def f(x, y):
    return x**2 / y

xmin, xmax = 0.0, 1.4
ymin, ymax = 0.0, 5.0
hx, hy     = 0.1, 0.5

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: arrow direction = (1, f(x, y)), optional normalization.
normalize = True

# ============================================================
# PART 3 — Implementation
# ============================================================
def plot_direction_field(f, xmin, xmax, ymin, ymax, hx, hy, normalize):
    x = np.arange(xmin, xmax + hx, hx)
    y = np.arange(ymin, ymax + hy, hy)
    X, Y = np.meshgrid(x, y)

    U = np.ones_like(X)
    V = f(X, Y)

    if normalize:
        L = np.sqrt(U**2 + V**2)
        U, V = U / L, V / L

    plt.figure(figsize=(9, 6))
    plt.quiver(X, Y, U, V, angles='xy')
    plt.xlim(xmin, xmax); plt.ylim(ymin, ymax)
    plt.xlabel('x'); plt.ylabel('y'); plt.grid(True)
    plt.title("Direction field y'(x) = f(x, y)")
    plt.show()

# ============================================================
# PART 4 — Call
# ============================================================
plot_direction_field(f, xmin, xmax, ymin, ymax, hx, hy, normalize)
