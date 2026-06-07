# ============================================================
# TOPIC: Visualization — function f: R^2 -> R (surface / wireframe / contour lines)
# DESCRIPTION:
# Plots a scalar function f: R^2 -> R graphically.
# Use when a function of two variables needs to be examined as a surface,
# wireframe, or contour lines in 2D or 3D.
# USE WHEN:
# When a function of two variables needs to be visually compared or analyzed.
# EXAMPLE:
# f(x,y)=x^2+y^2 as a surface and contour lines.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ============================================================
# PART 1 — Inputs
# ============================================================
def f(x, y):                         # scalar function f(x,y)
    return x**2 + y**2               # change this for a different function

x_range  = (-5, 5)                   # x range
y_range  = (-5, 5)                   # y range
n_points = 50                        # grid resolution

# ============================================================
# PART 2 — Method selection
# ============================================================
# Options:
#   "surface"   -> coloured 3D surface plot
#   "wireframe" -> 3D wireframe (grid)
#   "contour2d" -> contour lines in the x-y plane
#   "contour3d" -> contour lines in 3D space
#   "all"       -> all four side by side
method = "all"

# ============================================================
# PART 3 — Implementation
# ============================================================
def visualize_2d_function(f, x_range, y_range, n_points, method):
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    def plot_surface(ax):
        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_title("Surface")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    def plot_wireframe(ax):
        ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2)
        ax.set_title("Wireframe (grid)")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    def plot_contour2d(ax):
        cont = ax.contour(X, Y, Z, cmap=cm.coolwarm)
        ax.clabel(cont, inline=True, fontsize=8)
        ax.set_title("Contour lines (2D)")
        ax.set_xlabel("x"); ax.set_ylabel("y")

    def plot_contour3d(ax):
        ax.contour(X, Y, Z, cmap=cm.coolwarm)
        ax.set_title("Contour lines (3D)")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    if method == "surface":
        fig = plt.figure(figsize=(7, 6))
        plot_surface(fig.add_subplot(111, projection='3d'))
    elif method == "wireframe":
        fig = plt.figure(figsize=(7, 6))
        plot_wireframe(fig.add_subplot(111, projection='3d'))
    elif method == "contour2d":
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_contour2d(ax)
    elif method == "contour3d":
        fig = plt.figure(figsize=(7, 6))
        plot_contour3d(fig.add_subplot(111, projection='3d'))
    elif method == "all":
        fig = plt.figure(figsize=(14, 10))
        plot_surface  (fig.add_subplot(2, 2, 1, projection='3d'))
        plot_wireframe(fig.add_subplot(2, 2, 2, projection='3d'))
        plot_contour2d(fig.add_subplot(2, 2, 3))
        plot_contour3d(fig.add_subplot(2, 2, 4, projection='3d'))
    else:
        raise ValueError("method must be 'surface', 'wireframe', 'contour2d', 'contour3d', or 'all'")

    plt.tight_layout()
    plt.show()

# ============================================================
# PART 4 — Call
# ============================================================
visualize_2d_function(f, x_range, y_range, n_points, method)
