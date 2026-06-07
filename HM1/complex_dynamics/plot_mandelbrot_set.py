# ============================================================
# TOPIC: Complex dynamics — Mandelbrot set
# DESCRIPTION:
# Computes a 2D array of iteration counts for the Mandelbrot set
# Z_{n+1} = Z_n^2 + C, Z_0 = 0, C in C. Stops as soon as |Z_n| > 2,
# and plots the escape time as an image (imshow).
# USE WHEN:
# When the Mandelbrot set should be rendered as an illustrative example
# of complex dynamics / iterative methods.
# EXAMPLE:
# 1500 px in each direction, 200 iterations, standard view window.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PART 1 — Inputs
# ============================================================
num_pixels = 1500   # resolution in x- and y-direction
max_iter   = 200    # max. iterations per point
x_min      = -2.0
x_max      =  0.7
y_min      = -1.4
y_max      =  1.4

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here.

# ============================================================
# PART 3 — Implementation
# ============================================================
def plot_mandelbrot_set(num_pixels, max_iter, x_min, x_max, y_min, y_max):
    re_axis = np.linspace(x_min, x_max, num_pixels, dtype=np.float64)
    im_axis = np.linspace(y_min, y_max, num_pixels, dtype=np.float64)
    Re, Im = np.meshgrid(re_axis, im_axis)
    C = Re + 1j * Im
    Z = np.zeros_like(C, dtype=np.complex128)
    iters = np.zeros_like(C, dtype=int)

    for k in range(max_iter):
        Z = Z ** 2 + C
        mask = (np.abs(Z) > 2) & (C != 0)
        iters[mask] = k
        Z[mask] = 0
        C[mask] = 0

    plt.figure()
    plt.imshow(iters, extent=(x_min, x_max, y_min, y_max), origin="lower")
    plt.xlabel("Re(C)")
    plt.ylabel("Im(C)")
    plt.title("Mandelbrot set (iterations until |Z| > 2)")
    plt.colorbar(label="Iterations")
    plt.tight_layout()
    plt.show()
    return iters

# ============================================================
# PART 4 — Call
# ============================================================
plot_mandelbrot_set(num_pixels, max_iter, x_min, x_max, y_min, y_max)
