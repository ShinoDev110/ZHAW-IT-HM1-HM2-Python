# ============================================================
# TOPIC: Natural cubic spline interpolation (own implementation)
# DESCRIPTION:
# Computes the coefficients a_i, b_i, c_i, d_i of the natural cubic
# spline function per the lecture-script algorithm (Ch. 6.2.3) and evaluates
# S(x) at arbitrary points xx. Plots the support points and the spline.
# USE WHEN:
# When the spline algorithm should be explicitly implemented from scratch
# (typical exercise task; do not use scipy).
# EXAMPLE:
# Interpolate 4 support points (4,6), (6,3), (8,9), (10,0) with natural
# cubic splines.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PART 1 — Inputs
# ============================================================
x_data = np.array([4, 6, 8, 10], dtype=float)
y_data = np.array([6, 3, 9, 0], dtype=float)
xx     = np.linspace(x_data[0], x_data[-1], 200)

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: natural cubic spline (S''(x_0) = S''(x_n) = 0).

# ============================================================
# PART 3 — Implementation
# ============================================================
def natural_cubic_spline(x, y, xx):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x) - 1                          # n intervals

    a = y[:-1].copy()
    h = np.diff(x)

    # Solve A c = z for c_1 ... c_{n-1}; c_0 = c_n = 0
    c = np.zeros(n + 1)
    if n >= 2:
        size = n - 1
        A = np.zeros((size, size))
        z = np.zeros(size)
        for i in range(size):
            ii = i + 1
            A[i, i] = 2 * (h[ii - 1] + h[ii])
            if i > 0:        A[i, i - 1] = h[ii - 1]
            if i < size - 1: A[i, i + 1] = h[ii]
            z[i] = 3 * (y[ii + 1] - y[ii]) / h[ii] - 3 * (y[ii] - y[ii - 1]) / h[ii - 1]
        c[1:n] = np.linalg.solve(A, z)

    b = np.zeros(n)
    d = np.zeros(n)
    for i in range(n):
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] / 3 * (c[i + 1] + 2 * c[i])
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    print(f"{'i':<3} {'a_i':<14} {'b_i':<14} {'c_i':<14} {'d_i':<14}")
    for i in range(n):
        print(f"{i:<3} {a[i]:<14.6f} {b[i]:<14.6f} {c[i]:<14.6f} {d[i]:<14.6f}")

    yy = np.zeros_like(xx)
    for k, xv in enumerate(xx):
        i = np.searchsorted(x, xv) - 1
        i = max(0, min(i, n - 1))
        dx = xv - x[i]
        yy[k] = a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3

    plt.figure(figsize=(9, 6))
    plt.plot(xx, yy, 'b-', label='Natural cubic spline')
    plt.plot(x, y, 'ro', markersize=8, label='Support points')
    plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.grid(True)
    plt.title('Natural cubic spline interpolation')
    plt.show()
    return yy

# ============================================================
# PART 4 — Call
# ============================================================
natural_cubic_spline(x_data, y_data, xx)
