# ============================================================
# TOPIC: Natürliche kubische Spline-Interpolation (eigene Implementation)
# DESCRIPTION:
# Berechnet die Koeffizienten a_i, b_i, c_i, d_i der natürlichen kubischen
# Splinefunktion gemäss Skript-Algorithmus (Kap. 6.2.3) und wertet S(x)
# an beliebigen Stellen xx aus. Plottet die Stützpunkte und den Spline.
# USE WHEN:
# Wenn der Spline-Algorithmus explizit selbst implementiert werden soll
# (typische Übungsaufgabe; nicht scipy verwenden).
# EXAMPLE:
# 4 Stützpunkte (4,6), (6,3), (8,9), (10,0) durch natürliche kubische
# Splines interpolieren.
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
    plt.plot(xx, yy, 'b-', label='Natürliche kubische Spline')
    plt.plot(x, y, 'ro', markersize=8, label='Stützpunkte')
    plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.grid(True)
    plt.title('Natürliche kubische Spline-Interpolation')
    plt.show()
    return yy

# ============================================================
# PART 4 — Call
# ============================================================
natural_cubic_spline(x_data, y_data, xx)
