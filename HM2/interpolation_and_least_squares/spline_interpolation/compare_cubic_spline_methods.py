# ============================================================
# TOPIC: Vergleich — eigene Spline, scipy.CubicSpline, polyfit-Polynom
# DESCRIPTION:
# Stellt die natürliche kubische Spline (eigene Implementation), die
# scipy-Spline und ein Interpolationspolynom hohen Grades (polyfit) für
# dieselben Daten in einer Grafik dar.
# USE WHEN:
# Wenn eine Aufgabe nach dem direkten Vergleich von Spline-Verfahren und
# Polynom-Interpolation fragt (z.B. um die Oszillation hoher Polynome zu
# zeigen).
# EXAMPLE:
# US-Bevölkerung 1900-2010 — eigene Spline vs. scipy.CubicSpline vs.
# Polynom Grad 11.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# ============================================================
# PART 1 — Inputs
# ============================================================
x_data = np.array([1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010], dtype=float)
y_data = np.array([75.995, 91.972, 105.711, 123.203, 131.669, 150.697,
                   179.323, 203.212, 226.505, 249.633, 281.422, 308.745])
xx = np.linspace(x_data[0], x_data[-1], 500)
x_shift = 1900   # zur Konditionsverbesserung bei polyfit

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: all three overlaid.

# ============================================================
# PART 3 — Implementation
# ============================================================
def compare_cubic_spline_methods(x_data, y_data, xx, x_shift):
    # 1) eigene natürliche kubische Spline
    def own_spline(x, y, xx):
        x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
        n = len(x) - 1
        a = y[:-1].copy(); h = np.diff(x)
        c = np.zeros(n + 1)
        if n >= 2:
            size = n - 1
            A = np.zeros((size, size)); z = np.zeros(size)
            for i in range(size):
                ii = i + 1
                A[i, i] = 2 * (h[ii - 1] + h[ii])
                if i > 0:        A[i, i - 1] = h[ii - 1]
                if i < size - 1: A[i, i + 1] = h[ii]
                z[i] = 3*(y[ii+1]-y[ii])/h[ii] - 3*(y[ii]-y[ii-1])/h[ii-1]
            c[1:n] = np.linalg.solve(A, z)
        b = np.zeros(n); d = np.zeros(n)
        for i in range(n):
            b[i] = (y[i+1]-y[i])/h[i] - h[i]/3*(c[i+1]+2*c[i])
            d[i] = (c[i+1]-c[i])/(3*h[i])
        yy = np.zeros_like(xx)
        for k, xv in enumerate(xx):
            i = max(0, min(np.searchsorted(x, xv)-1, n-1))
            dx = xv - x[i]
            yy[k] = a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3
        return yy

    yy_own    = own_spline(x_data, y_data, xx)
    cs        = interpolate.CubicSpline(x_data, y_data, bc_type='natural')
    yy_scipy  = cs(xx)
    coeffs    = np.polyfit(x_data - x_shift, y_data, len(x_data) - 1)
    yy_poly   = np.polyval(coeffs, xx - x_shift)

    plt.figure(figsize=(11, 6))
    plt.plot(x_data, y_data, 'ko', markersize=8, label='Datenpunkte')
    plt.plot(xx, yy_own,   'b-',  label='Eigene natürliche Spline')
    plt.plot(xx, yy_scipy, 'g--', label='scipy.CubicSpline (natural)')
    plt.plot(xx, yy_poly,  'r:',  label=f'Polynom Grad {len(x_data)-1}')
    plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.grid(True)
    plt.title('Vergleich Spline-Methoden vs. Polynom hohen Grades')
    plt.show()

# ============================================================
# PART 4 — Call
# ============================================================
compare_cubic_spline_methods(x_data, y_data, xx, x_shift)
