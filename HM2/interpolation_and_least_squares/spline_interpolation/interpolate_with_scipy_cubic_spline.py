# ============================================================
# TOPIC: Kubische Spline-Interpolation mit scipy.interpolate.CubicSpline
# DESCRIPTION:
# Verwendet scipy.interpolate.CubicSpline für schnelles Spline-Fitten.
# Randbedingungen wählbar via bc_type: 'natural', 'clamped', 'not-a-knot',
# 'periodic'.
# USE WHEN:
# Wenn die Spline-Interpolation mit einer fertigen Library gemacht werden
# darf — z.B. als Vergleich zur eigenen Implementation.
# EXAMPLE:
# US-Bevölkerung 1900-2010 mit natürlichem kubischem Spline interpolieren.
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

# ============================================================
# PART 2 — Method selection
# ============================================================
# Options for bc_type:
#   "natural"     -> S''(x_0) = S''(x_n) = 0
#   "clamped"     -> S'(x_0) = S'(x_n) = 0
#   "not-a-knot"  -> S''' stetig in x_1, x_{n-1}
#   "periodic"    -> periodische Splinefunktion
bc_type = "natural"

# ============================================================
# PART 3 — Implementation
# ============================================================
def cubic_spline_scipy(x_data, y_data, xx, bc_type):
    cs = interpolate.CubicSpline(x_data, y_data, bc_type=bc_type)
    yy = cs(xx)

    plt.figure(figsize=(9, 6))
    plt.plot(xx, yy, 'b-', label=f'scipy.CubicSpline ({bc_type})')
    plt.plot(x_data, y_data, 'ro', markersize=8, label='Datenpunkte')
    plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.grid(True)
    plt.title('Kubische Spline-Interpolation mit scipy')
    plt.show()
    return yy

# ============================================================
# PART 4 — Call
# ============================================================
cubic_spline_scipy(x_data, y_data, xx, bc_type)
