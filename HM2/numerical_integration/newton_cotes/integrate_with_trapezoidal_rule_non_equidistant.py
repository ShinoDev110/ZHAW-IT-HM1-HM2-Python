# ============================================================
# TOPIC: Numerical Integration — trapezoidal rule for NON-equidistant nodes
# DESCRIPTION:
# Approximates int_{x_0}^{x_n} f(x) dx from a tabulated data series
# (x_i, y_i) with arbitrary spacing via Tf_neq = sum_{i=0}^{n-1}
# (y_i + y_{i+1})/2 * (x_{i+1} - x_i).
# USE WHEN:
# The nodes are NOT uniformly distributed, e.g. for measurement data
# or physical tables.
# EXAMPLE:
# Earth mass from m = int_0^6370 rho(r) * 4 pi r^2 dr with a given density
# table at non-equidistant radii.
# ============================================================

import numpy as np

# ============================================================
# PART 1 — Inputs
# ============================================================
# Non-equidistant x values and corresponding y values of the integrand
x_data = np.array([0, 800, 1200, 1400, 2000, 3000, 3400, 3600, 4000, 5000, 5500, 6370], dtype=float)
rho    = np.array([13000, 12900, 12700, 12000, 11650, 10600, 9900, 5500, 5300, 4750, 4500, 3300], dtype=float)

# Here: integrand y = rho * 4 pi r^2, with r in METERS (km -> m)
x_int  = x_data * 1000
y_data = rho * 4 * np.pi * x_int**2

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: trapezoidal sum over non-equidistant subintervals.

# ============================================================
# PART 3 — Implementation
# ============================================================
def integrate_trapezoidal_non_equidistant(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = np.diff(x)
    Tf = np.sum((y[:-1] + y[1:]) / 2.0 * dx)
    print(f"Integral = {Tf}")
    return Tf

# ============================================================
# PART 4 — Call
# ============================================================
integrate_trapezoidal_non_equidistant(x_int, y_data)
