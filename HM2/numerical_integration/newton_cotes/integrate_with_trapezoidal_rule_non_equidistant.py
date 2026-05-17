# ============================================================
# TOPIC: Numerische Integration — Trapezregel für NICHT-äquidistante Stützpunkte
# DESCRIPTION:
# Approximiert int_{x_0}^{x_n} f(x) dx aus einer tabellierten Wertereihe
# (x_i, y_i) mit beliebigen Abständen via Tf_neq = sum_{i=0}^{n-1}
# (y_i + y_{i+1})/2 * (x_{i+1} - x_i).
# USE WHEN:
# Wenn die Stützstellen NICHT gleichmässig verteilt sind, z.B. bei
# Messdaten oder physikalischen Tabellen.
# EXAMPLE:
# Erdmasse aus m = int_0^6370 rho(r) * 4 pi r^2 dr mit gegebener Dichte-
# Tabelle bei nicht-äquidistanten Radien berechnen.
# ============================================================

import numpy as np

# ============================================================
# PART 1 — Inputs
# ============================================================
# Nicht-äquidistante x-Werte und zugehörige y-Werte des Integranden
x_data = np.array([0, 800, 1200, 1400, 2000, 3000, 3400, 3600, 4000, 5000, 5500, 6370], dtype=float)
rho    = np.array([13000, 12900, 12700, 12000, 11650, 10600, 9900, 5500, 5300, 4750, 4500, 3300], dtype=float)

# Hier: Integrand y = rho * 4 pi r^2, mit r in METERN (km -> m)
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
