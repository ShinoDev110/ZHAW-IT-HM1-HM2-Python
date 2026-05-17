# ============================================================
# TOPIC: Numerische Integration — summierte Trapezregel (äquidistant)
# DESCRIPTION:
# Approximiert int_a^b f(x) dx mit Tf(h) = h * ((f(a)+f(b))/2 +
# sum_{i=1}^{n-1} f(x_i)), äquidistante Subintervalle.
# USE WHEN:
# Wenn ein bestimmtes Integral mit der summierten Trapezregel
# (Newton-Cotes Ordnung 1) approximiert werden soll und f als
# auswertbare Funktion vorliegt.
# EXAMPLE:
# Berechne int_2^4 1/x dx mit n = 4.
# ============================================================

import numpy as np

# ============================================================
# PART 1 — Inputs
# ============================================================
def f(x):
    return 1 / x

a = 2.0
b = 4.0
n = 4

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: summed trapezoidal rule, equidistant subintervals.

# ============================================================
# PART 3 — Implementation
# ============================================================
def integrate_summed_trapezoidal(f, a, b, n):
    h = (b - a) / n
    x = np.array([a + i * h for i in range(n + 1)])
    y = f(x)
    Tf = h * ((y[0] + y[-1]) / 2 + np.sum(y[1:-1]))
    print(f"a = {a}, b = {b}, n = {n}, h = {h}")
    print(f"T(f) = {Tf}")
    return Tf

# ============================================================
# PART 4 — Call
# ============================================================
integrate_summed_trapezoidal(f, a, b, n)
