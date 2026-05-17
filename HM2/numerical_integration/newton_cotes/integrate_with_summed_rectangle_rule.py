# ============================================================
# TOPIC: Numerische Integration — summierte Rechteck-/Mittelpunktsregel
# DESCRIPTION:
# Approximiert int_a^b f(x) dx mit Rf(h) = h * sum_{i=0}^{n-1} f(x_i + h/2),
# d.h. n äquidistante Subintervalle und Auswertung in der Mitte.
# USE WHEN:
# Wenn ein bestimmtes Integral mit der summierten Mittelpunktsregel
# (Newton-Cotes Ordnung 0) approximiert werden soll.
# EXAMPLE:
# Berechne int_2^4 1/x dx näherungsweise mit n = 4.
# ============================================================

import numpy as np

# ============================================================
# PART 1 — Inputs
# ============================================================
def f(x):
    return 1 / x

a = 2.0          # untere Grenze
b = 4.0          # obere Grenze
n = 4            # Anzahl Subintervalle

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: summed midpoint rule.

# ============================================================
# PART 3 — Implementation
# ============================================================
def integrate_summed_rectangle(f, a, b, n):
    h = (b - a) / n
    midpoints = np.array([a + (i + 0.5) * h for i in range(n)])
    Rf = h * np.sum(f(midpoints))
    print(f"a = {a}, b = {b}, n = {n}, h = {h}")
    print(f"R(f) = {Rf}")
    return Rf

# ============================================================
# PART 4 — Call
# ============================================================
integrate_summed_rectangle(f, a, b, n)
