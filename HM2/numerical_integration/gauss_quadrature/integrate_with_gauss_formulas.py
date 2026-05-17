# ============================================================
# TOPIC: Numerische Integration — Gauss-Formeln (n = 1, 2, 3)
# DESCRIPTION:
# Approximiert int_a^b f(x) dx mit den Gauss-Quadraturformeln für 1, 2
# oder 3 Stützstellen. Stützstellen und Gewichte sind so optimiert, dass
# die Fehlerordnung maximal wird.
# USE WHEN:
# Wenn ein Integral mit möglichst wenigen Funktionsauswertungen sehr
# genau berechnet werden soll.
# EXAMPLE:
# Berechne int_0^0.5 exp(-x^2) dx mit der Gauss-Formel G3.
# ============================================================

import numpy as np

# ============================================================
# PART 1 — Inputs
# ============================================================
def f(x):
    return np.exp(-x**2)

a = 0.0
b = 0.5

# ============================================================
# PART 2 — Method selection
# ============================================================
# Options:
#   "G1"  -> 1 Stützstelle
#   "G2"  -> 2 Stützstellen
#   "G3"  -> 3 Stützstellen
#   "all" -> alle drei
method = "all"

# ============================================================
# PART 3 — Implementation
# ============================================================
def integrate_gauss_formulas(f, a, b, method):
    mid  = (a + b) / 2
    half = (b - a) / 2

    def G1():
        return (b - a) * f(mid)

    def G2():
        return half * (f(-1/np.sqrt(3) * half + mid)
                       + f( 1/np.sqrt(3) * half + mid))

    def G3():
        return half * (5/9 * f(-np.sqrt(0.6) * half + mid)
                       + 8/9 * f(mid)
                       + 5/9 * f( np.sqrt(0.6) * half + mid))

    if method == "G1":
        print(f"G1(f) = {G1()}")
    elif method == "G2":
        print(f"G2(f) = {G2()}")
    elif method == "G3":
        print(f"G3(f) = {G3()}")
    elif method == "all":
        print(f"G1(f) = {G1()}")
        print(f"G2(f) = {G2()}")
        print(f"G3(f) = {G3()}")
    else:
        raise ValueError("method must be 'G1', 'G2', 'G3', or 'all'")

# ============================================================
# PART 4 — Call
# ============================================================
integrate_gauss_formulas(f, a, b, method)
