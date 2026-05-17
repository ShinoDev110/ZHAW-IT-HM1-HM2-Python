# ============================================================
# TOPIC: Funktionen — Polynom p(x), p'(x) und Stammfunktion via Horner
# DESCRIPTION:
# Wertet ein Polynom p(x) = a_0 + a_1 x + ... + a_n x^n auf einem
# Gitter aus, berechnet zusätzlich p'(x) und die Stammfunktion P(x)
# mit P(0) = 0. Die Auswertung erfolgt durchgehend mit dem
# Horner-Schema.
# USE WHEN:
# Wenn ein Polynom inklusive Ableitung und Integral numerisch effizient
# auf einem Gitter gebraucht wird (z.B. zum Plotten).
# EXAMPLE:
# p(x) = 1 + 2x + 3x^2 - x^3 auf [-2, 2] mit 1000 Stützstellen.
# ============================================================

import numpy as np

# ============================================================
# PART 1 — Inputs
# ============================================================
koeffizienten        = [1.0, 2.0, 3.0, -1.0]   # [a0, a1, a2, a3]
x_min                = -2.0
x_max                = 2.0
anzahl_stuetzstellen = 1000

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here (Horner). Es werden immer p, p' und P berechnet.

# ============================================================
# PART 3 — Implementation
# ============================================================
def _horner(koeff, x_array):
    ergebnis = np.zeros_like(x_array, dtype=float)
    for a in koeff[::-1]:
        ergebnis = ergebnis * x_array + a
    return ergebnis

def evaluate_polynomial_with_horner(koeffizienten, x_min, x_max, anzahl_stuetzstellen):
    koeff = np.array(koeffizienten, dtype=float)
    if koeff.ndim != 1 or koeff.size == 0:
        raise ValueError("koeffizienten muss ein nicht-leerer 1D-Vektor sein.")
    if x_min >= x_max:
        raise ValueError("x_min muss kleiner als x_max sein.")

    grad   = koeff.size - 1
    xs     = np.linspace(x_min, x_max, anzahl_stuetzstellen)

    p_vals = _horner(koeff, xs)

    if grad == 0:
        dp_koeff = np.array([0.0])
    else:
        dp_koeff = np.array([k * koeff[k] for k in range(1, grad + 1)], dtype=float)
    dp_vals = _horner(dp_koeff, xs)

    P_koeff = np.zeros(grad + 2, dtype=float)
    for k in range(grad + 1):
        P_koeff[k + 1] = koeff[k] / (k + 1)
    P_vals = _horner(P_koeff, xs)

    print(f"p(x) bei x_min={x_min}: {p_vals[0]}")
    print(f"p(x) bei x_max={x_max}: {p_vals[-1]}")
    print(f"p'(x) bei x_min={x_min}: {dp_vals[0]}")
    print(f"P(x) bei x_max={x_max} (mit P(0)=0): {P_vals[-1]}")
    return xs, p_vals, dp_vals, P_vals

# ============================================================
# PART 4 — Call
# ============================================================
evaluate_polynomial_with_horner(koeffizienten, x_min, x_max, anzahl_stuetzstellen)
