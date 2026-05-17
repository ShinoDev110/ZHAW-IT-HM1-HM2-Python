# ============================================================
# TOPIC: Fixpunktiteration — a-priori Iterationsschätzung
# DESCRIPTION:
# Schätzt nach Banach die maximale Anzahl Iterationen, die nötig sind,
# damit |x_n - x̄| < toleranz, aus der Lipschitz-Konstanten alpha und der
# ersten Schrittweite |x_1 - x_0|.
# USE WHEN:
# Wenn vor dem Start einer Fixpunktiteration abgeschätzt werden soll,
# wie viele Schritte für eine gewünschte Genauigkeit nötig sind.
# EXAMPLE:
# F(x) = exp(x) - exp(1), Intervall [-3, -2], Startwert x0 = -2.5,
# Toleranz 1e-5.
# ============================================================

from math import ceil
import numpy as np
from sympy import diff, log, sympify

# ============================================================
# PART 1 — Inputs
# ============================================================
funktion  = "exp(x) - exp(1)"  # F(x) der Fixpunktiteration
x_0       = {"x": -2.5}        # Startwert x0
intervall = [-3, -2]           # Intervall, auf dem alpha geschätzt wird
toleranz  = 1e-5               # gewünschte Toleranz

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here. alpha wird stets als max|F'(x)| über das Intervall
# diskret approximiert; n folgt aus n >= log(tol·(1-alpha)/|x1-x0|) / log(alpha).

# ============================================================
# PART 3 — Implementation
# ============================================================
def _get_alpha(funktion, interval):
    f = sympify(funktion)
    symbols = list(f.free_symbols)
    if not symbols:
        raise ValueError("Keine Unbekannte in Funktion gefunden.")
    if len(interval) != 2:
        raise ValueError("Intervall muss aus zwei Werten bestehen.")
    s = symbols[0]
    werte = []
    for t in np.linspace(interval[0], interval[1], 100):
        werte.append(diff(f, s).subs(s, t).evalf())
    return np.max(np.abs(werte))

def _a_priori(alpha, x_0, x_1, tol):
    return log((tol * (1 - alpha)) / np.abs(x_1 - x_0)) / log(alpha)

def estimate_iterations_a_priori(funktion, x_0, intervall, toleranz):
    alpha = _get_alpha(funktion, intervall)
    print(f"alpha (Lipschitz) ~= {alpha}")

    x0_val = x_0[list(x_0.keys())[0]]
    x1_val = sympify(funktion).subs(x_0).evalf()

    n_real = _a_priori(alpha, x0_val, x1_val, toleranz)
    print(f"Anzahl Iterationsschritte: {n_real} bzw. {ceil(n_real)}")
    return n_real, ceil(n_real)

# ============================================================
# PART 4 — Call
# ============================================================
estimate_iterations_a_priori(funktion, x_0, intervall, toleranz)
