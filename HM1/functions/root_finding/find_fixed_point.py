# ============================================================
# TOPIC: Nullstellenverfahren — Fixpunktiteration x_{k+1} = F(x_k)
# DESCRIPTION:
# Allgemeine Fixpunktiteration einer symbolisch gegebenen Funktion F(x).
# Wahl zwischen Abbruch nach Toleranz |x_{k+1} - x_k| < tol oder fester
# Anzahl Iterationen.
# USE WHEN:
# Wenn ein Fixpunkt x̄ einer Iterationsvorschrift F(x) gesucht ist
# und F im Suchbereich kontrahierend ist (|F'(x)| < 1).
# EXAMPLE:
# F(x) = (230x^4 + 18x^3 + 9x^2 - 9) / 221, Startwert x0 = 0.
# ============================================================

import numpy as np
from sympy import sympify

# ============================================================
# PART 1 — Inputs
# ============================================================
funktion     = "(230 * x**4 + 18 * x**3 + 9 * x**2 - 9) / 221"  # F(x), bereits nach x aufgelöst
x_0          = {"x": 0}    # Startwert x0
toleranz     = 1e-6        # Toleranz für Abbruchkriterium
iterationen  = 10          # max. Iterationen (für Methode "iters")

# ============================================================
# PART 2 — Method selection
# ============================================================
# method:
#   "tol"   -> Abbruch bei |x_{k+1} - x_k| < toleranz
#   "iters" -> immer iterationen Schritte ausführen
method = "tol"

# ============================================================
# PART 3 — Implementation
# ============================================================
def _fixpunktiteration_schritt(fx, wert):
    return fx.subs(wert).evalf()

def find_fixed_point(method, funktion, x_0, toleranz, iterationen):
    fx = sympify(funktion)
    symbols = list(fx.free_symbols)
    if len(symbols) == 0:
        raise ValueError("Keine Unbekannte in Funktion gefunden.")
    s = symbols[0]
    previous_value = x_0[str(s)]

    if method == "tol":
        next_x_value = -100
        count = 0
        while np.abs(next_x_value - previous_value) > toleranz:
            if count > 0:
                previous_value = next_x_value
            next_x_value = _fixpunktiteration_schritt(fx, {s: previous_value})
            count += 1
            print(f"{s}_{count} = {next_x_value}")
        return next_x_value
    elif method == "iters":
        next_x_value = -100
        for i in range(iterationen):
            if i > 0:
                previous_value = next_x_value
            next_x_value = _fixpunktiteration_schritt(fx, {s: previous_value})
            print(f"{s}_{i + 1} = {next_x_value}")
        return next_x_value
    else:
        raise ValueError(f"Unbekannte Methode: {method!r}")

# ============================================================
# PART 4 — Call
# ============================================================
find_fixed_point(method, funktion, x_0, toleranz, iterationen)
