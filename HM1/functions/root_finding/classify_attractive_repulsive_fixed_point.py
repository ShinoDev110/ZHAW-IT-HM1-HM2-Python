# ============================================================
# TOPIC: Fixpunkt-Analyse — anziehend vs. abstossend
# DESCRIPTION:
# Klassifiziert einen Fixpunkt x̄ von F(x) als anziehend / abstossend
# anhand von |F'(x̄)|: wahlweise an einer konkreten Stelle oder als
# Schranke auf einem Intervall.
# USE WHEN:
# Wenn beurteilt werden soll, ob eine Fixpunktiteration in der Nähe
# eines Fixpunkts konvergiert (|F'(x̄)| < 1) oder divergiert (> 1).
# EXAMPLE:
# F(x) = (230x^4 + 18x^3 + 9x^2 - 9)/221 am Fixpunkt 0; oder
# F(x) = exp(x) - exp(1) auf dem Intervall [-3, -2].
# ============================================================

from sympy import diff, exp, symbols

# ============================================================
# PART 1 — Inputs
# ============================================================
x = symbols("x")

# Funktion und Fixpunkt für Methode "point"
funktion_point = (230 * x**4 + 18 * x**3 + 9 * x**2 - 9) / 221
fixpunkt       = 0

# Funktion und Intervall für Methode "interval"
funktion_interval = exp(x) - exp(1)
intervall_start   = -3
intervall_ende    = -2

# ============================================================
# PART 2 — Method selection
# ============================================================
# method:
#   "point"    -> |F'(x̄)| an einem konkreten bekannten Fixpunkt
#   "interval" -> Verhalten auf einem Intervall [a, b]
method = "point"

# ============================================================
# PART 3 — Implementation
# ============================================================
def _classify_at_point(funktion, var, fixpunkt):
    ableitung = diff(funktion, var)
    wert = ableitung.subs(var, fixpunkt)
    print(f"F'(x) = {ableitung}, F'({fixpunkt}) = {wert.evalf()}\n")
    if abs(wert) < 1:
        print(f"Anziehender Fixpunkt, weil |F'(x̄)| = {abs(wert.evalf())} < 1")
    else:
        print(f"Abstossender Fixpunkt, weil |F'(x̄)| = {abs(wert.evalf())} > 1")

def _classify_on_interval(funktion, var, a, b):
    ableitung = diff(funktion, var)
    fa = ableitung.subs(var, a)
    fb = ableitung.subs(var, b).evalf()
    if fa < 1 and fb < 1:
        print(f"Anziehender Fixpunkt im Intervall [{a}, {b}]\n")
        print(f"Beweis: f'({a}) = {fa} <= f'(x̄) <= f'({b}) = {fb} < 1")
    elif 1 < fa and 1 < fb:
        print(f"Abstossender Fixpunkt im Intervall [{a}, {b}]\n")
        print(f"Beweis: 1 < f'({a}) = {fa} <= f'(x̄) <= f'({b}) = {fb}")
    else:
        print(f"Uneindeutig: f'({a}) = {fa}, f'({b}) = {fb} (Vorzeichen / Schranken inkonsistent)")

def classify_attractive_repulsive_fixed_point(method, x_sym,
                                              funktion_point, fixpunkt,
                                              funktion_interval, a, b):
    if method == "point":
        _classify_at_point(funktion_point, x_sym, fixpunkt)
    elif method == "interval":
        _classify_on_interval(funktion_interval, x_sym, a, b)
    else:
        raise ValueError(f"Unbekannte Methode: {method!r}")

# ============================================================
# PART 4 — Call
# ============================================================
classify_attractive_repulsive_fixed_point(method, x,
                                          funktion_point, fixpunkt,
                                          funktion_interval, intervall_start, intervall_ende)
