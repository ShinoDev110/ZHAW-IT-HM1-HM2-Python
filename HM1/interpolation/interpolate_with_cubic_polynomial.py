# ============================================================
# TOPIC: Interpolation — kubische Polynom-Interpolation (4 Stützstellen)
# DESCRIPTION:
# Bestimmt die Koeffizienten eines kubischen Polynoms
#   p(t) = a t^3 + b t^2 + c t + d
# mit t = jahr - referenzjahr, das genau 4 gegebene (jahr, messwert)-
# Paare exakt interpoliert (Vandermonde-System). Anschliessend
# Auswertung an einer Zielstelle.
# USE WHEN:
# Wenn aus genau 4 Datenpunkten ein eindeutiges Interpolationspolynom
# 3. Grades konstruiert und ausgewertet werden soll.
# EXAMPLE:
# Jahre [1997, 1999, 2006, 2010], Werte [150, 104, 172, 152],
# Referenz 1997, Auswertung bei 2003.
# ============================================================

import numpy as np

# ============================================================
# PART 1 — Inputs
# ============================================================
jahreswerte    = np.array([1997.0, 1999.0, 2006.0, 2010.0])
messwerte      = np.array([ 150.0,  104.0,  172.0,  152.0])
referenzjahr   = 1997.0
auswertestelle = 2003.0

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here (Vandermonde-System für 4 Punkte).

# ============================================================
# PART 3 — Implementation
# ============================================================
def _berechne_koeffizienten(jahreswerte, messwerte, referenzjahr):
    t = jahreswerte - referenzjahr
    vandermonde = np.vstack([t ** 3, t ** 2, t, np.ones_like(t)]).T
    return np.linalg.solve(vandermonde, messwerte)

def _auswerten(jahr, koeffizienten, referenzjahr):
    a, b, c, d = koeffizienten
    t = np.array(jahr, dtype=float) - referenzjahr
    return ((a * t + b) * t + c) * t + d

def interpolate_with_cubic_polynomial(jahreswerte, messwerte, referenzjahr, auswertestelle):
    koeff = _berechne_koeffizienten(jahreswerte, messwerte, referenzjahr)
    a, b, c, d = koeff
    print(f"p(t) = {a} t^3 + {b} t^2 + {c} t + {d}")
    print(f"mit t = jahr - {referenzjahr}")
    wert = float(_auswerten(auswertestelle, koeff, referenzjahr))
    print(f"\np({auswertestelle}) = {wert}")
    return koeff, wert

# ============================================================
# PART 4 — Call
# ============================================================
interpolate_with_cubic_polynomial(jahreswerte, messwerte, referenzjahr, auswertestelle)
