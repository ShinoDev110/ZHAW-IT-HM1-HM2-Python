# ============================================================
# TOPIC: Numerische Differentiation — Differenzenquotient + h-Extrapolation
# DESCRIPTION:
# Approximiert f'(x0) mit dem Vorwärts-, Rückwärts- oder zentralen
# Differenzenquotienten für eine Folge halbierter Schrittweiten h_j = h0/2^j
# und verbessert das Resultat mit Richardson-/Romberg-Extrapolation
# ("h-Algorithmus"): T_{j,k} = (r^k T_{j+1,k-1} - T_{j,k-1})/(r^k - 1) mit
# r = 2 (Vorwärts/Rückwärts, Fehler O(h)) bzw. r = 4 (zentral, Fehler O(h^2)).
# USE WHEN:
# Wenn eine Ableitung numerisch genau aus Funktionswerten bestimmt werden soll
# (typische Aufgabe: "Vorwärtsdifferenz und Extrapolation für h = 2, 1, 0.5").
# EXAMPLE:
# v(t) = 2000·ln(10000/(10000-100t)) - 9.8t, a(30) = v'(30): h0 = 2, 2 Stufen
# -> T(0,2) ~= 18.7714.
# ============================================================

import numpy as np

# ============================================================
# PART 1 — Inputs
# ============================================================
def f(t):
    return 2000.0 * np.log(10000.0 / (10000.0 - 100.0 * t)) - 9.8 * t   # v(t)

x0 = 30.0      # Stelle, an der f'(x0) gesucht ist
h0 = 2.0       # grösste Schrittweite h_0
m  = 2         # Anzahl Extrapolationsstufen -> h_j = h0/2^j, j = 0..m

def df_exact(t):
    return 2000.0 * 100.0 / (10000.0 - 100.0 * t) - 9.8   # v'(t), nur zum Vergleich

use_exact = True   # df_exact für Fehlerausgabe verwenden?

# ============================================================
# PART 2 — Method selection
# ============================================================
# kind:
#   "forward"  -> (f(x+h) - f(x)) / h          (Fehler O(h),  r = 2)
#   "backward" -> (f(x) - f(x-h)) / h          (Fehler O(h),  r = 2)
#   "central"  -> (f(x+h) - f(x-h)) / (2h)     (Fehler O(h^2), r = 4)
kind = "forward"

# ============================================================
# PART 3 — Implementation
# ============================================================
def _diff_quotient(f, x0, h, kind):
    if kind == "forward":
        return (f(x0 + h) - f(x0)) / h
    if kind == "backward":
        return (f(x0) - f(x0 - h)) / h
    if kind == "central":
        return (f(x0 + h) - f(x0 - h)) / (2.0 * h)
    raise ValueError(f"Unbekannte kind: {kind!r}")

def differentiate_numerically_with_extrapolation(f, x0, h0, m, kind,
                                                 df_exact=None, use_exact=False):
    r = 4 if kind == "central" else 2
    T = np.zeros((m + 1, m + 1))

    for j in range(m + 1):
        h_j = h0 / 2**j
        T[j, 0] = _diff_quotient(f, x0, h_j, kind)

    for k in range(1, m + 1):
        for j in range(m + 1 - k):
            T[j, k] = (r**k * T[j + 1, k - 1] - T[j, k - 1]) / (r**k - 1)

    print("============================================================")
    print(f"Numerische Differentiation — {kind} + Extrapolation (r = {r})")
    print("============================================================")
    print(f"f'({x0}) gesucht, h_j = {h0}/2^j\n")
    print("Extrapolationsschema T_{j,k} (Zeile j = Schrittweite, Spalte k = Stufe):")
    for j in range(m + 1):
        row = [f"{T[j, k]:.4f}" for k in range(m + 1 - j)]
        print(f"  h={h0/2**j:<6g} " + "   ".join(row))

    best = T[0, m]
    print(f"\nBester Wert T(0,{m}) = {best:.6f}")
    if use_exact and df_exact is not None:
        exact = df_exact(x0)
        print(f"Exakt    f'({x0}) = {exact:.6f}")
        print(f"|Fehler| = {abs(best - exact):.3e}")
    return best

# ============================================================
# PART 4 — Call
# ============================================================
differentiate_numerically_with_extrapolation(f, x0, h0, m, kind, df_exact, use_exact)
