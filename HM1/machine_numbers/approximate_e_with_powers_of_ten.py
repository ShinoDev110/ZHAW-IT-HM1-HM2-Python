# ============================================================
# TOPIC: Maschinenzahlen — Approximation von e via (1 + 1/10^n)^(10^n)
# DESCRIPTION:
# Berechnet die klassische Approximation e ~= (1 + 1/10^n)^(10^n) für
# wachsendes n und vergleicht mit math.e (absolute und relative
# Abweichung). Demonstriert, dass die Approximation ab einem gewissen
# n wieder schlechter wird (Rundungsfehler!).
# USE WHEN:
# Wenn der Effekt grosser Exponenten und kleiner Inkremente auf
# Gleitkommazahlen sichtbar gemacht werden soll.
# EXAMPLE:
# n = 1, 2, ..., 16 -> Annäherung an e und ab wann sie wieder driftet.
# ============================================================

import math

# ============================================================
# PART 1 — Inputs
# ============================================================
n_max = 16   # Exponentenbereich n = 1..n_max

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here. Es wird immer ein Sweep von n=1 bis n_max ausgeführt.

# ============================================================
# PART 3 — Implementation
# ============================================================
def _approximate_e(n):
    nenner = 10 ** n
    return (1.0 + 1.0 / nenner) ** nenner

def _abs_dev(x, ref):
    return abs(x - ref)

def _rel_dev(x, ref):
    return abs(x - ref) / abs(ref)

def approximate_e_with_powers_of_ten(n_max):
    print(f"Referenzwert math.e = {math.e}")
    print(f"{'n':>3} | {'(1+1/10^n)^(10^n)':>26} | {'abs dev':>14} | {'rel dev':>14}")
    print("-" * 70)
    for n in range(1, n_max + 1):
        e_approx = _approximate_e(n)
        print(f"{n:>3} | {e_approx:>26.16f} | {_abs_dev(e_approx, math.e):>14.4e} | {_rel_dev(e_approx, math.e):>14.4e}")

# ============================================================
# PART 4 — Call
# ============================================================
approximate_e_with_powers_of_ten(n_max)
