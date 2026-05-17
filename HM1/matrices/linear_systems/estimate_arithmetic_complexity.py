# ============================================================
# TOPIC: Lineare Systeme — geschätzter Aufwand (Mult/Div) klassischer Algos
# DESCRIPTION:
# Liefert Richtwerte für die Anzahl arithmetischer Operationen (Mult/Div):
#   Gauss-Elimination:           ~ (2/3) n^3
#   Vorwärts- + Rückwärtseinsetzen: ~ 2 n^2
# als kleine Merkformeln, je für eine Liste von Systemgrössen n.
# USE WHEN:
# Wenn zur Abschätzung gezeigt werden soll, wie sich der Rechenaufwand
# mit wachsendem n entwickelt.
# EXAMPLE:
# n = 10, 100, 1000, 10000.
# ============================================================

# ============================================================
# PART 1 — Inputs
# ============================================================
groessen = [10, 100, 1000, 10_000]

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here. Beide Formeln werden für jede Grösse ausgegeben.

# ============================================================
# PART 3 — Implementation
# ============================================================
def _aufwand_gauss(n):
    return (2.0 / 3.0) * (n ** 3)

def _aufwand_einsetzen(n):
    return 2.0 * (n ** 2)

def estimate_arithmetic_complexity(groessen):
    print(f"{'n':>8} | {'Gauss (~2/3 n^3)':>22} | {'Einsetzen (~2 n^2)':>22}")
    print("-" * 60)
    for n in groessen:
        print(f"{n:>8} | {_aufwand_gauss(n):>22.4e} | {_aufwand_einsetzen(n):>22.4e}")

# ============================================================
# PART 4 — Call
# ============================================================
estimate_arithmetic_complexity(groessen)
