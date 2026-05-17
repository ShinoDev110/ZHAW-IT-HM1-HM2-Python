# ============================================================
# TOPIC: Maschinenzahlen — Auflösung von (1 + 10^-n) testen
# DESCRIPTION:
# Berechnet 1 + 10^-n für n = 1..N und beobachtet, ab welcher Stelle
# das Ergebnis in Gleitkommazahlen wieder zu 1 wird. Zeigt, wo der
# Übergang zur Maschinengenauigkeit liegt.
# USE WHEN:
# Wenn die effektive Maschinengenauigkeit (resp. der Verlust kleiner
# Beiträge) empirisch sichtbar gemacht werden soll.
# EXAMPLE:
# n = 1..60 -> (1 + 10^-n) und Vergleich mit 1.
# ============================================================

# ============================================================
# PART 1 — Inputs
# ============================================================
anzahl_stellen = 18  # max. untersuchte Nachkommastelle n

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here.

# ============================================================
# PART 3 — Implementation
# ============================================================
def test_precision_loss_at_1_plus_eps(anzahl_stellen):
    print(f"{'n':>3} | {'1 + 10^-n':>26} | {'== 1?':>6}")
    print("-" * 45)
    for n in range(1, anzahl_stellen + 1):
        eps = 1.0 / (10 ** n)
        val = 1.0 + eps
        print(f"{n:>3} | {val:>26.20f} | {val == 1.0!s:>6}")

# ============================================================
# PART 4 — Call
# ============================================================
test_precision_loss_at_1_plus_eps(anzahl_stellen)
