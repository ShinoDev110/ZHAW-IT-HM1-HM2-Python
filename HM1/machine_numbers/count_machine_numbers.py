# ============================================================
# TOPIC: Maschinenzahlen — Anzahl darstellbarer Maschinenzahlen
# DESCRIPTION:
# Berechnet die Anzahl verschiedener Maschinenzahlen in einem
# Gleitkomma-System M(B, n, e_min, e_max) nach der Formel
# 2 · B^(n-1) · (B - 1) · (e_max - e_min + 1) + 1, hier vereinfacht
# über Basis, Mantissenstellen, Exponentenstellen und Vorzeichen-Bit.
# USE WHEN:
# Wenn die Anzahl darstellbarer Maschinenzahlen eines hypothetischen
# Gleitkommaformats geschätzt werden soll.
# EXAMPLE:
# Basis 2, 15 Mantissenstellen, 5 Exponentenstellen, mit Vorzeichen.
# ============================================================

# ============================================================
# PART 1 — Inputs
# ============================================================
basis            = 2   # Basis B
mantisse_stellen = 15  # Anzahl Stellen für Mantisse
exponent_stellen = 5   # Anzahl Stellen für Exponent
vorzeichen       = 1   # Vorzeichen-Bit vorhanden? (1 = ja, 0 = nein)

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here.

# ============================================================
# PART 3 — Implementation
# ============================================================
def count_machine_numbers(base, mantisse, exponent, vorzeichen_exp):
    anzahl = base ** mantisse * (base ** (exponent + vorzeichen_exp) - 1) + 1
    print(f"Anzahl verschiedener Maschinenzahlen: {anzahl}")
    return anzahl

# ============================================================
# PART 4 — Call
# ============================================================
count_machine_numbers(basis, mantisse_stellen, exponent_stellen, vorzeichen)
