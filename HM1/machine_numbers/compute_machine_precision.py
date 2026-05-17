# ============================================================
# TOPIC: Maschinenzahlen — Maschinengenauigkeit eps
# DESCRIPTION:
# Berechnet die Maschinengenauigkeit eps = (B/2) · B^(-n) eines
# Gleitkommasystems mit Basis B und n Mantissenstellen.
# USE WHEN:
# Wenn die Auflösungsgrenze "1 + eps != 1" eines Zahlenformats
# bestimmt werden soll.
# EXAMPLE:
# Basis 10, n = 2 Mantissenstellen -> eps = 0.05.
# ============================================================

# ============================================================
# PART 1 — Inputs
# ============================================================
basis            = 10  # Basis B
mantisse_stellen = 2   # Anzahl Mantissenstellen n

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here.

# ============================================================
# PART 3 — Implementation
# ============================================================
def compute_machine_precision(base, m_stellen):
    eps = (base / 2) * base ** (-m_stellen)
    print(f"Maschinengenauigkeit eps: {eps}")
    return eps

# ============================================================
# PART 4 — Call
# ============================================================
compute_machine_precision(basis, mantisse_stellen)
