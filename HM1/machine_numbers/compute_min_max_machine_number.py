# ============================================================
# TOPIC: Maschinenzahlen — kleinste / grösste positive Maschinenzahl
# DESCRIPTION:
# Berechnet x_min und x_max einer Maschinenzahlenmenge M(B, n, e_min, e_max).
# Druckt zusätzlich die Zwischenrechnung mit Mantissen-/Exponentenstellen.
# USE WHEN:
# Wenn für ein gegebenes Gleitkommaformat die kleinste und grösste
# darstellbare positive Zahl bestimmt werden soll.
# EXAMPLE:
# Basis = 2, n = 20 Mantissenstellen, e_max-Stellen = 4 -> x_min und x_max.
# ============================================================

# ============================================================
# PART 1 — Inputs
# ============================================================
basis            = 2    # Basis B des Zahlensystems
mantisse_stellen = 20   # Anzahl Mantissenstellen n
exponent_stellen = 4    # Anzahl Stellen für den Exponenten
basis_exponent   = 2    # Basis für den Exponenten (üblicherweise = Basis)
debug            = True # detaillierte Zwischenausgaben

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here. Beide Werte (x_min, x_max) werden immer berechnet.

# ============================================================
# PART 3 — Implementation
# ============================================================
def _get_min(base_exponent, exp_stellen, debug=False):
    max_zahl_im_system = base_exponent - 1
    smallest_binary_number = "-" + f"{max_zahl_im_system}" * exp_stellen
    if debug:
        print(f"Kleinste Zahl im {base_exponent}-System mit {exp_stellen} Stellen: 0.1 * {base_exponent}^{smallest_binary_number}")
    return int(smallest_binary_number, base_exponent)

def _get_max(base_exponent, exp_stellen, debug=False):
    max_zahl_im_system = base_exponent - 1
    largest_binary_number = f"{max_zahl_im_system}" * exp_stellen
    if debug:
        print(f"Max Zahl im {base_exponent}-System mit {exp_stellen} Stellen: {base_exponent}^{largest_binary_number}")
    return int(largest_binary_number, base_exponent)

def compute_min_max_machine_number(base, base_exponent, m_stellen, exp_stellen, debug=False):
    l_min = _get_min(base_exponent, exp_stellen, debug)
    if debug:
        print(f"B^(e_min - 1): {base}^({l_min} - 1) = {base}^({l_min - 1})")
        print(f"Kleinste positive Maschinenzahl: {base ** (l_min - 1)}\n")
    x_min = base ** l_min - 1

    l_max = _get_max(base_exponent, exp_stellen, debug)
    if debug:
        print(f"B^e_max - B^(e_max - n): {base}^{l_max} - {base}^({l_max} - {m_stellen}) = {base}^{l_max} - {base}^{l_max - m_stellen}")
        print(f"Max Positive Maschinenzahl: {(base ** l_max) - (base ** (l_max - m_stellen))}")
    x_max = (base ** l_max) - (base ** (l_max - m_stellen))

    return x_min, x_max

# ============================================================
# PART 4 — Call
# ============================================================
compute_min_max_machine_number(basis, basis_exponent, mantisse_stellen, exponent_stellen, debug)
