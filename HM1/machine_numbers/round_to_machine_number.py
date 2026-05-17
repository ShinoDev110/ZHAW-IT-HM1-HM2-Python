# ============================================================
# TOPIC: Maschinenzahlen — Dezimalzahl in IEEE-artige Bitdarstellung
# DESCRIPTION:
# Wandelt eine reelle Zahl in das Format |Vorzeichen|Exponent (biased)|
# Mantisse| mit wählbarer Anzahl Exponenten- und Mantissenbits um.
# Subnormale / Inf werden nicht behandelt.
# USE WHEN:
# Wenn aus einer Dezimalzahl die binäre Maschinenzahl-Darstellung
# nachgerechnet werden soll (z.B. fiktives 12-Bit-Format).
# EXAMPLE:
# value = 2.0, 4 Exponentenbits, 7 Mantissenbits -> Bitstring.
# ============================================================

import math

# ============================================================
# PART 1 — Inputs
# ============================================================
value  = 2.0  # zu konvertierende Dezimalzahl
e_bits = 4    # Anzahl Exponentenbits (biased)
m_bits = 7    # Anzahl Mantissenbits

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here.

# ============================================================
# PART 3 — Implementation
# ============================================================
def round_to_machine_number(val, exp_bits, man_bits):
    if val == 0:
        sign = "0"
        exponent = "0" * exp_bits
        mantissa = "0" * man_bits
        result = f"|{sign}|{exponent}|{mantissa}|"
        print(result)
        return sign, exponent, mantissa, result

    sign = "1" if val < 0 else "0"
    bias = (1 << (exp_bits - 1)) - 1

    m, e = math.frexp(abs(val))

    E = e + bias
    if E <= 0 or E >= (1 << exp_bits) - 1:
        raise ValueError("Underflow/Overflow for this format (no subnormals/inf handled).")

    exponent = format(E, f"0{exp_bits}b")

    frac = m
    bits = []
    for _ in range(man_bits):
        frac *= 2
        if frac >= 1:
            bits.append("1")
            frac -= 1
        else:
            bits.append("0")

    mantissa = "".join(bits)

    result = (
        f"Vorzeichen: {sign}\n"
        f"Exponent (biased): {exponent}\n"
        f"Mantisse: 0.{mantissa}\n"
        f"|{sign}|{exponent}|{mantissa}|"
    )
    print(result)
    return sign, exponent, mantissa, result

# ============================================================
# PART 4 — Call
# ============================================================
round_to_machine_number(value, e_bits, m_bits)
