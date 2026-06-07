# ============================================================
# TOPIC: Machine numbers — decimal number to IEEE-like bit representation
# DESCRIPTION:
# Converts a real number into the format |sign|exponent (biased)|
# mantissa| with a selectable number of exponent and mantissa bits.
# Subnormals / Inf are not handled.
# USE WHEN:
# When the binary machine-number representation of a decimal number
# should be recalculated (e.g. a fictitious 12-bit format).
# EXAMPLE:
# value = 2.0, 4 exponent bits, 7 mantissa bits -> bit string.
# ============================================================

import math

# ============================================================
# PART 1 — Inputs
# ============================================================
value  = 2.0  # decimal number to convert
e_bits = 4    # number of exponent bits (biased)
m_bits = 7    # number of mantissa bits

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
        f"Sign: {sign}\n"
        f"Exponent (biased): {exponent}\n"
        f"Mantissa: 0.{mantissa}\n"
        f"|{sign}|{exponent}|{mantissa}|"
    )
    print(result)
    return sign, exponent, mantissa, result

# ============================================================
# PART 4 — Call
# ============================================================
round_to_machine_number(value, e_bits, m_bits)
