# ============================================================
# TOPIC: Machine numbers — count of representable machine numbers
# DESCRIPTION:
# Computes the number of distinct machine numbers in a
# floating-point system M(B, n, e_min, e_max) per the formula
# 2 · B^(n-1) · (B - 1) · (e_max - e_min + 1) + 1, simplified here
# via base, mantissa digits, exponent digits, and sign bit.
# USE WHEN:
# When the count of representable machine numbers of a hypothetical
# floating-point format should be estimated.
# EXAMPLE:
# Base 2, 15 mantissa digits, 5 exponent digits, with sign.
# ============================================================

# ============================================================
# PART 1 — Inputs
# ============================================================
base             = 2   # base B
mantissa_digits  = 15  # number of digits for mantissa
exponent_digits  = 5   # number of digits for exponent
sign             = 1   # sign bit present? (1 = yes, 0 = no)

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here.

# ============================================================
# PART 3 — Implementation
# ============================================================
def count_machine_numbers(base, mantissa, exponent, sign_exp):
    count = base ** mantissa * (base ** (exponent + sign_exp) - 1) + 1
    print(f"Number of distinct machine numbers: {count}")
    return count

# ============================================================
# PART 4 — Call
# ============================================================
count_machine_numbers(base, mantissa_digits, exponent_digits, sign)
