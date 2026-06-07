# ============================================================
# TOPIC: Machine numbers — machine precision eps
# DESCRIPTION:
# Computes the machine precision eps = (B/2) · B^(-n) of a
# floating-point system with base B and n mantissa digits.
# USE WHEN:
# When the resolution limit "1 + eps != 1" of a number format
# should be determined.
# EXAMPLE:
# Base 10, n = 2 mantissa digits -> eps = 0.05.
# ============================================================

# ============================================================
# PART 1 — Inputs
# ============================================================
base             = 10  # base B
mantissa_digits  = 2   # number of mantissa digits n

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here.

# ============================================================
# PART 3 — Implementation
# ============================================================
def compute_machine_precision(base, m_digits):
    eps = (base / 2) * base ** (-m_digits)
    print(f"Machine precision eps: {eps}")
    return eps

# ============================================================
# PART 4 — Call
# ============================================================
compute_machine_precision(base, mantissa_digits)
