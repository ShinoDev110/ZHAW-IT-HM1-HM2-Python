# ============================================================
# TOPIC: Machine numbers — smallest / largest positive machine number
# DESCRIPTION:
# Computes x_min and x_max of a machine number set M(B, n, e_min, e_max).
# Also prints intermediate results with mantissa/exponent digits.
# USE WHEN:
# When the smallest and largest representable positive number of a
# given floating-point format should be determined.
# EXAMPLE:
# Base = 2, n = 20 mantissa digits, e_max digits = 4 -> x_min and x_max.
# ============================================================

# ============================================================
# PART 1 — Inputs
# ============================================================
base             = 2    # base B of the number system
mantissa_digits  = 20   # number of mantissa digits n
exponent_digits  = 4    # number of digits for the exponent
exponent_base    = 2    # base for the exponent (usually = base)
debug            = True # detailed intermediate output

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here. Both values (x_min, x_max) are always computed.

# ============================================================
# PART 3 — Implementation
# ============================================================
def _get_min(exp_base, exp_digits, debug=False):
    max_digit_in_system = exp_base - 1
    smallest_binary_number = "-" + f"{max_digit_in_system}" * exp_digits
    if debug:
        print(f"Smallest number in the {exp_base}-system with {exp_digits} digits: 0.1 * {exp_base}^{smallest_binary_number}")
    return int(smallest_binary_number, exp_base)

def _get_max(exp_base, exp_digits, debug=False):
    max_digit_in_system = exp_base - 1
    largest_binary_number = f"{max_digit_in_system}" * exp_digits
    if debug:
        print(f"Largest number in the {exp_base}-system with {exp_digits} digits: {exp_base}^{largest_binary_number}")
    return int(largest_binary_number, exp_base)

def compute_min_max_machine_number(base, exp_base, m_digits, exp_digits, debug=False):
    l_min = _get_min(exp_base, exp_digits, debug)
    if debug:
        print(f"B^(e_min - 1): {base}^({l_min} - 1) = {base}^({l_min - 1})")
        print(f"Smallest positive machine number: {base ** (l_min - 1)}\n")
    x_min = base ** l_min - 1

    l_max = _get_max(exp_base, exp_digits, debug)
    if debug:
        print(f"B^e_max - B^(e_max - n): {base}^{l_max} - {base}^({l_max} - {m_digits}) = {base}^{l_max} - {base}^{l_max - m_digits}")
        print(f"Largest positive machine number: {(base ** l_max) - (base ** (l_max - m_digits))}")
    x_max = (base ** l_max) - (base ** (l_max - m_digits))

    return x_min, x_max

# ============================================================
# PART 4 — Call
# ============================================================
compute_min_max_machine_number(base, exponent_base, mantissa_digits, exponent_digits, debug)
