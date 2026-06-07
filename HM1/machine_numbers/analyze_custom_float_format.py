# ============================================================
# TOPIC: Machine numbers — complete analysis of a bit format with bias
# DESCRIPTION:
# Analyses a freely defined floating-point format (sign | exponent
# with bias | mantissa) and delivers in a report: machine precision,
# largest/smallest exponent (taking bias into account), smallest/
# largest positive machine number (value + bit pattern) and the count of
# distinct machine numbers (incl. 0). Assumption: normalised mantissa
# 0.m1..mn with m1 != 0 (without hidden bit) or 1.m1..mn (with hidden bit).
# USE WHEN:
# When a custom standard (e.g. "IDDD-643") should be fully characterised
# rather than computing the individual values separately.
# EXAMPLE:
# IDDD-643: 1 sign bit, 5 exponent bits (bias 15), 10 mantissa bits,
# no hidden bit -> eps = 2^-10, e in [-15, 16], x_min = 2^-16,
# x_max = 65472, 32769 distinct machine numbers.
# ============================================================

# ============================================================
# PART 1 — Inputs
# ============================================================
base          = 2     # base B (2 = binary)
sign_bit      = True  # own sign bit present?
exponent_bits = 5     # number of exponent digits
bias          = 15    # bias of the exponent (stored exp - bias = actual exp)
mantissa_bits = 10    # number of mantissa digits n
hidden_bit    = False # True = 1.m1..mn (IEEE-like), False = 0.m1..mn with m1!=0

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here. All characteristic values are always computed.
# (Assumption: no reserved exponents for Inf/NaN — full range.)

# ============================================================
# PART 3 — Implementation
# ============================================================
def _digits(value, base, width):
    # value as digit sequence in the given base with fixed width (most significant first)
    ds = []
    x = value
    for _ in range(width):
        ds.append(x % base)
        x //= base
    ds.reverse()
    return "".join(str(d) for d in ds)

def analyze_custom_float_format(base, sign_bit, exp_bits, bias, m_bits, hidden_bit):
    print("============================================================")
    print("Analysis of a custom floating-point format")
    print("============================================================")
    total_bits = (1 if sign_bit else 0) + exp_bits + m_bits
    print(f"Format: {'1 sign | ' if sign_bit else ''}{exp_bits} exponent (bias {bias}) "
          f"| {m_bits} mantissa  ({total_bits} bits, base {base})")
    print(f"Mantissa: {'1.m1..mn (hidden bit)' if hidden_bit else '0.m1..mn with m1 != 0'}\n")

    # --- Exponent range (stored exponent 0 .. base^exp_bits - 1) ---
    e_min = 0 - bias
    e_max = (base ** exp_bits - 1) - bias
    n_exp = e_max - e_min + 1
    print(f"Smallest exponent e_min = 0 - {bias}                 = {e_min}")
    print(f"Largest  exponent e_max = ({base}^{exp_bits} - 1) - {bias} = {e_max}")
    print(f"Number of exponents     = {n_exp}\n")

    # --- Machine precision ---
    if hidden_bit:
        eps = base ** (-m_bits)
        print(f"Machine precision eps = B^(-n) = {base}^(-{m_bits}) = {eps:.6g}\n")
    else:
        eps = (base / 2) * base ** (-m_bits)
        print(f"Machine precision eps = (B/2)·B^(-n) = {eps:.6g}\n")

    # --- Smallest / largest positive machine number ---
    if hidden_bit:
        x_min = base ** e_min                                   # mantissa 1.0
        x_max = (base - base ** (1 - m_bits)) * base ** e_max   # mantissa (base - ulp)
        mant_min_digits = "0" * m_bits
        mant_max_digits = str(base - 1) * m_bits
    else:
        x_min = base ** (e_min - 1)                             # mantissa 0.10..0
        x_max = (1 - base ** (-m_bits)) * base ** e_max         # mantissa 0.11..1
        mant_min_digits = "1" + "0" * (m_bits - 1)
        mant_max_digits = "1" * m_bits

    sign_str = "0|" if sign_bit else ""
    exp_min_field = _digits(0, base, exp_bits)
    exp_max_field = _digits(base ** exp_bits - 1, base, exp_bits)
    print("Smallest positive machine number:")
    print(f"   Bit pattern |{sign_str}{exp_min_field}|{mant_min_digits}|  = {x_min:.6g}")
    print("Largest positive machine number:")
    print(f"   Bit pattern |{sign_str}{exp_max_field}|{mant_max_digits}|  = {x_max:.6g}\n")

    # --- Count of distinct machine numbers (incl. 0) ---
    if hidden_bit:
        mant_count = base ** m_bits                 # all m1..mn free (leading 1 implicit)
    else:
        mant_count = (base - 1) * base ** (m_bits - 1)   # m1 != 0
    sign_factor = 2 if sign_bit else 1
    total = sign_factor * mant_count * n_exp + 1
    print(f"Number of mantissas (normalised) = {mant_count}")
    print(f"Count of distinct machine numbers (incl. 0):")
    print(f"   {sign_factor} · {mant_count} · {n_exp} + 1 = {total}")
    return {"eps": eps, "e_min": e_min, "e_max": e_max,
            "x_min": x_min, "x_max": x_max, "count": total}

# ============================================================
# PART 4 — Call
# ============================================================
analyze_custom_float_format(base, sign_bit, exponent_bits, bias,
                            mantissa_bits, hidden_bit)
