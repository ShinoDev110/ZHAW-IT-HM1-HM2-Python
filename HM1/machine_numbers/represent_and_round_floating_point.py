# ============================================================
# TOPIC: Machine numbers — rounding vs. truncation + cancellation
# DESCRIPTION:
# Represents a real number as a normalised floating-point number 0.m1..mn · B^e
# with n mantissa digits, once by truncation and once by rounding,
# and gives the absolute and relative error in each case. Optionally the
# difference p - q of two numbers is formed to make cancellation visible
# (relative error of the difference for both rounding types).
# USE WHEN:
# When the representation error of a number in finite mantissa length is
# sought, or to show that rounding is better for subtraction.
# EXAMPLE:
# p = 9890.9, q = 9890.1, base 10, n = 4 -> p-q (truncate) = 0 (100% error),
# p-q (round) = 1 (25% error). Alternative: x = sqrt(3), base 2, n = 5.
# ============================================================

from math import floor, log

# ============================================================
# PART 1 — Inputs
# ============================================================
value_1 = 9890.9   # first number p
value_2 = 9890.1   # second number q (only for mode = "pair")
base    = 10       # base B of the floating-point system (2 or 10 typical)
n_mant  = 4        # number of mantissa digits n of the normalised representation

# Alternative example (exercise 2014.1b): sqrt(3) in 5 binary digits
# from math import sqrt; value_1 = sqrt(3); base = 2; n_mant = 5; mode = "single"

# ============================================================
# PART 2 — Method selection
# ============================================================
# mode:
#   "single" -> represent value_1 only (rounding vs. truncation + error)
#   "pair"   -> additionally difference value_1 - value_2 (show cancellation)
mode = "pair"

# ============================================================
# PART 3 — Implementation
# ============================================================
def _to_base_digits(m_int, base, n):
    # n digits (most significant first) of m_int in the given base
    digits = []
    x = m_int
    for _ in range(n):
        digits.append(x % base)
        x //= base
    digits.reverse()
    return digits

def _digit_str(digits):
    return "".join(str(d) for d in digits)

def _normalize(value, base):
    # value = sign * mantissa * base^e  with  1/base <= mantissa < 1
    if value == 0:
        return 0, 0.0, 0
    sign = -1 if value < 0 else 1
    a = abs(value)
    e = floor(log(a) / log(base)) + 1
    mant = a / base ** e
    while mant >= 1.0:            # rounding guard at the boundaries
        mant /= base; e += 1
    while mant < 1.0 / base:
        mant *= base; e -= 1
    return sign, mant, e

def _represent(value, base, n, rounding):
    # returns (represented value, sign, mantissa digits, exponent)
    sign, mant, e = _normalize(value, base)
    if sign == 0:
        return 0.0, 0, [0] * n, 0
    scaled = mant * base ** n
    if rounding == "round":
        m_int = floor(scaled + 0.5)
    else:  # "truncate"
        m_int = floor(scaled)
    if m_int >= base ** n:        # carry, e.g. 0.999 -> 1.000
        m_int //= base
        e += 1
    rep_value = sign * m_int * base ** (e - n)
    digits = _to_base_digits(m_int, base, n)
    return rep_value, sign, digits, e

def _print_value_report(name, value, base, n):
    print(f"--- {name} = {value} (exact) ---")
    results = {}
    for rounding in ("truncate", "round"):
        rep, sign, digits, e = _represent(value, base, n, rounding)
        abs_err = abs(rep - value)
        rel_err = abs_err / abs(value) if value != 0 else 0.0
        sign_str = "-" if sign < 0 else ""
        label = "Truncate" if rounding == "truncate" else "Round   "
        print(f"  {label}: {sign_str}0.{_digit_str(digits)} · {base}^{e} = {rep}")
        print(f"           abs. error = {abs_err:.6g}, rel. error = {rel_err:.6g}")
        results[rounding] = rep
    print()
    return results

def represent_and_round_floating_point(mode, value_1, value_2, base, n):
    print("============================================================")
    print(f"Normalised representation 0.m1..m{n} · {base}^e  (n = {n}, base {base})")
    print("============================================================\n")

    rep1 = _print_value_report("p (value_1)", value_1, base, n)
    if mode == "single":
        return rep1

    if mode != "pair":
        raise ValueError(f"Unknown mode: {mode!r}")

    rep2 = _print_value_report("q (value_2)", value_2, base, n)

    exact_diff = value_1 - value_2
    print("--- Difference p - q (cancellation) ---")
    print(f"  exact:        p - q = {exact_diff}")
    best = None
    for rounding, label in (("truncate", "Truncate"), ("round", "Round")):
        diff = rep1[rounding] - rep2[rounding]
        rel = abs(diff - exact_diff) / abs(exact_diff) if exact_diff != 0 else float("inf")
        print(f"  {label:>8}: p - q = {diff}, rel. error = {rel:.6g}")
        if best is None or rel < best[1]:
            best = (label, rel)
    print(f"\nConclusion: '{best[0]}' gives the smaller relative error of the difference")
    print("(Rounding is better here, as truncation leads to stronger cancellation).")
    return rep1, rep2

# ============================================================
# PART 4 — Call
# ============================================================
represent_and_round_floating_point(mode, value_1, value_2, base, n_mant)
