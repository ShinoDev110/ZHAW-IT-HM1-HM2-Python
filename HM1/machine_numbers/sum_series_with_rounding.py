# ============================================================
# TOPIC: Machine numbers — summation in both directions with rounding
# DESCRIPTION:
# Sums the series Σ 1/i^2 once upward (i = 1..n) and once
# downward (i = n..1), rounding each partial sum to a fixed
# number of decimal places. Shows how summation order affects
# the result with floating-point numbers.
# USE WHEN:
# When the influence of rounding errors and summation order
# should be demonstrated (classic script example).
# EXAMPLE:
# n = 300 terms, rounded to 5 decimal places.
# ============================================================

# ============================================================
# PART 1 — Inputs
# ============================================================
term_count       = 300  # n = upper limit
rounding_digits  = 5    # decimal places per partial sum

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here. Both directions (upward AND downward) are
# always computed and compared.

# ============================================================
# PART 3 — Implementation
# ============================================================
def sum_series_with_rounding(term_count, rounding_digits):
    def series_term(idx):
        return 1.0 / (idx ** 2)

    total = 0.0
    for i in range(1, term_count + 1):
        total += round(series_term(i), rounding_digits)
        total = round(total, rounding_digits)
    sum_upward = total

    total = 0.0
    for i in range(term_count, 0, -1):
        total += round(series_term(i), rounding_digits)
        total = round(total, rounding_digits)
    sum_downward = total

    print(f"Sum upward   (i=1..{term_count}):  {sum_upward}")
    print(f"Sum downward (i={term_count}..1):  {sum_downward}")
    print(f"Difference:                         {sum_upward - sum_downward}")
    return sum_upward, sum_downward

# ============================================================
# PART 4 — Call
# ============================================================
sum_series_with_rounding(term_count, rounding_digits)
