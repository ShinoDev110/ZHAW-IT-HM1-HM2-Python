# ============================================================
# TOPIC: Machine numbers — testing resolution of (1 + 10^-n)
# DESCRIPTION:
# Computes 1 + 10^-n for n = 1..N and observes at which point
# the result in floating-point numbers becomes 1 again. Shows
# where the transition to machine precision lies.
# USE WHEN:
# When the effective machine precision (resp. the loss of small
# contributions) should be made empirically visible.
# EXAMPLE:
# n = 1..60 -> (1 + 10^-n) and comparison with 1.
# ============================================================

# ============================================================
# PART 1 — Inputs
# ============================================================
digit_count = 18  # max. investigated decimal place n

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here.

# ============================================================
# PART 3 — Implementation
# ============================================================
def test_precision_loss_at_1_plus_eps(digit_count):
    print(f"{'n':>3} | {'1 + 10^-n':>26} | {'== 1?':>6}")
    print("-" * 45)
    for n in range(1, digit_count + 1):
        eps = 1.0 / (10 ** n)
        val = 1.0 + eps
        print(f"{n:>3} | {val:>26.20f} | {val == 1.0!s:>6}")

# ============================================================
# PART 4 — Call
# ============================================================
test_precision_loss_at_1_plus_eps(digit_count)
