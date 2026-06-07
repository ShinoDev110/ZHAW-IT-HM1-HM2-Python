# ============================================================
# TOPIC: Linear Systems — estimated cost (Mult/Div) of classical algorithms
# DESCRIPTION:
# Provides reference values for the number of arithmetic operations (Mult/Div):
#   Gauss elimination:              ~ (2/3) n^3
#   Forward + back substitution:    ~ 2 n^2
# as small rule-of-thumb formulas, each for a list of system sizes n.
# USE WHEN:
# When an estimate should show how the arithmetic cost grows
# with increasing n.
# EXAMPLE:
# n = 10, 100, 1000, 10000.
# ============================================================

# ============================================================
# PART 1 — Inputs
# ============================================================
sizes = [10, 100, 1000, 10_000]

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here. Both formulas are printed for each size.

# ============================================================
# PART 3 — Implementation
# ============================================================
def _cost_gauss(n):
    return (2.0 / 3.0) * (n ** 3)

def _cost_substitution(n):
    return 2.0 * (n ** 2)

def estimate_arithmetic_complexity(sizes):
    print(f"{'n':>8} | {'Gauss (~2/3 n^3)':>22} | {'Substitution (~2 n^2)':>22}")
    print("-" * 60)
    for n in sizes:
        print(f"{n:>8} | {_cost_gauss(n):>22.4e} | {_cost_substitution(n):>22.4e}")

# ============================================================
# PART 4 — Call
# ============================================================
estimate_arithmetic_complexity(sizes)
