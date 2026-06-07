# ============================================================
# TOPIC: Machine numbers — comparing machine precision of two computers
# DESCRIPTION:
# Computes the machine precision eps = (B/2) · B^(-n) for several
# floating-point formats and determines which computer computes more
# accurately (smaller eps). Allows different bases (binary, decimal, hex, ...).
# USE WHEN:
# When two computers with different bases/mantissa lengths should be
# compared ("Which machine computes more accurately?").
# EXAMPLE:
# 46-digit binary arithmetic vs. 14-digit decimal arithmetic
# -> eps = 2^-46 ~= 1.42e-14 < 5·10^-14, so the binary machine is more accurate.
# ============================================================

# ============================================================
# PART 1 — Inputs
# ============================================================
# Each machine: name, base B, number of mantissa digits n.
machines = [
    {"name": "Binary (46-digit)",   "base": 2,  "mantissa": 46},
    {"name": "Decimal (14-digit)", "base": 10, "mantissa": 14},
]

# Further example (exercise 2020_2.1c): 5-digit binary vs. 2-digit hex
# machines = [
#     {"name": "Binary (5-digit)", "base": 2,  "mantissa": 5},
#     {"name": "Hex (2-digit)",    "base": 16, "mantissa": 2},
# ]

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here: eps = (B/2) · B^(-n) for each machine, then comparison.

# ============================================================
# PART 3 — Implementation
# ============================================================
def _eps(base, mantissa):
    return (base / 2) * base ** (-mantissa)

def compare_machine_precision(machines):
    print("============================================================")
    print("Comparison of machine precision  eps = (B/2)·B^(-n)")
    print("============================================================")
    results = []
    for m in machines:
        eps = _eps(m["base"], m["mantissa"])
        results.append((m["name"], m["base"], m["mantissa"], eps))
        print(f"  {m['name']:<24}: B={m['base']:<3} n={m['mantissa']:<3} "
              f"-> eps = {eps:.6g}")

    eps_values = [r[3] for r in results]
    eps_min = min(eps_values)
    most_accurate = [r[0] for r in results if r[3] == eps_min]

    print("\nSmallest eps => most accurate machine.")
    if len(most_accurate) == len(results):
        print("=> All machines compute equally accurately (eps identical).")
    elif len(most_accurate) > 1:
        print(f"=> Equally accurate (smallest eps): {', '.join(most_accurate)}")
    else:
        print(f"=> Most accurate machine: {most_accurate[0]} (eps = {eps_min:.6g})")
    return results

# ============================================================
# PART 4 — Call
# ============================================================
compare_machine_precision(machines)
