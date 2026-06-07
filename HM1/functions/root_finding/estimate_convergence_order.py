# ============================================================
# TOPIC: Root-finding methods — convergence order from iteration sequence
# DESCRIPTION:
# Estimates from an iteration sequence {x_k} the convergence order p
# and the constant C in the model e_{k+1} ~= C · e_k^p, with
# e_k = |x_k - x*|. If x* is unknown, x* ~= x_last is used.
# USE WHEN:
# When empirically distinguishing whether a method converges linearly,
# quadratically, or cubically.
# EXAMPLE:
# Newton sequence xs = [0.5, 0.8, 0.93, 0.991, 0.99988, 1.0] -> p ~= 2.
# ============================================================

import math

# ============================================================
# PART 1 — Inputs
# ============================================================
sequence_x      = [0.5, 0.8, 0.93, 0.991, 0.99988, 1.0]
exact_solution  = 1.0  # set to None if unknown -> last x is used instead

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here.

# ============================================================
# PART 3 — Implementation
# ============================================================
def estimate_convergence_order(sequence_x, exact_solution=None):
    if len(sequence_x) < 3:
        print("Sequence too short (at least 3 values needed).")
        return None, None

    if exact_solution is None:
        exact_solution = sequence_x[-1]

    errors = [abs(x - exact_solution) for x in sequence_x]
    errors = [e for e in errors if e > 0]
    if len(errors) < 3:
        print("Too few non-zero errors.")
        return None, None

    p_values = []
    for k in range(1, len(errors) - 1):
        e_km1, e_k, e_kp1 = errors[k - 1], errors[k], errors[k + 1]
        denominator = math.log(e_k / e_km1)
        numerator = math.log(e_kp1 / e_k)
        if denominator != 0:
            p_values.append(numerator / denominator)

    if not p_values:
        print("Could not estimate p.")
        return None, None

    p_estimated = sum(p_values) / len(p_values)
    e_k, e_kp1 = errors[-2], errors[-1]
    C_estimated = e_kp1 / (e_k ** p_estimated) if e_k != 0 else None

    print(f"Estimated convergence order p ~= {p_estimated}")
    print(f"Estimated constant          C ~= {C_estimated}")
    return p_estimated, C_estimated

# ============================================================
# PART 4 — Call
# ============================================================
estimate_convergence_order(sequence_x, exact_solution)
