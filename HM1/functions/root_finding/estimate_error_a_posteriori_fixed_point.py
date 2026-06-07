# ============================================================
# TOPIC: Fixed-point iteration — a-posteriori error estimate (scalar)
# DESCRIPTION:
# Estimates per Banach the error of a scalar fixed-point iteration from two
# consecutive iterates: |x̄ - x_{k+1}| <= (alpha/(1-alpha)) ·
# |x_{k+1} - x_k|. The Lipschitz constant alpha is either given directly
# or estimated as max|F'(x)| over an interval.
# USE WHEN:
# When after some iterations it should be assessed how accurate the current
# approximation x_{k+1} already is ("How accurate is the estimate?").
# EXAMPLE:
# F(x) = 1/(cos(x+pi/4)-1)+2 on [1,2], x_k = 1.3376, x_{k+1} = 1.3441
# -> |x̄ - 1.3441| <= (alpha/(1-alpha)) · 0.0065.
# ============================================================

import numpy as np
from sympy import diff, sympify

# ============================================================
# PART 1 — Inputs
# ============================================================
function    = "1/(cos(x + pi/4) - 1) + 2"  # F(x) of the fixed-point iteration
x_prev      = 1.3376    # x_k       (previous iterate)
x_curr      = 1.3441    # x_{k+1}   (current iterate)
interval    = [1.0, 2.0]  # interval for estimating alpha (max|F'|)
alpha_given = 0.5       # fixed Lipschitz constant (only for "given_alpha")

# ============================================================
# PART 2 — Method selection
# ============================================================
# method:
#   "from_interval" -> alpha = max|F'(x)| discrete over the interval
#   "given_alpha"   -> alpha = alpha_given (e.g. given from a sub-problem)
method = "from_interval"

# ============================================================
# PART 3 — Implementation
# ============================================================
def _get_alpha_from_interval(function, interval):
    f = sympify(function)
    symbols = list(f.free_symbols)
    if not symbols:
        raise ValueError("No unknown found in function.")
    s = symbols[0]
    df = diff(f, s)
    values = [abs(float(df.subs(s, t).evalf()))
              for t in np.linspace(interval[0], interval[1], 200)]
    return max(values), df

def estimate_error_a_posteriori_fixed_point(method, function, x_prev, x_curr,
                                            interval, alpha_given):
    print("============================================================")
    print("A-posteriori error estimate of the fixed-point iteration")
    print("============================================================")
    if method == "from_interval":
        alpha, df = _get_alpha_from_interval(function, interval)
        print(f"F(x)  = {function}")
        print(f"F'(x) = {df}")
        print(f"alpha = max|F'(x)| on [{interval[0]}, {interval[1]}] ~= {alpha:.6g}")
    elif method == "given_alpha":
        alpha = alpha_given
        print(f"alpha (given) = {alpha:.6g}")
    else:
        raise ValueError(f"Unknown method: {method!r}")

    if alpha >= 1:
        raise ValueError(f"alpha = {alpha} >= 1 -> no contraction, "
                         "a-posteriori estimate not valid.")

    step = abs(x_curr - x_prev)
    factor = alpha / (1 - alpha)
    bound = factor * step
    print(f"\n|x_(k+1) - x_k| = |{x_curr} - {x_prev}| = {step:.6g}")
    print(f"Factor alpha/(1-alpha) = {factor:.6g}")
    print(f"|x̄ - x_(k+1)| <= (alpha/(1-alpha))·|x_(k+1)-x_k| = {bound:.6g}")
    print(f"\n=> The approximation x_(k+1) = {x_curr} is accurate to about {bound:.3g}.")
    return bound

# ============================================================
# PART 4 — Call
# ============================================================
estimate_error_a_posteriori_fixed_point(method, function, x_prev, x_curr,
                                        interval, alpha_given)
