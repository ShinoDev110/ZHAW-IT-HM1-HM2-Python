# ============================================================
# TOPIC: Fixed-point analysis — attractive vs. repulsive
# DESCRIPTION:
# Classifies a fixed point x̄ of F(x) as attractive / repulsive
# based on |F'(x̄)|: either at a concrete point or as a
# bound over an interval.
# USE WHEN:
# When deciding whether a fixed-point iteration near a fixed point
# converges (|F'(x̄)| < 1) or diverges (> 1).
# EXAMPLE:
# F(x) = (230x^4 + 18x^3 + 9x^2 - 9)/221 at fixed point 0; or
# F(x) = exp(x) - exp(1) on the interval [-3, -2].
# ============================================================

from sympy import diff, exp, symbols

# ============================================================
# PART 1 — Inputs
# ============================================================
x = symbols("x")

# Function and fixed point for method "point"
function_point = (230 * x**4 + 18 * x**3 + 9 * x**2 - 9) / 221
fixed_point    = 0

# Function and interval for method "interval"
function_interval = exp(x) - exp(1)
interval_start    = -3
interval_end      = -2

# ============================================================
# PART 2 — Method selection
# ============================================================
# method:
#   "point"    -> |F'(x̄)| at a concrete known fixed point
#   "interval" -> behaviour on an interval [a, b]
method = "point"

# ============================================================
# PART 3 — Implementation
# ============================================================
def _classify_at_point(function, var, fixed_point):
    derivative = diff(function, var)
    value = derivative.subs(var, fixed_point)
    print(f"F'(x) = {derivative}, F'({fixed_point}) = {value.evalf()}\n")
    if abs(value) < 1:
        print(f"Attractive fixed point, because |F'(x̄)| = {abs(value.evalf())} < 1")
    else:
        print(f"Repulsive fixed point, because |F'(x̄)| = {abs(value.evalf())} > 1")

def _classify_on_interval(function, var, a, b):
    derivative = diff(function, var)
    fa = derivative.subs(var, a)
    fb = derivative.subs(var, b).evalf()
    if fa < 1 and fb < 1:
        print(f"Attractive fixed point on interval [{a}, {b}]\n")
        print(f"Proof: f'({a}) = {fa} <= f'(x̄) <= f'({b}) = {fb} < 1")
    elif 1 < fa and 1 < fb:
        print(f"Repulsive fixed point on interval [{a}, {b}]\n")
        print(f"Proof: 1 < f'({a}) = {fa} <= f'(x̄) <= f'({b}) = {fb}")
    else:
        print(f"Ambiguous: f'({a}) = {fa}, f'({b}) = {fb} (sign / bounds inconsistent)")

def classify_attractive_repulsive_fixed_point(method, x_sym,
                                              function_point, fixed_point,
                                              function_interval, a, b):
    if method == "point":
        _classify_at_point(function_point, x_sym, fixed_point)
    elif method == "interval":
        _classify_on_interval(function_interval, x_sym, a, b)
    else:
        raise ValueError(f"Unknown method: {method!r}")

# ============================================================
# PART 4 — Call
# ============================================================
classify_attractive_repulsive_fixed_point(method, x,
                                          function_point, fixed_point,
                                          function_interval, interval_start, interval_end)
