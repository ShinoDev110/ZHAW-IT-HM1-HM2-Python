# ============================================================
# TOPIC: Interpolation — cubic polynomial interpolation (4 nodes)
# DESCRIPTION:
# Determines the coefficients of a cubic polynomial
#   p(t) = a t^3 + b t^2 + c t + d
# with t = year - reference_year, that exactly interpolates 4 given
# (year, measurement) pairs (Vandermonde system). Then evaluates
# the polynomial at a target point.
# USE WHEN:
# When an unambiguous interpolation polynomial of degree 3 must be
# constructed from exactly 4 data points and evaluated.
# EXAMPLE:
# Years [1997, 1999, 2006, 2010], values [150, 104, 172, 152],
# reference 1997, evaluate at 2003.
# ============================================================

import numpy as np

# ============================================================
# PART 1 — Inputs
# ============================================================
year_values      = np.array([1997.0, 1999.0, 2006.0, 2010.0])
measurements     = np.array([ 150.0,  104.0,  172.0,  152.0])
reference_year   = 1997.0
evaluation_point = 2003.0

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here (Vandermonde system for 4 points).

# ============================================================
# PART 3 — Implementation
# ============================================================
def _compute_coefficients(year_values, measurements, reference_year):
    t = year_values - reference_year
    vandermonde = np.vstack([t ** 3, t ** 2, t, np.ones_like(t)]).T
    return np.linalg.solve(vandermonde, measurements)

def _evaluate(year, coefficients, reference_year):
    a, b, c, d = coefficients
    t = np.array(year, dtype=float) - reference_year
    return ((a * t + b) * t + c) * t + d

def interpolate_with_cubic_polynomial(year_values, measurements, reference_year, evaluation_point):
    coeff = _compute_coefficients(year_values, measurements, reference_year)
    a, b, c, d = coeff
    print(f"p(t) = {a} t^3 + {b} t^2 + {c} t + {d}")
    print(f"with t = year - {reference_year}")
    value = float(_evaluate(evaluation_point, coeff, reference_year))
    print(f"\np({evaluation_point}) = {value}")
    return coeff, value

# ============================================================
# PART 4 — Call
# ============================================================
interpolate_with_cubic_polynomial(year_values, measurements, reference_year, evaluation_point)
