# ============================================================
# TOPIC: Numerische Integration — Schrittweite h / Anzahl n bei Fehlertoleranz
# DESCRIPTION:
# Bestimmt das maximale h (und damit minimale n), damit der Fehler der
# summierten Rechteck-, Trapez- bzw. Simpsonregel unter eine vorgegebene
# Toleranz fällt. Nutzt die Fehlerabschätzungen aus Satz 7.1 mit
# numerisch ermittelten max|f''| bzw. max|f^(4)| auf [a, b].
# USE WHEN:
# Wenn vorab entschieden werden soll, wie viele Subintervalle nötig sind,
# um eine bestimmte Genauigkeit zu garantieren.
# EXAMPLE:
# Wie viele Subintervalle braucht es für int_1^2 ln(x^2) dx mit
# Fehler < 1e-5 bei Rechteck-, Trapez- und Simpsonregel?
# ============================================================

import numpy as np
import sympy as sp

# ============================================================
# PART 1 — Inputs
# ============================================================
x_sym = sp.Symbol('x')
f_sym = sp.ln(x_sym**2)        # zu integrierende Funktion (symbolisch)

a   = 1.0
b   = 2.0
tol = 1e-5

# ============================================================
# PART 2 — Method selection
# ============================================================
# Options:
#   "rectangle"   -> |error| <= h^2 / 24 * (b-a) * max|f''|
#   "trapezoidal" -> |error| <= h^2 / 12 * (b-a) * max|f''|
#   "simpson"     -> |error| <= h^4 / 2880 * (b-a) * max|f^(4)|
#   "all"         -> alle drei
method = "all"

# ============================================================
# PART 3 — Implementation
# ============================================================
def estimate_required_step_size(f_sym, x_sym, a, b, tol, method):
    f2_lam = sp.lambdify(x_sym, sp.diff(f_sym, x_sym, 2), "numpy")
    f4_lam = sp.lambdify(x_sym, sp.diff(f_sym, x_sym, 4), "numpy")

    xs = np.linspace(a, b, 10000)
    max_f2 = np.max(np.abs(f2_lam(xs)))
    max_f4 = np.max(np.abs(f4_lam(xs)))
    print(f"max|f''(x)|   auf [{a},{b}] = {max_f2}")
    print(f"max|f^(4)(x)| auf [{a},{b}] = {max_f4}\n")

    def report(name, h, n):
        print(f"{name:<18}  h <= {h:.6e}   n >= {int(np.ceil(n))}")

    if method in ("rectangle", "all"):
        h = np.sqrt(24 * tol / ((b - a) * max_f2))
        report("Rechteckregel", h, (b - a) / h)
    if method in ("trapezoidal", "all"):
        h = np.sqrt(12 * tol / ((b - a) * max_f2))
        report("Trapezregel", h, (b - a) / h)
    if method in ("simpson", "all"):
        h = (2880 * tol / ((b - a) * max_f4))**(1/4)
        report("Simpsonregel", h, (b - a) / h)

# ============================================================
# PART 4 — Call
# ============================================================
estimate_required_step_size(f_sym, x_sym, a, b, tol, method)
