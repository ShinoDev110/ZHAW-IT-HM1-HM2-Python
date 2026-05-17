# ============================================================
# TOPIC: Funktionen mit mehreren Variablen — Partielle Ableitungen & Linearisierung
# DESCRIPTION:
# Berechnet symbolisch partielle Ableitungen 1. Ordnung für mehrere Funktionen
# und wertet sie an gegebenen Punkten aus (Aufgabe 5.2).
# Linearisiert zusätzlich zwei weitere Funktionen via Jacobi-Matrix
# und berechnet die verallgemeinerte Tangentengleichung (Aufgabe 5.3).
# USE WHEN:
# Wenn partielle Ableitungen oder Linearisierungen von Funktionen mit
# mehreren Variablen berechnet werden sollen.
# ============================================================

import sympy as sp
import numpy as np

# ============================================================
# PART 1 — Inputs
# ============================================================
x, y = sp.symbols('x y')
x1, x2, x3 = sp.symbols('x1 x2 x3')

# --- Aufgabe 5.2: Funktionen und Auswertungspunkte ---
functions_5_2 = [
    # (Funktion, Variablen, Auswertungspunkt, Bezeichnung)
    (
        x**2 * y**4 + sp.exp(x) * sp.cos(y) + 10*x - 2*y**2 + 3,
        [x, y],
        {x: 0, y: 0},
        "f(x,y) = x^2*y^4 + exp(x)*cos(y) + 10x - 2y^2 + 3  bei (0, 0)"
    ),
    (
        x * y**2 * (sp.sin(x) + sp.sin(y)),
        [x, y],
        {x: sp.pi/2, y: sp.pi},
        "f(x,y) = x*y^2*(sin(x)+sin(y))  bei (pi/2, pi)"
    ),
    (
        sp.ln(x + y**2) - sp.exp(2*x*y) + 3*x,
        [x, y],
        {x: 1, y: 0},
        "f(x,y) = ln(x+y^2) - exp(2xy) + 3x  bei (1, 0)"
    ),
]

# --- Aufgabe 5.3: Funktionen zur Linearisierung ---
linearize_cases = [
    # (vektorwertige Funktion als Matrix, Variablen, Entwicklungspunkt, Bezeichnung)
    (
        sp.Matrix([
            sp.sin(x2 + 2*x3),
            sp.cos(2*x1 + x2),
        ]),
        sp.Matrix([x1, x2, x3]),
        sp.Matrix([sp.pi/4, 0, sp.pi]),
        "f(x1,x2,x3) = (sin(x2+2*x3), cos(2*x1+x2))  bei (pi/4, 0, pi)"
    ),
    (
        sp.Matrix([
            sp.exp(x1**2 + x2**2),
        ]),
        sp.Matrix([x1, x2]),
        sp.Matrix([sp.Rational(1, 2), sp.Rational(1, 2)]),
        "f(x1,x2) = exp(x1^2+x2^2)  bei (0.5, 0.5)  [Tangentialebene]"
    ),
]

# ============================================================
# PART 2 — Method selection
# ============================================================
# Welcher Teil soll berechnet werden?
# options: "partial_derivatives"  — nur Aufgabe 5.2
#          "linearize"            — nur Aufgabe 5.3
#          "all"                  — beide Aufgaben
mode = "all"

# ============================================================
# PART 3 — Implementation
# ============================================================
def compute_partial_derivatives_and_linearize(functions_5_2, linearize_cases, mode):

    def section(title):
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)

    # --- Aufgabe 5.2: Partielle Ableitungen ---
    if mode in ("partial_derivatives", "all"):
        section("Aufgabe 5.2 - Partielle Ableitungen 1. Ordnung")
        for f_expr, vars_, point, label in functions_5_2:
            print(f"\n{label}")
            for v in vars_:
                df = sp.diff(f_expr, v)
                df_simplified = sp.simplify(df)
                df_at_point   = df_simplified.subs(point)
                df_numeric    = float(df_at_point.evalf())
                print(f"  df/d{v} = {df_simplified}")
                print(f"  df/d{v} an Auswertungspunkt = {df_at_point} ~ {df_numeric:.6f}")

    # --- Aufgabe 5.3: Linearisierung via Jacobi-Matrix ---
    if mode in ("linearize", "all"):
        section("Aufgabe 5.3 - Linearisierung (Jacobi-Matrix + Tangentengleichung)")
        for f_vec, X_vec, x0_vec, label in linearize_cases:
            print(f"\n{label}")

            # Jacobi-Matrix symbolisch
            Df = f_vec.jacobian(X_vec)
            print(f"\nJacobi-Matrix Df(x):")
            sp.pprint(Df)

            # Df(x0) auswerten
            subs_dict = {X_vec[i]: x0_vec[i] for i in range(len(X_vec))}
            Df_x0     = Df.subs(subs_dict)
            f_x0      = f_vec.subs(subs_dict)

            print(f"\nDf(x0) =")
            sp.pprint(sp.simplify(Df_x0))
            print(f"\nf(x0) =")
            sp.pprint(sp.simplify(f_x0))

            # Linearisierung: g(x) = f(x0) + Df(x0) * (x - x0)
            dx = X_vec - x0_vec
            g  = sp.simplify(f_x0 + Df_x0 * dx)
            print(f"\nLinearisierung g(x) = f(x0) + Df(x0)*(x - x0) =")
            sp.pprint(g)

            # Numerische Werte für Df_x0 und f_x0
            Df_x0_num = np.array(Df_x0.evalf().tolist(), dtype=float)
            f_x0_num  = np.array(f_x0.evalf().tolist(), dtype=float).flatten()
            x0_num    = np.array(x0_vec.evalf().tolist(), dtype=float).flatten()
            print(f"\nNumerisch: Df(x0) =\n{Df_x0_num}")
            print(f"f(x0) = {f_x0_num}")

# ============================================================
# PART 4 — Call
# ============================================================
compute_partial_derivatives_and_linearize(functions_5_2, linearize_cases, mode)
