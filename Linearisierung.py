# ============================================================
# TOPIC: Linearisierung — beliebige Funktion f: R^n -> R^m an Stelle x0
# DESCRIPTION:
# Linearisiert eine Funktion f: R^n -> R^m an einer Stelle x^(0) mit der
# verallgemeinerten Tangentengleichung g(x) = f(x^(0)) + Df(x^(0)) · (x - x^(0)).
# USE WHEN:
# Wenn eine Linearisierung oder eine Tangentialapproximation einer mehrdimensionalen
# Funktion benötigt wird.
# EXAMPLE:
# Linearisierung von f(x1,x2,x3) an x^(0)=(1.5,3,2.5)^T.
# ============================================================

import sympy as sp

# ============================================================
# PART 1 — Inputs
# ============================================================
# Symbolic variables (add more here if you need higher dimensions)
x1, x2, x3 = sp.symbols('x1 x2 x3')   # symbolische Variablen

# Variable vector (order must match the function arguments)
X = sp.Matrix([x1,
               x2,
               x3])                  # Variablenvektor

# Function f(x) as a sympy Matrix — works for any number of components m
f = sp.Matrix([
    x1 + x2**2 - x3**2 - 13,                 # erste Funktionskomponente
    sp.ln(x2 / 4) + sp.exp(0.5*x3 - 1) - 1,  # zweite Funktionskomponente
    (x2 - 3)**2 - x3**3 + 7                  # dritte Funktionskomponente
])

# Evaluation point x^(0)
x0 = sp.Matrix([1.5,                        # erste Koordinate von x^(0)
                3,                          # zweite Koordinate von x^(0)
                2.5])                       # dritte Koordinate von x^(0)

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: build Jacobi-Matrix, evaluate at x0, apply
# generalized tangent equation g(x) = f(x0) + Df(x0) · (x - x0).

# ============================================================
# PART 3 — Implementation
# ============================================================
def linearize_function(f, X, x0):
    # Jacobi matrix (works for any n input variables and any m output components)
    Df = f.jacobian(X)

    # Substitution map X -> x0
    subs_map = [(X[i], x0[i]) for i in range(len(X))]

    # Evaluate at x0
    f_x0  = f.subs(subs_map)
    Df_x0 = Df.subs(subs_map)

    # Generalized tangent equation
    g = sp.simplify(f_x0 + Df_x0 * (X - x0))

    # Output
    print("Symbolische Jacobi-Matrix Df(x):")
    sp.pprint(Df)
    print(f"\nf(x^(0)) =")
    sp.pprint(f_x0.evalf())
    print(f"\nDf(x^(0)) =")
    sp.pprint(Df_x0.evalf())
    print("\nLinearisierung g(x) = f(x^(0)) + Df(x^(0)) · (x - x^(0)):")
    sp.pprint(g)

    return g

# ============================================================
# PART 4 — Call
# ============================================================
linearize_function(f, X, x0)
