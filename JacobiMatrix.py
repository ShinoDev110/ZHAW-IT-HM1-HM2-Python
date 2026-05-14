# ============================================================
# TOPIC: Jacobi-Matrix einer Funktion f: R^n -> R^m (symbolisch + an x0 ausgewertet)
# DESCRIPTION:
# Berechnet die Jacobi-Matrix einer Funktion f: R^n -> R^m symbolisch und
# wertet sie an einem Punkt x0 aus.
# USE WHEN:
# Wenn die Jacobi-Matrix einer Funktion gesucht ist, zum Beispiel zur Kontrolle
# einer handgerechneten Lösung.
# EXAMPLE:
# Jacobi-Matrix von f(x1,x2,x3) am Punkt (1,2,3)^T.
# ============================================================

import sympy as sp

# ============================================================
# PART 1 — Inputs
# ============================================================
# Symbolic variables (add more here if needed)
x1, x2, x3 = sp.symbols('x1 x2 x3')   # symbolische Variablen

# --- Problem a) ---
X_a  = sp.Matrix([x1,
                  x2])                  # Variablenvektor für Aufgabe a)
f_a  = sp.Matrix([5*x1*x2,              # Funktion für Aufgabe a)
                  x1**2 * x2**2 + x1 + 2*x2])
x0_a = sp.Matrix([1,                    # Auswertungspunkt für Aufgabe a)
                  2])

# --- Problem b) ---
X_b  = sp.Matrix([x1,
                  x2,
                  x3])                 # Variablenvektor für Aufgabe b)
f_b  = sp.Matrix([sp.ln(x1**2 + x2**2) + x3**2,      # Funktion für Aufgabe b)
                  sp.exp(x2**2 + x3**2) + x1**2,
                  1/(x3**2 + x1**2) + x2**2])
x0_b = sp.Matrix([1,                   # Auswertungspunkt für Aufgabe b)
                  2,
                  3])

# ============================================================
# PART 2 — Method selection
# ============================================================
# Options:
#   "a"    -> compute Jacobian only for problem a)
#   "b"    -> compute Jacobian only for problem b)
#   "both" -> compute both
method = "both"

# ============================================================
# PART 3 — Implementation
# ============================================================
def compute_jacobian_matrix(method, f_a, X_a, x0_a, f_b, X_b, x0_b):

    def solve(name, f, X, x0):
        # General formula: works for any n (vector size) and m (number of components)
        Df    = f.jacobian(X)
        subs  = [(X[i], x0[i]) for i in range(len(X))]
        Df_x0 = sp.simplify(Df.subs(subs))

        print(f"=== Problem {name}  (f: R^{len(X)} -> R^{f.shape[0]}) ===")
        print("Symbolische Jacobi-Matrix Df(x):")
        sp.pprint(Df)
        print(f"\nAusgewertet an x0 = {list(x0)}:")
        sp.pprint(Df_x0)
        print(f"\nNumerisch:")
        sp.pprint(Df_x0.evalf())
        print()

    if method == "a":
        solve("a", f_a, X_a, x0_a)
    elif method == "b":
        solve("b", f_b, X_b, x0_b)
    elif method == "both":
        solve("a", f_a, X_a, x0_a)
        solve("b", f_b, X_b, x0_b)
    else:
        raise ValueError("method must be 'a', 'b', or 'both'")

# ============================================================
# PART 4 — Call
# ============================================================
compute_jacobian_matrix(method, f_a, X_a, x0_a, f_b, X_b, x0_b)
