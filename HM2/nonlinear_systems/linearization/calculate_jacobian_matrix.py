# ============================================================
# TOPIC: Jacobian matrix of a function f: R^n -> R^m (symbolic + evaluated at x0)
# DESCRIPTION:
# Computes the Jacobian matrix of a function f: R^n -> R^m symbolically and
# evaluates it at a point x0.
# USE WHEN:
# When the Jacobian matrix of a function is required, for example to verify
# a hand-computed solution.
# EXAMPLE:
# Jacobian matrix of f(x1,x2,x3) at the point (1,2,3)^T.
# ============================================================

import sympy as sp

# ============================================================
# PART 1 — Inputs
# ============================================================
# Symbolic variables (add more here if needed)
x1, x2, x3 = sp.symbols('x1 x2 x3')   # symbolic variables

# --- Problem a) ---
X_a  = sp.Matrix([x1,
                  x2])                  # variable vector for problem a)
f_a  = sp.Matrix([5*x1*x2,              # function for problem a)
                  x1**2 * x2**2 + x1 + 2*x2])
x0_a = sp.Matrix([1,                    # evaluation point for problem a)
                  2])

# --- Problem b) ---
X_b  = sp.Matrix([x1,
                  x2,
                  x3])                 # variable vector for problem b)
f_b  = sp.Matrix([sp.ln(x1**2 + x2**2) + x3**2,      # function for problem b)
                  sp.exp(x2**2 + x3**2) + x1**2,
                  1/(x3**2 + x1**2) + x2**2])
x0_b = sp.Matrix([1,                   # evaluation point for problem b)
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
        print("Symbolic Jacobian matrix Df(x):")
        sp.pprint(Df)
        print(f"\nEvaluated at x0 = {list(x0)}:")
        sp.pprint(Df_x0)
        print(f"\nNumerical:")
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
