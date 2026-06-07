# ============================================================
# TOPIC: Eigenvalues — characteristic polynomial det(A - lambdaI)
# DESCRIPTION:
# Builds via sympy.charpoly the symbolic form of p(lambda) = det(A - lambdaI)
# and prints both the expanded and the factorized
# representation.
# USE WHEN:
# When the characteristic polynomial is needed symbolically (e.g. for
# manual determination of eigenvalues).
# EXAMPLE:
# A = [[1,1,1],[0,2,0],[1,-1,1]].
# ============================================================

import numpy as np
import sympy as sp

# ============================================================
# PART 1 — Inputs
# ============================================================
A = np.array([[1.0,  1.0, 1.0],
              [0.0,  2.0, 0.0],
              [1.0, -1.0, 1.0]])
debug = True

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here. Both forms (expanded + factorized)
# are always printed.

# ============================================================
# PART 3 — Implementation
# ============================================================
def compute_characteristic_polynomial(A, debug=False):
    A_sym = sp.Matrix(A)
    n = A_sym.shape[0]
    lam = sp.Symbol("lambda")
    p = A_sym.charpoly(lam).as_expr()
    p_fact = sp.factor(p)
    if debug:
        print("A =")
        print(np.array(A_sym, dtype=float))
        print("--------------------------------------------------")
        print(f"p(lambda) = det(A - lambda I_{n}) =")
        print(sp.sstr(sp.expand(p)))
        print("--------------------------------------------------")
        print("Factorized form =")
        print(sp.sstr(p_fact))
    return p, p_fact

# ============================================================
# PART 4 — Call
# ============================================================
compute_characteristic_polynomial(A, debug)
