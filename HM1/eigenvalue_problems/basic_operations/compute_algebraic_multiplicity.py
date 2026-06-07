# ============================================================
# TOPIC: Eigenvalues — algebraic multiplicity of each eigenvalue
# DESCRIPTION:
# Computes all eigenvalues of a matrix, rounds them to a selectable
# number of decimal places, sorts them (real before complex) and counts
# repetitions as algebraic multiplicity.
# USE WHEN:
# When it needs to be assessed whether an eigenvalue occurs multiple times
# and whether A could be diagonalizable.
# EXAMPLE:
# A = [[1,1,1],[0,2,0],[1,-1,1]].
# ============================================================

from collections import Counter
import numpy as np

# ============================================================
# PART 1 — Inputs
# ============================================================
A = np.array([[1, 1, 1],
              [0, 2, 0],
              [1, -1, 1]])
decimals = 10

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here.

# ============================================================
# PART 3 — Implementation
# ============================================================
def _algebraic_multiplicity(matrix, decimals=10):
    eigenvalues = np.linalg.eigvals(matrix)
    eigenvalues = np.round(eigenvalues, decimals)
    eigenvalues = [val.real if np.isclose(val.imag, 0, atol=10**(-decimals)) else val for val in eigenvalues]
    return dict(Counter(eigenvalues))

def compute_algebraic_multiplicity(matrix, decimals=10):
    mult = _algebraic_multiplicity(matrix, decimals)
    print("Matrix A:")
    print(matrix)
    print("--------------------------------------------------")
    print("Eigenvalues ->  Multiplicity:")
    def sort_key(x):
        if isinstance(x, complex):
            return (1, x.real, x.imag)
        return (0, float(x), 0.0)
    for lam in sorted(mult.keys(), key=sort_key):
        m = mult[lam]
        if isinstance(lam, complex):
            lam_str = f"{lam.real:.{decimals}f} {lam.imag:+.{decimals}f}i"
        else:
            lam_str = f"{float(lam):.{decimals}f}".rstrip("0").rstrip(".")
        print(f"  lambda = {lam_str:<3}  ->  {m}")
    return mult

# ============================================================
# PART 4 — Call
# ============================================================
compute_algebraic_multiplicity(A, decimals)
