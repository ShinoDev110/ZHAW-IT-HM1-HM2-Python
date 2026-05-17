# ============================================================
# TOPIC: Eigenwerte — algebraische Vielfachheit jedes Eigenwerts
# DESCRIPTION:
# Berechnet alle Eigenwerte einer Matrix, rundet sie auf eine wählbare
# Anzahl Dezimalstellen, sortiert (reell vor komplex) und zählt
# Wiederholungen als algebraische Vielfachheit.
# USE WHEN:
# Wenn beurteilt werden soll, ob ein Eigenwert mehrfach auftritt und ob
# A diagonalisierbar sein könnte.
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
def _algebraische_vielfachheit(matrix, decimals=10):
    ew = np.linalg.eigvals(matrix)
    ew = np.round(ew, decimals)
    ew = [val.real if np.isclose(val.imag, 0, atol=10**(-decimals)) else val for val in ew]
    return dict(Counter(ew))

def compute_algebraic_multiplicity(matrix, decimals=10):
    mult = _algebraische_vielfachheit(matrix, decimals)
    print("Matrix A:")
    print(matrix)
    print("--------------------------------------------------")
    print("Eigenwerte ->  Vielfachheit:")
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
