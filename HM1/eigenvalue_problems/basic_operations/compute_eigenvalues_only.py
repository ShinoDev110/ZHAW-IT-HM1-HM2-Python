# ============================================================
# TOPIC: Eigenwerte — Eigenwerte einer Matrix (nur Werte)
# DESCRIPTION:
# Berechnet alle Eigenwerte einer quadratischen Matrix via
# numpy.linalg.eigvals (ohne Eigenvektoren).
# USE WHEN:
# Wenn nur die Eigenwerte selbst von Interesse sind (z.B. zur Schätzung
# des Spektralradius oder als Plausibilitätscheck).
# EXAMPLE:
# A = [[1,1,1],[0,2,0],[1,-1,1]].
# ============================================================

import numpy as np

# ============================================================
# PART 1 — Inputs
# ============================================================
A = np.array([[1, 1, 1],
              [0, 2, 0],
              [1, -1, 1]])

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here.

# ============================================================
# PART 3 — Implementation
# ============================================================
def compute_eigenvalues_only(A):
    ew = np.linalg.eigvals(A)
    print(f"Eigenwerte: {ew}")
    return ew

# ============================================================
# PART 4 — Call
# ============================================================
compute_eigenvalues_only(A)
