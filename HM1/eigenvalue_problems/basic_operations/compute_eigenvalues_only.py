# ============================================================
# TOPIC: Eigenvalues — eigenvalues of a matrix (values only)
# DESCRIPTION:
# Computes all eigenvalues of a square matrix via
# numpy.linalg.eigvals (without eigenvectors).
# USE WHEN:
# When only the eigenvalues themselves are of interest (e.g. for estimating
# the spectral radius or as a plausibility check).
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
    eigenvalues = np.linalg.eigvals(A)
    print(f"Eigenvalues: {eigenvalues}")
    return eigenvalues

# ============================================================
# PART 4 — Call
# ============================================================
compute_eigenvalues_only(A)
