# ============================================================
# TOPIC: Eigenvalues — Spectrum of a matrix
# DESCRIPTION:
# Prints the complete spectrum (all eigenvalues) of a matrix and
# returns its cardinality.
# USE WHEN:
# When the spectrum {lambda_1, ..., lambda_n} of a matrix is needed
# as a set (e.g. to directly verify spectral theorems).
# EXAMPLE:
# A = [[1,0,0],[2,3,0],[0,1,2]].
# ============================================================

import numpy as np

# ============================================================
# PART 1 — Inputs
# ============================================================
matrix = np.array([[1, 0, 0],
                   [2, 3, 0],
                   [0, 1, 2]])
debug = True

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here.

# ============================================================
# PART 3 — Implementation
# ============================================================
def compute_spectrum(A, debug=False):
    eigenvalues = np.linalg.eigvals(A)
    if debug:
        print(f"Spectrum: {eigenvalues}\n")
    print(f"Cardinality of the spectrum: {eigenvalues.size}")
    return eigenvalues

# ============================================================
# PART 4 — Call
# ============================================================
compute_spectrum(matrix, debug)
