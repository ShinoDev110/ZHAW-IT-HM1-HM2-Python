# ============================================================
# TOPIC: Matrices — test for orthogonality
# DESCRIPTION:
# Checks whether a square matrix is orthogonal by testing
# A · A^T ~= I (np.allclose).
# USE WHEN:
# When you need to verify that a rotation matrix or a Q-matrix
# from QR decomposition is indeed orthogonal.
# EXAMPLE:
# Rotation matrix [[2/3,-2/3,-1/3],[1/3,2/3,-2/3],[2/3,1/3,2/3]] -> True.
# ============================================================

import numpy as np

# ============================================================
# PART 1 — Inputs
# ============================================================
A = np.array([[2/3, -2/3, -1/3],
              [1/3,  2/3, -2/3],
              [2/3,  1/3,  2/3]])

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here.

# ============================================================
# PART 3 — Implementation
# ============================================================
def check_orthogonal_matrix(A):
    A = np.array(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix (n x n).")
    is_orth = np.allclose(np.dot(A, A.T), np.eye(A.shape[0]))
    print(f"Orthogonal: {is_orth}")
    return is_orth

# ============================================================
# PART 4 — Call
# ============================================================
check_orthogonal_matrix(A)
