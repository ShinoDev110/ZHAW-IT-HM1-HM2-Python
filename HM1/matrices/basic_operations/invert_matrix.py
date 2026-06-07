# ============================================================
# TOPIC: Matrices — inverse of a matrix
# DESCRIPTION:
# Computes the inverse A^-1 of a square matrix using
# numpy.linalg.inv.
# USE WHEN:
# When the inverse is genuinely needed; otherwise prefer
# solving A x = b directly rather than inverting explicitly.
# EXAMPLE:
# 3x3 matrix [[-4,1,0],[3,-2,1],[5,2,-1]] -> inverse.
# ============================================================

import numpy as np

# ============================================================
# PART 1 — Inputs
# ============================================================
matrix = np.array([[-4.0,  1.0,  0.0],
                   [ 3.0, -2.0,  1.0],
                   [ 5.0,  2.0, -1.0]])

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here.

# ============================================================
# PART 3 — Implementation
# ============================================================
def invert_matrix(A):
    A_inv = np.linalg.inv(A)
    print(f"Inverse:\n{A_inv}")
    return A_inv

# ============================================================
# PART 4 — Call
# ============================================================
invert_matrix(matrix)
