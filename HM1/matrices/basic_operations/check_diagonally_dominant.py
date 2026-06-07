# ============================================================
# TOPIC: Matrices — check strict diagonal dominance
# DESCRIPTION:
# Checks the row-sum and column-sum criterion for strict diagonal dominance:
# |a_ii| > Σ_{j≠i} |a_ij| (row) or |a_jj| > Σ_{i≠j} |a_ij| (column).
# USE WHEN:
# Before applying the Jacobi or Gauss-Seidel method to verify
# whether sufficient convergence conditions are satisfied.
# EXAMPLE:
# Test matrix [[-8,5,2],[5,9,-1],[4,-8,7]] for diagonal dominance.
# ============================================================

import numpy as np

# ============================================================
# PART 1 — Inputs
# ============================================================
matrix = np.array([[-8.0,  5.0,  2.0],
                   [ 5.0,  9.0, -1.0],
                   [ 4.0, -8.0,  7.0]])
debug = True

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here. Both the row-sum AND column-sum criterion are checked.

# ============================================================
# PART 3 — Implementation
# ============================================================
def _check_row_dominance(A, debug=False):
    rows = A.shape[0]
    for i in range(rows):
        diagonal = abs(A[i, i])
        non_diagonal_sum = np.sum(np.abs(A[i, :])) - diagonal
        if diagonal <= non_diagonal_sum:
            return False
    if debug:
        print("Row-sum criterion satisfied")
        print("for all i = 1, ...., n |a_ii| > sum^n _j=1, j != i |a_i,j|\n")
    return True

def _check_column_dominance(A, debug=False):
    cols = A.shape[1]
    for j in range(cols):
        diagonal = abs(A[j, j])
        non_diagonal_sum = np.sum(np.abs(A[:, j])) - diagonal
        if diagonal <= non_diagonal_sum:
            return False
    if debug:
        print("Column-sum criterion satisfied")
        print("for all j = 1, ...., n |a_jj| > sum^n _i=1, i != j |a_i,j|\n")
    return True

def check_diagonally_dominant(A, debug=False):
    z = _check_row_dominance(A, debug)
    s = _check_column_dominance(A, debug)
    is_dd = z or s
    print(f"Strict diagonal dominance: {is_dd}")
    print("If A is diagonally dominant, the Jacobi method (simultaneous iteration)")
    print("and the Gauss-Seidel method (successive iteration) both converge for Ax = b.")
    return is_dd

# ============================================================
# PART 4 — Call
# ============================================================
check_diagonally_dominant(matrix, debug)
