# ============================================================
# TOPIC: Matrices — matrix norms and condition numbers
# DESCRIPTION:
# Prints ||A||_1, ||A||_2, ||A||_inf and (if A is square) the
# condition numbers cond_1(A), cond_2(A), cond_inf(A).
# USE WHEN:
# When assessing the stability of a linear system of equations Ax = b.
# EXAMPLE:
# 3x3 matrix [[240,120,80],[60,180,170],[60,90,500]].
# ============================================================

import numpy as np
import numpy.linalg as lin

np.set_printoptions(precision=6, suppress=True)

# ============================================================
# PART 1 — Inputs
# ============================================================
A = np.array([[240.0, 120.0,  80.0],
              [ 60.0, 180.0, 170.0],
              [ 60.0,  90.0, 500.0]])
debug = False  # additionally print A^-1 (only for square A)

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here. All three norms are always computed, and (if
# square) all three condition numbers are computed as well.

# ============================================================
# PART 3 — Implementation
# ============================================================
def compute_matrix_norm_and_condition(A, debug=False):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"A must be 2D. Got shape {A.shape}.")
    m, n = A.shape
    print("Matrix A:")
    print(A)
    print("------------------------------------------------------------")
    for name, ord_val in [("1", 1), ("2", 2), ("inf", np.inf)]:
        print(f"||A||_{name:>9} = {lin.norm(A, ord=ord_val)}")
    print("------------------------------------------------------------")
    if m == n:
        for name, p in [("1", 1), ("2", 2), ("inf", np.inf)]:
            print(f"cond_{name:>9}(A) = {lin.cond(A, p=p)}")
        if debug:
            print("------------------------------------------------------------")
            print("A^-1 (reference):")
            print(lin.inv(A))
    else:
        print("cond_p(A): only defined for square matrices (skipped).")

# ============================================================
# PART 4 — Call
# ============================================================
compute_matrix_norm_and_condition(A, debug)
