# ============================================================
# TOPIC: Linear Systems — Cramer's rule
# DESCRIPTION:
# Solves Ax = b via x_i = det(A_i) / det(A), where A_i is the matrix A
# with the i-th column replaced by b. Practical only for very small
# systems (2x2, 3x3), otherwise too expensive.
# USE WHEN:
# When the system is small and a symbolic/formula-oriented approach is
# desired (no pivoting/elimination needed).
# EXAMPLE:
# 3x3 system A = [[2,1,-1],[-3,-1,2],[-2,1,2]], b = [8,-11,-3].
# ============================================================

import numpy as np

np.set_printoptions(precision=6, suppress=True)

# ============================================================
# PART 1 — Inputs
# ============================================================
A = np.array([[ 2.0,  1.0, -1.0],
              [-3.0, -1.0,  2.0],
              [-2.0,  1.0,  2.0]])

b = np.array([ 8.0, -11.0, -3.0])

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here.

# ============================================================
# PART 3 — Implementation
# ============================================================
def solve_linear_system_with_cramer(A, b):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Cramer: matrix must be square.")
    if b.size != n:
        raise ValueError("Cramer: dimensions do not match.")
    det_A = np.linalg.det(A)
    if det_A == 0.0:
        raise ValueError("Cramer: det(A) = 0, no unique solution.")

    x = np.zeros(n)
    for i in range(n):
        A_i = A.copy()
        A_i[:, i] = b
        det_A_i = np.linalg.det(A_i)
        x[i] = det_A_i / det_A
        print(f"x_{i+1} = det(A_{i+1}) / det(A) = {det_A_i} / {det_A} = {x[i]}")

    print(f"\nSolution: x = {x}")
    print(f"check A@x ~= b: {np.allclose(A @ x, b)}")
    return x

# ============================================================
# PART 4 — Call
# ============================================================
solve_linear_system_with_cramer(A, b)
