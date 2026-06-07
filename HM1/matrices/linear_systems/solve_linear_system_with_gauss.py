# ============================================================
# TOPIC: Linear Systems — Gauss elimination with column pivoting
# DESCRIPTION:
# Solves Ax = b via Gauss elimination with row-pivot search; also returns
# the determinant (product of the diagonal of U times
# (-1)^number_of_row_swaps) and the upper triangular matrix U.
# USE WHEN:
# When a direct, stable solver without separate L/R storage is needed
# or the determinant is required as a by-product.
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
def solve_linear_system_with_gauss(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square.")
    if b.size != n:
        raise ValueError("Dimensions of A and b are not compatible.")

    U = A.copy()
    rhs = b.copy()
    num_permutations = 0

    for k in range(n - 1):
        pivot_row = k + np.argmax(np.abs(U[k:, k]))
        if U[pivot_row, k] == 0.0:
            raise ValueError("Singular matrix — no unique linear system solution.")
        if pivot_row != k:
            U[[k, pivot_row], :] = U[[pivot_row, k], :]
            rhs[[k, pivot_row]]  = rhs[[pivot_row, k]]
            num_permutations += 1

        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            U[i, k:] -= factor * U[k, k:]
            rhs[i]   -= factor * rhs[k]

    det = ((-1) ** num_permutations) * float(np.prod(np.diag(U)))

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = np.dot(U[i, i + 1:], x[i + 1:])
        x[i] = (rhs[i] - s) / U[i, i]

    print("U =\n", U)
    print(f"det(A) = {det}")
    print(f"x = {x}")
    print(f"check A@x ~= b: {np.allclose(A @ x, b)}")
    return U, det, x

# ============================================================
# PART 4 — Call
# ============================================================
solve_linear_system_with_gauss(A, b)
