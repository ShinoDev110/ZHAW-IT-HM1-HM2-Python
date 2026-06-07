# ============================================================
# TOPIC: Linear Systems — solving via pre-computed L, R
# DESCRIPTION:
# Solves Ax = b using an already available LR decomposition A = L·R by
# solving Ly = b forward and Rx = y backward. Useful when
# A is decomposed once and needs to be solved for many different b's.
# USE WHEN:
# When A is fixed and x is needed for multiple right-hand sides b —
# decompose LR once, then only 2 triangular solves per b.
# EXAMPLE:
# A = [[2,2,-1],[1,-1,0],[2,0,1]], b = [-1/3, -11/3, 2/3].
# ============================================================

import numpy as np

np.set_printoptions(precision=6, suppress=True)

# ============================================================
# PART 1 — Inputs
# ============================================================
A = np.array([[2.0,  2.0, -1.0],
              [1.0, -1.0,  0.0],
              [2.0,  0.0,  1.0]])

b = np.array([-1.0 / 3.0,
              -11.0 / 3.0,
               2.0 / 3.0])

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here (LR without pivoting + forward/back substitution).

# ============================================================
# PART 3 — Implementation
# ============================================================
def _lr_decompose(A):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    R = A.copy()
    L = np.eye(n)
    for k in range(n - 1):
        if R[k, k] == 0.0:
            raise ValueError("Zero pivot in LR decomposition without pivoting.")
        for i in range(k + 1, n):
            mult = R[i, k] / R[k, k]
            L[i, k] = mult
            R[i, k:] -= mult * R[k, k:]
    return L, R

def solve_linear_system_with_lr(A, b):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    L, R = _lr_decompose(A)
    n = L.shape[0]

    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(R[i, i + 1:], x[i + 1:])) / R[i, i]

    print("L =\n", L)
    print("R =\n", R)
    print(f"y = {y}")
    print(f"x = {x}")
    print(f"check A@x ~= b: {np.allclose(A @ x, b)}")
    return L, R, y, x

# ============================================================
# PART 4 — Call
# ============================================================
solve_linear_system_with_lr(A, b)
