# ============================================================
# TOPIC: Matrix decomposition — LR decomposition without pivoting
# DESCRIPTION:
# Computes the LR decomposition A = L · R of a square matrix without
# pivot selection and simultaneously solves Ax = b via forward and
# backward substitution.
# USE WHEN:
# When solving a linear system of equations with a matrix whose
# pivot elements in Gaussian elimination are non-zero.
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

b = np.array([[-1.0  / 3.0],
              [-11.0 / 3.0],
              [  2.0 / 3.0]])

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here. A is always decomposed into L·R AND Ax=b is solved.

# ============================================================
# PART 3 — Implementation
# ============================================================
def _as_col(v):
    v = np.asarray(v, dtype=float)
    return v.reshape(-1, 1) if v.ndim == 1 else v

def _forward_sub(L, b, unit_diag=True):
    L = np.asarray(L, dtype=float)
    b = _as_col(b)
    n = L.shape[0]
    y = np.zeros((n, 1), dtype=float)
    for i in range(n):
        s = float(np.dot(L[i, :i], y[:i, 0]))
        denom = 1.0 if unit_diag else L[i, i]
        y[i, 0] = (b[i, 0] - s) / denom
    return y

def _back_sub(U, y):
    U = np.asarray(U, dtype=float)
    y = _as_col(y)
    n = U.shape[0]
    x = np.zeros((n, 1), dtype=float)
    for i in range(n - 1, -1, -1):
        s = float(np.dot(U[i, i+1:], x[i+1:, 0]))
        x[i, 0] = (y[i, 0] - s) / U[i, i]
    return x

def _lr_decompose(A, eps=1e-14):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square (n,n).")
    n = A.shape[0]
    R = A.copy()
    L = np.eye(n, dtype=float)
    for i in range(n):
        if abs(R[i, i]) < eps:
            raise ZeroDivisionError("LR without pivoting: pivot is 0 / too small.")
        for j in range(i + 1, n):
            factor = R[j, i] / R[i, i]
            L[j, i] = factor
            R[j, i:] = R[j, i:] - factor * R[i, i:]
    return L, R

def decompose_with_lr(A, b):
    A = np.asarray(A, dtype=float)
    b = _as_col(b)
    L, R = _lr_decompose(A)
    y = _forward_sub(L, b, unit_diag=True)
    x = _back_sub(R, y)

    print("A =\n", A)
    print("b =\n", b, "\n")
    print("=== LR (A = L·R) ===")
    print("L =\n", L)
    print("R =\n", R)
    print("y =\n", y)
    print("x =\n", x)
    print("check L@R == A:", np.allclose(L @ R, A))
    print("check A@x == b:", np.allclose(A @ x, b))
    return L, R, y, x

# ============================================================
# PART 4 — Call
# ============================================================
decompose_with_lr(A, b)
