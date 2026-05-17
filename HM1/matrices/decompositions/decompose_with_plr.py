# ============================================================
# TOPIC: Matrix-Zerlegung — PLR-Zerlegung mit Zeilen-Pivotisierung
# DESCRIPTION:
# Berechnet P·A = L·R mit Permutationsmatrix P, normierter unterer
# Dreiecksmatrix L und oberer Dreiecksmatrix R. Löst zusätzlich Ax = b
# über z = P·b, L·y = z, R·x = y.
# USE WHEN:
# Wenn A nicht regulär nullpivotierbar ist (Standard-Fall): PLR ist die
# stabile LR-Zerlegung mit Pivotsuche.
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
# Only one method here. Es wird immer P·A=L·R zerlegt UND Ax=b gelöst.

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

def _plr_decompose(A, eps=1e-14):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A muss quadratisch (n,n) sein.")
    n = A.shape[0]
    R = A.copy()
    L = np.eye(n, dtype=float)
    p = np.arange(n)
    for i in range(n):
        pivot = i + int(np.argmax(np.abs(R[i:, i])))
        if abs(R[pivot, i]) < eps:
            raise ZeroDivisionError("PLR: Matrix singulär / Pivot zu klein.")
        if pivot != i:
            R[[i, pivot], :] = R[[pivot, i], :]
            L[[i, pivot], :i] = L[[pivot, i], :i]
            p[[i, pivot]] = p[[pivot, i]]
        for j in range(i + 1, n):
            factor = R[j, i] / R[i, i]
            L[j, i] = factor
            R[j, i:] = R[j, i:] - factor * R[i, i:]
    P = np.eye(n, dtype=int)[p]
    return P, L, R

def decompose_with_plr(A, b):
    A = np.asarray(A, dtype=float)
    b = _as_col(b)
    P, L, R = _plr_decompose(A)
    y = _forward_sub(L, P @ b, unit_diag=True)
    x = _back_sub(R, y)

    print("A =\n", A)
    print("b =\n", b, "\n")
    print("=== PLR (P·A = L·R) ===")
    print("P =\n", P)
    print("L =\n", L)
    print("R =\n", R)
    print("y =\n", y)
    print("x =\n", x)
    print("check P@A == L@R:", np.allclose(P @ A, L @ R))
    print("check A@x == b:", np.allclose(A @ x, b))
    return P, L, R, y, x

# ============================================================
# PART 4 — Call
# ============================================================
decompose_with_plr(A, b)
