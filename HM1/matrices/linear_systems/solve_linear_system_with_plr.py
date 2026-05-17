# ============================================================
# TOPIC: Lineare Systeme — Lösen mit PLR-Zerlegung (P·A = L·R)
# DESCRIPTION:
# Löst Ax = b über P·A = L·R mit Zeilenpivot-Suche: zuerst z = P·b,
# dann L·y = z (vorwärts) und R·x = y (rückwärts).
# USE WHEN:
# Standardweg für direkten LGS-Löser; stabiler als reine LR-Zerlegung.
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
# Only one method here.

# ============================================================
# PART 3 — Implementation
# ============================================================
def _plr_decompose(A):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    R = A.copy()
    L = np.eye(n)
    p = np.arange(n)
    for k in range(n - 1):
        pivot = k + int(np.argmax(np.abs(R[k:, k])))
        if R[pivot, k] == 0.0:
            raise ValueError("PLR: Matrix singulär / Pivot 0.")
        if pivot != k:
            R[[k, pivot], :]  = R[[pivot, k], :]
            L[[k, pivot], :k] = L[[pivot, k], :k]
            p[[k, pivot]]     = p[[pivot, k]]
        for i in range(k + 1, n):
            mult = R[i, k] / R[k, k]
            L[i, k] = mult
            R[i, k:] -= mult * R[k, k:]
    P = np.eye(n, dtype=int)[p]
    return P, L, R

def solve_linear_system_with_plr(A, b):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    P, L, R = _plr_decompose(A)
    n = L.shape[0]

    z = P @ b
    y = np.zeros(n)
    for i in range(n):
        y[i] = z[i] - np.dot(L[i, :i], y[:i])

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(R[i, i + 1:], x[i + 1:])) / R[i, i]

    print("P =\n", P)
    print("L =\n", L)
    print("R =\n", R)
    print(f"z = P·b = {z}")
    print(f"y = {y}")
    print(f"x = {x}")
    print(f"check A@x ~= b: {np.allclose(A @ x, b)}")
    return P, L, R, y, x

# ============================================================
# PART 4 — Call
# ============================================================
solve_linear_system_with_plr(A, b)
