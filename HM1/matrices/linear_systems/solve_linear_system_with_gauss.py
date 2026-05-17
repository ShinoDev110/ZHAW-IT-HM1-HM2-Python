# ============================================================
# TOPIC: Lineare Systeme — Gauss-Elimination mit Spaltenpivot
# DESCRIPTION:
# Löst Ax = b per Gauss-Elimination mit Zeilenpivot-Suche; liefert
# zusätzlich die Determinante (Produkt der Diagonale von U mal
# (-1)^Anzahl_Zeilenvertauschungen) und die obere Dreiecksmatrix U.
# USE WHEN:
# Wenn ein direkter, stabiler Löser ohne separate L/R-Speicherung
# gebraucht wird oder die Determinante als Nebenprodukt benötigt wird.
# EXAMPLE:
# 3x3-System A = [[2,1,-1],[-3,-1,2],[-2,1,2]], b = [8,-11,-3].
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
        raise ValueError("Matrix A muss quadratisch sein.")
    if b.size != n:
        raise ValueError("Dimensionen von A und b sind nicht kompatibel.")

    U = A.copy()
    rhs = b.copy()
    anzahl_perm = 0

    for k in range(n - 1):
        pivot_row = k + np.argmax(np.abs(U[k:, k]))
        if U[pivot_row, k] == 0.0:
            raise ValueError("Singuläre Matrix – kein eindeutiges LGS.")
        if pivot_row != k:
            U[[k, pivot_row], :] = U[[pivot_row, k], :]
            rhs[[k, pivot_row]]  = rhs[[pivot_row, k]]
            anzahl_perm += 1

        for i in range(k + 1, n):
            faktor = U[i, k] / U[k, k]
            U[i, k:] -= faktor * U[k, k:]
            rhs[i]   -= faktor * rhs[k]

    det = ((-1) ** anzahl_perm) * float(np.prod(np.diag(U)))

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
