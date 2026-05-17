# ============================================================
# TOPIC: Lineare Systeme — Lösen über vorberechnete L,R
# DESCRIPTION:
# Löst Ax = b über eine schon vorhandene LR-Zerlegung A = L·R, indem
# Ly = b vorwärts und Rx = y rückwärts gelöst werden. Hilfreich, wenn
# A einmal zerlegt und mit vielen verschiedenen b's gelöst werden soll.
# USE WHEN:
# Wenn A fix ist und x für mehrere rechte Seiten b benötigt wird —
# einmal LR zerlegen, dann pro b nur 2 Dreieckslösungen.
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
# Only one method here (LR ohne Pivot + Vor/Rückwärtseinsetzen).

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
            raise ValueError("Nullpivot in LR ohne Pivotwahl.")
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
