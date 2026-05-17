# ============================================================
# TOPIC: Matrix-Zerlegung — QR-Zerlegung via numpy.linalg.qr
# DESCRIPTION:
# Berechnet A = Q·R mit numpy.linalg.qr (intern Householder/Givens)
# und löst Ax = b via x = R^-1 · Q^T · b (Rückwärtseinsetzen).
# USE WHEN:
# Wenn eine stabile Zerlegung gebraucht wird; insbesondere für
# Ausgleichsprobleme oder schlecht konditionierte Matrizen.
# EXAMPLE:
# A = [[2,2,-1],[1,-1,0],[2,0,1]], b = [-1/3, -11/3, 2/3].
# ============================================================

import numpy as np

np.set_printoptions(precision=10, suppress=True)

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
# Only one method here. Verwendet wird die in numpy eingebaute QR.

# ============================================================
# PART 3 — Implementation
# ============================================================
def _as_col(v):
    v = np.asarray(v, dtype=float)
    return v.reshape(-1, 1) if v.ndim == 1 else v

def _back_sub(U, y):
    U = np.asarray(U, dtype=float)
    y = _as_col(y)
    n = U.shape[0]
    x = np.zeros((n, 1), dtype=float)
    for i in range(n - 1, -1, -1):
        s = float(np.dot(U[i, i+1:], x[i+1:, 0]))
        x[i, 0] = (y[i, 0] - s) / U[i, i]
    return x

def decompose_with_qr(A, b):
    A = np.asarray(A, dtype=float)
    b = _as_col(b)
    Q, R = np.linalg.qr(A)
    x = _back_sub(R, Q.T @ b)

    print("A =\n", A)
    print("b =\n", b, "\n")
    print("=== QR (A = Q·R) ===")
    print("Q =\n", Q)
    print("R =\n", R)
    print("x =\n", x)
    print("check Q@R == A:", np.allclose(Q @ R, A))
    print("check A@x == b:", np.allclose(A @ x, b))
    return Q, R, x

# ============================================================
# PART 4 — Call
# ============================================================
decompose_with_qr(A, b)
