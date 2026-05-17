# ============================================================
# TOPIC: Matrizen — Matrixnormen und Konditionszahlen
# DESCRIPTION:
# Druckt ||A||_1, ||A||_2, ||A||_inf und (falls A quadratisch) die
# Konditionszahlen cond_1(A), cond_2(A), cond_inf(A).
# USE WHEN:
# Wenn die Stabilität eines linearen Gleichungssystems Ax = b
# eingeschätzt werden soll.
# EXAMPLE:
# 3x3-Matrix [[240,120,80],[60,180,170],[60,90,500]].
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
debug = False  # zusätzlich A^-1 ausgeben (nur für quadratische A)

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here. Es werden immer alle drei Normen und (falls
# quadratisch) alle drei Konditionszahlen berechnet.

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
            print("A^-1 (Referenz):")
            print(lin.inv(A))
    else:
        print("cond_p(A): nur für quadratische Matrizen definiert (übersprungen).")

# ============================================================
# PART 4 — Call
# ============================================================
compute_matrix_norm_and_condition(A, debug)
