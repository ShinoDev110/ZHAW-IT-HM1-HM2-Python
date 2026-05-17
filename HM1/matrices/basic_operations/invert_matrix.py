# ============================================================
# TOPIC: Matrizen — Inverse einer Matrix
# DESCRIPTION:
# Berechnet die Inverse A^-1 einer quadratischen Matrix mit
# numpy.linalg.inv.
# USE WHEN:
# Wenn ausnahmsweise wirklich die Inverse benötigt wird; sonst lieber
# A x = b direkt lösen statt explizit zu invertieren.
# EXAMPLE:
# 3x3-Matrix [[-4,1,0],[3,-2,1],[5,2,-1]] -> Inverse.
# ============================================================

import numpy as np

# ============================================================
# PART 1 — Inputs
# ============================================================
matrix = np.array([[-4.0,  1.0,  0.0],
                   [ 3.0, -2.0,  1.0],
                   [ 5.0,  2.0, -1.0]])

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here.

# ============================================================
# PART 3 — Implementation
# ============================================================
def invert_matrix(A):
    A_inv = np.linalg.inv(A)
    print(f"Inverse:\n{A_inv}")
    return A_inv

# ============================================================
# PART 4 — Call
# ============================================================
invert_matrix(matrix)
