# ============================================================
# TOPIC: Eigenwerte — Spektrum einer Matrix
# DESCRIPTION:
# Druckt das vollständige Spektrum (alle Eigenwerte) einer Matrix und
# gibt seine Mächtigkeit zurück.
# USE WHEN:
# Wenn das Spektrum {lambda_1, ..., lambda_n} einer Matrix als Menge gebraucht
# wird (z.B. um Spektralsätze direkt nachzulesen).
# EXAMPLE:
# A = [[1,0,0],[2,3,0],[0,1,2]].
# ============================================================

import numpy as np

# ============================================================
# PART 1 — Inputs
# ============================================================
matrix = np.array([[1, 0, 0],
                   [2, 3, 0],
                   [0, 1, 2]])
debug = True

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here.

# ============================================================
# PART 3 — Implementation
# ============================================================
def compute_spectrum(A, debug=False):
    eigenwerte = np.linalg.eigvals(A)
    if debug:
        print(f"Spektrum: {eigenwerte}\n")
    print(f"Mächtigkeit des Spektrums: {eigenwerte.size}")
    return eigenwerte

# ============================================================
# PART 4 — Call
# ============================================================
compute_spectrum(matrix, debug)
