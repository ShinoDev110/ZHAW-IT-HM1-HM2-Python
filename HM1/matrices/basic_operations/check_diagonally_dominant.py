# ============================================================
# TOPIC: Matrizen — strikte Diagonaldominanz prüfen
# DESCRIPTION:
# Prüft Zeilen- bzw. Spaltensummenkriterium für strikte Diagonaldominanz:
# |a_ii| > Σ_{j≠i} |a_ij| (Zeile) oder |a_jj| > Σ_{i≠j} |a_ij| (Spalte).
# USE WHEN:
# Wenn vor dem Jacobi- oder Gauss-Seidel-Verfahren geprüft werden soll,
# ob hinreichende Konvergenzbedingungen erfüllt sind.
# EXAMPLE:
# Matrix [[-8,5,2],[5,9,-1],[4,-8,7]] auf Diagonaldominanz testen.
# ============================================================

import numpy as np

# ============================================================
# PART 1 — Inputs
# ============================================================
matrix = np.array([[-8.0,  5.0,  2.0],
                   [ 5.0,  9.0, -1.0],
                   [ 4.0, -8.0,  7.0]])
debug = True

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here. Zeilen- UND Spaltensummenkriterium werden geprüft.

# ============================================================
# PART 3 — Implementation
# ============================================================
def _check_zeilendominanz(A, debug=False):
    rows = A.shape[0]
    for i in range(rows):
        diagonal = abs(A[i, i])
        non_diagonal_sum = np.sum(np.abs(A[i, :])) - diagonal
        if diagonal <= non_diagonal_sum:
            return False
    if debug:
        print("Zeilensummenkriterium erfüllt")
        print("für alle i = 1, ...., n |a_ii| > sum^n _j=1, j != i |a_i,j|\n")
    return True

def _check_spaltendominanz(A, debug=False):
    cols = A.shape[1]
    for j in range(cols):
        diagonal = abs(A[j, j])
        non_diagonal_sum = np.sum(np.abs(A[:, j])) - diagonal
        if diagonal <= non_diagonal_sum:
            return False
    if debug:
        print("Spaltensummenkriterium erfüllt")
        print("für alle j = 1, ...., n |a_jj| > sum^n _i=1, i != j |a_i,j|\n")
    return True

def check_diagonally_dominant(A, debug=False):
    z = _check_zeilendominanz(A, debug)
    s = _check_spaltendominanz(A, debug)
    is_dd = z or s
    print(f"Strikte Diagonaldominanz: {is_dd}")
    print("Falls A diagonal dominant ist, konvergiert das Gesamtschrittverfahren (Jacobi)")
    print("und auch das Einzelschrittverfahren (Gauss-Seidel) für Ax = b.")
    return is_dd

# ============================================================
# PART 4 — Call
# ============================================================
check_diagonally_dominant(matrix, debug)
