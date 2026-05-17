# ============================================================
# TOPIC: Eigenwerte — Spektralradius der Jacobi- und Gauss-Seidel-Matrix
# DESCRIPTION:
# Zerlegt A in D, L, R, bildet die Iterationsmatrizen B_J = -D^-1(L+R)
# und B_GS = -(L+D)^-1 R und druckt jeweils rho(B) = max |eigvals(B)|.
# USE WHEN:
# Wenn vor dem Einsatz iterativer LGS-Löser geprüft werden soll, ob
# Konvergenz garantiert ist (rho < 1) und welcher Löser schneller wird.
# EXAMPLE:
# A = [[1,0,0],[2,3,0],[0,1,2]] -> rho(B_J), rho(B_GS).
# ============================================================

import numpy as np
import numpy.linalg as lin

# ============================================================
# PART 1 — Inputs
# ============================================================
A = np.array([[1.0, 0.0, 0.0],
              [2.0, 3.0, 0.0],
              [0.0, 1.0, 2.0]])
debug = True

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here. Es werden immer rho(Jacobi) UND rho(Gauss-Seidel)
# berechnet.

# ============================================================
# PART 3 — Implementation
# ============================================================
def _A_to_LDR(A):
    A = np.asarray(A, dtype=float)
    D = np.diag(np.diag(A))
    L = np.tril(A) - D
    R = np.triu(A) - D
    return L, D, R

def _B_jacobi(L, D, R):
    return -lin.solve(D, L + R)

def _B_gauss_seidel(L, D, R):
    return -lin.solve(L + D, R)

def _spektral_radius(M, debug=False):
    ew = lin.eigvals(M)
    abs_ew = np.abs(ew)
    if debug:
        print("Eigenwerte:", ew)
        print("Beträge:", abs_ew)
        print()
    return float(np.max(abs_ew))

def compute_spectral_radius(A, debug=False):
    L, D, R = _A_to_LDR(A)
    B_J  = _B_jacobi(L, D, R)
    B_GS = _B_gauss_seidel(L, D, R)
    rho_J  = _spektral_radius(B_J, debug=debug)
    rho_GS = _spektral_radius(B_GS, debug=debug)
    print(f"Spektralradius Jacobi  rho(B) = {rho_J}")
    print(f"Spektralradius GS      rho(B) = {rho_GS}")
    return rho_J, rho_GS

# ============================================================
# PART 4 — Call
# ============================================================
compute_spectral_radius(A, debug)
