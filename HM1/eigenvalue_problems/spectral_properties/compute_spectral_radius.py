# ============================================================
# TOPIC: Eigenvalues — Spectral radius of the Jacobi and Gauss-Seidel iteration matrix
# DESCRIPTION:
# Decomposes A into D, L, R, forms the iteration matrices B_J = -D^-1(L+R)
# and B_GS = -(L+D)^-1 R and prints rho(B) = max |eigvals(B)| for each.
# USE WHEN:
# When checking before using iterative linear system solvers whether
# convergence is guaranteed (rho < 1) and which solver will be faster.
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
# Only one method here. Both rho(Jacobi) AND rho(Gauss-Seidel)
# are always computed.

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

def _spectral_radius(M, debug=False):
    eigenvalues = lin.eigvals(M)
    abs_eigenvalues = np.abs(eigenvalues)
    if debug:
        print("Eigenvalues:", eigenvalues)
        print("Magnitudes:", abs_eigenvalues)
        print()
    return float(np.max(abs_eigenvalues))

def compute_spectral_radius(A, debug=False):
    L, D, R = _A_to_LDR(A)
    B_J  = _B_jacobi(L, D, R)
    B_GS = _B_gauss_seidel(L, D, R)
    rho_J  = _spectral_radius(B_J, debug=debug)
    rho_GS = _spectral_radius(B_GS, debug=debug)
    print(f"Spectral radius Jacobi  rho(B) = {rho_J}")
    print(f"Spectral radius GS      rho(B) = {rho_GS}")
    return rho_J, rho_GS

# ============================================================
# PART 4 — Call
# ============================================================
compute_spectral_radius(A, debug)
