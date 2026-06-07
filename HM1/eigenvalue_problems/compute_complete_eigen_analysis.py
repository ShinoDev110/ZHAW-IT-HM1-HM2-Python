# ============================================================
# TOPIC: Eigenvalues — complete eigen analysis (alg. + geom. multiplicity)
# DESCRIPTION:
# For a square matrix A: numpy.eig provides eigenvalues and
# eigenvectors; the Gauss algorithm brings A - lambdaI into row echelon form
# (REF), from which the geometric multiplicity n - rank(A - lambdaI)
# follows. Output sorted (real before complex).
# USE WHEN:
# When algebraic AND geometric multiplicity including associated eigenvectors
# are needed for every eigenvalue at a glance.
# EXAMPLE:
# A = [[1,1,1],[0,2,0],[1,-1,1]].
# ============================================================

import numpy as np
from numpy.linalg import matrix_rank
from collections import Counter

np.set_printoptions(precision=6, suppress=True)

# ============================================================
# PART 1 — Inputs
# ============================================================
A = np.array([[1, 1, 1],
              [0, 2, 0],
              [1, -1, 1]], dtype=float)

decimals = 10
debug    = True

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here. Both alg. + geom. multiplicity are always
# computed and printed together.

# ============================================================
# PART 3 — Implementation
# ============================================================
def _round_eigenvalue(val, decimals=10):
    val = complex(val)
    if np.isclose(val.imag, 0, atol=10 ** (-decimals)):
        return float(np.round(val.real, decimals))
    return complex(np.round(val.real, decimals), np.round(val.imag, decimals))

def _format_lambda(lam, decimals=10):
    if isinstance(lam, complex):
        return f"{lam.real:.{decimals}f} {lam.imag:+.{decimals}f}i"
    return f"{float(lam):.{decimals}f}".rstrip("0").rstrip(".")

def _algebraic_multiplicity(A, decimals=10):
    eigenvalues = np.linalg.eigvals(A)
    eigenvalues = [_round_eigenvalue(v, decimals) for v in eigenvalues]
    return dict(Counter(eigenvalues))

def _matrix_to_ref(matrix, debug=False):
    matrix = np.array(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Must be a square matrix (n,n).")
    n = matrix.shape[0]
    if debug:
        print(f"---- Bringing A - lambdaI_{n} to row echelon form (Gauss)")
    for i in range(n):
        if matrix[i, i] == 0:
            continue
        for j in range(i + 1, n):
            if matrix[j, i] != 0:
                factor = matrix[j, i] / matrix[i, i]
                if debug:
                    print("---- Step")
                    print("Current:")
                    print(matrix)
                matrix[j] = matrix[j] - factor * matrix[i]
                if debug:
                    print(f"R{j+1} := R{j+1} - ({factor}) * R{i+1}")
                    print("New:")
                    print(matrix, "\n")
    if debug:
        print("---- Result REF:")
        print(matrix, "\n")
    return matrix

def _geometric_multiplicity(A, debug=False, eigenvalues=None, decimals=10):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    gm = {}
    if eigenvalues is None:
        eigenvalues = list(np.linalg.eigvals(A))
    eigenvalues = [_round_eigenvalue(v, decimals) for v in eigenvalues]
    for lam in eigenvalues:
        diff = A - float(lam) * np.eye(n) if not isinstance(lam, complex) else A - lam * np.eye(n)
        rank = matrix_rank(diff)
        gm[lam] = n - rank
        if debug:
            print(f"-- Geometric multiplicity for lambda = {_format_lambda(lam, decimals)}")
            print(f"rank(A - lambdaI) = {rank}  ->  n - rank = {n} - {rank} = {gm[lam]}")
            print("A - lambdaI =")
            print(diff, "\n")
            _matrix_to_ref(diff, debug=True)
    return gm

def compute_complete_eigen_analysis(A, debug=False, decimals=10):
    A = np.asarray(A, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    eigenvalues_clean = [_round_eigenvalue(v, decimals) for v in eigenvalues]

    def sort_key(i):
        lam = eigenvalues_clean[i]
        if isinstance(lam, complex):
            return (1, lam.real, lam.imag)
        return (0, float(lam), 0.0)
    order = sorted(range(len(eigenvalues_clean)), key=sort_key)

    if debug:
        print("=" * 50)
        print("Eigenvalues & Eigenvectors")
        print("-" * 50)
        print("Matrix A:")
        print(A)
        print("-" * 50)
        print("Algebraic Multiplicity:")
        am = _algebraic_multiplicity(A, decimals)
        for lam in sorted(am.keys(), key=lambda x: (isinstance(x, complex),
                                                     float(np.real(x)), float(np.imag(x)))):
            print(f"  lambda = {_format_lambda(lam, decimals):<20} -> m_a = {am[lam]}")
        print("-" * 50)

    for idx, i in enumerate(order, start=1):
        lam = eigenvalues_clean[i]
        v = eigenvectors[:, i].reshape(-1, 1)
        scale = np.max(np.abs(v))
        v_scaled = v / scale if scale != 0 else v
        if debug:
            print(f"[{idx}] lambda = {_format_lambda(lam, decimals)}")
            _geometric_multiplicity(A, debug=True, eigenvalues=[lam], decimals=decimals)
            print("Eigenvector (raw):")
            print(v)
            print("Eigenvector (scaled, max(|v_i|)=1):")
            print(v_scaled)
            print("-" * 50)

    return eigenvalues_clean, eigenvectors

# ============================================================
# PART 4 — Call
# ============================================================
compute_complete_eigen_analysis(A, debug, decimals)
