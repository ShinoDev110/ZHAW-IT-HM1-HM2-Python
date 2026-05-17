# ============================================================
# TOPIC: Eigenwerte — vollständige Eigenanalyse (alg. + geom. Vielfachheit)
# DESCRIPTION:
# Für eine quadratische Matrix A: numpy.eig liefert Eigenwerte und
# Eigenvektoren; der Gauss-Algorithmus bringt A - lambdaI in Zeilenstufenform
# (ZSF), wodurch sich die geometrische Vielfachheit n - rg(A - lambdaI)
# ergibt. Ausgabe sortiert (reell vor komplex).
# USE WHEN:
# Wenn für jeden Eigenwert algebraische UND geometrische Vielfachheit
# inklusive zugehöriger Eigenvektoren auf einen Blick gebraucht werden.
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
# Only one method here. Es werden alg. + geom. Vielfachheit immer zusammen
# berechnet und ausgegeben.

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

def _algebraische_vielfachheit(A, decimals=10):
    ew = np.linalg.eigvals(A)
    ew = [_round_eigenvalue(v, decimals) for v in ew]
    return dict(Counter(ew))

def _matrix_in_zsf(matrix, debug=False):
    matrix = np.array(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Muss eine quadratische Matrix sein (n,n).")
    n = matrix.shape[0]
    if debug:
        print(f"---- A - lambdaI_{n} auf ZSF bringen (Gauss)")
    for i in range(n):
        if matrix[i, i] == 0:
            continue
        for j in range(i + 1, n):
            if matrix[j, i] != 0:
                factor = matrix[j, i] / matrix[i, i]
                if debug:
                    print("---- Schritt")
                    print("Aktuell:")
                    print(matrix)
                matrix[j] = matrix[j] - factor * matrix[i]
                if debug:
                    print(f"Z{j+1} := Z{j+1} - ({factor}) * Z{i+1}")
                    print("Neu:")
                    print(matrix, "\n")
    if debug:
        print("---- Ergebnis ZSF:")
        print(matrix, "\n")
    return matrix

def _geometrische_vielfachheit(A, debug=False, eigenvalues=None, decimals=10):
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
            print(f"-- Geometrische Vielfachheit zu lambda = {_format_lambda(lam, decimals)}")
            print(f"rg(A - lambdaI) = {rank}  ->  n - rg = {n} - {rank} = {gm[lam]}")
            print("A - lambdaI =")
            print(diff, "\n")
            _matrix_in_zsf(diff, debug=True)
    return gm

def compute_complete_eigen_analysis(A, debug=False, decimals=10):
    A = np.asarray(A, dtype=float)
    ew, ev = np.linalg.eig(A)
    ew_clean = [_round_eigenvalue(v, decimals) for v in ew]

    def sort_key(i):
        lam = ew_clean[i]
        if isinstance(lam, complex):
            return (1, lam.real, lam.imag)
        return (0, float(lam), 0.0)
    order = sorted(range(len(ew_clean)), key=sort_key)

    if debug:
        print("=" * 50)
        print("Eigenwerte & Eigenvektoren")
        print("-" * 50)
        print("Matrix A:")
        print(A)
        print("-" * 50)
        print("Algebraische Vielfachheit:")
        am = _algebraische_vielfachheit(A, decimals)
        for lam in sorted(am.keys(), key=lambda x: (isinstance(x, complex),
                                                     float(np.real(x)), float(np.imag(x)))):
            print(f"  lambda = {_format_lambda(lam, decimals):<20} -> m_a = {am[lam]}")
        print("-" * 50)

    for idx, i in enumerate(order, start=1):
        lam = ew_clean[i]
        v = ev[:, i].reshape(-1, 1)
        scale = np.max(np.abs(v))
        v_scaled = v / scale if scale != 0 else v
        if debug:
            print(f"[{idx}] lambda = {_format_lambda(lam, decimals)}")
            _geometrische_vielfachheit(A, debug=True, eigenvalues=[lam], decimals=decimals)
            print("Eigenvektor (roh):")
            print(v)
            print("Eigenvektor (skaliert, max(|v_i|)=1):")
            print(v_scaled)
            print("-" * 50)

    return ew_clean, ev

# ============================================================
# PART 4 — Call
# ============================================================
compute_complete_eigen_analysis(A, debug, decimals)
