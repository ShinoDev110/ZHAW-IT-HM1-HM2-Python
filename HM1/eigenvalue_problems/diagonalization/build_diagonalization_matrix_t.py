# ============================================================
# TOPIC: Diagonalization — build T from eigenvectors and D = T^-1 A T
# DESCRIPTION:
# Builds the transformation matrix T from the columns of the eigenvectors,
# checks rank(T) == n and computes D = T^-1 A T. Optionally also compare
# with a manually provided T matrix.
# USE WHEN:
# When it should be checked whether a matrix A is diagonalizable and
# what diagonal form results.
# EXAMPLE:
# A = [[2,0,1],[7,-5,9],[6,-6,9]].
# ============================================================

import numpy as np

np.set_printoptions(precision=6, suppress=True)

# ============================================================
# PART 1 — Inputs
# ============================================================
A = np.array([[2.0,  0.0, 1.0],
              [7.0, -5.0, 9.0],
              [6.0, -6.0, 9.0]])

tol   = 1e-10
debug = True

# Optional: manually provided T (for method "manual"):
T_given = np.array([[ 3.0, 1.0, 1.0],
                    [-1.0, 1.0, 2.0],
                    [-3.0, 0.0, 1.0]])

# ============================================================
# PART 2 — Method selection
# ============================================================
# method:
#   "auto"   -> build T from eigenvectors of A, check D = T^-1 A T
#   "manual" -> compare with provided T_given
method = "auto"

# ============================================================
# PART 3 — Implementation
# ============================================================
def _line(title=""):
    if title:
        print(f"\n{'='*70}\n{title}\n{'='*70}")
    else:
        print("-" * 70)

def _is_diagonal(M, tol=1e-10):
    M = np.asarray(M, dtype=float)
    off = M.copy()
    np.fill_diagonal(off, 0.0)
    return np.all(np.abs(off) <= tol)

def _make_T_from_eigenvectors(A, tol=1e-10):
    vals, vecs = np.linalg.eig(A)
    vals_clean = []
    for v in vals:
        if abs(v.imag) < tol:
            vals_clean.append(float(v.real))
        else:
            vals_clean.append(v)
    rank_T = np.linalg.matrix_rank(vecs, tol=tol)
    return vecs, np.array(vals_clean, dtype=object), rank_T

def _diagonalize_with_T(A, T, tol=1e-10, debug=False):
    try:
        T_inv = np.linalg.inv(T)
    except np.linalg.LinAlgError:
        return None, False
    D = T_inv @ A @ T
    if debug:
        print("\nT^-1 =\n", T_inv)
        print("\nD = T^-1 A T =\n", D)
        print(f"\nIs diagonal (tol={tol})? -> {_is_diagonal(D, tol)}")
    return D, _is_diagonal(D, tol)

def build_diagonalization_matrix_t(method, A, T_given, tol=1e-10, debug=False):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")
    n = A.shape[0]

    if method == "auto":
        _line("Automatic diagonalization check (build T from eigenvectors)")
        print("A =\n", A)
        T, eigenvalues, rank_T = _make_T_from_eigenvectors(A, tol=tol)
        _line("Eigenvalues / Eigenvectors via numpy.linalg.eig")
        print("Eigenvalues (unordered) =")
        for i, lam in enumerate(eigenvalues, start=1):
            print(f"  lambda{i} = {lam}")
        print("\nT (columns = eigenvectors) =\n", T)
        print(f"\nrank(T) = {rank_T} (needs {n} for diagonalizability)")
        if rank_T < n:
            _line("Result")
            print("Not diagonalizable (too few linearly independent eigenvectors).")
            return None, eigenvalues, T
        D, diag_ok = _diagonalize_with_T(A, T, tol=tol, debug=debug)
        _line("Result")
        if diag_ok:
            print("Diagonalizable.")
            print("\nD = T^-1 A T =\n", D)
            print("\nDiagonal of D (should be eigenvalues):")
            for i, d in enumerate(np.diag(D), start=1):
                print(f"  D[{i},{i}] = {d}")
        else:
            print("T invertible, but D numerically not diagonal (rounding/complex).")
            print("\nD =\n", D)
        return D, eigenvalues, T

    if method == "manual":
        _line("Manual check with provided T")
        D, ok = _diagonalize_with_T(A, T_given, tol=tol, debug=debug)
        print("\nDiagonal with this T" if ok else "\nNOT diagonal with this T")
        return D, None, T_given

    raise ValueError(f"Unknown method: {method!r}")

# ============================================================
# PART 4 — Call
# ============================================================
build_diagonalization_matrix_t(method, A, T_given, tol, debug)
