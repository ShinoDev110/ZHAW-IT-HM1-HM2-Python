# ============================================================
# TOPIC: Diagonalisierung — T aus Eigenvektoren bauen und D = T^-1 A T
# DESCRIPTION:
# Baut die Transformationsmatrix T aus den Spalten der Eigenvektoren,
# prüft rank(T) == n und berechnet D = T^-1 A T. Wahlweise auch mit
# einer manuell vorgegebenen T-Matrix vergleichen.
# USE WHEN:
# Wenn geprüft werden soll, ob eine Matrix A diagonalisierbar ist und
# welche Diagonalform sich ergibt.
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

# Optional: manuell vorgegebenes T (für Methode "manual"):
T_given = np.array([[ 3.0, 1.0, 1.0],
                    [-1.0, 1.0, 2.0],
                    [-3.0, 0.0, 1.0]])

# ============================================================
# PART 2 — Method selection
# ============================================================
# method:
#   "auto"   -> T aus Eigenvektoren von A bauen, D = T^-1 A T prüfen
#   "manual" -> mit vorgegebenem T_given vergleichen
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
        raise ValueError("A muss quadratisch sein.")
    n = A.shape[0]

    if method == "auto":
        _line("Automatic diagonalization check (build T from eigenvectors)")
        print("A =\n", A)
        T, eigenwerte, rank_T = _make_T_from_eigenvectors(A, tol=tol)
        _line("Eigenwerte / Eigenvektoren via numpy.linalg.eig")
        print("Eigenwerte (ungeordnet) =")
        for i, lam in enumerate(eigenwerte, start=1):
            print(f"  lambda{i} = {lam}")
        print("\nT (Spalten = Eigenvektoren) =\n", T)
        print(f"\nrank(T) = {rank_T} (braucht {n} für Diagonalisierbarkeit)")
        if rank_T < n:
            _line("Result")
            print("Nicht diagonalisierbar (zu wenige linear unabhängige Eigenvektoren).")
            return None, eigenwerte, T
        D, diag_ok = _diagonalize_with_T(A, T, tol=tol, debug=debug)
        _line("Result")
        if diag_ok:
            print("Diagonalisierbar.")
            print("\nD = T^-1 A T =\n", D)
            print("\nDiagonal von D (sollte Eigenwerte sein):")
            for i, d in enumerate(np.diag(D), start=1):
                print(f"  D[{i},{i}] = {d}")
        else:
            print("T invertierbar, aber D numerisch nicht diagonal (Rundung/komplex).")
            print("\nD =\n", D)
        return D, eigenwerte, T

    if method == "manual":
        _line("Manual check with provided T")
        D, ok = _diagonalize_with_T(A, T_given, tol=tol, debug=debug)
        print("\nDiagonal with this T" if ok else "\nNOT diagonal with this T")
        return D, None, T_given

    raise ValueError(f"Unbekannte Methode: {method!r}")

# ============================================================
# PART 4 — Call
# ============================================================
build_diagonalization_matrix_t(method, A, T_given, tol, debug)
