# ============================================================
# TOPIC: Eigenwerte — QR-Iteration für alle Eigenwerte
# DESCRIPTION:
# Wiederholtes A_{k+1} = R_k · Q_k (mit Q_k R_k = A_k) bis A_k
# quasi-obere-Dreiecksform annimmt. Wahlweise mit fester Iterationszahl
# oder Toleranz-Abbruch, ggf. mit Rayleigh-Shift. Liest die Eigenwerte
# direkt vom (quasi-)oberen Dreieck ab.
# USE WHEN:
# Wenn alle Eigenwerte einer Matrix iterativ und stabil bestimmt werden
# sollen (oft Standardmethode in numerischen Bibliotheken).
# EXAMPLE:
# A = [[1,1,0],[3,-1,2],[2,-1,3]].
# ============================================================

import numpy as np
import numpy.linalg as lin

np.set_printoptions(precision=6, suppress=True)

# ============================================================
# PART 1 — Inputs
# ============================================================
A = np.array([[1.0,  1.0, 0.0],
              [3.0, -1.0, 2.0],
              [2.0, -1.0, 3.0]])

iters    = 200    # für Methode "fixed"
tol      = 1e-8   # für Methode "tol"
max_iter = 1000   # Sicherheitslimit für "tol"
shift    = False  # Rayleigh-Shift am Ende des Quadrats
debug    = True

# ============================================================
# PART 2 — Method selection
# ============================================================
# method:
#   "fixed" -> immer iters Schritte
#   "tol"   -> Abbruch wenn ||offdiag||_F <= tol oder max_iter erreicht
method = "fixed"

# ============================================================
# PART 3 — Implementation
# ============================================================
def _line(title=""):
    if title:
        print(f"\n{'='*70}\n{title}\n{'='*70}")
    else:
        print("-" * 70)

def _is_square(A):
    return A.ndim == 2 and A.shape[0] == A.shape[1]

def _offdiag_norm(A):
    B = A.copy()
    np.fill_diagonal(B, 0.0)
    return float(lin.norm(B, ord="fro"))

def _eigenvalues_from_quasitriangular(A):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    vals = []
    i = 0
    eps = 1e-12
    while i < n:
        if i < n - 1 and abs(A[i + 1, i]) > eps:
            block = A[i:i+2, i:i+2]
            vals.extend(list(lin.eigvals(block)))
            i += 2
        else:
            vals.append(A[i, i])
            i += 1
    out = []
    for v in vals:
        if isinstance(v, complex) and abs(v.imag) < 1e-10:
            out.append(float(v.real))
        else:
            out.append(v)
    return out

def _qr_fixed(A, iters, shift=False, debug=False):
    A = np.asarray(A, dtype=float)
    if not _is_square(A):
        raise ValueError("A must be square (n,n).")
    Ak = A.copy()
    n = Ak.shape[0]
    for k in range(1, iters + 1):
        mu = Ak[n-1, n-1] if shift else 0.0
        Q, R = lin.qr(Ak - mu * np.eye(n))
        Ak = R @ Q + mu * np.eye(n)
        if debug:
            _line(f"Iteration {k}  (shift mu = {mu})" if shift else f"Iteration {k}")
            print("Q =\n", Q)
            print("\nR =\n", R)
            print("\nA_k = RQ (+ muI) =\n", Ak)
    return Ak

def _qr_tol(A, tol=1e-8, max_iter=1000, shift=False, debug=False):
    A = np.asarray(A, dtype=float)
    if not _is_square(A):
        raise ValueError("A must be square (n,n).")
    Ak = A.copy()
    n = Ak.shape[0]
    for k in range(1, max_iter + 1):
        if _offdiag_norm(Ak) <= tol:
            return Ak, k
        mu = Ak[n-1, n-1] if shift else 0.0
        Q, R = lin.qr(Ak - mu * np.eye(n))
        Ak = R @ Q + mu * np.eye(n)
        if debug:
            _line(f"Iteration {k}  (shift mu = {mu})" if shift else f"Iteration {k}")
            print("A_k =\n", Ak)
    return Ak, max_iter

def compute_eigenvalues_with_qr_iteration(method, A, iters, tol, max_iter, shift, debug=False):
    _line("QR-Verfahren (Eigenwerte über QR-Iteration)")
    print("A =\n", A)
    if method == "fixed":
        A_final = _qr_fixed(A, iters, shift=shift, debug=debug)
    elif method == "tol":
        A_final, _ = _qr_tol(A, tol=tol, max_iter=max_iter, shift=shift, debug=debug)
    else:
        raise ValueError(f"Unbekannte Methode: {method!r}")
    _line("Result")
    print("A_final (quasi obere Dreiecksmatrix) =\n", A_final)
    vals = _eigenvalues_from_quasitriangular(A_final)
    print("\nEigenwerte ~=")
    for i, v in enumerate(vals, start=1):
        print(f"  lambda{i} ~= {v}")
    print("\nnp.linalg.eigvals(A) =", lin.eigvals(A))
    return A_final, vals

# ============================================================
# PART 4 — Call
# ============================================================
compute_eigenvalues_with_qr_iteration(method, A, iters, tol, max_iter, shift, debug)
