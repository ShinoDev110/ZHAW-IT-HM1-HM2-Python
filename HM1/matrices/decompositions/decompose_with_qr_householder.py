# ============================================================
# TOPIC: Matrix decomposition — QR via Householder reflections
# DESCRIPTION:
# Custom implementation of the QR decomposition A = Q·R of a square
# matrix using Householder reflections (instead of np.linalg.qr).
# Optionally prints a report with Q, R and consistency checks.
# USE WHEN:
# When the Householder method needs to be traced step by step
# explicitly (exam problem / teaching purpose).
# EXAMPLE:
# A = [[2,0,1],[7,-5,9],[6,-6,9]].
# ============================================================

import numpy as np

np.set_printoptions(precision=6, suppress=True)

# ============================================================
# PART 1 — Inputs
# ============================================================
A = np.array([
    [2.0,  0.0, 1.0],
    [7.0, -5.0, 9.0],
    [6.0, -6.0, 9.0],
])
debug = False

# ============================================================
# PART 2 — Method selection
# ============================================================
# output:
#   "report"    -> print Q, R + consistency checks verbosely
#   "minimal"   -> print only Q and R
#   "vs_numpy"  -> compare Householder Q,R against numpy.linalg.qr
output = "report"

# ============================================================
# PART 3 — Implementation
# ============================================================
def _is_square(A):
    return A.ndim == 2 and A.shape[0] == A.shape[1]

def _is_orthogonal(Q, tol=1e-10):
    n = Q.shape[0]
    return np.linalg.norm(Q.T @ Q - np.eye(n), ord=np.inf) < tol

def _matrix_deviation(A, B, ord_norm=np.inf):
    return float(np.linalg.norm(A - B, ord=ord_norm))

def _sign(x):
    return 1.0 if x >= 0 else -1.0

def _qr_householder(A, debug=False):
    A = np.array(A, dtype=float, copy=True)
    if not _is_square(A):
        raise ValueError("A must be square (n x n).")
    n = A.shape[0]
    Q = np.eye(n)
    R = A.copy()
    for k in range(n - 1):
        x = R[k:, k].reshape(-1, 1)
        norm_x = float(np.linalg.norm(x, ord=2))
        if norm_x == 0.0:
            continue
        e1 = np.zeros_like(x)
        e1[0, 0] = 1.0
        v = x + _sign(float(x[0, 0])) * norm_x * e1
        norm_v = float(np.linalg.norm(v, ord=2))
        if norm_v == 0.0:
            continue
        v = v / norm_v
        Hk_block = np.eye(x.shape[0]) - 2.0 * (v @ v.T)
        Hk = np.eye(n)
        Hk[k:, k:] = Hk_block
        R = Hk @ R
        Q = Q @ Hk
        if debug:
            print(f"--------------------- step k={k+1}")
            print("x = R[k:, k] =\n", x)
            print("v (normalised) =\n", v)
            print("H_k =\n", Hk)
            print("R = H_k * R =\n", R)
            print("Q = Q * H_k =\n", Q, "\n")
    return Q, R

def _print_qr_report(A, Q, R):
    print("============================================================")
    print("QR decomposition (Householder)")
    print("============================================================")
    print("A =\n", A, "\n")
    print("Q =\n", Q, "\n")
    print("R =\n", R, "\n")
    print("Checks")
    print("------")
    print("Q orthogonal?  ", _is_orthogonal(Q))
    print("R upper-tri?   ", np.allclose(R, np.triu(R), atol=1e-10))
    print("||A - Q R||_inf  =", _matrix_deviation(A, Q @ R, np.inf))
    print("||A - Q R||_2  =", _matrix_deviation(A, Q @ R, 2))
    print("============================================================\n")

def decompose_with_qr_householder(A, output, debug=False):
    if output == "report":
        Q, R = _qr_householder(A, debug=debug)
        _print_qr_report(A, Q, R)
    elif output == "minimal":
        Q, R = _qr_householder(A, debug=debug)
        print("Q=\n", Q)
        print("\nR=\n", R)
    elif output == "vs_numpy":
        Qh, Rh = _qr_householder(A, debug=False)
        Qn, Rn = np.linalg.qr(A)
        print("Householder ||A-QR||_inf:", _matrix_deviation(A, Qh @ Rh, np.inf))
        print("NumPy       ||A-QR||_inf:", _matrix_deviation(A, Qn @ Rn, np.inf))
        print("\nHouseholder R=\n", Rh)
        print("\nNumPy       R=\n", Rn)
        Q, R = Qh, Rh
    else:
        raise ValueError(f"Unknown output choice: {output!r}")
    return Q, R

# ============================================================
# PART 4 — Call
# ============================================================
decompose_with_qr_householder(A, output, debug)
