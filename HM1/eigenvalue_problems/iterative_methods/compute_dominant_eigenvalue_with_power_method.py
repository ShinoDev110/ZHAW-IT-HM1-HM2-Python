# ============================================================
# TOPIC: Eigenvalues — Von-Mises / Power Method (dominant eigenvalue)
# DESCRIPTION:
# Power iteration v_{k+1} = A v_k / ||A v_k||, with Rayleigh quotient
# lambda ~= v^T A v / (v^T v). Finds the dominant eigenvalue (largest
# in magnitude) and the corresponding eigenvector. Optionally uses a
# fixed number of iterations or a convergence threshold ||v_{k+1} - v_k|| < tol.
# USE WHEN:
# When only the dominant eigenvalue / eigenvector is needed (e.g.
# PageRank-like problems, quick approximation of the spectral radius).
# EXAMPLE:
# A = [[1,1,0],[3,-1,2],[2,-1,3]], v0 = [1,0,0].
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

v0 = np.array([1.0,
               0.0,
               0.0])

iters    = 3       # for method "fixed"
tol      = 1e-4    # for method "tol"
max_iter = 1000    # safety limit for "tol"
debug    = True

# ============================================================
# PART 2 — Method selection
# ============================================================
# method:
#   "fixed" -> always iters steps
#   "tol"   -> stop when ||v_{k+1} - v_k||_2 <= tol or max_iter reached
method = "fixed"

# ============================================================
# PART 3 — Implementation
# ============================================================
def _line(title=""):
    if title:
        print(f"\n{'='*60}\n{title}\n{'='*60}")
    else:
        print(f"{'-'*60}")

def _col(v):
    return np.asarray(v, dtype=float).reshape(-1, 1)

def _norm2(v):
    return float(lin.norm(v, ord=2))

def _rayleigh(A, v):
    Av = A @ v
    return float((v.T @ Av) / (v.T @ v))

def _power_fixed(A, v0, iters, debug=False):
    A = np.asarray(A, dtype=float)
    v = np.asarray(v0, dtype=float).reshape(-1)
    lam = None
    for k in range(1, iters + 1):
        Av = A @ v
        v_next = Av / _norm2(Av)
        lam = _rayleigh(A, v)
        if debug:
            _line(f"Iteration {k}")
            print("v (old) =\n", _col(v))
            print("A*v     =\n", _col(Av))
            print(f"||A*v||2 = {_norm2(Av)}")
            print("v (new, normalized) =\n", _col(v_next))
            print(f"lambda (Rayleigh) ~= {lam}")
        v = v_next
    return lam, v

def _power_tol(A, v0, tol, max_iter=1000, debug=False):
    A = np.asarray(A, dtype=float)
    v_prev = np.asarray(v0, dtype=float).reshape(-1)
    v_next = np.full_like(v_prev, np.inf, dtype=float)
    lam = None
    k = 0
    for k in range(1, max_iter + 1):
        if lin.norm(v_next - v_prev, ord=2) <= tol:
            break
        if k > 1:
            v_prev = v_next
        Av = A @ v_prev
        v_next = Av / _norm2(Av)
        lam = _rayleigh(A, v_prev)
        if debug:
            _line(f"Iteration {k}")
            print("v (old) =\n", _col(v_prev))
            print("A*v     =\n", _col(Av))
            print(f"||A*v||2 = {_norm2(Av)}")
            print("v (new, normalized) =\n", _col(v_next))
            print(f"Delta v (2-norm) = {float(lin.norm(v_next - v_prev, ord=2))}")
            print(f"lambda (Rayleigh) ~= {lam}")
    return lam, v_next, k

def compute_dominant_eigenvalue_with_power_method(method, A, v0, iters, tol, max_iter, debug=False):
    _line("Von Mises: dominant eigenvalue (max |lambda|)")
    print("A =\n", A)
    print("v0 =\n", _col(v0))
    if method == "fixed":
        lam, v = _power_fixed(A, v0, iters, debug=debug)
    elif method == "tol":
        lam, v, _ = _power_tol(A, v0, tol, max_iter=max_iter, debug=debug)
    else:
        raise ValueError(f"Unknown method: {method!r}")
    _line("Result")
    print(f"lambda_max ~= {lam}")
    print("Eigenvector (normalized) v ~=\n", _col(v))
    print("============================================================")
    print("Verification:")
    print("A*v ~= lambda*v ?\n", _col(A @ v), "\nvs\n", _col(lam * v))
    return lam, v

# ============================================================
# PART 4 — Call
# ============================================================
compute_dominant_eigenvalue_with_power_method(method, A, v0, iters, tol, max_iter, debug)
