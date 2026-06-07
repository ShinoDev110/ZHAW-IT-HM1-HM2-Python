# ============================================================
# TOPIC: Iterative Solvers — Gauss-Seidel method for Ax = b
# DESCRIPTION:
# Iterative Gauss-Seidel solver: (D+L) x^(k+1) = -R x^(k) + b with
# B = -(D+L)^-1 R and C = (D+L)^-1 b. Optionally provides a-priori /
# a-posteriori stop bounds per iteration.
# USE WHEN:
# When Ax = b is to be solved iteratively and A is strictly
# diagonally dominant (Gauss-Seidel generally converges faster than
# Jacobi).
# EXAMPLE:
# 3x3 system with A = [[4,-1,1],[-2,5,1],[1,-2,5]], b = [5,11,12].
# ============================================================

import numpy as np
import numpy.linalg as lin
from math import ceil

np.set_printoptions(precision=6, suppress=True)

# ============================================================
# PART 1 — Inputs
# ============================================================
A = np.array([[ 4.0, -1.0, 1.0],
              [-2.0,  5.0, 1.0],
              [ 1.0, -2.0, 5.0]])

b = np.array([[5.0],
              [11.0],
              [12.0]])

x0 = np.array([[0.0],
               [0.0],
               [0.0]])

norm     = np.inf  # 1, 2 or np.inf
tol      = 1e-2    # tolerance for stop criterion
max_iter = 10_000  # max. iterations
debug    = True    # print intermediate results

# ============================================================
# PART 2 — Method selection
# ============================================================
# Choose stop mode:
#   "fixed"        -> always max_iter steps
#   "aposteriori"  -> stop when a-posteriori bound <= tol
#   "apriori"      -> stop when k >= a-priori iteration requirement n
#   "both"         -> stop when (a-posteriori <= tol) OR (k >= n)
stop_mode = "both"

# ============================================================
# PART 3 — Implementation
# ============================================================
def _as_col(v):
    v = np.asarray(v, dtype=float)
    return v.reshape(-1, 1) if v.ndim == 1 else v

def _A_to_LDR(A, debug=False):
    A = np.asarray(A, dtype=float)
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    R = np.triu(A, 1)
    if debug:
        print("-- decompose A into D, L, R")
        print("D =\n", D)
        print("L =\n", L)
        print("R =\n", R, "\n")
    return L, D, R

def _gs_BC_from_LDR(L, D, R, b, debug=False):
    M_inv = lin.inv(D + L)
    B = -M_inv @ R
    C = M_inv @ _as_col(b)
    if debug:
        print("-- Gauss-Seidel: B and C")
        print("B = -(D+L)^-1 R =\n", B)
        print("C = (D+L)^-1 b  =\n", C, "\n")
    return B, C

def _gs_step(B, C, x_prev, debug=False):
    x_prev = _as_col(x_prev)
    if debug:
        print("B * x + C =")
        print(B, "\n*\n", x_prev, "\n+\n", C)
    return B @ x_prev + C

def _a_posteriori(B, x_n, x_n_minus_1, norm=np.inf, debug=False):
    B_norm = lin.norm(B, ord=norm)
    if B_norm >= 1:
        raise ValueError(f"||B|| = {B_norm} >= 1 -> estimate not valid (no contraction).")
    diff_norm = lin.norm(_as_col(x_n) - _as_col(x_n_minus_1), ord=norm)
    err = (B_norm / (1 - B_norm)) * diff_norm
    if debug:
        print("-- A-posteriori")
        print(f"||B|| = {B_norm}")
        print(f"||x^n - x^(n-1)|| = {diff_norm}")
        print(f"||x^n - x̄|| <= (||B||/(1-||B||)) * ||x^n - x^(n-1)|| = {err}\n")
    return err

def _a_priori_iterations(B, x0, x1, tol, norm=np.inf, debug=False):
    B_norm = lin.norm(B, ord=norm)
    if B_norm >= 1:
        raise ValueError(f"||B|| = {B_norm} >= 1 -> estimate not valid (no contraction).")
    step_norm = lin.norm(_as_col(x1) - _as_col(x0), ord=norm)
    if step_norm == 0:
        return 0.0, 0
    q = (tol * (1 - B_norm)) / step_norm
    if q <= 0 or B_norm <= 0:
        return float("inf"), 0
    n_real = np.log(q) / np.log(B_norm)
    n_int = max(0, ceil(n_real))
    if debug:
        print("-- A-priori")
        print(f"||B|| = {B_norm}")
        print(f"||x^1 - x^0|| = {step_norm}")
        print(f"tol = {tol}")
        print(f"q = tol*(1-||B||)/||x1-x0|| = {q}")
        print(f"n >= log(q) / log(||B||) = {n_real}")
        print(f"n = {n_int}\n")
    return n_real, n_int

def _print_result(B, C, xs, info, A, b):
    print("\n=== Gauss-Seidel Matrices ===")
    print("B =\n", B)
    print("C =\n", C, "\n")
    print("=== Stop Info ===")
    for k, v in info.items():
        print(f"{k}: {v}")
    k_last = info["iterations_done"]
    x_star = xs[k_last]
    print(f"\nLast approximation x^{k_last} =\n{x_star}")
    print("check A@x == b:", np.allclose(A @ x_star, b))

def solve_with_gauss_seidel(A, b, x0, tol, max_iter, norm=np.inf,
                            stop_mode="aposteriori", debug=False):
    print("A =\n", A)
    print("b =\n", b)
    print("x0 =\n", x0, "\n")

    L, D, R = _A_to_LDR(A, debug=debug)
    B, C = _gs_BC_from_LDR(L, D, R, b, debug=debug)

    stop_mode = (stop_mode or "aposteriori").lower()
    x = _as_col(x0)
    xs = [x.copy()]

    B_norm = lin.norm(B, ord=norm)
    if stop_mode in ("aposteriori", "apriori", "both") and B_norm >= 1:
        raise ValueError(f"||B|| = {B_norm} >= 1 -> a-priori/a-posteriori stop not valid (no contraction).")

    n_apriori_real = None
    n_apriori_int  = None
    last_post      = None
    stop_reason    = None

    if stop_mode in ("apriori", "both") and max_iter >= 1:
        x1 = _gs_step(B, C, x, debug=False)
        xs.append(x1.copy())
        n_apriori_real, n_apriori_int = _a_priori_iterations(B, xs[0], xs[1], tol, norm=norm, debug=debug)
        if debug:
            print(f"Target iterations from a-priori: n = {n_apriori_real} -> ceil = {n_apriori_int}\n")
        if n_apriori_int <= 1:
            stop_reason = "a-priori reached (n<=1)"
            info = {
                "stop_mode": stop_mode, "stop_reason": stop_reason,
                "iterations_done": 1, "B_norm": B_norm,
                "a_priori_n_real": n_apriori_real, "a_priori_n_int": n_apriori_int,
                "a_posteriori_last": None,
            }
            _print_result(B, C, xs, info, A, b)
            return B, C, xs
        x = x1
        if stop_mode == "both":
            last_post = _a_posteriori(B, xs[1], xs[0], norm=norm, debug=False)
            if last_post <= tol:
                stop_reason = "a-posteriori reached (after k=1)"
                info = {
                    "stop_mode": stop_mode, "stop_reason": stop_reason,
                    "iterations_done": 1, "B_norm": B_norm,
                    "a_priori_n_real": n_apriori_real, "a_priori_n_int": n_apriori_int,
                    "a_posteriori_last": last_post,
                }
                _print_result(B, C, xs, info, A, b)
                return B, C, xs
        start_k = 2
    else:
        start_k = 1

    for k in range(start_k, max_iter + 1):
        if debug:
            print(f"--------------------- Iteration {k}")
            print(f"x^{k} = B * x^{k-1} + C")
        x_new = _gs_step(B, C, x, debug=False)
        xs.append(x_new.copy())
        if debug:
            print("=\n", x_new, "\n")

        if stop_mode in ("aposteriori", "both"):
            last_post = _a_posteriori(B, xs[k], xs[k - 1], norm=norm, debug=False)
            if debug:
                print(f"A-posteriori bound (k={k}): {last_post}")
            if last_post <= tol:
                stop_reason = f"a-posteriori reached (k={k})"
                break

        if stop_mode in ("apriori", "both") and n_apriori_int is not None:
            if k >= n_apriori_int:
                stop_reason = f"a-priori reached (k={k} >= {n_apriori_int})"
                break

        x = x_new

    if stop_reason is None:
        stop_reason = f"max_iter reached (k={len(xs)-1})"

    info = {
        "stop_mode": stop_mode, "stop_reason": stop_reason,
        "iterations_done": len(xs) - 1, "B_norm": B_norm,
        "a_priori_n_real": n_apriori_real, "a_priori_n_int": n_apriori_int,
        "a_posteriori_last": last_post,
    }
    _print_result(B, C, xs, info, A, b)
    return B, C, xs

# ============================================================
# PART 4 — Call
# ============================================================
solve_with_gauss_seidel(A, b, x0, tol, max_iter, norm=norm,
                        stop_mode=stop_mode, debug=debug)
