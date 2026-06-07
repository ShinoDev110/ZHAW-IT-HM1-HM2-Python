# ============================================================
# TOPIC: Iterative Solvers — relaxation method (damped Jacobi / JOR)
# DESCRIPTION:
# Generalised Jacobi method with relaxation parameter omega. Instead of
# D x = -(L+R) x + b the form (1/omega) D x = -(((omega-1)/omega) D + L + R) x + b
# is used. This yields the fixed-point iteration x^(k+1) = B_omega x^(k) + c_omega
# with B_omega = (1-omega) I - omega D^-1 (L+R) and c_omega = omega D^-1 b.
# For omega = 1 this reduces exactly to the classical Jacobi method.
# USE WHEN:
# When it should be investigated whether a relaxation parameter omega != 1
# improves the convergence of the Jacobi method (smaller ||B|| or rho(B)).
# EXAMPLE:
# A = [[7,-2,-2],[-2,7,-2],[-2,-2,7]], b = [5,-13,14], omega = 1.15,
# stop via a-posteriori bound in the 1-norm < 1e-9.
# ============================================================

import numpy as np
import numpy.linalg as lin
from math import ceil

np.set_printoptions(precision=6, suppress=True)

# ============================================================
# PART 1 — Inputs
# ============================================================
A = np.array([[ 7.0, -2.0, -2.0],
              [-2.0,  7.0, -2.0],
              [-2.0, -2.0,  7.0]])              # coefficient matrix

b = np.array([[  5.0],
              [-13.0],
              [ 14.0]])                          # right-hand side

x0 = np.array([[0.0],
               [0.0],
               [0.0]])                          # initial vector

omega    = 1.15    # relaxation parameter (omega = 1 -> classical Jacobi)
tol      = 1e-9    # tolerance for stop criterion
max_iter = 10_000  # maximum number of iterations
norm     = 1       # norm used (1, 2 or np.inf); check uses 1-norm
debug    = True    # print intermediate results

# List of omega values for comparison mode "compare"
omega_list = [1.0, 1.05, 1.15, 1.25, 1.5]

# ============================================================
# PART 2 — Method selection
# ============================================================
# mode:
#   "solve"   -> solves the system with the omega set above
#   "compare" -> tests all omega from omega_list and shows ||B||, rho(B)
#                and required iterations (shows whether omega is better)
mode = "solve"

# stop_mode (only for mode = "solve"):
#   "fixed"        -> always max_iter steps
#   "aposteriori"  -> stop when a-posteriori bound <= tol
#   "apriori"      -> stop when k >= a-priori iteration requirement n
#   "both"         -> stop when (a-posteriori <= tol) OR (k >= n)
stop_mode = "aposteriori"

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

def _relax_BC_from_LDR(L, D, R, b, omega, debug=False):
    # B_omega = (1-omega) I - omega D^-1 (L+R),  c_omega = omega D^-1 b
    D_inv = lin.inv(D)
    n = D.shape[0]
    B = (1.0 - omega) * np.eye(n) - omega * D_inv @ (L + R)
    C = omega * D_inv @ _as_col(b)
    if debug:
        print(f"-- Relaxation (omega = {omega}): B and c")
        print("B_omega = (1-omega) I - omega D^-1 (L+R) =\n", B)
        print("c_omega = omega D^-1 b =\n", C, "\n")
    return B, C

def _spectral_radius(B):
    return float(np.max(np.abs(lin.eigvals(B))))

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

def _relax_step(B, C, x_prev):
    return B @ _as_col(x_prev) + C

def _iterate(B, C, x0, tol, max_iter, norm, stop_mode, debug=False):
    stop_mode = (stop_mode or "aposteriori").lower()
    x = _as_col(x0)
    xs = [x.copy()]
    B_norm = lin.norm(B, ord=norm)

    if stop_mode in ("aposteriori", "apriori", "both") and B_norm >= 1:
        raise ValueError(f"||B|| = {B_norm} >= 1 -> stop criteria not valid (no contraction).")

    n_apriori_int = None
    last_post = None
    stop_reason = None

    if stop_mode in ("apriori", "both") and max_iter >= 1:
        x1 = _relax_step(B, C, x)
        xs.append(x1.copy())
        _, n_apriori_int = _a_priori_iterations(B, xs[0], xs[1], tol, norm=norm, debug=debug)
        x = x1
        start_k = 2
    else:
        start_k = 1

    for k in range(start_k, max_iter + 1):
        x_new = _relax_step(B, C, x)
        xs.append(x_new.copy())
        if debug:
            print(f"--- Iteration {k}: x^{k} =\n{x_new.ravel()}")

        if stop_mode in ("aposteriori", "both"):
            last_post = _a_posteriori(B, xs[k], xs[k - 1], norm=norm, debug=False)
            if last_post <= tol:
                stop_reason = f"a-posteriori reached (k={k})"
                break
        if stop_mode in ("apriori", "both") and n_apriori_int is not None and k >= n_apriori_int:
            stop_reason = f"a-priori reached (k={k} >= {n_apriori_int})"
            break
        x = x_new

    if stop_reason is None:
        stop_reason = f"max_iter reached (k={len(xs) - 1})"
    return xs, stop_reason, last_post

def solve_with_relaxation(mode, A, b, x0, omega, tol, max_iter, norm,
                          stop_mode, omega_list, debug=False):
    print("A =\n", A)
    print("b =\n", b)
    print("x0 =\n", x0, "\n")
    L, D, R = _A_to_LDR(A, debug=debug)

    if mode == "compare":
        print("=== Comparison of different omega values (omega = 1 is classical Jacobi) ===")
        print(f"{'omega':>6} | {'||B||':>10} | {'rho(B)':>10} | {'Iterations':>11} | Convergence")
        print("-" * 64)
        for w in omega_list:
            B, C = _relax_BC_from_LDR(L, D, R, b, w, debug=False)
            B_norm = lin.norm(B, ord=norm)
            rho = _spectral_radius(B)
            if B_norm >= 1:
                print(f"{w:>6.3f} | {B_norm:>10.6f} | {rho:>10.6f} | {'-':>11} | "
                      f"||B|| >= 1 (a-posteriori not valid)")
                continue
            xs, reason, _ = _iterate(B, C, x0, tol, max_iter, norm, "aposteriori", debug=False)
            iters = len(xs) - 1
            print(f"{w:>6.3f} | {B_norm:>10.6f} | {rho:>10.6f} | {iters:>11d} | {reason}")
        print("\nSmaller rho(B) or ||B|| => faster convergence.")
        return None

    if mode != "solve":
        raise ValueError(f"Unknown mode: {mode!r}")

    B, C = _relax_BC_from_LDR(L, D, R, b, omega, debug=debug)
    B_norm = lin.norm(B, ord=norm)
    rho = _spectral_radius(B)
    print(f"||B||_{('inf' if norm == np.inf else norm)} = {B_norm}")
    print(f"rho(B) (spectral radius) = {rho}\n")

    xs, reason, last_post = _iterate(B, C, x0, tol, max_iter, norm, stop_mode, debug=debug)
    x_star = xs[-1]

    print("\n=== Result: relaxation method ===")
    print(f"omega = {omega}")
    print(f"Stop reason: {reason}")
    print(f"Number of iterations n = {len(xs) - 1}")
    if last_post is not None:
        print(f"Last a-posteriori bound = {last_post}")
    print(f"Solution vector x =\n{x_star}")
    print("check A@x == b:", np.allclose(A @ x_star, _as_col(b)))
    return B, C, xs

# ============================================================
# PART 4 — Call
# ============================================================
solve_with_relaxation(mode, A, b, x0, omega, tol, max_iter, norm,
                      stop_mode, omega_list, debug)
