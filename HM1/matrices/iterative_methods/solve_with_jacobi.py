# ============================================================
# TOPIC: Iterative Löser — Jacobi-Verfahren für Ax = b
# DESCRIPTION:
# Iterativer Jacobi-Löser mit Splittung A = D + L + R und Update
# x^(k+1) = -D^-1 (L+R) x^(k) + D^-1 b. Liefert auf Wunsch a-priori
# / a-posteriori Abbruchschranken pro Iteration.
# USE WHEN:
# Wenn ein lineares Gleichungssystem Ax = b iterativ gelöst werden soll
# und A strikt diagonal-dominant ist (|D| > |L+R| zeilenweise).
# EXAMPLE:
# 3x3-System mit A = [[4,-1,1],[-2,5,1],[1,-2,5]], b = [5,11,12].
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
              [ 1.0, -2.0, 5.0]])              # Koeffizientenmatrix

b = np.array([[5.0],
              [11.0],
              [12.0]])                          # rechte Seite

x0 = np.array([[0.0],
               [0.0],
               [0.0]])                          # Startvektor

tol      = 1e-4    # Toleranz für Abbruchkriterium
max_iter = 50      # Maximale Iterationsanzahl
norm     = np.inf  # Verwendete Norm (1, 2 oder np.inf)
debug    = True    # Zwischenresultate ausgeben

# ============================================================
# PART 2 — Method selection
# ============================================================
# Stop-Mode wählen:
#   "fixed"        -> immer max_iter Schritte
#   "aposteriori"  -> Abbruch wenn a-posteriori Schranke <= tol
#   "apriori"      -> Abbruch wenn k >= a-priori Iterationsbedarf n
#   "both"         -> Abbruch wenn (a-posteriori <= tol) ODER (k >= n)
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
        print("-- A in D, L, R zerlegen")
        print("D =\n", D)
        print("L =\n", L)
        print("R =\n", R, "\n")
    return L, D, R

def _jacobi_BC_from_LDR(L, D, R, b, debug=False):
    D_inv = lin.inv(D)
    B = -D_inv @ (L + R)
    C = D_inv @ _as_col(b)
    if debug:
        print("-- Jacobi: B und C")
        print("B = -D^-1(L+R) =\n", B)
        print("C = D^-1 b     =\n", C, "\n")
    return B, C

def _a_posteriori(B, x_n, x_n_minus_1, norm=np.inf, debug=False):
    B_norm = lin.norm(B, norm)
    if B_norm >= 1:
        raise ValueError(f"||B|| = {B_norm} >= 1 -> Abschätzung nicht gültig (keine Kontraktion).")
    diff_norm = lin.norm(_as_col(x_n) - _as_col(x_n_minus_1), norm)
    err = (B_norm / (1 - B_norm)) * diff_norm
    if debug:
        print("-- A-posteriori")
        print(f"||B|| = {B_norm}")
        print(f"||x^n - x^(n-1)|| = {diff_norm}")
        print(f"||x^n - x̄|| <= (||B||/(1-||B||)) * ||x^n - x^(n-1)|| = {err}\n")
    return err

def _a_priori_iterations(B, x0, x1, tol, norm=np.inf, debug=False):
    B_norm = lin.norm(B, norm)
    if B_norm >= 1:
        raise ValueError(f"||B|| = {B_norm} >= 1 -> Abschätzung nicht gültig (keine Kontraktion).")
    step_norm = lin.norm(_as_col(x1) - _as_col(x0), norm)
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

def _jacobi_step(B, C, x_prev, debug=False):
    x_prev = _as_col(x_prev)
    if debug:
        print("B * x + C =")
        print(B, "\n*\n", x_prev, "\n+\n", C)
    return B @ x_prev + C

def _print_result(B, C, xs, info):
    print("\n=== Jacobi Matrices ===")
    print("B =\n", B)
    print("C =\n", C, "\n")
    print("=== Abbruch Info ===")
    for k, v in info.items():
        print(f"{k}: {v}")
    k_last = info["iterations_done"]
    print(f"\nLetzte Näherung x^{k_last} =\n{xs[k_last]}")

def solve_linear_system_with_jacobi(A, b, x0, tol, max_iter, norm=np.inf,
                                    stop_mode="aposteriori", debug=False):
    print("A =\n", A)
    print("b =\n", b)
    print("x0 =\n", x0, "\n")

    L, D, R = _A_to_LDR(A, debug=debug)
    B, C = _jacobi_BC_from_LDR(L, D, R, b, debug=debug)

    stop_mode = (stop_mode or "aposteriori").lower()
    x = _as_col(x0)
    xs = [x.copy()]

    B_norm = lin.norm(B, norm)
    if stop_mode in ("aposteriori", "apriori", "both") and B_norm >= 1:
        raise ValueError(f"||B|| = {B_norm} >= 1 -> Abbruchkriterien nicht gültig (keine Kontraktion).")

    n_apriori_real = None
    n_apriori_int  = None
    last_post      = None
    stop_reason    = None

    if stop_mode in ("apriori", "both") and max_iter >= 1:
        x1 = _jacobi_step(B, C, x, debug=debug)
        xs.append(x1.copy())
        n_apriori_real, n_apriori_int = _a_priori_iterations(B, xs[0], xs[1], tol, norm=norm, debug=debug)
        if debug:
            print(f"Target iterations from a-priori: n = {n_apriori_real} -> ceil = {n_apriori_int}\n")
        if n_apriori_int <= 1:
            stop_reason = "a-priori erreicht (n<=1)"
            _print_result(B, C, xs, {
                "stop_mode": stop_mode, "stop_reason": stop_reason,
                "iterations_done": 1, "B_norm": B_norm,
                "a_priori_n_real": n_apriori_real, "a_priori_n_int": n_apriori_int,
                "a_posteriori_last": None,
            })
            return B, C, xs
        x = x1
        if stop_mode == "both":
            last_post = _a_posteriori(B, xs[1], xs[0], norm=norm, debug=False)
            if last_post <= tol:
                stop_reason = "a-posteriori erreicht (nach k=1)"
                _print_result(B, C, xs, {
                    "stop_mode": stop_mode, "stop_reason": stop_reason,
                    "iterations_done": 1, "B_norm": B_norm,
                    "a_priori_n_real": n_apriori_real, "a_priori_n_int": n_apriori_int,
                    "a_posteriori_last": last_post,
                })
                return B, C, xs
        start_k = 2
    else:
        start_k = 1

    for k in range(start_k, max_iter + 1):
        if debug:
            print(f"--------------------- Iteration {k}")
            print(f"x^{k} = B * x^{k-1} + C")
        x_new = _jacobi_step(B, C, x, debug=False)
        xs.append(x_new.copy())
        if debug:
            print("=\n", x_new, "\n")

        if stop_mode in ("aposteriori", "both"):
            last_post = _a_posteriori(B, xs[k], xs[k - 1], norm=norm, debug=False)
            if debug:
                print(f"A-posteriori bound (k={k}): {last_post}")
            if last_post <= tol:
                stop_reason = f"a-posteriori erreicht (k={k})"
                break

        if stop_mode in ("apriori", "both") and n_apriori_int is not None:
            if k >= n_apriori_int:
                stop_reason = f"a-priori erreicht (k={k} >= {n_apriori_int})"
                break

        x = x_new

    if stop_reason is None:
        stop_reason = f"max_iter erreicht (k={len(xs)-1})"

    _print_result(B, C, xs, {
        "stop_mode": stop_mode, "stop_reason": stop_reason,
        "iterations_done": len(xs) - 1, "B_norm": B_norm,
        "a_priori_n_real": n_apriori_real, "a_priori_n_int": n_apriori_int,
        "a_posteriori_last": last_post,
    })
    return B, C, xs

# ============================================================
# PART 4 — Call
# ============================================================
solve_linear_system_with_jacobi(A, b, x0, tol, max_iter, norm=norm,
                                stop_mode=stop_mode, debug=debug)
