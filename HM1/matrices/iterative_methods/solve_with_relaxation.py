# ============================================================
# TOPIC: Iterative Löser — Relaxationsverfahren (gedämpftes Jacobi / JOR)
# DESCRIPTION:
# Verallgemeinertes Jacobi-Verfahren mit Relaxationsparameter omega. Statt
# D x = -(L+R) x + b wird (1/omega) D x = -(((omega-1)/omega) D + L + R) x + b
# verwendet. Daraus folgt die Fixpunktiteration x^(k+1) = B_omega x^(k) + c_omega
# mit B_omega = (1-omega) I - omega D^-1 (L+R) und c_omega = omega D^-1 b.
# Für omega = 1 ergibt sich exakt das klassische Jacobi-Verfahren.
# USE WHEN:
# Wenn untersucht werden soll, ob ein Relaxationsparameter omega != 1 die
# Konvergenz des Jacobi-Verfahrens verbessert (kleineres ||B|| bzw. rho(B)).
# EXAMPLE:
# A = [[7,-2,-2],[-2,7,-2],[-2,-2,7]], b = [5,-13,14], omega = 1.15,
# Abbruch über a-posteriori-Schranke in der 1-Norm < 1e-9.
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
              [-2.0, -2.0,  7.0]])              # Koeffizientenmatrix

b = np.array([[  5.0],
              [-13.0],
              [ 14.0]])                          # rechte Seite

x0 = np.array([[0.0],
               [0.0],
               [0.0]])                          # Startvektor

omega    = 1.15    # Relaxationsparameter (omega = 1 -> klassisches Jacobi)
tol      = 1e-9    # Toleranz für Abbruchkriterium
max_iter = 10_000  # Maximale Iterationsanzahl
norm     = 1       # Verwendete Norm (1, 2 oder np.inf); Prüfung nutzt 1-Norm
debug    = True    # Zwischenresultate ausgeben

# Liste der omega-Werte für den Vergleichsmodus "compare"
omega_list = [1.0, 1.05, 1.15, 1.25, 1.5]

# ============================================================
# PART 2 — Method selection
# ============================================================
# mode:
#   "solve"   -> löst das System mit dem oben gesetzten omega
#   "compare" -> testet alle omega aus omega_list und zeigt ||B||, rho(B)
#                und benötigte Iterationen (so sieht man, ob omega besser ist)
mode = "solve"

# stop_mode (nur für mode = "solve"):
#   "fixed"        -> immer max_iter Schritte
#   "aposteriori"  -> Abbruch wenn a-posteriori Schranke <= tol
#   "apriori"      -> Abbruch wenn k >= a-priori Iterationsbedarf n
#   "both"         -> Abbruch wenn (a-posteriori <= tol) ODER (k >= n)
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
        print("-- A in D, L, R zerlegen")
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
        print(f"-- Relaxation (omega = {omega}): B und c")
        print("B_omega = (1-omega) I - omega D^-1 (L+R) =\n", B)
        print("c_omega = omega D^-1 b =\n", C, "\n")
    return B, C

def _spectral_radius(B):
    return float(np.max(np.abs(lin.eigvals(B))))

def _a_posteriori(B, x_n, x_n_minus_1, norm=np.inf, debug=False):
    B_norm = lin.norm(B, ord=norm)
    if B_norm >= 1:
        raise ValueError(f"||B|| = {B_norm} >= 1 -> Abschätzung nicht gültig (keine Kontraktion).")
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
        raise ValueError(f"||B|| = {B_norm} >= 1 -> Abschätzung nicht gültig (keine Kontraktion).")
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
        raise ValueError(f"||B|| = {B_norm} >= 1 -> Abbruchkriterien nicht gültig (keine Kontraktion).")

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
                stop_reason = f"a-posteriori erreicht (k={k})"
                break
        if stop_mode in ("apriori", "both") and n_apriori_int is not None and k >= n_apriori_int:
            stop_reason = f"a-priori erreicht (k={k} >= {n_apriori_int})"
            break
        x = x_new

    if stop_reason is None:
        stop_reason = f"max_iter erreicht (k={len(xs) - 1})"
    return xs, stop_reason, last_post

def solve_with_relaxation(mode, A, b, x0, omega, tol, max_iter, norm,
                          stop_mode, omega_list, debug=False):
    print("A =\n", A)
    print("b =\n", b)
    print("x0 =\n", x0, "\n")
    L, D, R = _A_to_LDR(A, debug=debug)

    if mode == "compare":
        print("=== Vergleich verschiedener omega (omega = 1 ist klassisches Jacobi) ===")
        print(f"{'omega':>6} | {'||B||':>10} | {'rho(B)':>10} | {'Iterationen':>11} | Konvergenz")
        print("-" * 64)
        for w in omega_list:
            B, C = _relax_BC_from_LDR(L, D, R, b, w, debug=False)
            B_norm = lin.norm(B, ord=norm)
            rho = _spectral_radius(B)
            if B_norm >= 1:
                print(f"{w:>6.3f} | {B_norm:>10.6f} | {rho:>10.6f} | {'-':>11} | "
                      f"||B|| >= 1 (a-posteriori ungültig)")
                continue
            xs, reason, _ = _iterate(B, C, x0, tol, max_iter, norm, "aposteriori", debug=False)
            iters = len(xs) - 1
            print(f"{w:>6.3f} | {B_norm:>10.6f} | {rho:>10.6f} | {iters:>11d} | {reason}")
        print("\nKleineres rho(B) bzw. ||B|| => schnellere Konvergenz.")
        return None

    if mode != "solve":
        raise ValueError(f"Unbekannter mode: {mode!r}")

    B, C = _relax_BC_from_LDR(L, D, R, b, omega, debug=debug)
    B_norm = lin.norm(B, ord=norm)
    rho = _spectral_radius(B)
    print(f"||B||_{('inf' if norm == np.inf else norm)} = {B_norm}")
    print(f"rho(B) (Spektralradius) = {rho}\n")

    xs, reason, last_post = _iterate(B, C, x0, tol, max_iter, norm, stop_mode, debug=debug)
    x_star = xs[-1]

    print("\n=== Resultat Relaxationsverfahren ===")
    print(f"omega = {omega}")
    print(f"Abbruchgrund: {reason}")
    print(f"Anzahl Iterationen n = {len(xs) - 1}")
    if last_post is not None:
        print(f"Letzte a-posteriori Schranke = {last_post}")
    print(f"Lösungsvektor x =\n{x_star}")
    print("check A@x == b:", np.allclose(A @ x_star, _as_col(b)))
    return B, C, xs

# ============================================================
# PART 4 — Call
# ============================================================
solve_with_relaxation(mode, A, b, x0, omega, tol, max_iter, norm,
                      stop_mode, omega_list, debug)
