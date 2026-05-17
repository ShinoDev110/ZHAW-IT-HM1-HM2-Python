# ============================================================
# TOPIC: Nichtlineare Gleichungssysteme — Gedämpftes Newton-Verfahren (3D-System)
# DESCRIPTION:
# Löst ein nichtlineares 3×3-Gleichungssystem mit dem gedämpften Newton-Verfahren.
# Gibt pro Iterationsschritt k ||f(x^(k))||₂ und ||x^(k) - x^(k-1)||₂ aus.
# USE WHEN:
# Wenn das gedämpfte Newton-Verfahren auf ein 3D-System angewendet werden soll
# (z.B. wenn Standard-Newton divergiert oder schlecht konditioniert ist).
# EXAMPLE:
# Lösung von f(x1,x2,x3) = (x1+x2²-x3²-13, ln(x2/4)+e^(0.5x3-1)-1,
#                             (x2-3)²-x3³+7) ausgehend von x^(0)=(1.5,3,2.5)^T.
# ============================================================

import numpy as np
import sympy as sp

# ============================================================
# PART 1 — Inputs
# ============================================================
x1, x2, x3 = sp.symbols('x1 x2 x3')   # symbolische Variablen
X = sp.Matrix([x1, x2, x3])            # Vektor der Unbekannten

f_sym = sp.Matrix([
    x1 + x2**2 - x3**2 - 13,                          # f1
    sp.ln(x2 / 4) + sp.exp(0.5 * x3 - 1) - 1,        # f2
    (x2 - 3)**2 - x3**3 + 7                            # f3
])

x0       = np.array([1.5, 3.0, 2.5], dtype=float)    # Startvektor
tol      = 1e-5                                        # Abbruch wenn ||f(x^(k))||₂ < tol
max_iter = 50                                          # maximale Iterationszahl
k_max    = 4                                           # maximale Dämpfungsstufe (Halbierungen)

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: gedämpftes Newton-Verfahren.

# ============================================================
# PART 3 — Implementation
# ============================================================
def solve_3d_nonlinear_system_damped_newton(f_sym, X, x0, tol, max_iter, k_max):
    Df_sym = f_sym.jacobian(X)
    syms   = list(X)
    f_lam  = sp.lambdify(syms, f_sym,  "numpy")
    Df_lam = sp.lambdify(syms, Df_sym, "numpy")

    n = len(syms)

    def f_eval(x_vec):
        return np.array(f_lam(*x_vec), dtype=float).reshape(-1)

    def Df_eval(x_vec):
        return np.array(Df_lam(*x_vec), dtype=float).reshape(n, n)

    x     = x0.copy()
    x_old = x.copy()

    print(f"Startvektor x^(0) = {x}")
    print(f"Toleranz = {tol},  kmax = {k_max}\n")
    header = f"{'k':<4} {'||f(x^(k))||_2':<18} {'||x^(k)-x^(k-1)||_2':<22} {'x^(k)'}"
    print(header)
    print("-" * 75)

    f0_norm = np.linalg.norm(f_eval(x), 2)
    print(f"{'0':<4} {f0_norm:<18.6e} {'---':<22} {x}")

    for iteration in range(1, max_iter + 1):
        fx  = f_eval(x)
        Dfx = Df_eval(x)

        try:
            delta = np.linalg.solve(Dfx, -fx)
        except np.linalg.LinAlgError:
            print("Jacobi-Matrix ist singulär – Abbruch.")
            break

        # Dämpfung: finde kleinstes k mit ||f(x + delta/2^k)|| < ||f(x)||
        err_curr = np.linalg.norm(fx, 2)
        k_found  = None
        for k in range(k_max + 1):
            x_try = x + delta / (2**k)
            if np.linalg.norm(f_eval(x_try), 2) < err_curr:
                k_found = k
                break
        if k_found is None:
            k_found = 0   # kein besserer Schritt gefunden → voller Schritt

        x_old = x.copy()
        x     = x + delta / (2**k_found)

        f_norm = np.linalg.norm(f_eval(x), 2)
        dx_norm = np.linalg.norm(x - x_old, 2)
        print(f"{iteration:<4} {f_norm:<18.6e} {dx_norm:<22.6e} {np.round(x, 8)}")

        if f_norm < tol:
            print(f"\nKonvergiert nach {iteration} Iterationen.")
            print(f"Lösung: x* = {x}")
            print(f"Residuum: ||f(x*)|| = {f_norm:.6e}")
            return x

    print(f"\nKeine Konvergenz nach {max_iter} Iterationen.")
    return x

# ============================================================
# PART 4 — Call
# ============================================================
solve_3d_nonlinear_system_damped_newton(f_sym, X, x0, tol, max_iter, k_max)
