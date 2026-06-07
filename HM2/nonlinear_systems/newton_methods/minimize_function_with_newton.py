# ============================================================
# TOPIC: Newton-Verfahren für Systeme — Minimum einer Funktion g(x,y,...)
# DESCRIPTION:
# Bestimmt eine Minimalstelle einer skalaren Funktion mehrerer Variablen, indem
# das nichtlineare Gleichungssystem grad g = 0 mit dem Newton-Verfahren für
# Systeme gelöst wird. Die Jacobi-Matrix von grad g ist die Hesse-Matrix von g.
# Iteriert bis ||grad g||_inf < tol und prüft über die Hesse-Eigenwerte, ob
# tatsächlich ein Minimum vorliegt.
# USE WHEN:
# Wenn das Minimum (oder ein stationärer Punkt) einer differenzierbaren
# Funktion g(x,y,...) gesucht ist (Optimierung via Nullstelle des Gradienten).
# EXAMPLE:
# g(x,y) = x^4 - 2x^2 y + 10x^2 - 8.4x + y^2 + 5.764, Start (0.2, 0.8)
# -> Minimum bei (0.42, 0.1764), g = 4.0.
# ============================================================

import numpy as np
import sympy as sp

# ============================================================
# PART 1 — Inputs
# ============================================================
x, y = sp.symbols('x y')          # Variablen von g
syms = [x, y]                      # Reihenfolge der Unbekannten

g_sym = x**4 - 2*x**2*y + 10*x**2 - 8.4*x + y**2 + 5.764   # zu minimierende Funktion

x0       = np.array([0.2, 0.8], dtype=float)   # Startvektor
tol      = 1e-8                                 # Abbruch: ||grad g||_inf < tol
max_iter = 100                                  # maximale Iterationszahl

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: Newton ohne Dämpfung auf grad g = 0
# (H(x) · delta = -grad g(x),  x <- x + delta).

# ============================================================
# PART 3 — Implementation
# ============================================================
def minimize_function_with_newton(g_sym, syms, x0, tol, max_iter):
    grad = sp.Matrix([sp.diff(g_sym, s) for s in syms])
    H    = grad.jacobian(sp.Matrix(syms))
    print("============================================================")
    print("Minimierung über Newton-Verfahren (grad g = 0)")
    print("============================================================")
    print("grad g =")
    sp.pprint(grad)
    print("\nHesse-Matrix H = Jacobi(grad g) =")
    sp.pprint(H)

    grad_l = sp.lambdify(syms, grad, "numpy")
    H_l    = sp.lambdify(syms, H, "numpy")
    g_l    = sp.lambdify(syms, g_sym, "numpy")
    nv = len(syms)

    def grad_eval(v): return np.array(grad_l(*v), dtype=float).reshape(-1)
    def H_eval(v):    return np.array(H_l(*v), dtype=float).reshape(nv, nv)

    v = np.array(x0, dtype=float)
    err = np.linalg.norm(grad_eval(v), np.inf)
    print(f"\nn = 0   x = {v}   ||grad g||_inf = {err:.6e}")

    n = 0
    while err > tol and n < max_iter:
        delta = np.linalg.solve(H_eval(v), -grad_eval(v))
        v = v + delta
        err = np.linalg.norm(grad_eval(v), np.inf)
        n += 1
        print(f"n = {n}   x = {v}   ||grad g||_inf = {err:.6e}")

    g_min = float(g_l(*v))
    eig = np.linalg.eigvals(H_eval(v))
    art = "Minimum" if np.all(eig > 0) else ("Maximum" if np.all(eig < 0) else "Sattelpunkt")
    print(f"\nStationärer Punkt: x = {v}")
    print(f"g(x) = {g_min:.10g}")
    print(f"Hesse-Eigenwerte = {eig}  ->  {art}")
    return v, g_min

# ============================================================
# PART 4 — Call
# ============================================================
minimize_function_with_newton(g_sym, syms, x0, tol, max_iter)
