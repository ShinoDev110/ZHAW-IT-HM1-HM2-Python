# ============================================================
# TOPIC: Nullstellenverfahren — Sekantenverfahren (1D)
# DESCRIPTION:
# Sekantenverfahren x_{k+1} = x_k - f(x_k)(x_k - x_{k-1}) / (f(x_k) - f(x_{k-1}))
# für eine symbolisch gegebene Funktion. Approximiert f' durch die
# Sekante; braucht zwei Startwerte. Abbruch per Toleranz oder Iterationszahl.
# USE WHEN:
# Wenn eine Nullstelle gesucht ist, aber die Ableitung nicht analytisch
# vorliegt oder zu teuer auszuwerten ist.
# EXAMPLE:
# f(x) = (x^2 + 1)^2 - 10 - 5/((x-1)^2 + 1), Startwerte x0=1.6, x1=1.7.
# ============================================================

import sympy as sp
from sympy import sympify

# ============================================================
# PART 1 — Inputs
# ============================================================
x = sp.Symbol("x")
funktion_1 = (x**2 + 1)**2 - 10
funktion_2 = 5 / ((x - 1)**2 + 1)
funktion   = funktion_1 - funktion_2   # f(x), deren Nullstelle gesucht ist

x_0         = {"x": 1.6}   # erster Startwert
x_1         = {"x": 1.7}   # zweiter Startwert
toleranz    = 1e-6         # Toleranz für Abbruchkriterium
iterationen = 2            # max. Iterationen (für Methode "iters")
debug       = True         # Zwischenwerte anzeigen

# ============================================================
# PART 2 — Method selection
# ============================================================
# method:
#   "tol"   -> Abbruch bei |x_{k+1} - x_k| < toleranz
#   "iters" -> immer iterationen Schritte ausführen
method = "tol"

# ============================================================
# PART 3 — Implementation
# ============================================================
def _secant_tol(fx, x0, x1, tol, debug=False, max_iter=1000):
    fx = sympify(fx)
    symbols = sorted(fx.free_symbols, key=lambda s: s.name)
    if not symbols:
        raise ValueError("Keine Unbekannte in Funktion gefunden.")
    s = symbols[0]
    x_nm1 = float(x0[str(s)])
    x_n   = float(x1[str(s)])
    for k in range(max_iter):
        f_nm1 = float(fx.subs({s: x_nm1}).evalf())
        f_n   = float(fx.subs({s: x_n}).evalf())
        denom = f_n - f_nm1
        if denom == 0:
            raise ZeroDivisionError("Sekantenverfahren: f(x_n) - f(x_{n-1}) = 0.")
        x_np1 = x_n - f_n * (x_n - x_nm1) / denom
        if debug:
            print(f"---- Iteration {k + 1}")
            print(f"{s}_{k} = {x_nm1}")
            print(f"{s}_{k+1} = {x_n}")
            print(f"f({s}_{k}) = {f_nm1}")
            print(f"f({s}_{k+1}) = {f_n}")
            print(f"{s}_{k+2} = {x_np1}\n")
        if abs(x_np1 - x_n) <= tol:
            return x_np1
        x_nm1, x_n = x_n, x_np1
    raise RuntimeError("Maximale Iterationen erreicht.")

def _secant_iters(fx, x0, x1, iters, debug=False):
    fx = sympify(fx)
    symbols = sorted(fx.free_symbols, key=lambda s: s.name)
    if not symbols:
        raise ValueError("Keine Unbekannte in Funktion gefunden.")
    s = symbols[0]
    x_nm1 = float(x0[str(s)])
    x_n   = float(x1[str(s)])
    for k in range(iters):
        f_nm1 = float(fx.subs({s: x_nm1}).evalf())
        f_n   = float(fx.subs({s: x_n}).evalf())
        denom = f_n - f_nm1
        if denom == 0:
            raise ZeroDivisionError("Sekantenverfahren: f(x_n) - f(x_{n-1}) = 0.")
        x_np1 = x_n - f_n * (x_n - x_nm1) / denom
        if debug:
            print(f"---- Iteration {k + 1}")
            print(f"{s}_{k} = {x_nm1}")
            print(f"{s}_{k+1} = {x_n}")
            print(f"f({s}_{k}) = {f_nm1}")
            print(f"f({s}_{k+1}) = {f_n}")
            print(f"{s}_{k+2} = {x_np1}\n")
        x_nm1, x_n = x_n, x_np1
    return x_n

def find_root_with_secant(method, funktion, x_0, x_1, toleranz, iterationen, debug=False):
    if method == "tol":
        return _secant_tol(funktion, x_0, x_1, toleranz, debug)
    if method == "iters":
        return _secant_iters(funktion, x_0, x_1, iterationen, debug)
    raise ValueError(f"Unbekannte Methode: {method!r}")

# ============================================================
# PART 4 — Call
# ============================================================
find_root_with_secant(method, funktion, x_0, x_1, toleranz, iterationen, debug)
