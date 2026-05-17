# ============================================================
# TOPIC: Nullstellenverfahren — Newton-Verfahren (1D)
# DESCRIPTION:
# Klassisches Newton-Verfahren x_{k+1} = x_k - f(x_k)/f'(x_k) für eine
# symbolisch gegebene Funktion. Abbruch wahlweise per Toleranz oder
# fester Iterationszahl, mit ausführlichem Debug-Output.
# USE WHEN:
# Wenn eine Nullstelle einer differenzierbaren Funktion f(x) gesucht ist
# und ein guter Startwert verfügbar ist (quadratische Konvergenz).
# EXAMPLE:
# f(x) = exp(x) - (sqrt(x) + 2), Startwert x0 = 0.5.
# ============================================================

from sympy import diff, sympify

# ============================================================
# PART 1 — Inputs
# ============================================================
funktion    = "exp(x) - (sqrt(x) + 2)"  # f(x); für Fixpunkt: "F(x) - x"
x_0         = {"x": 0.5}                # Startwert x0
toleranz    = 1e-7                      # Toleranz für Abbruchkriterium
iterationen = 5                         # max. Iterationen (für Methode "iters")
debug       = True                      # Zwischenwerte anzeigen

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
def _newton_tol(fx, x0, tol, debug=False):
    fx = sympify(fx)
    symbols = list(fx.free_symbols)
    if not symbols:
        raise ValueError("Keine Unbekannte in Funktion gefunden.")
    s = symbols[0]
    ableitung = diff(fx)
    xn = x0[str(s)]
    i = 0
    while True:
        x_prev = xn
        xn = x_prev - (fx.subs({str(s): x_prev}).evalf() /
                       ableitung.subs({str(s): x_prev}).evalf())
        if debug:
            print(f"---- Iteration {i + 1}")
            print(f"f({s}_{i}) = {fx.subs({str(s): x_prev}).evalf()}")
            print(f"f'({s}_{i}) = {ableitung.subs({str(s): x_prev}).evalf()}")
            print(f"{s}_{i + 1} = {xn}\n")
        if abs(xn - x_prev) < tol:
            break
        i += 1
    return xn

def _newton_iters(fx, x0, iters, debug=False):
    fx = sympify(fx)
    symbols = list(fx.free_symbols)
    if not symbols:
        raise ValueError("Keine Unbekannte in Funktion gefunden.")
    s = symbols[0]
    ableitung = diff(fx)
    state = dict(x0)
    for i in range(iters):
        xk = state[str(s)] - (fx.subs(state).evalf() / ableitung.subs(state).evalf())
        if debug:
            print(f"---- Iteration {i + 1}")
            print(f"f({s}_{i}) = {fx.subs(state).evalf()}")
            print(f"f'({s}_{i}) = {ableitung.subs(state).evalf()}")
            print(f"{s}_{i + 1} = {xk}\n")
        state[str(s)] = xk
    return state[str(s)]

def find_root_with_newton(method, funktion, x_0, toleranz, iterationen, debug=False):
    if method == "tol":
        return _newton_tol(funktion, x_0, toleranz, debug)
    if method == "iters":
        return _newton_iters(funktion, x_0, iterationen, debug)
    raise ValueError(f"Unbekannte Methode: {method!r}")

# ============================================================
# PART 4 — Call
# ============================================================
find_root_with_newton(method, funktion, x_0, toleranz, iterationen, debug)
