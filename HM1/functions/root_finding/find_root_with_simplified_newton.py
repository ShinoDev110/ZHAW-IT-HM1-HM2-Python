# ============================================================
# TOPIC: Nullstellenverfahren — vereinfachtes Newton-Verfahren
# DESCRIPTION:
# Vereinfachtes Newton: x_{k+1} = x_k - f(x_k) / f'(x_0). Die Ableitung
# wird nur einmal am Startwert x0 ausgewertet und dann als konstant
# angenommen. Spart Rechenaufwand, konvergiert aber nur linear.
# USE WHEN:
# Wenn f'(x_0) bekannt ist und Berechnungen von f'(x_k) teuer sind, der
# Startwert nahe genug an der Nullstelle liegt.
# EXAMPLE:
# f(x) = e^(x^2) + x^-3 - 10, f'(x) = 2x e^(x^2) - 3 x^-4, x0 = 1.5.
# ============================================================

import math

# ============================================================
# PART 1 — Inputs
# ============================================================
def f(x):
    return math.exp(x * x) + x ** -3 - 10.0

def df(x):
    return 2.0 * x * math.exp(x * x) - 3.0 * x ** -4

x0            = 1.5
toleranz      = 1e-8
maximale_iter = 50

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here.

# ============================================================
# PART 3 — Implementation
# ============================================================
def find_root_with_simplified_newton(f, df, x0, tol=1e-8, max_iter=50):
    df0 = float(df(x0))
    if df0 == 0.0:
        raise ValueError("f'(x0) = 0, vereinfachtes Newton nicht anwendbar.")

    x_alt = float(x0)
    folge = [x_alt]
    iterationen = 0

    while iterationen < max_iter:
        fx = float(f(x_alt))
        x_neu = x_alt - fx / df0
        folge.append(x_neu)
        iterationen += 1
        print(f"k={iterationen} | x_k = {x_neu} | f(x_k) = {f(x_neu)}")
        if abs(x_neu - x_alt) < tol:
            break
        x_alt = x_neu

    print(f"\nNäherungslösung: x ~= {x_neu}")
    print(f"Iterationen:     {iterationen}")
    return x_neu, iterationen, folge

# ============================================================
# PART 4 — Call
# ============================================================
find_root_with_simplified_newton(f, df, x0, toleranz, maximale_iter)
