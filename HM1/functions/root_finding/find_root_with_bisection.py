# ============================================================
# TOPIC: Nullstellenverfahren — Bisektion (Intervallhalbierung)
# DESCRIPTION:
# Klassische Intervallhalbierung für f(x) = 0. Voraussetzung: f stetig
# auf [a, b] und f(a)·f(b) < 0 (Vorzeichenwechsel). Liefert Mittelpunkt
# des letzten Intervalls als Näherung und Iterationszahl.
# USE WHEN:
# Wenn ein robuster (aber linear konvergenter) Nullstellenlöser ohne
# Ableitungsinformation benötigt wird.
# EXAMPLE:
# f(x) = e^(x^2) + x^-3 - 10 auf [0.5, 2.0], tol = 1e-8.
# ============================================================

import math

# ============================================================
# PART 1 — Inputs
# ============================================================
def f(x):
    return math.exp(x * x) + x ** -3 - 10.0

intervall_links  = 0.5
intervall_rechts = 2.0
toleranz         = 1e-8
maximale_iter    = 100

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here.

# ============================================================
# PART 3 — Implementation
# ============================================================
def find_root_with_bisection(f, a, b, tol=1e-8, max_iter=100):
    a, b = float(a), float(b)
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("Bisektion: kein Vorzeichenwechsel im Startintervall.")

    iterationen = 0
    while (b - a) / 2.0 > tol and iterationen < max_iter:
        mitte = 0.5 * (a + b)
        fm = f(mitte)
        if fm == 0.0:
            print(f"Exakter Treffer bei x = {mitte}, iterations = {iterationen + 1}")
            return mitte, iterationen + 1
        if fa * fm < 0:
            b, fb = mitte, fm
        else:
            a, fa = mitte, fm
        iterationen += 1

    x_naeherung = 0.5 * (a + b)
    print(f"Nullstellen-Näherung: x ~= {x_naeherung}")
    print(f"Iterationen: {iterationen}")
    print(f"f(x) ~= {f(x_naeherung)}")
    return x_naeherung, iterationen

# ============================================================
# PART 4 — Call
# ============================================================
find_root_with_bisection(f, intervall_links, intervall_rechts, toleranz, maximale_iter)
