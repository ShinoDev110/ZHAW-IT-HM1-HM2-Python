# ============================================================
# TOPIC: Funktionen — symbolisches Ableiten und Integrieren
# DESCRIPTION:
# Bildet die Ableitung und die Stammfunktion einer symbolisch
# definierten Funktion und wertet diese ggf. an einer Stelle aus.
# USE WHEN:
# Wenn zu einer gegebenen Funktion f(x) schnell f'(x) bzw. ∫f(x) dx
# benötigt wird (z.B. zur Vorbereitung des Newton-Verfahrens).
# EXAMPLE:
# f(x) = sqrt(1 - x), Ableitung und Integral, ausgewertet bei x = 0.8.
# ============================================================

from sympy import diff, integrate, sympify

# ============================================================
# PART 1 — Inputs
# ============================================================
fx     = "sqrt(1 - x)"          # Funktion als String
symbol = "x"                    # Variable, nach der abgeleitet/integriert wird
werte  = {"x": 0.8, "y": 0.6}   # Werte für die Auswertung

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here. Ableitung UND Integral werden immer ausgegeben.

# ============================================================
# PART 3 — Implementation
# ============================================================
def differentiate_and_integrate_symbolic(fx, symbol, werte):
    fx_sym = sympify(fx)
    sym = sympify(symbol)

    ableitung   = diff(fx_sym, sym)
    stammfunkt  = integrate(fx_sym, sym)

    print(f"Funktion:           {fx_sym}")
    print(f"Ableitung:          {ableitung}")
    print(f"Stammfunktion:      {stammfunkt}")
    print()
    print(f"f(x)  ausgewertet:  {fx_sym.subs(werte).evalf()}")
    print(f"f'(x) ausgewertet:  {ableitung.subs(werte).evalf()}")
    return ableitung, stammfunkt

# ============================================================
# PART 4 — Call
# ============================================================
differentiate_and_integrate_symbolic(fx, symbol, werte)
