# ============================================================
# TOPIC: Fixpunktiteration — a-posteriori Fehlerabschätzung (skalar)
# DESCRIPTION:
# Schätzt nach Banach den Fehler einer skalaren Fixpunktiteration aus zwei
# aufeinanderfolgenden Iterierten: |x̄ - x_{k+1}| <= (alpha/(1-alpha)) ·
# |x_{k+1} - x_k|. Die Lipschitz-Konstante alpha wird entweder direkt
# vorgegeben oder als max|F'(x)| über ein Intervall geschätzt.
# USE WHEN:
# Wenn nach einigen Iterationen beurteilt werden soll, wie genau die aktuelle
# Näherung x_{k+1} bereits ist ("Wie genau ist der Schätzwert?").
# EXAMPLE:
# F(x) = 1/(cos(x+pi/4)-1)+2 auf [1,2], x_k = 1.3376, x_{k+1} = 1.3441
# -> |x̄ - 1.3441| <= (alpha/(1-alpha)) · 0.0065.
# ============================================================

import numpy as np
from sympy import diff, sympify

# ============================================================
# PART 1 — Inputs
# ============================================================
funktion    = "1/(cos(x + pi/4) - 1) + 2"  # F(x) der Fixpunktiteration
x_prev      = 1.3376    # x_k       (vorherige Iterierte)
x_curr      = 1.3441    # x_{k+1}   (aktuelle Iterierte)
intervall   = [1.0, 2.0]  # Intervall zur Schätzung von alpha (max|F'|)
alpha_given = 0.5       # fest vorgegebene Lipschitzkonstante (nur "given_alpha")

# ============================================================
# PART 2 — Method selection
# ============================================================
# method:
#   "from_interval" -> alpha = max|F'(x)| diskret über das Intervall
#   "given_alpha"   -> alpha = alpha_given (z.B. aus Teilaufgabe gegeben)
method = "from_interval"

# ============================================================
# PART 3 — Implementation
# ============================================================
def _get_alpha_from_interval(funktion, interval):
    f = sympify(funktion)
    symbole = list(f.free_symbols)
    if not symbole:
        raise ValueError("Keine Unbekannte in Funktion gefunden.")
    s = symbole[0]
    df = diff(f, s)
    werte = [abs(float(df.subs(s, t).evalf()))
             for t in np.linspace(interval[0], interval[1], 200)]
    return max(werte), df

def estimate_error_a_posteriori_fixed_point(method, funktion, x_prev, x_curr,
                                            intervall, alpha_given):
    print("============================================================")
    print("A-posteriori Fehlerabschätzung der Fixpunktiteration")
    print("============================================================")
    if method == "from_interval":
        alpha, df = _get_alpha_from_interval(funktion, intervall)
        print(f"F(x)  = {funktion}")
        print(f"F'(x) = {df}")
        print(f"alpha = max|F'(x)| auf [{intervall[0]}, {intervall[1]}] ~= {alpha:.6g}")
    elif method == "given_alpha":
        alpha = alpha_given
        print(f"alpha (vorgegeben) = {alpha:.6g}")
    else:
        raise ValueError(f"Unbekannte Methode: {method!r}")

    if alpha >= 1:
        raise ValueError(f"alpha = {alpha} >= 1 -> keine Kontraktion, "
                         "a-posteriori-Abschätzung nicht gültig.")

    step = abs(x_curr - x_prev)
    faktor = alpha / (1 - alpha)
    bound = faktor * step
    print(f"\n|x_(k+1) - x_k| = |{x_curr} - {x_prev}| = {step:.6g}")
    print(f"Faktor alpha/(1-alpha) = {faktor:.6g}")
    print(f"|x̄ - x_(k+1)| <= (alpha/(1-alpha))·|x_(k+1)-x_k| = {bound:.6g}")
    print(f"\n=> Die Näherung x_(k+1) = {x_curr} ist auf etwa {bound:.3g} genau.")
    return bound

# ============================================================
# PART 4 — Call
# ============================================================
estimate_error_a_posteriori_fixed_point(method, funktion, x_prev, x_curr,
                                        intervall, alpha_given)
