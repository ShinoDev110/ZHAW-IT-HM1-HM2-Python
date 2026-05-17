# ============================================================
# TOPIC: Funktionen — Konditionszahl einer reellen Funktion
# DESCRIPTION:
# Berechnet κ_f(x) = | x · f'(x) / f(x) | wahlweise mit analytischer
# Ableitung, zentraler numerischer Ableitung oder als Verlauf über ein
# Intervall (Gitter), um schlecht konditionierte Stellen zu finden.
# USE WHEN:
# Wenn beurteilt werden soll, wie stark relative Eingabefehler in x
# in relative Ausgabefehler in f(x) verstärkt werden.
# EXAMPLE:
# f(x) = exp(x) - 3x bei x0 = 2 mit analytischer Ableitung;
# zusätzlich Sweep über [0.1, 5.0].
# ============================================================

import math
import numpy as np

# ============================================================
# PART 1 — Inputs
# ============================================================
x0 = 2.0  # Auswertungsstelle für die punktuelle Konditionszahl
h  = 1e-6 # Schrittweite für die numerische Ableitung (nur Methode "numeric")

def f(x: float) -> float:
    # Beispiel-Funktion (anpassen)
    return math.exp(x) - 3.0 * x

def df(x: float) -> float:
    # Analytische Ableitung (für Methode "analytic")
    return math.exp(x) - 3.0

x_grid = np.linspace(0.1, 5.0, 20)  # Gitter für Methode "grid"

# ============================================================
# PART 2 — Method selection
# ============================================================
# method:
#   "analytic" -> κ_f(x0) mit gegebener Ableitung df
#   "numeric"  -> κ_f(x0) mit zentraler numerischer Ableitung (kein df nötig)
#   "grid"     -> κ_f auf x_grid auswerten, kritische Stellen finden
method = "grid"

# ============================================================
# PART 3 — Implementation
# ============================================================
def _kappa_analytic(f, df, x):
    fx = float(f(x))
    if fx == 0.0:
        return math.inf
    return abs(x * float(df(x)) / fx)

def _ableitung_zentral(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2.0 * h)

def _kappa_numeric(f, x, h=1e-6):
    fx = float(f(x))
    if fx == 0.0:
        return math.inf
    return abs(x * float(_ableitung_zentral(f, x, h)) / fx)

def _drucke_report(x, fx, dfx, kappa):
    print("============================================================")
    print("Konditionszahl einer reellen Funktion")
    print("============================================================")
    print(f"x      = {x}")
    print(f"f(x)   = {fx}")
    print(f"f'(x)  = {dfx}")
    print(f"kappa  = |x f'(x) / f(x)| = {kappa}")
    print()

def compute_condition_number_of_function(method, f, df, x0, h, x_grid):
    if method == "analytic":
        fx, dfx = f(x0), df(x0)
        _drucke_report(x0, fx, dfx, _kappa_analytic(f, df, x0))
    elif method == "numeric":
        fx, dfx = f(x0), _ableitung_zentral(f, x0, h)
        _drucke_report(x0, fx, dfx, _kappa_numeric(f, x0, h))
    elif method == "grid":
        print("x\tkappa")
        print("-" * 30)
        for x in x_grid:
            k = _kappa_analytic(f, df, float(x))
            print(f"{x:.3f}\t{k:.6g}")
    else:
        raise ValueError(f"Unbekannte Methode: {method!r}")

# ============================================================
# PART 4 — Call
# ============================================================
compute_condition_number_of_function(method, f, df, x0, h, x_grid)
