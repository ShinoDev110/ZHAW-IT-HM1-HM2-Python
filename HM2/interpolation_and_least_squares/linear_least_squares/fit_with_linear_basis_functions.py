# ============================================================
# TOPIC: Lineares Ausgleichsproblem — beliebige Basisfunktionen
# DESCRIPTION:
# Fittet f(x) = sum_j lambda_j * phi_j(x) mit FREI wählbaren Basisfunktionen
# phi_j (z.B. 1/x, x, x^2, ...) nach der Methode der kleinsten Quadrate über
# die Normalgleichungen und stabil via QR. Optional wird y vorher transformiert
# (Kehrwert 1/y oder log y), um nichtlineare Ansätze wie 1/(a+b*x^2) auf ein
# lineares Problem zurückzuführen. Liefert Koeffizienten und Fehlerfunktional.
# USE WHEN:
# Wenn der Ansatz eine Linearkombination beliebiger Basisfunktionen ist (kein
# reines Polynom) oder per Kehrwert/Logarithmus linearisiert werden kann.
# EXAMPLE:
# Daten (-2,-5.5),(-1,-5),(1,5),(2,5.5) mit f(x)=alpha/x + beta*x
# -> alpha = 3, beta = 2, Fehlerfunktional E = 0.
# ============================================================

import numpy as np

np.set_printoptions(precision=6, suppress=True)

# ============================================================
# PART 1 — Inputs
# ============================================================
x_data = np.array([-2.0, -1.0, 1.0, 2.0])      # x-Werte
y_data = np.array([-5.5, -5.0, 5.0, 5.5])      # y-Werte

# Basisfunktionen phi_j(x) (vektorisierte lambdas) + Beschriftungen
basis_functions = [lambda x: 1.0 / x, lambda x: x]
basis_labels    = ["1/x", "x"]

# Alternatives Beispiel (PNG-Aufgabe 2b): y = 1/(a + b*x^2)  ->  1/y = a + b*x^2
# x_data = np.array([0,1,2,3,4,5], dtype=float)
# y_data = np.array([0.54,0.44,0.28,0.18,0.12,0.08])
# basis_functions = [lambda x: np.ones_like(x), lambda x: x**2]
# basis_labels    = ["1", "x^2"];  y_transform = "reciprocal"

# ============================================================
# PART 2 — Method selection
# ============================================================
# y_transform:
#   "none"       -> direkt f(x) = sum lambda_j phi_j(x) fitten
#   "reciprocal" -> 1/y fitten (für Ansatz y = 1/(sum lambda_j phi_j(x)))
#   "log"        -> ln(y) fitten (für Ansatz y = exp(sum lambda_j phi_j(x)))
y_transform = "none"

# ============================================================
# PART 3 — Implementation
# ============================================================
def _forward(y, transform):
    if transform == "none":       return y.copy()
    if transform == "reciprocal": return 1.0 / y
    if transform == "log":        return np.log(y)
    raise ValueError(f"Unbekannte y_transform: {transform!r}")

def _inverse(lin, transform):
    if transform == "none":       return lin
    if transform == "reciprocal": return 1.0 / lin
    if transform == "log":        return np.exp(lin)
    raise ValueError(f"Unbekannte y_transform: {transform!r}")

def _design_matrix(basis, x):
    return np.column_stack([np.asarray(phi(x), dtype=float) * np.ones_like(x) for phi in basis])

def _model_label(labels, lam):
    terms = [f"{lam[j]:+.6g}*{labels[j]}" for j in range(len(labels))]
    return " ".join(terms).lstrip("+")

def fit_with_linear_basis_functions(x_data, y_data, basis_functions, basis_labels,
                                    y_transform):
    y_fit = _forward(y_data, y_transform)
    A = _design_matrix(basis_functions, x_data)

    print("============================================================")
    print("Lineares Ausgleichsproblem mit beliebigen Basisfunktionen")
    print("============================================================")
    print(f"Basis: {basis_labels}   y-Transformation: {y_transform}")
    print("Designmatrix A =\n", A)

    # Normalgleichungen
    AtA = A.T @ A
    lam_ne = np.linalg.solve(AtA, A.T @ y_fit)
    # QR (stabiler)
    Q, R = np.linalg.qr(A)
    lam_qr = np.linalg.solve(R, Q.T @ y_fit)

    print("\n-- Normalgleichungen  (A^T A) lambda = A^T y")
    print(f"lambda = {lam_ne}   cond(A^T A) = {np.linalg.cond(AtA):.4e}")
    print("-- QR-Zerlegung")
    print(f"lambda = {lam_qr}   cond(R) = {np.linalg.cond(R):.4e}")

    lam = lam_qr
    for name, val in zip(basis_labels, lam):
        print(f"   Koeffizient zu {name:>6} = {val:.10g}")

    # Fehlerfunktional im (linearen) Fit-Raum und im Originalraum
    E_fit = float(np.linalg.norm(y_fit - A @ lam, 2)**2)
    model_data = _inverse(A @ lam, y_transform)
    E_orig = float(np.linalg.norm(y_data - model_data, 2)**2)
    print(f"\nFehlerfunktional E (Fit-Raum)      = ||y_fit - A·lambda||^2 = {E_fit:.6e}")
    print(f"Fehlerfunktional E (Originaldaten) = sum (y_i - f(x_i))^2   = {E_orig:.6e}")
    print(f"Modell: f(x) = {_model_label(basis_labels, lam)}"
          + ("" if y_transform == "none" else f"   (auf {y_transform} angewandt)"))
    return lam, E_orig

# ============================================================
# PART 4 — Call
# ============================================================
fit_with_linear_basis_functions(x_data, y_data, basis_functions, basis_labels, y_transform)
