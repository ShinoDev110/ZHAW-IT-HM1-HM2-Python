# ============================================================
# TOPIC: Lineares Ausgleichsproblem — Polynom-Fit via Normalgleichungen / QR / polyfit
# DESCRIPTION:
# Löst min ||A·lambda - y||^2 für einen Polynom-Ansatz, einmal über die
# Normalgleichungen (A^T A·lambda = A^T y), einmal stabil via QR-Zerlegung,
# und einmal mit numpy.polyfit. Berechnet Konditionszahlen von A^T A und R
# sowie das Fehlerfunktional E(f) = ||y - A·lambda||^2.
# USE WHEN:
# Wenn ein Polynom (oder allgemein eine Linearkombination von Basisfunktionen)
# im Sinne der kleinsten Fehlerquadrate an Daten angepasst werden soll und
# Konditionsfragen relevant sind.
# EXAMPLE:
# Wasserdichte rho(T) durch ein Polynom 2. Grades f(T) = aT^2 + bT + c fitten,
# Konditionszahlen vergleichen.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PART 1 — Inputs
# ============================================================
x_data = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
y_data = np.array([999.9, 999.7, 998.2, 995.7, 992.2, 988.1, 983.2, 977.8, 971.8, 965.3, 958.4])
degree = 2

# ============================================================
# PART 2 — Method selection
# ============================================================
# Options:
#   "normal_equations" -> A^T A·lambda = A^T y
#   "qr_decomposition" -> A = QR, R·lambda = Q^T y
#   "numpy_polyfit"    -> np.polyfit
#   "all"              -> alle drei
method = "all"

# ============================================================
# PART 3 — Implementation
# ============================================================
def fit_polynomial_least_squares(x_data, y_data, degree, method):
    A = np.column_stack([x_data**(degree - k) for k in range(degree + 1)])
    xs = np.linspace(x_data.min(), x_data.max(), 500)

    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, 'ko', markersize=8, label='Datenpunkte')

    runs = ["normal_equations", "qr_decomposition", "numpy_polyfit"] if method == "all" else [method]

    for m in runs:
        if m == "normal_equations":
            AtA = A.T @ A
            lam = np.linalg.solve(AtA, A.T @ y_data)
            cond = np.linalg.cond(AtA)
            style = '--'; key = 'Normalgleichungen'; cond_str = f'cond(A^T A) = {cond:.4e}'
        elif m == "qr_decomposition":
            Q, R = np.linalg.qr(A)
            lam = np.linalg.solve(R, Q.T @ y_data)
            cond = np.linalg.cond(R)
            style = '-.'; key = 'QR-Zerlegung'; cond_str = f'cond(R) = {cond:.4e}'
        elif m == "numpy_polyfit":
            lam = np.polyfit(x_data, y_data, degree)
            style = ':'; key = 'numpy.polyfit'; cond_str = '(keine Matrix exponiert)'
        else:
            raise ValueError("method invalid")

        err = np.linalg.norm(y_data - np.polyval(lam, x_data), 2)**2
        print(f"--- {key} ---")
        print(f"lambda = {lam}")
        print(f"{cond_str}")
        print(f"E(f) = ||y - A·lambda||^2 = {err:.6e}\n")
        plt.plot(xs, np.polyval(lam, xs), style, label=key)

    plt.xlabel('T [°C]'); plt.ylabel('Dichte [g/l]')
    plt.legend(); plt.grid(True)
    plt.title(f'Polynom-Fit Grad {degree}')
    plt.show()

# ============================================================
# PART 4 — Call
# ============================================================
fit_polynomial_least_squares(x_data, y_data, degree, method)
