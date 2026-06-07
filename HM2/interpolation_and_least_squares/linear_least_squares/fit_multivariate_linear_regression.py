# ============================================================
# TOPIC: Multivariate linear regression — y = λ1·x1 + ... + λn·xn + λ_{n+1}
# DESCRIPTION:
# Solves the overdetermined system A·lambda = y with multiple input features
# and an intercept via normal equations or QR decomposition.
# USE WHEN:
# When y is to be fitted as a linear combination of several independent variables
# plus a constant (multiple regression).
# EXAMPLE:
# Mass of escaped hydrocarbon vapors as a function of tank temperature,
# fuel temperature, tank pressure, and fuel pressure.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PART 1 — Inputs
# ============================================================
X_features = np.array([
    [33, 53, 3.32, 3.42], [31, 36, 3.10, 3.26], [33, 51, 3.18, 3.18],
    [37, 51, 3.39, 3.08], [36, 54, 3.20, 3.41], [35, 35, 3.03, 3.03],
    [59, 56, 4.78, 4.57], [60, 60, 4.72, 4.72], [59, 60, 4.60, 4.41],
    [60, 60, 4.53, 4.53], [34, 35, 2.90, 2.95], [60, 59, 4.40, 4.36],
    [60, 62, 4.31, 4.42], [60, 36, 4.27, 3.94], [62, 38, 4.41, 3.49],
    [62, 61, 4.39, 4.39], [90, 64, 7.32, 6.70], [90, 60, 7.32, 7.20],
    [92, 92, 7.45, 7.45], [91, 92, 7.27, 7.26], [61, 62, 3.91, 4.08],
    [59, 42, 3.75, 3.45], [88, 65, 6.48, 5.80], [91, 89, 6.70, 6.60],
    [63, 62, 4.30, 4.30], [60, 61, 4.02, 4.10], [60, 62, 4.02, 3.89],
    [59, 62, 3.98, 4.02], [59, 62, 4.39, 4.53], [37, 35, 2.75, 2.64],
    [35, 35, 2.59, 2.59], [37, 37, 2.73, 2.59]
], dtype=float)
y_data = np.array([29,24,26,22,27,21,33,34,32,34,20,36,34,23,24,32,
                   40,46,55,52,29,22,31,45,37,37,33,27,34,19,16,22], dtype=float)

# ============================================================
# PART 2 — Method selection
# ============================================================
# Options:
#   "normal_equations" -> A^T A·lambda = A^T y
#   "qr_decomposition" -> via QR (better conditioned)
method = "qr_decomposition"

# ============================================================
# PART 3 — Implementation
# ============================================================
def fit_multivariate_linear_regression(X_features, y_data, method):
    n = X_features.shape[0]
    A = np.column_stack([X_features, np.ones(n)])

    if method == "normal_equations":
        lam = np.linalg.solve(A.T @ A, A.T @ y_data)
    elif method == "qr_decomposition":
        Q, R = np.linalg.qr(A)
        lam = np.linalg.solve(R, Q.T @ y_data)
    else:
        raise ValueError("method must be 'normal_equations' or 'qr_decomposition'")

    y_fit = A @ lam
    err   = np.linalg.norm(y_data - y_fit, 2)**2

    print(f"Method: {method}")
    print(f"lambda = {lam}")
    print(f"E(f) = ||y - A·lambda||^2 = {err:.4e}")

    plt.figure(figsize=(11, 5))
    idx = np.arange(n)
    plt.plot(idx, y_data, 'ko', label='Measurement points')
    plt.plot(idx, y_fit,  'rx-', label='Fit')
    plt.xlabel('Index'); plt.ylabel('y'); plt.legend(); plt.grid(True)
    plt.title('Multivariate Linear Regression')
    plt.show()

# ============================================================
# PART 4 — Call
# ============================================================
fit_multivariate_linear_regression(X_features, y_data, method)
