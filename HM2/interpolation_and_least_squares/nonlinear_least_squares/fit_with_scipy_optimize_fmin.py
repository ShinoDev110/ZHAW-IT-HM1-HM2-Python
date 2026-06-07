# ============================================================
# TOPIC: Nonlinear fit with scipy.optimize.fmin (direct minimization)
# DESCRIPTION:
# Minimizes the error functional E(lambda) = sum (y_i - f(lambda, x_i))^2
# directly with scipy.optimize.fmin (Nelder-Mead). No gradient required.
# USE WHEN:
# When a ready-made optimizer is to be used as a comparison or quick alternative
# to the Gauss-Newton method (or as a sanity check).
# EXAMPLE:
# Fit f(x) = λ0 + λ1·10^(λ2+λ3·x) / (1 + 10^(λ2+λ3·x)) with scipy.optimize.fmin
# and compare to the Gauss-Newton solution.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin

# ============================================================
# PART 1 — Inputs
# ============================================================
x_data = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5])
y_data = np.array([159.57209984, 159.8851819, 159.89378952, 160.30305273, 160.84630757,
                   160.94703969, 161.56961845, 162.31468058, 162.32140561, 162.88880047,
                   163.53234609, 163.85817086, 163.55339958, 163.86393263, 163.90535931,
                   163.44385491])

# Model as a plain Python function f(lambda_vec, x_array)
def model(p, x):
    return p[0] + p[1] * 10**(p[2] + p[3]*x) / (1 + 10**(p[2] + p[3]*x))

lambda_0 = np.array([100.0, 120.0, 3.0, -1.0])

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: scipy.optimize.fmin (Nelder-Mead simplex).

# ============================================================
# PART 3 — Implementation
# ============================================================
def fit_with_scipy_fmin(x_data, y_data, model, lambda_0):
    def error_functional(p):
        return np.sum((y_data - model(p, x_data))**2)

    p_opt = fmin(error_functional, lambda_0, disp=True)
    print(f"\nSolution lambda = {p_opt}")
    print(f"E(lambda) = {error_functional(p_opt):.6e}")

    xs = np.linspace(x_data.min(), x_data.max(), 300)
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, 'ko', markersize=8, label='Data points')
    plt.plot(xs, model(p_opt, xs), 'b-', label='scipy.optimize.fmin fit')
    plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.grid(True)
    plt.title('Fit with scipy.optimize.fmin')
    plt.show()
    return p_opt

# ============================================================
# PART 4 — Call
# ============================================================
fit_with_scipy_fmin(x_data, y_data, model, lambda_0)
