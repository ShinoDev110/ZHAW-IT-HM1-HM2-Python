# ============================================================
# TOPIC: Log Linearization — reduce nonlinear exponential fit to a linear fit
# DESCRIPTION:
# Transforms an exponential model by taking the logarithm into a linear
# least-squares problem and solves it via normal equations. Optional extrapolation
# to a future x-point. Semi-logarithmic plot.
# USE WHEN:
# When the function to be fitted has an exponential form (f(x) = a·exp(b·x)
# or N(t) = 10^(θ1+θ2·t)) and can be linearized by taking the logarithm.
# EXAMPLE:
# Moore's Law: fit log10(N) = θ1 + (t-1970)·θ2 to transistor counts
# and extrapolate to 2015.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PART 1 — Inputs
# ============================================================
x_data = np.array([1971,1972,1974,1978,1982,1985,1989,1993,1997,1999,2000,2002,2003], dtype=float)
y_data = np.array([2250,2500,5000,29000,120000,275000,1180000,3100000,7500000,
                   24000000,42000000,220000000,410000000], dtype=float)
x_shift       = 1970
x_extrapolate = 2015

# ============================================================
# PART 2 — Method selection
# ============================================================
# Options:
#   "exp"      -> fit f(x) = a · exp(b · (x - x_shift))
#   "power_10" -> fit N(t) = 10^(theta1 + theta2 · (t - x_shift))
method = "power_10"

# ============================================================
# PART 3 — Implementation
# ============================================================
def fit_via_log_linearization(x_data, y_data, x_shift, x_extrapolate, method):
    xs = x_data - x_shift

    if method == "exp":
        y_log = np.log(y_data)
        A = np.column_stack([xs, np.ones_like(xs)])
        lam = np.linalg.solve(A.T @ A, A.T @ y_log)
        b, ln_a = lam; a = np.exp(ln_a)
        print(f"a = {a:.6f}, b = {b:.6f}")
        def f(xv):
            return a * np.exp(b * (xv - x_shift))
        label = f'y = {a:.4g}·exp({b:.4g}·(x-{x_shift}))'
    elif method == "power_10":
        y_log = np.log10(y_data)
        A = np.column_stack([np.ones_like(xs), xs])
        lam = np.linalg.solve(A.T @ A, A.T @ y_log)
        theta1, theta2 = lam
        print(f"theta1 = {theta1:.6f}, theta2 = {theta2:.6f}")
        if theta2 > 0:
            print(f"Doubling time = log10(2)/theta2 = {np.log10(2)/theta2:.3f} years")
        def f(xv):
            return 10**(theta1 + theta2 * (xv - x_shift))
        label = f'log10(N) = {theta1:.4g}+{theta2:.4g}·(t-{x_shift})'
    else:
        raise ValueError("method must be 'exp' or 'power_10'")

    x_dense = np.linspace(x_data.min(), x_extrapolate, 500)
    print(f"Extrapolation at x = {x_extrapolate}: y = {f(x_extrapolate):.4e}")
    print(f"E(f) = {np.linalg.norm(y_data - f(x_data), 2)**2:.4e}")

    plt.figure(figsize=(10, 6))
    plt.semilogy(x_data, y_data, 'ko', markersize=8, label='Data points')
    plt.semilogy(x_dense, f(x_dense), 'b-', label=label)
    plt.semilogy(x_extrapolate, f(x_extrapolate), 'r*', markersize=14, label=f'Extrapolation @ {x_extrapolate}')
    plt.xlabel('x'); plt.ylabel('y (log)'); plt.legend(); plt.grid(True, which='both')
    plt.title('Log-Linearized Fit')
    plt.show()

# ============================================================
# PART 4 — Call
# ============================================================
fit_via_log_linearization(x_data, y_data, x_shift, x_extrapolate, method)
