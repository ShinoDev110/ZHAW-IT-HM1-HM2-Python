# ============================================================
# TOPIC: Linear least-squares problem — arbitrary basis functions
# DESCRIPTION:
# Fits f(x) = sum_j lambda_j * phi_j(x) with FREELY selectable basis functions
# phi_j (e.g. 1/x, x, x^2, ...) using the least-squares method via
# the normal equations and stably via QR. Optionally y is transformed beforehand
# (reciprocal 1/y or log y) to reduce nonlinear approaches like 1/(a+b*x^2) to a
# linear problem. Returns coefficients and error functional.
# USE WHEN:
# When the approach is a linear combination of arbitrary basis functions (not a
# pure polynomial) or can be linearized via reciprocal/logarithm.
# EXAMPLE:
# Data (-2,-5.5),(-1,-5),(1,5),(2,5.5) with f(x)=alpha/x + beta*x
# -> alpha = 3, beta = 2, error functional E = 0.
# ============================================================

import numpy as np

np.set_printoptions(precision=6, suppress=True)

# ============================================================
# PART 1 — Inputs
# ============================================================
x_data = np.array([-2.0, -1.0, 1.0, 2.0])      # x-values
y_data = np.array([-5.5, -5.0, 5.0, 5.5])      # y-values

# Basis functions phi_j(x) (vectorized lambdas) + labels
basis_functions = [lambda x: 1.0 / x, lambda x: x]
basis_labels    = ["1/x", "x"]

# Alternative example (PNG task 2b): y = 1/(a + b*x^2)  ->  1/y = a + b*x^2
# x_data = np.array([0,1,2,3,4,5], dtype=float)
# y_data = np.array([0.54,0.44,0.28,0.18,0.12,0.08])
# basis_functions = [lambda x: np.ones_like(x), lambda x: x**2]
# basis_labels    = ["1", "x^2"];  y_transform = "reciprocal"

# ============================================================
# PART 2 — Method selection
# ============================================================
# y_transform:
#   "none"       -> directly fit f(x) = sum lambda_j phi_j(x)
#   "reciprocal" -> fit 1/y (for approach y = 1/(sum lambda_j phi_j(x)))
#   "log"        -> fit ln(y) (for approach y = exp(sum lambda_j phi_j(x)))
y_transform = "none"

# ============================================================
# PART 3 — Implementation
# ============================================================
def _forward(y, transform):
    if transform == "none":       return y.copy()
    if transform == "reciprocal": return 1.0 / y
    if transform == "log":        return np.log(y)
    raise ValueError(f"Unknown y_transform: {transform!r}")

def _inverse(lin, transform):
    if transform == "none":       return lin
    if transform == "reciprocal": return 1.0 / lin
    if transform == "log":        return np.exp(lin)
    raise ValueError(f"Unknown y_transform: {transform!r}")

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
    print("Linear least-squares problem with arbitrary basis functions")
    print("============================================================")
    print(f"Basis: {basis_labels}   y-transform: {y_transform}")
    print("Design matrix A =\n", A)

    # Normal equations
    AtA = A.T @ A
    lam_ne = np.linalg.solve(AtA, A.T @ y_fit)
    # QR (more stable)
    Q, R = np.linalg.qr(A)
    lam_qr = np.linalg.solve(R, Q.T @ y_fit)

    print("\n-- Normal equations  (A^T A) lambda = A^T y")
    print(f"lambda = {lam_ne}   cond(A^T A) = {np.linalg.cond(AtA):.4e}")
    print("-- QR decomposition")
    print(f"lambda = {lam_qr}   cond(R) = {np.linalg.cond(R):.4e}")

    lam = lam_qr
    for name, val in zip(basis_labels, lam):
        print(f"   Coefficient for {name:>6} = {val:.10g}")

    # Error functional in the (linear) fit space and in the original space
    E_fit = float(np.linalg.norm(y_fit - A @ lam, 2)**2)
    model_data = _inverse(A @ lam, y_transform)
    E_orig = float(np.linalg.norm(y_data - model_data, 2)**2)
    print(f"\nError functional E (fit space)       = ||y_fit - A·lambda||^2 = {E_fit:.6e}")
    print(f"Error functional E (original data)   = sum (y_i - f(x_i))^2   = {E_orig:.6e}")
    print(f"Model: f(x) = {_model_label(basis_labels, lam)}"
          + ("" if y_transform == "none" else f"   (applied to {y_transform})"))
    return lam, E_orig

# ============================================================
# PART 4 — Call
# ============================================================
fit_with_linear_basis_functions(x_data, y_data, basis_functions, basis_labels, y_transform)
