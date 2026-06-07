# ============================================================
# TOPIC: Nonlinear systems of equations — Parameter fitting for soil pressure model
# DESCRIPTION:
# Determines the constants k1, k2, k3 in the soil pressure model
#   p(r) = k1·e^(k2·r) + k3·r
# via damped Newton method from 3 measurement points.
# Subsequently determines the minimum radius r* at which a circular disk
# can bear a load of 500 N without sinking more than 30 cm.
# USE WHEN:
# When model constants are to be determined from measurement points via Newton
# and a subsequent prediction with the fitted model is requested.
# EXAMPLE:
# Measurement points: (r=1, p=10), (r=2, p=12), (r=3, p=15) [N/cm²]
# Initial vector k^(0) = (10, 0.1, -1)^T
# Goal: minimum radius for a load of 500 N at sinking depth < 30 cm.
# ============================================================

import numpy as np
import sympy as sp

# ============================================================
# PART 1 — Inputs
# ============================================================
# Measurement points (radius in cm, pressure in N/cm²)
data_points = [
    (1, 10),   # r=1 cm -> p=10 N/cm²  (sinking depth = 30 cm)
    (2, 12),   # r=2 cm -> p=12 N/cm²
    (3, 15),   # r=3 cm -> p=15 N/cm²
]

k0       = np.array([10.0, 0.1, -1.0], dtype=float)  # initial vector for k1, k2, k3
tol      = 1e-8                                        # stop threshold for Newton
max_iter = 100                                         # maximum number of iterations
k_max    = 4                                           # damping levels

target_force = 500.0    # N   — minimum load capacity of the disk
depth        = 30.0     # cm  — maximum sinking depth (same as in measurement points)

# Search range for minimum radius (bisection specification)
r_search_min = 1.0      # cm
r_search_max = 100.0    # cm

# ============================================================
# PART 2 — Method selection
# ============================================================
# Which part should be executed?
# options: "fit"       — parameter fitting only (k1, k2, k3)
#          "predict"   — parameter fitting + prediction of minimum radius
mode = "predict"

# ============================================================
# PART 3 — Implementation
# ============================================================
def fit_soil_pressure_model_damped_newton(data_points, k0, tol, max_iter, k_max,
                                          target_force, r_search_min, r_search_max,
                                          mode):

    # Build the nonlinear system of equations f(k1,k2,k3) = 0
    # from the 3 measurement points: k1·e^(k2·ri) + k3·ri - pi = 0
    def f_eval(k_vec):
        k1, k2, k3 = k_vec
        return np.array([
            k1 * np.exp(k2 * r) + k3 * r - p
            for r, p in data_points
        ], dtype=float)

    def Df_eval(k_vec):
        k1, k2, k3 = k_vec
        Df = np.zeros((3, 3), dtype=float)
        for i, (r, _) in enumerate(data_points):
            Df[i, 0] = np.exp(k2 * r)          # df_i/dk1
            Df[i, 1] = k1 * r * np.exp(k2 * r) # df_i/dk2
            Df[i, 2] = r                         # df_i/dk3
        return Df

    # Damped Newton method
    def damped_newton(f, Df, x0):
        x = x0.copy()
        print(f"Initial vector k^(0) = {x}")
        print(f"Tolerance = {tol},  kmax = {k_max}\n")
        print(f"{'Iter':<6} {'||f(k)||_2':<18} {'k1':<14} {'k2':<14} {'k3':<14}")
        print("-" * 70)
        print(f"{'0':<6} {np.linalg.norm(f(x),2):<18.6e} {x[0]:<14.6f} {x[1]:<14.6f} {x[2]:<14.6f}")

        for iteration in range(1, max_iter + 1):
            fx  = f(x)
            Dfx = Df(x)
            try:
                delta = np.linalg.solve(Dfx, -fx)
            except np.linalg.LinAlgError:
                print("Jacobian matrix singular — stopping.")
                break

            err_curr = np.linalg.norm(fx, 2)
            k_found  = None
            for k in range(k_max + 1):
                if np.linalg.norm(f(x + delta / (2**k)), 2) < err_curr:
                    k_found = k
                    break
            if k_found is None:
                k_found = 0

            x = x + delta / (2**k_found)
            f_norm = np.linalg.norm(f(x), 2)
            print(f"{iteration:<6} {f_norm:<18.6e} {x[0]:<14.6f} {x[1]:<14.6f} {x[2]:<14.6f}")

            if f_norm < tol:
                print(f"\nConverged after {iteration} iterations.")
                return x

        print(f"No convergence after {max_iter} iterations.")
        return x

    # --- Parameter fitting ---
    print("=" * 60)
    print("Parameter fitting: k1, k2, k3")
    print("=" * 60)
    k_sol = damped_newton(f_eval, Df_eval, k0)
    k1, k2, k3 = k_sol
    print(f"\nFound parameters:")
    print(f"  k1 = {k1:.6f}")
    print(f"  k2 = {k2:.6f}")
    print(f"  k3 = {k3:.6f}")

    # Verification at measurement points
    print("\nVerification at measurement points:")
    for r, p_meas in data_points:
        p_model = k1 * np.exp(k2 * r) + k3 * r
        print(f"  r={r}: p_measured={p_meas}, p_model={p_model:.4f}")

    if mode == "fit":
        return k_sol

    # --- Prediction: minimum radius for load of 500 N ---
    print("\n" + "=" * 60)
    print(f"Prediction: minimum radius for load = {target_force} N")
    print("=" * 60)
    # Equilibrium: load = pressure x area
    # target_force = p(r) · π · r²   =>   g(r) = k1·e^(k2·r) + k3·r - target_force/(π·r²) = 0
    # Bisection on [r_search_min, r_search_max]
    def g(r):
        area = np.pi * r**2    # area of the circular disk in cm²
        return k1 * np.exp(k2 * r) + k3 * r - target_force / area

    # Check sign at search range boundaries
    ga = g(r_search_min)
    gb = g(r_search_max)
    if ga * gb > 0:
        print(f"No sign change found in interval [{r_search_min}, {r_search_max}].")
        print(f"  g({r_search_min}) = {ga:.4f},  g({r_search_max}) = {gb:.4f}")
        print("  Please adjust r_search_min / r_search_max.")
        return k_sol

    # Bisection
    a, b = r_search_min, r_search_max
    for _ in range(100):
        mid = (a + b) / 2
        if g(mid) * g(a) <= 0:
            b = mid
        else:
            a = mid
        if abs(b - a) < 1e-8:
            break

    r_star = (a + b) / 2
    p_star = k1 * np.exp(k2 * r_star) + k3 * r_star
    F_star = p_star * np.pi * r_star**2

    print(f"\nMinimum radius: r* = {r_star:.4f} cm")
    print(f"Pressure at r*:     p(r*) = {p_star:.4f} N/cm^2")
    print(f"Load at r*:  F = p(r*)*pi*r*^2 = {F_star:.2f} N  (target: {target_force} N)")

    return k_sol

# ============================================================
# PART 4 — Call
# ============================================================
fit_soil_pressure_model_damped_newton(
    data_points, k0, tol, max_iter, k_max,
    target_force, r_search_min, r_search_max,
    mode
)
