# ============================================================
# TOPIC: Numerical Integration — comparison of rectangle / trapezoidal / Simpson / Gauss / Romberg
# DESCRIPTION:
# Computes the same definite integral with all covered methods and shows the
# value and absolute error against a reference value
# (scipy.integrate.quad as a high-accuracy reference).
# USE WHEN:
# A task asks for a comparison of the accuracy of various quadrature methods.
# EXAMPLE:
# Compare all methods for int_0^0.5 exp(-x^2) dx with n = 3 (Romberg: m = 3).
# ============================================================

import numpy as np
from scipy.integrate import quad

# ============================================================
# PART 1 — Inputs
# ============================================================
def f(x):
    return np.exp(-x**2)

a = 0.0
b = 0.5
n = 3        # subintervals for Newton-Cotes
m = 3        # levels for Romberg

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: run all and print comparison.

# ============================================================
# PART 3 — Implementation
# ============================================================
def compare_quadrature_methods(f, a, b, n, m):
    exact, _ = quad(f, a, b)
    h = (b - a) / n

    # Rectangle
    mids = np.array([a + (i + 0.5) * h for i in range(n)])
    Rf = h * np.sum(f(mids))

    # Trapezoidal
    x = np.array([a + i * h for i in range(n + 1)])
    y = f(x)
    Tf = h * ((y[0] + y[-1]) / 2 + np.sum(y[1:-1]))

    # Simpson
    midpts = (x[:-1] + x[1:]) / 2.0
    Sf = h / 3 * (0.5 * (y[0] + y[-1]) + np.sum(y[1:-1]) + 2 * np.sum(f(midpts)))

    # Gauss
    mid_g, half_g = (a + b) / 2, (b - a) / 2
    G1f = (b - a) * f(mid_g)
    G2f = half_g * (f(-1/np.sqrt(3) * half_g + mid_g) + f(1/np.sqrt(3) * half_g + mid_g))
    G3f = half_g * (5/9 * f(-np.sqrt(0.6) * half_g + mid_g)
                    + 8/9 * f(mid_g)
                    + 5/9 * f(np.sqrt(0.6) * half_g + mid_g))

    # Romberg
    T = np.zeros((m + 1, m + 1))
    for j in range(m + 1):
        nj = 2**j; hj = (b - a) / nj
        xj = np.array([a + i * hj for i in range(nj + 1)])
        yj = f(xj)
        T[j, 0] = hj * ((yj[0] + yj[-1]) / 2 + np.sum(yj[1:-1]))
    for k in range(1, m + 1):
        for j in range(m + 1 - k):
            T[j, k] = (4**k * T[j + 1, k - 1] - T[j, k - 1]) / (4**k - 1)
    Romberg = T[0, m]

    results = [
        (f"Rectangle (n={n})",   Rf),
        (f"Trapezoidal (n={n})", Tf),
        (f"Simpson (n={n})",     Sf),
        ("Gauss G1",             G1f),
        ("Gauss G2",             G2f),
        ("Gauss G3",             G3f),
        (f"Romberg (m={m})",     Romberg),
    ]

    print(f"Reference (scipy.quad): {exact:.12f}\n")
    print(f"{'Method':<22} {'Value':<22} {'|Error|':<14}")
    print("-" * 60)
    for name, val in results:
        print(f"{name:<22} {val:<22.12f} {abs(val - exact):.6e}")

# ============================================================
# PART 4 — Call
# ============================================================
compare_quadrature_methods(f, a, b, n, m)
