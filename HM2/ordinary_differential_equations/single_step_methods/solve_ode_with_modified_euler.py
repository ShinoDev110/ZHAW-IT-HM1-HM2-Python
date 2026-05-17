# ============================================================
# TOPIC: DGL — modifiziertes Euler-Verfahren (Heun) für y'(x) = f(x, y)
# DESCRIPTION:
# Berechnet k1 = f(x_i, y_i), prädiziert mit klassischem Euler-Schritt,
# wertet die Steigung k2 am Endpunkt aus und nimmt den Durchschnitt
# (k1 + k2) / 2 als effektive Steigung. Konvergenzordnung p = 2.
# USE WHEN:
# Wenn eine DGL 1. Ordnung mit einem Heun-artigen Verfahren 2. Ordnung
# gelöst werden soll.
# EXAMPLE:
# Löse y' = t^2 + 0.1 * y mit y(-1.5) = 0 auf [-1.5, 1.5] mit n = 5.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PART 1 — Inputs
# ============================================================
def f(x, y):
    return x**2 + 0.1 * y

a, b = -1.5, 1.5
y0   = 0.0
n    = 5

def y_exact(x):
    return -10*x**2 - 200*x - 2000 + 1722.5 * np.exp(0.05 * (2*x + 3))

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: modified Euler (Heun).

# ============================================================
# PART 3 — Implementation
# ============================================================
def solve_ode_modified_euler(f, a, b, y0, n, y_exact=None):
    h = (b - a) / n
    x = np.array([a + i * h for i in range(n + 1)])
    y = np.zeros(n + 1)
    y[0] = y0

    print(f"{'i':<3} {'x_i':<10} {'y_i':<14}")
    print(f"{0:<3} {x[0]:<10.4f} {y[0]:<14.6f}")
    for i in range(n):
        k1 = f(x[i], y[i])
        y_euler = y[i] + h * k1
        k2 = f(x[i + 1], y_euler)
        y[i + 1] = y[i] + h * (k1 + k2) / 2
        print(f"{i+1:<3} {x[i+1]:<10.4f} {y[i+1]:<14.6f}")

    plt.figure(figsize=(9, 6))
    plt.plot(x, y, 'g-+', label='Mod. Euler')
    if y_exact is not None:
        xs = np.linspace(a, b, 300)
        plt.plot(xs, y_exact(xs), 'r-', label='Exakt')
        err = np.abs(y_exact(x) - y)
        print("\nAbsoluter Fehler:")
        for i in range(n + 1):
            print(f"  i = {i}, x = {x[i]:.4f}, err = {err[i]:.6e}")
    plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.grid(True)
    plt.title("Modifiziertes Euler-Verfahren (Heun)")
    plt.show()
    return x, y

# ============================================================
# PART 4 — Call
# ============================================================
solve_ode_modified_euler(f, a, b, y0, n, y_exact)
