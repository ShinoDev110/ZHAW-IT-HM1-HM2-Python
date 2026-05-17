# ============================================================
# TOPIC: DGL — klassisches Euler-Verfahren für y'(x) = f(x, y)
# DESCRIPTION:
# Löst das Anfangswertproblem y'(x) = f(x, y), y(a) = y0 auf [a, b] mit
# n Schritten via y_{i+1} = y_i + h * f(x_i, y_i). Plottet die Lösung,
# optional zusammen mit der exakten Lösung und/oder dem Richtungsfeld.
# USE WHEN:
# Wenn eine DGL 1. Ordnung mit dem einfachsten Einschrittverfahren
# (Konvergenzordnung p = 1) gelöst werden soll.
# EXAMPLE:
# Löse y' = t^2 + 0.1*y mit y(-1.5) = 0 auf [-1.5, 1.5] mit n = 5.
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

# Optionale exakte Lösung (auf None setzen, wenn nicht verfügbar)
def y_exact(x):
    return -10*x**2 - 200*x - 2000 + 1722.5 * np.exp(0.05 * (2*x + 3))

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: classical (explicit) Euler.

# ============================================================
# PART 3 — Implementation
# ============================================================
def solve_ode_euler(f, a, b, y0, n, y_exact=None):
    h = (b - a) / n
    x = np.array([a + i * h for i in range(n + 1)])
    y = np.zeros(n + 1)
    y[0] = y0

    print(f"{'i':<3} {'x_i':<10} {'y_i':<14}")
    print(f"{0:<3} {x[0]:<10.4f} {y[0]:<14.6f}")
    for i in range(n):
        y[i + 1] = y[i] + h * f(x[i], y[i])
        print(f"{i+1:<3} {x[i+1]:<10.4f} {y[i+1]:<14.6f}")

    plt.figure(figsize=(9, 6))
    plt.plot(x, y, 'm-+', label='Euler')
    if y_exact is not None:
        xs = np.linspace(a, b, 300)
        plt.plot(xs, y_exact(xs), 'r-', label='Exakt')
        err = np.abs(y_exact(x) - y)
        print("\nAbsoluter Fehler |y(x_i) - y_i|:")
        for i in range(n + 1):
            print(f"  i = {i}, x = {x[i]:.4f}, err = {err[i]:.6e}")
    plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.grid(True)
    plt.title("Euler-Verfahren")
    plt.show()
    return x, y

# ============================================================
# PART 4 — Call
# ============================================================
solve_ode_euler(f, a, b, y0, n, y_exact)
