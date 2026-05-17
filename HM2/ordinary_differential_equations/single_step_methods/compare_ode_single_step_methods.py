# ============================================================
# TOPIC: DGL — Vergleich Euler / Mittelpunkt / mod. Euler / RK4
# DESCRIPTION:
# Wendet alle vier klassischen Einschrittverfahren auf dieselbe DGL an
# und plottet (1) die Lösungen zusammen sowie (2) den globalen Fehler
# |y(x_i) - y_i| in einer logarithmischen Grafik.
# USE WHEN:
# Wenn eine Aufgabe nach dem direkten Vergleich der Konvergenzordnungen
# fragt oder veranschaulichen will, wie schnell der Fehler mit der
# Methode abnimmt.
# EXAMPLE:
# Vergleiche alle vier Verfahren für y' = x^2 / y mit y(0) = 2 auf [0, 10],
# h = 0.1.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PART 1 — Inputs
# ============================================================
def f(x, y):
    return x**2 / y

a, b = 0.0, 10.0
y0   = 2.0
h    = 0.1

def y_exact(x):
    return np.sqrt(2 * x**3 / 3 + 4)

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: run all four methods.

# ============================================================
# PART 3 — Implementation
# ============================================================
def compare_ode_single_step_methods(f, a, b, y0, h, y_exact):
    n = int(round((b - a) / h))
    x = np.array([a + i * h for i in range(n + 1)])

    def run_euler():
        y = np.zeros(n + 1); y[0] = y0
        for i in range(n):
            y[i+1] = y[i] + h * f(x[i], y[i])
        return y

    def run_midpoint():
        y = np.zeros(n + 1); y[0] = y0
        for i in range(n):
            x_h = x[i] + h/2
            y_h = y[i] + h/2 * f(x[i], y[i])
            y[i+1] = y[i] + h * f(x_h, y_h)
        return y

    def run_mod_euler():
        y = np.zeros(n + 1); y[0] = y0
        for i in range(n):
            k1 = f(x[i], y[i])
            y_eu = y[i] + h * k1
            k2 = f(x[i+1], y_eu)
            y[i+1] = y[i] + h * (k1 + k2) / 2
        return y

    def run_rk4():
        y = np.zeros(n + 1); y[0] = y0
        for i in range(n):
            k1 = f(x[i],     y[i])
            k2 = f(x[i]+h/2, y[i] + h/2*k1)
            k3 = f(x[i]+h/2, y[i] + h/2*k2)
            k4 = f(x[i]+h,   y[i] + h*k3)
            y[i+1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        return y

    runs = [
        ("Euler",       run_euler(),     'm'),
        ("Mittelpunkt", run_midpoint(),  'b'),
        ("Mod. Euler",  run_mod_euler(), 'g'),
        ("RK4",         run_rk4(),       'k'),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    xs = np.linspace(a, b, 500)
    ax1.plot(xs, y_exact(xs), 'r-', label='Exakt')
    for name, y, col in runs:
        ax1.plot(x, y, col + '-', label=name)
    ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.legend(); ax1.grid(True)
    ax1.set_title("Lösungen")

    for name, y, col in runs:
        err = np.abs(y_exact(x) - y)
        ax2.semilogy(x, err, col + '-', label=name)
    ax2.set_xlabel('x'); ax2.set_ylabel('|Fehler| (log)'); ax2.legend(); ax2.grid(True, which='both')
    ax2.set_title("Globaler Fehler")
    plt.tight_layout(); plt.show()

# ============================================================
# PART 4 — Call
# ============================================================
compare_ode_single_step_methods(f, a, b, y0, h, y_exact)
