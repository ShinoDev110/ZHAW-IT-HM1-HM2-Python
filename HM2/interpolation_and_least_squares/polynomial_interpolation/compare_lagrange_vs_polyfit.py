# ============================================================
# TOPIC: Vergleich Lagrange-Interpolation vs. polyfit/polyval
# DESCRIPTION:
# Plottet die Lagrange-Interpolation (eigene Implementation) und den
# polyfit-Fit (mit Mittelwertzentrierung) in derselben Grafik. Beide
# Verfahren sollten identische Polynome liefern; numerische Unterschiede
# zeigen Konditionsprobleme.
# USE WHEN:
# Wenn eine Aufgabe nach dem Vergleich der beiden Interpolationsmethoden
# fragt.
# EXAMPLE:
# Vergleich beider Verfahren für die Computer-Haushalte-Zeitreihe auf
# x in [1981, 2010].
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PART 1 — Inputs
# ============================================================
x_data = np.array([1981, 1984, 1989, 1993, 1997, 2000, 2001, 2003, 2004, 2010], dtype=float)
y_data = np.array([0.5, 8.2, 15, 22.9, 36.6, 51, 56.3, 61.8, 65, 76.7], dtype=float)
x_plot_range = (1981, 2010)
y_plot_range = (-100, 250)
plot_step    = 0.1

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: run both and overlay.

# ============================================================
# PART 3 — Implementation
# ============================================================
def compare_lagrange_polyfit(x_data, y_data, x_plot_range, y_plot_range, plot_step):
    x_plot = np.arange(x_plot_range[0], x_plot_range[1] + plot_step, plot_step)
    n_pts  = len(x_data)

    def lagrange_eval(xv):
        total = 0.0
        for i in range(n_pts):
            li = 1.0
            for j in range(n_pts):
                if j != i:
                    li *= (xv - x_data[j]) / (x_data[i] - x_data[j])
            total += li * y_data[i]
        return total

    y_lagrange = np.array([lagrange_eval(xv) for xv in x_plot])

    xm     = x_data.mean()
    coeffs = np.polyfit(x_data - xm, y_data, n_pts - 1)
    y_polyfit = np.polyval(coeffs, x_plot - xm)

    diff_max = np.max(np.abs(y_lagrange - y_polyfit))
    print(f"Max. Differenz Lagrange vs. polyfit auf Plotgitter: {diff_max:.4e}")

    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, 'ko', markersize=8, label='Datenpunkte')
    plt.plot(x_plot, y_lagrange, 'b-',  label='Lagrange (eigene)')
    plt.plot(x_plot, y_polyfit,  'r--', label='numpy.polyfit (zentriert)')
    plt.xlim(x_plot_range); plt.ylim(y_plot_range)
    plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.grid(True)
    plt.title('Vergleich: Lagrange vs. polyfit')
    plt.show()

# ============================================================
# PART 4 — Call
# ============================================================
compare_lagrange_polyfit(x_data, y_data, x_plot_range, y_plot_range, plot_step)
