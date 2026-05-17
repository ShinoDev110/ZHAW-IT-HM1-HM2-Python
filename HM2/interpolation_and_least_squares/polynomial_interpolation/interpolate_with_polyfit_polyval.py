# ============================================================
# TOPIC: Polynom-Interpolation mit numpy.polyfit / numpy.polyval
# DESCRIPTION:
# Berechnet die Koeffizienten eines Interpolationspolynoms (Grad = n_pts - 1)
# mit numpy.polyfit und wertet es aus. Optional kann x vor dem Fit
# zentriert werden (x - x.mean()), um die Kondition zu verbessern.
# USE WHEN:
# Wenn ein Polynom-Fit mit numpy gemacht werden soll und Konditionsprobleme
# bei rohen Eingabewerten (z.B. Jahreszahlen) untersucht werden sollen.
# EXAMPLE:
# Anteil US-Haushalte mit Computer 1981-2010 durch Polynom Grad 9 fitten und
# auf x in [1975, 2020] plotten.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PART 1 — Inputs
# ============================================================
x_data = np.array([1981, 1984, 1989, 1993, 1997, 2000, 2001, 2003, 2004, 2010], dtype=float)
y_data = np.array([0.5, 8.2, 15, 22.9, 36.6, 51, 56.3, 61.8, 65, 76.7], dtype=float)
x_plot_range = (1975, 2020)
y_plot_range = (-100, 250)
plot_step    = 0.1

# ============================================================
# PART 2 — Method selection
# ============================================================
# Options:
#   "raw"           -> polyfit directly on x_data
#   "mean_centered" -> subtract x.mean() before polyfit
#   "both"          -> both side by side
method = "both"

# ============================================================
# PART 3 — Implementation
# ============================================================
def interpolate_polyfit_polyval(x_data, y_data, x_plot_range, y_plot_range, plot_step, method):
    n      = len(x_data) - 1
    x_plot = np.arange(x_plot_range[0], x_plot_range[1] + plot_step, plot_step)

    def fit(centered):
        if centered:
            xm = x_data.mean()
            coeffs = np.polyfit(x_data - xm, y_data, n)
            y_at_pts = np.polyval(coeffs, x_data - xm)
            y_plot   = np.polyval(coeffs, x_plot - xm)
        else:
            coeffs = np.polyfit(x_data, y_data, n)
            y_at_pts = np.polyval(coeffs, x_data)
            y_plot   = np.polyval(coeffs, x_plot)
        return coeffs, y_at_pts, y_plot

    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, 'ko', markersize=8, label='Datenpunkte')

    runs = ["raw", "mean_centered"] if method == "both" else [method]
    for m in runs:
        coeffs, y_at_pts, y_plot = fit(m == "mean_centered")
        err = np.max(np.abs(y_at_pts - y_data))
        print(f"--- {m} ---")
        print(f"Max. Fehler an Datenpunkten: {err:.4e}")
        plt.plot(x_plot, y_plot, label=f'Polynom Grad {n} ({m})')

    plt.xlim(x_plot_range); plt.ylim(y_plot_range)
    plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.grid(True)
    plt.title('Polynom-Interpolation mit polyfit/polyval')
    plt.show()

# ============================================================
# PART 4 — Call
# ============================================================
interpolate_polyfit_polyval(x_data, y_data, x_plot_range, y_plot_range, plot_step, method)
