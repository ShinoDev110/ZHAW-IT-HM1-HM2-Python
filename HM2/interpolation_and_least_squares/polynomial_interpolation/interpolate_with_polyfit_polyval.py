# ============================================================
# TOPIC: Polynomial interpolation with numpy.polyfit / numpy.polyval
# DESCRIPTION:
# Computes the coefficients of an interpolation polynomial (degree = n_pts - 1)
# with numpy.polyfit and evaluates it. Optionally x can be centered before the
# fit (x - x.mean()) to improve conditioning.
# USE WHEN:
# When a polynomial fit with numpy should be done and conditioning problems
# with raw input values (e.g. year numbers) should be investigated.
# EXAMPLE:
# Fit the share of US households with a computer 1981-2010 by a degree-9
# polynomial and plot on x in [1975, 2020].
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
    plt.plot(x_data, y_data, 'ko', markersize=8, label='Data points')

    runs = ["raw", "mean_centered"] if method == "both" else [method]
    for m in runs:
        coeffs, y_at_pts, y_plot = fit(m == "mean_centered")
        err = np.max(np.abs(y_at_pts - y_data))
        print(f"--- {m} ---")
        print(f"Max. error at data points: {err:.4e}")
        plt.plot(x_plot, y_plot, label=f'Polynomial degree {n} ({m})')

    plt.xlim(x_plot_range); plt.ylim(y_plot_range)
    plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.grid(True)
    plt.title('Polynomial interpolation with polyfit/polyval')
    plt.show()

# ============================================================
# PART 4 — Call
# ============================================================
interpolate_polyfit_polyval(x_data, y_data, x_plot_range, y_plot_range, plot_step, method)
