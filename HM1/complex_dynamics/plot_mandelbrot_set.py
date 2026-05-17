# ============================================================
# TOPIC: Komplexe Dynamik — Mandelbrot-Menge
# DESCRIPTION:
# Berechnet ein 2D-Array mit Iterationszahlen für die Mandelbrot-Menge
# Z_{n+1} = Z_n^2 + C, Z_0 = 0, C ∈ ℂ. Bricht ab, sobald |Z_n| > 2,
# und plottet die Fluchtgeschwindigkeit als Bild (imshow).
# USE WHEN:
# Wenn die Mandelbrot-Menge als anschauliches Beispiel komplexer
# Dynamik / iterativer Verfahren gerendert werden soll.
# EXAMPLE:
# 1500 px in jeder Richtung, 200 Iterationen, Standard-Ausschnitt.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PART 1 — Inputs
# ============================================================
anzahl_pixel = 1500   # Auflösung in x- und y-Richtung
max_iter     = 200    # max. Iterationen pro Punkt
x_min        = -2.0
x_max        =  0.7
y_min        = -1.4
y_max        =  1.4

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here.

# ============================================================
# PART 3 — Implementation
# ============================================================
def plot_mandelbrot_set(anzahl_pixel, max_iter, x_min, x_max, y_min, y_max):
    re_axis = np.linspace(x_min, x_max, anzahl_pixel, dtype=np.float64)
    im_axis = np.linspace(y_min, y_max, anzahl_pixel, dtype=np.float64)
    Re, Im = np.meshgrid(re_axis, im_axis)
    C = Re + 1j * Im
    Z = np.zeros_like(C, dtype=np.complex128)
    iters = np.zeros_like(C, dtype=int)

    for k in range(max_iter):
        Z = Z ** 2 + C
        mask = (np.abs(Z) > 2) & (C != 0)
        iters[mask] = k
        Z[mask] = 0
        C[mask] = 0

    plt.figure()
    plt.imshow(iters, extent=(x_min, x_max, y_min, y_max), origin="lower")
    plt.xlabel("Re(C)")
    plt.ylabel("Im(C)")
    plt.title("Mandelbrot-Menge (Iterationen bis |Z| > 2)")
    plt.colorbar(label="Iterationen")
    plt.tight_layout()
    plt.show()
    return iters

# ============================================================
# PART 4 — Call
# ============================================================
plot_mandelbrot_set(anzahl_pixel, max_iter, x_min, x_max, y_min, y_max)
