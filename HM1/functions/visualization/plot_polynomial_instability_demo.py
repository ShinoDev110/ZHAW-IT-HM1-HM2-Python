# ============================================================
# TOPIC: Visualisierung — numerische Instabilität bei (x - 2)^7
# DESCRIPTION:
# Vergleicht die ausmultiplizierte Form
#   x^7 - 14x^6 + 84x^5 - 280x^4 + 560x^3 - 672x^2 + 448x - 128
# mit der kompakten Form (x - 2)^7 in einem sehr engen Intervall um
# x = 2 und plottet beide. Demonstriert eindrucksvoll Auslöschung.
# USE WHEN:
# Wenn der Effekt von katastrophaler Auslöschung in Polynomdarstellungen
# anschaulich gemacht werden soll.
# EXAMPLE:
# x ∈ [1.99, 2.01], 501 Stützstellen.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PART 1 — Inputs
# ============================================================
x_min = 1.99
x_max = 2.01
n_pts = 501
y_min = -0.01
y_max =  0.01

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here.

# ============================================================
# PART 3 — Implementation
# ============================================================
def plot_polynomial_instability_demo(x_min, x_max, n_pts, y_min, y_max):
    xs = np.linspace(x_min, x_max, n_pts)
    ausmultipliziert = (
        (xs ** 7) - 14 * (xs ** 6) + 84 * (xs ** 5) - 280 * (xs ** 4)
        + 560 * (xs ** 3) - 672 * (xs ** 2) + 448 * xs - 128
    )
    kompakt = (xs - 2.0) ** 7

    plt.figure()
    plt.plot(xs, ausmultipliziert, label="f1(x) ausmultipliziert")
    plt.plot(xs, kompakt,          label="f2(x) = (x-2)^7")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Numerische Instabilität: (x-2)^7 ausmultipliziert vs. kompakt")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ============================================================
# PART 4 — Call
# ============================================================
plot_polynomial_instability_demo(x_min, x_max, n_pts, y_min, y_max)
