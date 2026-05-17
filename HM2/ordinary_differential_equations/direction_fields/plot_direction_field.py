# ============================================================
# TOPIC: DGL — Richtungsfeld einer DGL 1. Ordnung y'(x) = f(x, y) plotten
# DESCRIPTION:
# Zeichnet das Richtungsfeld einer DGL 1. Ordnung auf einem Gitter mit
# Schrittweiten hx, hy via np.meshgrid und plt.quiver. Pfeile zeigen die
# Steigung f(x, y) in jedem Gitterpunkt.
# USE WHEN:
# Wenn das Richtungsfeld einer DGL veranschaulicht werden soll, oft als
# Grundlage für die spätere Einzeichnung numerischer Lösungen.
# EXAMPLE:
# Richtungsfeld für y' = x^2 / y auf [0, 1.4] x [0, 5] mit hx = hy = 0.2.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PART 1 — Inputs
# ============================================================
def f(x, y):
    return x**2 / y

xmin, xmax = 0.0, 1.4
ymin, ymax = 0.0, 5.0
hx, hy     = 0.1, 0.5

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: arrow direction = (1, f(x, y)), optional normalization.
normalize = True

# ============================================================
# PART 3 — Implementation
# ============================================================
def plot_direction_field(f, xmin, xmax, ymin, ymax, hx, hy, normalize):
    x = np.arange(xmin, xmax + hx, hx)
    y = np.arange(ymin, ymax + hy, hy)
    X, Y = np.meshgrid(x, y)

    U = np.ones_like(X)
    V = f(X, Y)

    if normalize:
        L = np.sqrt(U**2 + V**2)
        U, V = U / L, V / L

    plt.figure(figsize=(9, 6))
    plt.quiver(X, Y, U, V, angles='xy')
    plt.xlim(xmin, xmax); plt.ylim(ymin, ymax)
    plt.xlabel('x'); plt.ylabel('y'); plt.grid(True)
    plt.title("Richtungsfeld y'(x) = f(x, y)")
    plt.show()

# ============================================================
# PART 4 — Call
# ============================================================
plot_direction_field(f, xmin, xmax, ymin, ymax, hx, hy, normalize)
