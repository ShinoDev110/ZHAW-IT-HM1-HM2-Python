# ============================================================
# TOPIC: Lagrange-Interpolation — interpoliert y-Werte an gegebenen x-Stellen
# DESCRIPTION:
# Eigene Implementierung der Lagrange-Interpolationsformel P_n(x) =
# sum(l_i(x) * y_i). Akzeptiert x_int als Skalar ODER Vektor und plottet
# das Resultat.
# USE WHEN:
# Wenn die Lagrange-Interpolation manuell programmiert werden soll
# (ohne numpy.polyfit) — typischer Klausur-/Übungsstil.
# EXAMPLE:
# Atmosphärendruck bei einer fehlenden Höhe (3750 m) aus einer Messreihe
# schätzen.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PART 1 — Inputs
# ============================================================
x_data = np.array([0, 2500, 5000, 10000], dtype=float)   # Stützstellen
y_data = np.array([1013, 747, 540, 226], dtype=float)    # Stützwerte
x_int  = 3750                                            # Skalar ODER Array

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: Lagrange polynomial via product formula.

# ============================================================
# PART 3 — Implementation
# ============================================================
def lagrange_int(x, y, x_int):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_int_arr = np.atleast_1d(np.asarray(x_int, dtype=float))
    n = len(x)

    def eval_at(xv):
        total = 0.0
        for i in range(n):
            li = 1.0
            for j in range(n):
                if j != i:
                    li *= (xv - x[j]) / (x[i] - x[j])
            total += li * y[i]
        return total

    y_int = np.array([eval_at(xv) for xv in x_int_arr])

    # Plot
    xs = np.linspace(x.min(), x.max(), 500)
    ys = np.array([eval_at(xv) for xv in xs])
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, 'b-', label='Lagrange-Polynom')
    plt.plot(x, y, 'ro', label='Stützpunkte')
    plt.plot(x_int_arr, y_int, 'k*', markersize=12, label='interpoliert')
    plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.grid(True)
    plt.title('Lagrange-Interpolation')
    plt.show()

    print(f"x_int = {x_int_arr}")
    print(f"y_int = {y_int}")
    return y_int

# ============================================================
# PART 4 — Call
# ============================================================
lagrange_int(x_data, y_data, x_int)
