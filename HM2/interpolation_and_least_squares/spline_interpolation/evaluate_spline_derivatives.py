# ============================================================
# TOPIC: Spline-Interpolation — Ableitungen (Geschwindigkeit / Beschleunigung)
# DESCRIPTION:
# Baut die natürliche kubische Splinefunktion s_k(t) = a_k + b_k(t-t_k)
# + c_k(t-t_k)^2 + d_k(t-t_k)^3 und wertet an einer Stelle den Funktionswert,
# die 1. Ableitung (Geschwindigkeit) und die 2. Ableitung (Beschleunigung) aus.
# Zusätzlich wird der Verlauf der 1. Ableitung graphisch dargestellt.
# USE WHEN:
# Wenn aus einer Spline-Bahn x(t) die Geschwindigkeit und Beschleunigung zu
# einem bestimmten Zeitpunkt gefragt sind (z.B. Bewegung eines Werkstücks).
# EXAMPLE:
# t = [0, 0.5, 2, 3], x = [1, 2, 2.5, 0], gesucht s, s' und s'' bei t = 1.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PART 1 — Inputs
# ============================================================
t_data = np.array([0.0, 0.5, 2.0, 3.0])   # Stützstellen (Zeiten)
x_data = np.array([1.0, 2.0, 2.5, 0.0])   # Stützwerte (Positionen)
t_query = 1.0                              # Auswertungsstelle

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: natürlicher kubischer Spline (s''(t_0) = s''(t_n) = 0).

# ============================================================
# PART 3 — Implementation
# ============================================================
def _natural_cubic_coeffs(t, x):
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(t) - 1
    a = x[:-1].copy()
    h = np.diff(t)
    c = np.zeros(n + 1)
    if n >= 2:
        size = n - 1
        A = np.zeros((size, size))
        z = np.zeros(size)
        for i in range(size):
            ii = i + 1
            A[i, i] = 2 * (h[ii - 1] + h[ii])
            if i > 0:        A[i, i - 1] = h[ii - 1]
            if i < size - 1: A[i, i + 1] = h[ii]
            z[i] = 3 * (x[ii + 1] - x[ii]) / h[ii] - 3 * (x[ii] - x[ii - 1]) / h[ii - 1]
        c[1:n] = np.linalg.solve(A, z)
    b = np.zeros(n); d = np.zeros(n)
    for i in range(n):
        b[i] = (x[i + 1] - x[i]) / h[i] - h[i] / 3 * (c[i + 1] + 2 * c[i])
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])
    return a, b, c[:n], d

def _segment(t, t_query):
    n = len(t) - 1
    i = np.searchsorted(t, t_query) - 1
    return max(0, min(i, n - 1))

def evaluate_spline_derivatives(t_data, x_data, t_query):
    a, b, c, d = _natural_cubic_coeffs(t_data, x_data)
    n = len(t_data) - 1
    print("============================================================")
    print("Spline-Koeffizienten (a, b, c, d je Segment)")
    print("============================================================")
    print(f"{'i':<3}{'a_i':>12}{'b_i':>12}{'c_i':>12}{'d_i':>12}")
    for i in range(n):
        print(f"{i:<3}{a[i]:>12.6f}{b[i]:>12.6f}{c[i]:>12.6f}{d[i]:>12.6f}")

    i = _segment(t_data, t_query)
    dt = t_query - t_data[i]
    s   = a[i] + b[i]*dt + c[i]*dt**2 + d[i]*dt**3
    ds  = b[i] + 2*c[i]*dt + 3*d[i]*dt**2
    dds = 2*c[i] + 6*d[i]*dt
    print(f"\nAuswertung bei t = {t_query} (Segment {i}):")
    print(f"  Position      s(t)   = {s:.6f}")
    print(f"  Geschwindigkeit s'(t) = {ds:.6f}")
    print(f"  Beschleunigung  s''(t)= {dds:.6f}")

    # Verlauf der 1. Ableitung plotten
    tt = np.linspace(t_data[0], t_data[-1], 400)
    dd = np.empty_like(tt)
    for k, tv in enumerate(tt):
        j = _segment(t_data, tv)
        dtj = tv - t_data[j]
        dd[k] = b[j] + 2*c[j]*dtj + 3*d[j]*dtj**2
    plt.figure(figsize=(9, 6))
    plt.plot(tt, dd, 'b-', label="s'(t) (Geschwindigkeit)")
    plt.plot(t_query, ds, 'ro', markersize=9, label=f"s'({t_query}) = {ds:.3f}")
    plt.xlabel('t'); plt.ylabel("s'(t)"); plt.legend(); plt.grid(True)
    plt.title("Geschwindigkeitsverlauf des Splines")
    plt.tight_layout(); plt.show()
    return s, ds, dds

# ============================================================
# PART 4 — Call
# ============================================================
evaluate_spline_derivatives(t_data, x_data, t_query)
