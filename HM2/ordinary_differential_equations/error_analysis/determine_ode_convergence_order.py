# ============================================================
# TOPIC: DGL — Konvergenzordnung eines Einschrittverfahrens bestimmen
# DESCRIPTION:
# Löst ein AWP mit einem Einschrittverfahren (Euler / Mittelpunkt / mod.
# Euler / RK4 oder eigenem Butcher-Schema) für mehrere Schrittweiten h,
# berechnet jeweils den globalen Fehler |y_num(b) - y_exakt(b)| am Endpunkt
# und bestimmt aus einem doppeltlogarithmischen Plot die Konvergenzordnung p
# (Steigung der Geraden, gerundet auf eine ganze Zahl).
# USE WHEN:
# Wenn eine Aufgabe verlangt "berechnen Sie den Fehler für h in {...} und
# schliessen Sie auf die Konvergenzordnung" (loglog-Plot).
# EXAMPLE:
# y' = 2(1-x)y, y(0)=1, exakt e^(2x-x^2), eigenes 4-stufiges RK-Schema,
# h in {0.1, 0.01, 0.001} -> Ordnung p = 3.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PART 1 — Inputs
# ============================================================
def f(x, y):
    return 2 * (1 - x) * y          # rechte Seite y' = f(x, y)

a, b_end = 0.0, 3.0                 # Integrationsintervall [a, b]
y0       = 1.0                      # Anfangswert y(a)

def y_exact(x):
    return np.exp(2 * x - x**2)      # exakte Lösung (für den Fehler)

h_list = [0.1, 0.01, 0.001]         # zu testende Schrittweiten

# Eigenes Butcher-Schema (nur für method = "custom"); Summe(b) muss 1 sein
c_custom = np.array([0.0, 1/3, 1/3, 2/3])
A_custom = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [1/3, 0.0, 0.0, 0.0],
    [0.0, 1/3, 0.0, 0.0],
    [0.0, 1/3, 1/3, 0.0],
])
b_custom = np.array([1/4, 0.0, 0.0, 3/4])

# ============================================================
# PART 2 — Method selection
# ============================================================
# method:
#   "euler" | "midpoint" | "mod_euler" | "rk4"  -> klassische Verfahren
#   "custom"                                     -> Butcher-Schema oben
method = "custom"

# ============================================================
# PART 3 — Implementation
# ============================================================
def _butcher(method):
    if method == "euler":
        return np.array([0.0]), np.array([[0.0]]), np.array([1.0])
    if method == "midpoint":
        return np.array([0.0, 0.5]), np.array([[0.0, 0.0], [0.5, 0.0]]), np.array([0.0, 1.0])
    if method == "mod_euler":
        return np.array([0.0, 1.0]), np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([0.5, 0.5])
    if method == "rk4":
        c = np.array([0.0, 0.5, 0.5, 1.0])
        A = np.array([[0.0, 0.0, 0.0, 0.0],
                      [0.5, 0.0, 0.0, 0.0],
                      [0.0, 0.5, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0]])
        b = np.array([1/6, 1/3, 1/3, 1/6])
        return c, A, b
    if method == "custom":
        return c_custom, A_custom, b_custom
    raise ValueError(f"Unbekannte Methode: {method!r}")

def _integrate_endpoint(f, a, b_end, y0, n, c, A, bvec):
    s = len(bvec)
    h = (b_end - a) / n
    x = a
    y = float(y0)
    for _ in range(n):
        k = np.zeros(s)
        for j in range(s):
            y_arg = y + h * sum(A[j, m] * k[m] for m in range(j))
            k[j] = f(x + c[j] * h, y_arg)
        y = y + h * np.sum(bvec * k)
        x = x + h
    return y

def determine_ode_convergence_order(method, f, a, b_end, y0, y_exact, h_list):
    c, A, bvec = _butcher(method)
    assert abs(np.sum(bvec) - 1.0) < 1e-12, "Summe der b-Koeffizienten muss 1 sein"
    exact_end = y_exact(b_end)

    print("============================================================")
    print(f"Konvergenzordnung — Methode '{method}'")
    print("============================================================")
    print(f"y_exakt({b_end}) = {exact_end:.12g}\n")
    print(f"{'h':>10} | {'n':>7} | {'y_num(b)':>16} | {'globaler Fehler':>16} | {'p (lokal)':>9}")
    print("-" * 74)

    hs, errs = [], []
    prev = None
    for h in h_list:
        n = int(round((b_end - a) / h))
        y_end = _integrate_endpoint(f, a, b_end, y0, n, c, A, bvec)
        err = abs(y_end - exact_end)
        p_local = ""
        if prev is not None and err > 0 and prev[1] > 0:
            p_local = f"{np.log(prev[1]/err) / np.log(prev[0]/h):.3f}"
        print(f"{h:>10g} | {n:>7d} | {y_end:>16.10f} | {err:>16.3e} | {p_local:>9}")
        hs.append(h); errs.append(err); prev = (h, err)

    hs = np.array(hs); errs = np.array(errs)
    valid = errs > 0
    slope = np.polyfit(np.log(hs[valid]), np.log(errs[valid]), 1)[0]
    print(f"\nSteigung im loglog-Fit = {slope:.4f}  ->  Konvergenzordnung p ~= {round(slope)}")

    plt.figure(figsize=(8, 6))
    plt.loglog(hs, errs, 'bo-', label=f"globaler Fehler ({method})")
    plt.loglog(hs, errs[0] * (hs / hs[0])**round(slope), 'r--',
               label=f"Referenz Ordnung {round(slope)}")
    plt.gca().invert_xaxis()
    plt.xlabel("Schrittweite h"); plt.ylabel("globaler Fehler bei x = b")
    plt.title(f"Konvergenzordnung (Steigung ~= {slope:.2f})")
    plt.grid(True, which="both"); plt.legend()
    plt.tight_layout(); plt.show()
    return hs, errs, slope

# ============================================================
# PART 4 — Call
# ============================================================
determine_ode_convergence_order(method, f, a, b_end, y0, y_exact, h_list)
