# ============================================================
# TOPIC: Visualisierung — Konditionszahl K_f(x) plotten + Schwellen-Bereich
# DESCRIPTION:
# Zeichnet K_f(x) = |x f'(x)/f(x)| (wahlweise halblogarithmisch) über ein
# Intervall, markiert die Schwelle K = threshold und gibt die x-Bereiche aus,
# in denen f gut konditioniert ist (K <= threshold). Zusätzlich wird das
# Grenzverhalten von K an wählbaren Stellen (numerisch + sympy.limit) gezeigt.
# USE WHEN:
# Wenn die gut/schlecht konditionierten Bereiche einer Funktion gesucht sind
# (z.B. "für welche x ist K <= 1?") oder das Verhalten für x -> x0 / x -> R.
# EXAMPLE:
# f(x) = x·exp(x): K = |1+x|, gut konditioniert (K<=1) auf [-2, 0].
# f(x) = x^2·sin(x) halblogarithmisch auf [-2pi, 3pi], Grenzwert bei x->0.
# ============================================================

import numpy as np
from matplotlib import pyplot as plt
import sympy as sp

# ============================================================
# PART 1 — Inputs
# ============================================================
funktion     = "x*exp(x)"   # f(x) als String (sympy-Syntax)
x_min        = -5.0         # linke Intervallgrenze
x_max        = 5.0          # rechte Intervallgrenze
n_pts        = 2000        # Anzahl Stützstellen
threshold    = 1.0         # Schwelle für "gut konditioniert" (K <= threshold)
limit_points = [0.0]       # Stellen, an denen das Grenzverhalten untersucht wird

# ============================================================
# PART 2 — Method selection
# ============================================================
# y_scale:
#   "semilogy" -> log-Achse für K (entspricht "halblogarithmisch" der Aufgaben)
#   "linear"   -> lineare K-Achse
#   "loglog"   -> beide Achsen logarithmisch
y_scale = "semilogy"

# ============================================================
# PART 3 — Implementation
# ============================================================
def _build_K(funktion):
    x = sp.Symbol("x")
    f = sp.sympify(funktion, locals={"x": x})
    K = sp.Abs(x * sp.diff(f, x) / f)
    K_simpl = sp.simplify(K)
    f_num = sp.lambdify(x, K, modules=["numpy"])
    return x, f, K_simpl, f_num

def _threshold_regions(xs, Ks, threshold):
    # zusammenhängende x-Bereiche, in denen K <= threshold (und endlich)
    mask = np.isfinite(Ks) & (Ks <= threshold)
    regions = []
    start = None
    for i, ok in enumerate(mask):
        if ok and start is None:
            start = xs[i]
        elif not ok and start is not None:
            regions.append((start, xs[i - 1]))
            start = None
    if start is not None:
        regions.append((start, xs[-1]))
    return regions

def _limit_analysis(x, K_simpl, points):
    for p in points:
        print(f"-- Grenzverhalten bei x -> {p}")
        try:
            lim = sp.limit(K_simpl, x, p)
            print(f"   sympy.limit(K, x, {p}) = {lim}")
        except Exception as exc:
            print(f"   sympy.limit nicht bestimmbar ({exc})")
        for d in (1e-1, 1e-2, 1e-4, 1e-6, 1e-8):
            try:
                val = float(K_simpl.subs(x, p + d).evalf())
                print(f"   K({p}+{d:g}) = {val:.6g}")
            except Exception:
                pass
        print()

def plot_condition_number_of_function(funktion, x_min, x_max, n_pts,
                                      threshold, limit_points, y_scale):
    x, f, K_simpl, f_num = _build_K(funktion)
    print("============================================================")
    print("Konditionszahl-Analyse")
    print("============================================================")
    print(f"f(x)   = {f}")
    print(f"K_f(x) = |x f'(x)/f(x)| = {K_simpl}\n")

    xs = np.linspace(x_min, x_max, n_pts)
    with np.errstate(all="ignore"):
        Ks = np.abs(np.asarray(f_num(xs), dtype=float))
    Ks = np.where(np.isfinite(Ks), Ks, np.nan)

    regions = _threshold_regions(xs, Ks, threshold)
    print(f"Gut konditioniert (K <= {threshold}) auf:")
    if regions:
        for a, b in regions:
            print(f"   x in [{a:.4g}, {b:.4g}]")
    else:
        print("   (kein Bereich im betrachteten Intervall)")
    print()

    if limit_points:
        _limit_analysis(x, K_simpl, limit_points)

    # ---- Plot ----
    plt.figure()
    ys = Ks.copy()
    if y_scale in ("semilogy", "loglog"):
        ys = np.where(ys > 0, ys, np.nan)
    plt.plot(xs, ys, label="K_f(x)")
    plt.axhline(threshold, color="red", linestyle="--", label=f"K = {threshold}")
    for a, b in regions:
        plt.axvspan(a, b, color="green", alpha=0.15)
    if y_scale == "semilogy":
        plt.yscale("log")
    elif y_scale == "loglog":
        plt.yscale("log"); plt.xscale("log")
    plt.xlabel("x")
    plt.ylabel("K_f(x)")
    plt.title(f"Konditionszahl von f(x) = {funktion}")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return regions

# ============================================================
# PART 4 — Call
# ============================================================
plot_condition_number_of_function(funktion, x_min, x_max, n_pts,
                                  threshold, limit_points, y_scale)
