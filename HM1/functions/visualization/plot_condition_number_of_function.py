# ============================================================
# TOPIC: Visualization — plot condition number K_f(x) + threshold region
# DESCRIPTION:
# Plots K_f(x) = |x f'(x)/f(x)| (optionally semilogarithmic) over an
# interval, marks the threshold K = threshold and prints the x-regions
# where f is well-conditioned (K <= threshold). Additionally shows the
# limit behavior of K at selectable points (numerically + sympy.limit).
# USE WHEN:
# When the well/poorly conditioned regions of a function are sought
# (e.g. "for which x is K <= 1?") or the behavior for x -> x0 / x -> R.
# EXAMPLE:
# f(x) = x·exp(x): K = |1+x|, well-conditioned (K<=1) on [-2, 0].
# f(x) = x^2·sin(x) semilogarithmic on [-2pi, 3pi], limit at x->0.
# ============================================================

import numpy as np
from matplotlib import pyplot as plt
import sympy as sp

# ============================================================
# PART 1 — Inputs
# ============================================================
funktion     = "x*exp(x)"   # f(x) as string (sympy syntax)
x_min        = -5.0         # left interval boundary
x_max        = 5.0          # right interval boundary
n_pts        = 2000        # number of support points
threshold    = 1.0         # threshold for "well-conditioned" (K <= threshold)
limit_points = [0.0]       # points at which limit behavior is investigated

# ============================================================
# PART 2 — Method selection
# ============================================================
# y_scale:
#   "semilogy" -> log axis for K (corresponds to "semilogarithmic" in exercises)
#   "linear"   -> linear K axis
#   "loglog"   -> both axes logarithmic
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
    # contiguous x-regions where K <= threshold (and finite)
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
        print(f"-- Limit behavior at x -> {p}")
        try:
            lim = sp.limit(K_simpl, x, p)
            print(f"   sympy.limit(K, x, {p}) = {lim}")
        except Exception as exc:
            print(f"   sympy.limit not determinable ({exc})")
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
    print("Condition number analysis")
    print("============================================================")
    print(f"f(x)   = {f}")
    print(f"K_f(x) = |x f'(x)/f(x)| = {K_simpl}\n")

    xs = np.linspace(x_min, x_max, n_pts)
    with np.errstate(all="ignore"):
        Ks = np.abs(np.asarray(f_num(xs), dtype=float))
    Ks = np.where(np.isfinite(Ks), Ks, np.nan)

    regions = _threshold_regions(xs, Ks, threshold)
    print(f"Well-conditioned (K <= {threshold}) on:")
    if regions:
        for a, b in regions:
            print(f"   x in [{a:.4g}, {b:.4g}]")
    else:
        print("   (no region in the considered interval)")
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
    plt.title(f"Condition number of f(x) = {funktion}")
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
