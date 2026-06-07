# ============================================================
# TOPIC: Visualization — fixed-point iteration (F(x)-x plot + stability)
# DESCRIPTION:
# Plots F(x) - x, finds the roots (= fixed points) via bisection
# in sign-change intervals and classifies each fixed point
# via |F'(x*)| as attracting / repelling / neutral.
# USE WHEN:
# When a graphical overview of all fixed points of a function F(x)
# including stability is needed.
# EXAMPLE:
# F(x) = (exp(x) + exp(-x))/2 - 1.5 on [-1, 2.5].
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, convert_xor

# ============================================================
# PART 1 — Inputs
# ============================================================
expr_str = "(exp(x) + exp(-x))/2 - 1.5"   # F(x); fixed points = roots of F(x) - x

x_spec = {"kind": "linspace", "start": -1.0, "stop": 2.5, "n": 2000}

plot_cfg = {
    "title": "Roots of F(x) - x",
    "xlabel": "x", "ylabel": "F(x) - x", "label": "F(x) - x",
    "xlim": (-1.0, 2.5), "grid_both": True, "legend": True,
}

root_cfg = {
    "tol_root": 1e-10, "max_iter": 200, "dedup_eps": 1e-6, "mark_roots": True,
}

stability_cfg = {
    "neutral_eps": 1e-3,   # | |F'(x*)| - 1 | <= neutral_eps -> "neutral"
    "annotate": True,
}

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here. Fixed points are always found via bisection at
# sign changes; classification is done via F'(x*).

# ============================================================
# PART 3 — Implementation
# ============================================================
def _parse_expr(expr_str):
    x = sp.Symbol("x")
    allowed = {
        "x": x, "E": sp.E, "e": sp.E, "pi": sp.pi,
        "exp": sp.exp, "log": sp.log, "ln": sp.log,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "sqrt": sp.sqrt, "abs": sp.Abs,
    }
    transformations = standard_transformations + (convert_xor,)
    return x, parse_expr(expr_str, local_dict=allowed, transformations=transformations)

def _x_values(x_spec):
    kind = x_spec.get("kind", "linspace")
    n = int(x_spec.get("n", 1000))
    if kind == "linspace":
        return np.linspace(float(x_spec["start"]), float(x_spec["stop"]), n)
    if kind == "logspace":
        return np.logspace(float(x_spec["start"]), float(x_spec["stop"]), n)
    raise ValueError("x_spec.kind must be 'linspace' or 'logspace'")

def _apply_axes_settings(plot_cfg):
    if plot_cfg.get("xlim") is not None:
        plt.xlim(*plot_cfg["xlim"])
    if plot_cfg.get("ylim") is not None:
        plt.ylim(*plot_cfg["ylim"])
    plt.xlabel(plot_cfg.get("xlabel", "x"))
    plt.ylabel(plot_cfg.get("ylabel", "y"))
    plt.title(plot_cfg.get("title", "Plot"))
    plt.grid(True, which="both" if plot_cfg.get("grid_both", True) else "major")
    if plot_cfg.get("legend", True):
        plt.legend()
    plt.tight_layout()

def _bisection_root(fun, a, b, tol=1e-10, max_iter=200):
    fa, fb = fun(a), fun(b)
    if np.isnan(fa) or np.isnan(fb) or np.isinf(fa) or np.isinf(fb):
        return None
    if fa == 0: return float(a)
    if fb == 0: return float(b)
    if fa * fb > 0: return None
    left, right = float(a), float(b)
    for _ in range(max_iter):
        mid = (left + right) / 2.0
        fm = fun(mid)
        if np.isnan(fm) or np.isinf(fm): return None
        if abs(fm) <= tol or abs(right - left) <= tol:
            return float(mid)
        if fa * fm <= 0:
            right = mid; fb = fm
        else:
            left = mid; fa = fm
    return float((left + right) / 2.0)

def _find_roots(fun, xs, tol_root=1e-10, max_iter=200, dedup_eps=1e-6):
    ys = fun(xs)
    roots = []
    hit_idx = np.where(np.isfinite(ys) & (np.abs(ys) <= tol_root))[0]
    for i in hit_idx:
        roots.append(float(xs[i]))
    for i in range(len(xs) - 1):
        y1, y2 = ys[i], ys[i + 1]
        if not (np.isfinite(y1) and np.isfinite(y2)): continue
        if y1 == 0:
            roots.append(float(xs[i])); continue
        if y1 * y2 < 0:
            r = _bisection_root(fun, xs[i], xs[i + 1], tol=tol_root, max_iter=max_iter)
            if r is not None:
                roots.append(float(r))
    roots = sorted(roots)
    unique = []
    for r in roots:
        if not unique or abs(r - unique[-1]) > dedup_eps:
            unique.append(r)
    return unique

def _classify_fixed_point(fprime_val, neutral_eps=1e-3):
    if not np.isfinite(fprime_val):
        return "undetermined"
    a = abs(fprime_val)
    if abs(a - 1.0) <= neutral_eps:
        return "neutral"
    return "attracting" if a < 1.0 else "repelling"

def plot_fixed_point_iteration(expr_str, x_spec, plot_cfg, root_cfg, stability_cfg):
    x, expr = _parse_expr(expr_str)
    f = sp.lambdify(x, expr, modules=["numpy"])
    fprime = sp.lambdify(x, sp.diff(expr, x), modules=["numpy"])

    xs = _x_values(x_spec)
    ys = f(xs) - xs

    roots = _find_roots(
        fun=lambda t: f(t) - t, xs=xs,
        tol_root=root_cfg.get("tol_root", 1e-10),
        max_iter=root_cfg.get("max_iter", 200),
        dedup_eps=root_cfg.get("dedup_eps", 1e-6),
    )

    print("\nRoots of F(x) - x (fixed points) + stability:")
    if not roots:
        print("  (none found)")
    else:
        for i, r in enumerate(roots, start=1):
            try:
                fp = float(fprime(r))
            except Exception:
                fp = float("nan")
            status = _classify_fixed_point(fp, stability_cfg.get("neutral_eps", 1e-3))
            print(f"  x_{i} = {r:.12g} | F'(x_i) = {fp:.12g} | |F'(x_i)| = {abs(fp):.12g} -> {status}")

    plt.figure()
    plt.plot(xs, ys, label=plot_cfg.get("label", "F(x) - x"))
    plt.axhline(0, linewidth=1)

    if root_cfg.get("mark_roots", True) and roots:
        plt.scatter(roots, [0.0] * len(roots), label="Roots")
        if stability_cfg.get("annotate", True):
            for r in roots:
                try:
                    fp = float(fprime(r))
                except Exception:
                    fp = float("nan")
                status = _classify_fixed_point(fp, stability_cfg.get("neutral_eps", 1e-3))
                plt.annotate(status, (r, 0.0), textcoords="offset points", xytext=(6, 6))

    _apply_axes_settings(plot_cfg)
    plt.show()
    return roots

# ============================================================
# PART 4 — Call
# ============================================================
plot_fixed_point_iteration(expr_str, x_spec, plot_cfg, root_cfg, stability_cfg)
