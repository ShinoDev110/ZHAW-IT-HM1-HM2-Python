# ============================================================
# TOPIC: Visualisierung — Sekantenverfahren auf f(x) + Nullstellen-Karte
# DESCRIPTION:
# Zeichnet f(x), findet Nullstellen per Bisektion bei
# Vorzeichenwechseln und führt zusätzlich das Sekantenverfahren ab
# (x0, x1) aus. Klassifiziert jede Nullstelle als einfach oder
# mehrfach/abgeflacht über |f'(x*)|.
# USE WHEN:
# Wenn der Konvergenzpfad des Sekantenverfahrens und alle Nullstellen
# einer Funktion zusammen grafisch dargestellt werden sollen.
# EXAMPLE:
# f(x) = (exp(x)+exp(-x))/2 - 1.5 - x, Startwerte x0=1.4, x1=2.6.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, convert_xor

# ============================================================
# PART 1 — Inputs
# ============================================================
expr_str = "(exp(x)+exp(-x))/2 - 1.5 - x"

x_spec = {"kind": "linspace", "start": -1.0, "stop": 2.5, "n": 2000}

plot_cfg = {
    "title": "Sekantenverfahren + Nullstellen von f(x)",
    "xlabel": "x", "ylabel": "f(x)", "label": "f(x)",
    "xlim": (-1.0, 2.5), "grid_both": True, "legend": True,
}

root_cfg = {"tol_root": 1e-6, "max_iter": 10000, "dedup_eps": 1e-6, "mark_roots": True}

secant_cfg = {
    "x0": 1.4, "x1": 2.6, "tol": 1e-6, "max_iter": 100000,
    "print_iterations": True, "mark_secant_result": True,
    "plot_iteration_points": False,
}

classify_cfg = {"eps": 1e-6, "annotate": True}

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here. Sekantenverfahren startet immer mit (x0, x1)
# aus secant_cfg.

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

def _wrap_numeric(sympy_func):
    def f(t):
        y = sympy_func(t)
        if np.isscalar(y) or (isinstance(y, np.ndarray) and y.shape == ()):
            return float(y)
        return np.asarray(y, dtype=float)
    return f

def _x_values(x_spec):
    kind = x_spec.get("kind", "linspace")
    n = int(x_spec.get("n", 1000))
    if kind == "linspace":
        return np.linspace(float(x_spec["start"]), float(x_spec["stop"]), n)
    if kind == "logspace":
        return np.logspace(float(x_spec["start"]), float(x_spec["stop"]), n)
    raise ValueError("x_spec.kind must be 'linspace' or 'logspace'")

def _apply_axes_settings(plot_cfg):
    if plot_cfg.get("xlim") is not None: plt.xlim(*plot_cfg["xlim"])
    if plot_cfg.get("ylim") is not None: plt.ylim(*plot_cfg["ylim"])
    plt.xlabel(plot_cfg.get("xlabel", "x"))
    plt.ylabel(plot_cfg.get("ylabel", "y"))
    plt.title(plot_cfg.get("title", "Plot"))
    plt.grid(True, which="both" if plot_cfg.get("grid_both", True) else "major")
    if plot_cfg.get("legend", True): plt.legend()
    plt.tight_layout()

def _bisection_root(fun, a, b, tol=1e-10, max_iter=200):
    fa, fb = float(fun(a)), float(fun(b))
    if np.isnan(fa) or np.isnan(fb) or np.isinf(fa) or np.isinf(fb): return None
    if fa == 0: return float(a)
    if fb == 0: return float(b)
    if fa * fb > 0: return None
    left, right = float(a), float(b)
    for _ in range(max_iter):
        mid = (left + right) / 2.0
        fm = float(fun(mid))
        if np.isnan(fm) or np.isinf(fm): return None
        if abs(fm) <= tol or abs(right - left) <= tol:
            return float(mid)
        if fa * fm <= 0:
            right = mid; fb = fm
        else:
            left = mid; fa = fm
    return float((left + right) / 2.0)

def _find_nullstellen(fun, xs, tol_root=1e-10, max_iter=200, dedup_eps=1e-6):
    ys = np.asarray(fun(xs), dtype=float)
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

def _sekantenverfahren(fun, x0, x1, tol=1e-10, max_iter=50):
    x_prev, x_curr = float(x0), float(x1)
    f_prev, f_curr = float(fun(x_prev)), float(fun(x_curr))
    history = [(0, x_prev, f_prev), (1, x_curr, f_curr)]
    for k in range(2, max_iter + 1):
        denom = f_curr - f_prev
        if denom == 0 or not np.isfinite(denom): break
        x_next = x_curr - f_curr * (x_curr - x_prev) / denom
        f_next = float(fun(x_next))
        history.append((k, float(x_next), float(f_next)))
        if not np.isfinite(f_next) or not np.isfinite(x_next): break
        if abs(f_next) <= tol or abs(x_next - x_curr) <= tol:
            return float(x_next), history
        x_prev, f_prev = x_curr, f_curr
        x_curr, f_curr = x_next, f_next
    return float(x_curr), history

def _classify_root(fprime_val, eps=1e-6):
    if not np.isfinite(fprime_val): return "unbestimmt"
    return "mehrfach/abgeflacht" if abs(fprime_val) <= eps else "einfach"

def plot_secant_iteration(expr_str, x_spec, plot_cfg, root_cfg, secant_cfg, classify_cfg):
    x, expr = _parse_expr(expr_str)
    f      = _wrap_numeric(sp.lambdify(x, expr, modules=["numpy"]))
    fprime = _wrap_numeric(sp.lambdify(x, sp.diff(expr, x), modules=["numpy"]))

    xs = _x_values(x_spec)
    ys = f(xs)
    ys = np.where(np.isfinite(ys), ys, np.nan)

    roots = _find_nullstellen(f, xs,
                              tol_root=root_cfg.get("tol_root", 1e-10),
                              max_iter=root_cfg.get("max_iter", 200),
                              dedup_eps=root_cfg.get("dedup_eps", 1e-6))
    print("\nNullstellen von f(x) im Intervall:")
    if not roots:
        print("  (keine gefunden)")
    else:
        for i, r in enumerate(roots, start=1):
            fp = float(fprime(r))
            kind = _classify_root(fp, classify_cfg.get("eps", 1e-6))
            print(f"  x_{i} = {r:.12g} | f'(x_i) = {fp:.12g} -> {kind}")

    root_sec, hist = _sekantenverfahren(
        f, secant_cfg["x0"], secant_cfg["x1"],
        tol=secant_cfg.get("tol", 1e-10),
        max_iter=secant_cfg.get("max_iter", 50),
    )

    print("\nSekantenverfahren:")
    if secant_cfg.get("print_iterations", True):
        for k, xv, fv in hist:
            print(f"  k={k:>2} | x={xv:.12g} | f(x)={fv:.12g}")
    print(f"\n  Ergebnis: x* ~= {root_sec:.12g} | f(x*) ~= {float(f(root_sec)):.12g}")

    plt.figure()
    plt.plot(xs, ys, label=plot_cfg.get("label", "f(x)"))
    plt.axhline(0, linewidth=1)
    if root_cfg.get("mark_roots", True) and roots:
        plt.scatter(roots, [0.0] * len(roots), label="Nullstellen (Bisection)")
        if classify_cfg.get("annotate", True):
            for r in roots:
                fp = float(fprime(r))
                kind = _classify_root(fp, classify_cfg.get("eps", 1e-6))
                plt.annotate(kind, (r, 0.0), textcoords="offset points", xytext=(6, 6))
    if secant_cfg.get("mark_secant_result", True):
        plt.scatter([root_sec], [0.0], label="Sekantenverfahren Ergebnis")
    if secant_cfg.get("plot_iteration_points", True):
        xs_it = [h[1] for h in hist]
        ys_it = [h[2] for h in hist]
        plt.scatter(xs_it, ys_it, label="Iterationen (x_k, f(x_k))", s=20)
    _apply_axes_settings(plot_cfg)
    plt.show()
    return roots, root_sec, hist

# ============================================================
# PART 4 — Call
# ============================================================
plot_secant_iteration(expr_str, x_spec, plot_cfg, root_cfg, secant_cfg, classify_cfg)
