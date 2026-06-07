# ============================================================
# TOPIC: Visualization — Newton's method on f(x) + root map
# DESCRIPTION:
# Plots f(x), marks all roots (via bisection at sign changes)
# and additionally runs Newton's method starting from x0
# in either standard or simplified mode. Plots the
# iteration result and optionally the iteration points.
# USE WHEN:
# When the convergence path of Newton's method should be made
# visible graphically, together with the global root structure.
# EXAMPLE:
# f(x) = (exp(x)+exp(-x))/2 - 1.5 - x, x0 = 2.0.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, convert_xor

# ============================================================
# PART 1 — Inputs
# ============================================================
expr_str = "(exp(x)+exp(-x))/2 - 1.5 -x "

x_spec = {"kind": "linspace", "start": -3.0, "stop": 3.0, "n": 2000}

plot_cfg = {
    "title": "Newton's method + roots of f(x)",
    "xlabel": "x", "ylabel": "f(x)", "label": "f(x)",
    "xlim": (-3.0, 3.0), "grid_both": True, "legend": True,
}

root_cfg = {"tol_root": 1e-10, "dedup_eps": 1e-6, "mark_roots": True}

newton_cfg = {
    "x0": 2.0, "tol": 1e-10, "max_iter": 50,
    "print_iterations": True, "mark_newton_result": True,
    "plot_iteration_points": False,
}

# ============================================================
# PART 2 — Method selection
# ============================================================
# mode:
#   0 -> standard Newton's method (derivative at each step)
#   1 -> simplified Newton's method (derivative fixed at x0)
mode = 0

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

def _find_roots(fun, xs, tol_root=1e-10, dedup_eps=1e-6):
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
            a, b = float(xs[i]), float(xs[i + 1])
            fa, fb = float(y1), float(y2)
            for _ in range(200):
                m = (a + b) / 2.0
                fm = float(fun(m))
                if not np.isfinite(fm): break
                if abs(fm) <= tol_root or abs(b - a) <= tol_root:
                    roots.append(float(m)); break
                if fa * fm <= 0:
                    b, fb = m, fm
                else:
                    a, fa = m, fm
    roots = sorted(roots)
    unique = []
    for r in roots:
        if not unique or abs(r - unique[-1]) > dedup_eps:
            unique.append(r)
    return unique

def _newton_normal(fun, dfun, x0, tol=1e-10, max_iter=50):
    xk = float(x0)
    hist = [(0, xk, float(fun(xk)))]
    for k in range(1, max_iter + 1):
        fk = float(fun(xk))
        d  = float(dfun(xk))
        if not np.isfinite(d) or d == 0: break
        x_next = xk - fk / d
        f_next = float(fun(x_next))
        hist.append((k, float(x_next), float(f_next)))
        if abs(f_next) <= tol or abs(x_next - xk) <= tol:
            return float(x_next), hist
        xk = x_next
    return float(xk), hist

def _newton_simplified(fun, dfun_at_x0, x0, tol=1e-10, max_iter=50):
    d0 = float(dfun_at_x0)
    if not np.isfinite(d0) or d0 == 0:
        return float(x0), [(0, float(x0), float(fun(x0)))]
    xk = float(x0)
    hist = [(0, xk, float(fun(xk)))]
    for k in range(1, max_iter + 1):
        fk = float(fun(xk))
        x_next = xk - fk / d0
        f_next = float(fun(x_next))
        hist.append((k, float(x_next), float(f_next)))
        if abs(f_next) <= tol or abs(x_next - xk) <= tol:
            return float(x_next), hist
        xk = x_next
    return float(xk), hist

def plot_newton_iteration(expr_str, x_spec, plot_cfg, root_cfg, newton_cfg, mode):
    x, expr = _parse_expr(expr_str)
    f  = _wrap_numeric(sp.lambdify(x, expr, modules=["numpy"]))
    df = _wrap_numeric(sp.lambdify(x, sp.diff(expr, x), modules=["numpy"]))

    xs = _x_values(x_spec)
    ys = f(xs)
    ys = np.where(np.isfinite(ys), ys, np.nan)

    roots = _find_roots(f, xs,
                              tol_root=root_cfg.get("tol_root", 1e-10),
                              dedup_eps=root_cfg.get("dedup_eps", 1e-6))
    print("\nRoots of f(x) in the interval:")
    if not roots:
        print("  (none found)")
    else:
        for i, r in enumerate(roots, start=1):
            print(f"  x_{i} = {r:.12g}")

    x0       = float(newton_cfg.get("x0", 0.0))
    tol      = float(newton_cfg.get("tol", 1e-10))
    max_iter = int(newton_cfg.get("max_iter", 50))

    if mode == 0:
        root_n, hist = _newton_normal(f, df, x0, tol=tol, max_iter=max_iter)
        mode_name = "Newton (standard)"
    elif mode == 1:
        root_n, hist = _newton_simplified(f, float(df(x0)), x0, tol=tol, max_iter=max_iter)
        mode_name = "Newton (simplified)"
    else:
        raise ValueError("mode must be 0 (standard) or 1 (simplified)")

    print(f"\n{mode_name}:")
    if newton_cfg.get("print_iterations", True):
        for k, xv, fv in hist:
            print(f"  k={k:>2} | x={xv:.12g} | f(x)={fv:.12g}")
    print(f"\n  Result: x* ~= {root_n:.12g} | f(x*) ~= {float(f(root_n)):.12g}")

    plt.figure()
    plt.plot(xs, ys, label=plot_cfg.get("label", "f(x)"))
    plt.axhline(0, linewidth=1)
    if root_cfg.get("mark_roots", True) and roots:
        plt.scatter(roots, [0.0] * len(roots), label="Roots")
    if newton_cfg.get("mark_newton_result", True):
        plt.scatter([root_n], [0.0], label=mode_name)
    if newton_cfg.get("plot_iteration_points", False):
        xs_it = [h[1] for h in hist]
        ys_it = [h[2] for h in hist]
        plt.scatter(xs_it, ys_it, label="Iterations (x_k, f(x_k))", s=20)
    _apply_axes_settings(plot_cfg)
    plt.show()
    return roots, root_n, hist

# ============================================================
# PART 4 — Call
# ============================================================
plot_newton_iteration(expr_str, x_spec, plot_cfg, root_cfg, newton_cfg, mode)
