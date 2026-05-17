# ============================================================
# TOPIC: Visualisierung — generischer Funktionsplotter (linear/log)
# DESCRIPTION:
# Zeichnet beliebig viele Funktionen als Strings auf einem
# linspace/logspace-Gitter; unterstützt linlin, semilogy, loglog,
# symlog inkl. Umgang mit Polstellen und negativen y-Werten in Logplots.
# USE WHEN:
# Wenn mehrere Funktionen schnell auf einem gemeinsamen Achsensystem
# verglichen werden sollen (z.B. für Konditionsanalyse).
# EXAMPLE:
# f(x)=x, g(x)=exp(x), h(x)=x·exp(x) auf [0, 3] linear.
# ============================================================

import numpy as np
from matplotlib import pyplot as plt
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, convert_xor

# ============================================================
# PART 1 — Inputs
# ============================================================
x_min = 0.0
x_max = 3.0
n_pts = 1000

functions = [
    {"func": "x",        "label": "f(x) = x"},
    {"func": "exp(x)",   "label": "g(x) = e^x"},
    {"func": "x*exp(x)", "label": "h(x) = x e^x"},
]

# ============================================================
# PART 2 — Method selection
# ============================================================
# x_space:  "linspace" | "logspace" | "logspace_exp"
# y_scale:  "linear"   | "log" (= semilogy) | "loglog" | "symlog"
x_space = "linspace"
y_scale = "linear"

# ============================================================
# PART 3 — Implementation
# ============================================================
def _parse_function(expr_str):
    x = sp.Symbol("x")
    allowed = {
        "x": x, "E": sp.E, "e": sp.E, "pi": sp.pi,
        "exp": sp.exp, "log": sp.log, "ln": sp.log,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "sqrt": sp.sqrt, "abs": sp.Abs,
    }
    transformations = standard_transformations + (convert_xor,)
    expr = parse_expr(expr_str, local_dict=allowed, transformations=transformations)
    return sp.lambdify(x, expr, modules=["numpy"])

def _normalize_scales(cfg):
    xscale = (cfg.get("xscale") or "linear").lower()
    yscale = (cfg.get("yscale") or "linear").lower()
    if yscale == "semilogy":
        yscale = "log"
    if yscale == "loglog":
        xscale = "log"
        yscale = "log"
    return xscale, yscale

def _sanitize_log_bounds(x_min, x_max, x_eps, allow_shift):
    if x_max <= 0:
        raise ValueError("For log x, x_max must be > 0.")
    if x_min <= 0:
        if not allow_shift:
            raise ValueError("For log x, x_min must be > 0 (or set allow_shift_log_min=True).")
        x_min = x_eps
    if x_min >= x_max:
        raise ValueError("x_min must be < x_max.")
    return x_min, x_max

def _build_x_values(x_min, x_max, n_pts, x_space, x_eps, allow_shift):
    x_space = (x_space or "linspace").lower()
    if x_space == "linspace":
        return np.linspace(x_min, x_max, n_pts)
    if x_space == "logspace":
        x_min, x_max = _sanitize_log_bounds(x_min, x_max, x_eps, allow_shift)
        return np.logspace(np.log10(x_min), np.log10(x_max), n_pts)
    if x_space == "logspace_exp":
        return np.logspace(x_min, x_max, n_pts)
    raise ValueError('x_space must be "linspace", "logspace", or "logspace_exp"')

def plot_function_generic(functions, x_min, x_max, n_pts, x_space, y_scale):
    cfg = {
        "title": "Multiple functions (one plot)",
        "xlabel": "x", "ylabel": "value", "legend": True,
        "x_space": x_space,
        "xscale": "log" if x_space in ("logspace", "logspace_exp") else "linear",
        "yscale": y_scale, "linthresh": 1.0,
        "x_eps": 1e-12, "allow_shift_log_min": True,
    }
    xscale, yscale = _normalize_scales(cfg)
    x_eps       = float(cfg.get("x_eps", 1e-12))
    allow_shift = bool(cfg.get("allow_shift_log_min", True))

    if xscale == "log" and x_space == "linspace":
        x_min, x_max = _sanitize_log_bounds(x_min, x_max, x_eps, allow_shift)

    xs = _build_x_values(x_min, x_max, n_pts, x_space, x_eps, allow_shift)
    if xscale == "log" and np.any(xs <= 0):
        raise ValueError("xscale='log' requires all x > 0. Adjust x_min or x_eps.")

    plt.figure()
    for item in functions:
        f = _parse_function(item["func"])
        ys = f(xs)
        ys = np.where(np.isfinite(ys), ys, np.nan)
        if yscale == "log":
            ys = np.where(ys > 0, ys, np.nan)
        plt.plot(xs, ys, label=item.get("label", item["func"]))

    plt.xscale(xscale)
    if yscale == "symlog":
        plt.yscale("symlog", linthresh=cfg.get("linthresh", 1.0))
    else:
        plt.yscale(yscale)
    plt.xlim(xs.min(), xs.max())
    plt.xlabel(cfg["xlabel"])
    plt.ylabel(cfg["ylabel"])
    plt.title(cfg["title"])
    any_log = (plt.gca().get_xscale() != "linear") or (plt.gca().get_yscale() != "linear")
    plt.grid(True, which="both" if any_log else "major")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ============================================================
# PART 4 — Call
# ============================================================
plot_function_generic(functions, x_min, x_max, n_pts, x_space, y_scale)
