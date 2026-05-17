# ============================================================
# TOPIC: Partial differentiation — Wave equation (3D wireframe plot)
# DESCRIPTION:
# Plottet eine oder mehrere gegebene Funktionen f(x,t) als 3D-Wireframe.
# Verwenden, wenn eine Lösung einer partiellen Differentialgleichung wie
# der Wellengleichung räumlich-zeitlich dargestellt werden soll.
# USE WHEN:
# Wenn vorgegebene Funktionen in x und t visualisiert werden sollen.
# EXAMPLE:
# w(x,t)=sin(x+ct), v(x,t)=sin(x+ct)+cos(2x+2ct) bei c=1.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PART 1 — Inputs
# ============================================================
c = 1                                  # wave speed
x_range = (-5, 5)                      # x interval
t_range = (-5, 5)                      # t interval
n_points = 50                          # grid resolution per axis

# Define the candidate functions
def w(x, t):                            # erste vorgegebene Funktion
    return np.sin(x + c*t)

def v(x, t):                            # zweite vorgegebene Funktion
    return np.sin(x + c*t) + np.cos(2*x + 2*c*t)

# ============================================================
# PART 2 — Method selection
# ============================================================
# Options:
#   "w"    -> plot only w(x,t)
#   "v"    -> plot only v(x,t)
#   "both" -> plot w and v side by side
method = "both"

# ============================================================
# PART 3 — Implementation
# ============================================================
def plot_wave_wireframe(c, x_range, t_range, n_points, method):
    x = np.linspace(x_range[0], x_range[1], n_points)
    t = np.linspace(t_range[0], t_range[1], n_points)
    X, T = np.meshgrid(x, t)

    if method == "w":
        funcs = [(w, "w(x,t) = sin(x + ct)")]
    elif method == "v":
        funcs = [(v, "v(x,t) = sin(x + ct) + cos(2x + 2ct)")]
    elif method == "both":
        funcs = [(w, "w(x,t) = sin(x + ct)"),
                 (v, "v(x,t) = sin(x + ct) + cos(2x + 2ct)")]
    else:
        raise ValueError("method must be 'w', 'v', or 'both'")

    fig = plt.figure(figsize=(6*len(funcs), 5))
    for i, (f, title) in enumerate(funcs, start=1):
        Z = f(X, T)
        ax = fig.add_subplot(1, len(funcs), i, projection='3d')
        ax.plot_wireframe(X, T, Z, rstride=2, cstride=2)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_zlabel("value")
        ax.set_title(title)

    plt.tight_layout()
    plt.show()

# ============================================================
# PART 4 — Call
# ============================================================
plot_wave_wireframe(c, x_range, t_range, n_points, method)
