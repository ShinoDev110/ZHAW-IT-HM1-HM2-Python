# ============================================================
# TOPIC: Functions with several variables — 3D visualization and contour lines
# DESCRIPTION:
# Plots physical functions three-dimensionally (wireframe and surface)
# as well as 2D contour diagrams:
#   a) Projectile range W(v0, alpha) of oblique projectile motion
#   b) Ideal gas: p(V,T), V(p,T), T(p,V)
# USE WHEN:
# When 3D visualizations with wireframe, surface, and contour lines
# for several variables are required.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ============================================================
# PART 1 — Inputs
# ============================================================
g = 9.81    # gravitational acceleration [m/s^2]
R = 8.31    # universal gas constant [J/(mol*K)]

# Definition ranges
v0_range    = (0, 100)          # initial velocity [m/s]
alpha_range = (0, np.pi / 2)   # launch angle [rad], maximum at alpha = pi/4 (45 degrees)

V_range = (1e-3, 0.2)          # volume [m^3], lower bound >0 to avoid division by zero
T_range = (0, 1e4)             # temperature [K]

p_range_V  = (1e4, 1e5)        # pressure [Pa] for V(p,T)
T_range_V  = (0, 1e4)          # temperature [K] for V(p,T)

p_range_T  = (1e4, 1e6)        # pressure [Pa] for T(p,V)
V_range_T  = (0, 10)           # volume [m^3] for T(p,V), lower bound slightly >0

n_points = 60                  # grid points per axis

# ============================================================
# PART 2 — Method selection
# ============================================================
# Which function group should be plotted?
# options: "projectile"   — only projectile range W(v0, alpha)
#          "ideal_gas"    — only ideal gas functions p, V, T
#          "all"          — both groups in sequence
mode = "all"

# ============================================================
# PART 3 — Implementation
# ============================================================
def plot_projectile_range_and_ideal_gas(g, R, v0_range, alpha_range,
                                        V_range, T_range,
                                        p_range_V, T_range_V,
                                        p_range_T, V_range_T,
                                        n_points, mode):

    def plot_surface_wireframe_contour(X, Y, Z, xlabel, ylabel, zlabel, title):
        """Creates wireframe, surface, and contour plots for a 3D function."""
        fig = plt.figure(figsize=(15, 4))
        fig.suptitle(title, fontsize=13)

        # Wireframe
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax1.plot_wireframe(X, Y, Z, color='steelblue', linewidth=0.5, rstride=4, cstride=4)
        ax1.set_xlabel(xlabel); ax1.set_ylabel(ylabel); ax1.set_zlabel(zlabel)
        ax1.set_title("Wireframe")

        # Surface with colormap
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        surf = ax2.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor='none', alpha=0.9)
        ax2.set_xlabel(xlabel); ax2.set_ylabel(ylabel); ax2.set_zlabel(zlabel)
        ax2.set_title("Surface")
        fig.colorbar(surf, ax=ax2, shrink=0.5)

        # Contour lines (2D)
        ax3 = fig.add_subplot(1, 3, 3)
        cp = ax3.contourf(X, Y, Z, levels=20, cmap=cm.viridis)
        ax3.contour(X, Y, Z, levels=20, colors='white', linewidths=0.5, alpha=0.5)
        fig.colorbar(cp, ax=ax3)
        ax3.set_xlabel(xlabel); ax3.set_ylabel(ylabel)
        ax3.set_title("Contour lines")

        plt.tight_layout()
        plt.show()

    # --- a) Projectile range W(v0, alpha) ---
    # Maximum: W is maximized at alpha = pi/4 (45 degrees), since sin(2*pi/4) = sin(pi/2) = 1
    if mode in ("projectile", "all"):
        v0    = np.linspace(*v0_range,    n_points)
        alpha = np.linspace(*alpha_range, n_points)
        V0, A = np.meshgrid(v0, alpha)
        W = V0**2 * np.sin(2 * A) / g

        plot_surface_wireframe_contour(
            V0, A, W,
            xlabel="v0 [m/s]",
            ylabel="alpha [rad]",
            zlabel="W [m]",
            title="a) Projectile range  W(v0, alpha) = v0^2 * sin(2*alpha) / g\n"
                  "(Maximum at alpha = pi/4 = 45 degrees)"
        )

    # --- b) Ideal gas ---
    if mode in ("ideal_gas", "all"):
        # p(V, T) = R*T / V
        V_vals = np.linspace(V_range[0], V_range[1], n_points)
        T_vals = np.linspace(T_range[0], T_range[1], n_points)
        VV, TT = np.meshgrid(V_vals, T_vals)
        P = R * TT / VV
        plot_surface_wireframe_contour(
            VV, TT, P,
            xlabel="V [m^3]",
            ylabel="T [K]",
            zlabel="p [Pa]",
            title="b1) Pressure  p(V, T) = R*T / V"
        )

        # V(p, T) = R*T / p
        p_vals = np.linspace(p_range_V[0], p_range_V[1], n_points)
        T_vals2 = np.linspace(T_range_V[0], T_range_V[1], n_points)
        PP2, TT2 = np.meshgrid(p_vals, T_vals2)
        V_out = R * TT2 / PP2
        plot_surface_wireframe_contour(
            PP2, TT2, V_out,
            xlabel="p [Pa]",
            ylabel="T [K]",
            zlabel="V [m^3]",
            title="b2) Volume  V(p, T) = R*T / p"
        )

        # T(p, V) = p*V / R
        p_vals3 = np.linspace(p_range_T[0], p_range_T[1], n_points)
        V_vals3 = np.linspace(max(V_range_T[0], 1e-3), V_range_T[1], n_points)
        PP3, VV3 = np.meshgrid(p_vals3, V_vals3)
        T_out = PP3 * VV3 / R
        plot_surface_wireframe_contour(
            PP3, VV3, T_out,
            xlabel="p [Pa]",
            ylabel="V [m^3]",
            zlabel="T [K]",
            title="b3) Temperature  T(p, V) = p*V / R"
        )

# ============================================================
# PART 4 — Call
# ============================================================
plot_projectile_range_and_ideal_gas(
    g, R, v0_range, alpha_range,
    V_range, T_range,
    p_range_V, T_range_V,
    p_range_T, V_range_T,
    n_points, mode
)
