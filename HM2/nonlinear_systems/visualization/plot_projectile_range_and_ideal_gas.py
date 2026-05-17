# ============================================================
# TOPIC: Funktionen mit mehreren Variablen — 3D Darstellung und Höhenlinien
# DESCRIPTION:
# Stellt physikalische Funktionen dreidimensional (Wireframe und Surface)
# sowie als 2D-Höhenliniendiagramm dar:
#   a) Wurfweite W(v0, alpha) eines schiefen Wurfes
#   b) Ideales Gas: p(V,T), V(p,T), T(p,V)
# USE WHEN:
# Wenn 3D-Visualisierungen mit Wireframe, Surface und Konturlinien
# für mehrere Variablen verlangt werden.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ============================================================
# PART 1 — Inputs
# ============================================================
g = 9.81    # Erdbeschleunigung [m/s²]
R = 8.31    # universelle Gaskonstante [J/(mol·K)]

# Definitionsbereiche
v0_range    = (0, 100)          # Anfangsgeschwindigkeit [m/s]
alpha_range = (0, np.pi / 2)   # Abwurfwinkel [rad], Maximum bei alpha = pi/4 (45°)

V_range = (1e-3, 0.2)          # Volumen [m³], untere Grenze >0 um Division durch 0 zu vermeiden
T_range = (0, 1e4)             # Temperatur [K]

p_range_V  = (1e4, 1e5)        # Druck [Pa] für V(p,T)
T_range_V  = (0, 1e4)          # Temperatur [K] für V(p,T)

p_range_T  = (1e4, 1e6)        # Druck [Pa] für T(p,V)
V_range_T  = (0, 10)           # Volumen [m³] für T(p,V), untere Grenze leicht >0

n_points = 60                  # Gitterpunkte pro Achse

# ============================================================
# PART 2 — Method selection
# ============================================================
# Welche Funktionsgruppe soll dargestellt werden?
# options: "projectile"   — nur Wurfweite W(v0, alpha)
#          "ideal_gas"    — nur Idealgas-Funktionen p, V, T
#          "all"          — beide Gruppen nacheinander
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
        """Erstellt Wireframe-, Surface- und Kontur-Plot für eine 3D-Funktion."""
        fig = plt.figure(figsize=(15, 4))
        fig.suptitle(title, fontsize=13)

        # Wireframe
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax1.plot_wireframe(X, Y, Z, color='steelblue', linewidth=0.5, rstride=4, cstride=4)
        ax1.set_xlabel(xlabel); ax1.set_ylabel(ylabel); ax1.set_zlabel(zlabel)
        ax1.set_title("Wireframe")

        # Surface mit Colormap
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        surf = ax2.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor='none', alpha=0.9)
        ax2.set_xlabel(xlabel); ax2.set_ylabel(ylabel); ax2.set_zlabel(zlabel)
        ax2.set_title("Surface")
        fig.colorbar(surf, ax=ax2, shrink=0.5)

        # Höhenlinien (2D)
        ax3 = fig.add_subplot(1, 3, 3)
        cp = ax3.contourf(X, Y, Z, levels=20, cmap=cm.viridis)
        ax3.contour(X, Y, Z, levels=20, colors='white', linewidths=0.5, alpha=0.5)
        fig.colorbar(cp, ax=ax3)
        ax3.set_xlabel(xlabel); ax3.set_ylabel(ylabel)
        ax3.set_title("Höhenlinien")

        plt.tight_layout()
        plt.show()

    # --- a) Wurfweite W(v0, alpha) ---
    # Maximum: W wird maximal bei alpha = pi/4 (45°), da sin(2*pi/4) = sin(pi/2) = 1
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
            title="a) Wurfweite  W(v0, α) = v0² · sin(2α) / g\n"
                  "(Maximum bei α = π/4 = 45°)"
        )

    # --- b) Ideales Gas ---
    if mode in ("ideal_gas", "all"):
        # p(V, T) = R·T / V
        V_vals = np.linspace(V_range[0], V_range[1], n_points)
        T_vals = np.linspace(T_range[0], T_range[1], n_points)
        VV, TT = np.meshgrid(V_vals, T_vals)
        P = R * TT / VV
        plot_surface_wireframe_contour(
            VV, TT, P,
            xlabel="V [m³]",
            ylabel="T [K]",
            zlabel="p [Pa]",
            title="b1) Druck  p(V, T) = R·T / V"
        )

        # V(p, T) = R·T / p
        p_vals = np.linspace(p_range_V[0], p_range_V[1], n_points)
        T_vals2 = np.linspace(T_range_V[0], T_range_V[1], n_points)
        PP2, TT2 = np.meshgrid(p_vals, T_vals2)
        V_out = R * TT2 / PP2
        plot_surface_wireframe_contour(
            PP2, TT2, V_out,
            xlabel="p [Pa]",
            ylabel="T [K]",
            zlabel="V [m³]",
            title="b2) Volumen  V(p, T) = R·T / p"
        )

        # T(p, V) = p·V / R
        p_vals3 = np.linspace(p_range_T[0], p_range_T[1], n_points)
        V_vals3 = np.linspace(max(V_range_T[0], 1e-3), V_range_T[1], n_points)
        PP3, VV3 = np.meshgrid(p_vals3, V_vals3)
        T_out = PP3 * VV3 / R
        plot_surface_wireframe_contour(
            PP3, VV3, T_out,
            xlabel="p [Pa]",
            ylabel="V [m³]",
            zlabel="T [K]",
            title="b3) Temperatur  T(p, V) = p·V / R"
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
