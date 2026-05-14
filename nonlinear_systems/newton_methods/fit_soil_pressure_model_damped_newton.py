# ============================================================
# TOPIC: Nichtlineare Gleichungssysteme — Parameterbestimmung Bodendruckmodell
# DESCRIPTION:
# Bestimmt die Konstanten k1, k2, k3 im Bodendruckmodell
#   p(r) = k1·e^(k2·r) + k3·r
# via gedämpftem Newton-Verfahren aus 3 Messpunkten.
# Ermittelt anschliessend den minimalen Radius r*, ab dem eine runde Scheibe
# eine Last von 500 N aushält, ohne 30 cm tief einzusinken.
# USE WHEN:
# Wenn Modellkonstanten aus Messpunkten via Newton bestimmt werden sollen
# und eine anschliessende Vorhersage mit dem gefitteten Modell gefragt ist.
# EXAMPLE:
# Messpunkte: (r=1, p=10), (r=2, p=12), (r=3, p=15) [N/cm²]
# Startvektor k^(0) = (10, 0.1, -1)^T
# Gesucht: minimaler Radius für Last von 500 N bei Absinktiefe < 30 cm.
# ============================================================

import numpy as np
import sympy as sp

# ============================================================
# PART 1 — Inputs
# ============================================================
# Messpunkte (radius in cm, druck in N/cm²)
data_points = [
    (1, 10),   # r=1 cm → p=10 N/cm²  (Absinktiefe = 30 cm)
    (2, 12),   # r=2 cm → p=12 N/cm²
    (3, 15),   # r=3 cm → p=15 N/cm²
]

k0       = np.array([10.0, 0.1, -1.0], dtype=float)  # Startvektor für k1, k2, k3
tol      = 1e-8                                        # Abbruchschwelle für Newton
max_iter = 100                                         # maximale Iterationszahl
k_max    = 4                                           # Dämpfungsstufen

target_force = 500.0    # N   — Mindest-Traglast der Scheibe
depth        = 30.0     # cm  — maximale Absinktiefe (dieselbe wie in Messpunkten)

# Suchbereich für minimalen Radius (Bisektions-Vorgabe)
r_search_min = 1.0      # cm
r_search_max = 100.0    # cm

# ============================================================
# PART 2 — Method selection
# ============================================================
# Welcher Teil soll ausgeführt werden?
# options: "fit"       — nur Parameterbestimmung (k1, k2, k3)
#          "predict"   — Parameterbestimmung + Vorhersage des minimalen Radius
mode = "predict"

# ============================================================
# PART 3 — Implementation
# ============================================================
def fit_soil_pressure_model_damped_newton(data_points, k0, tol, max_iter, k_max,
                                          target_force, r_search_min, r_search_max,
                                          mode):

    # Aufstellen des nichtlinearen Gleichungssystems f(k1,k2,k3) = 0
    # aus den 3 Messpunkten: k1·e^(k2·ri) + k3·ri - pi = 0
    def f_eval(k_vec):
        k1, k2, k3 = k_vec
        return np.array([
            k1 * np.exp(k2 * r) + k3 * r - p
            for r, p in data_points
        ], dtype=float)

    def Df_eval(k_vec):
        k1, k2, k3 = k_vec
        Df = np.zeros((3, 3), dtype=float)
        for i, (r, _) in enumerate(data_points):
            Df[i, 0] = np.exp(k2 * r)          # ∂fi/∂k1
            Df[i, 1] = k1 * r * np.exp(k2 * r) # ∂fi/∂k2
            Df[i, 2] = r                         # ∂fi/∂k3
        return Df

    # Gedämpftes Newton-Verfahren
    def damped_newton(f, Df, x0):
        x = x0.copy()
        print(f"Startvektor k^(0) = {x}")
        print(f"Toleranz = {tol},  kmax = {k_max}\n")
        print(f"{'Iter':<6} {'||f(k)||_2':<18} {'k1':<14} {'k2':<14} {'k3':<14}")
        print("-" * 70)
        print(f"{'0':<6} {np.linalg.norm(f(x),2):<18.6e} {x[0]:<14.6f} {x[1]:<14.6f} {x[2]:<14.6f}")

        for iteration in range(1, max_iter + 1):
            fx  = f(x)
            Dfx = Df(x)
            try:
                delta = np.linalg.solve(Dfx, -fx)
            except np.linalg.LinAlgError:
                print("Jacobi-Matrix singulär – Abbruch.")
                break

            err_curr = np.linalg.norm(fx, 2)
            k_found  = None
            for k in range(k_max + 1):
                if np.linalg.norm(f(x + delta / (2**k)), 2) < err_curr:
                    k_found = k
                    break
            if k_found is None:
                k_found = 0

            x = x + delta / (2**k_found)
            f_norm = np.linalg.norm(f(x), 2)
            print(f"{iteration:<6} {f_norm:<18.6e} {x[0]:<14.6f} {x[1]:<14.6f} {x[2]:<14.6f}")

            if f_norm < tol:
                print(f"\nKonvergiert nach {iteration} Iterationen.")
                return x

        print(f"Keine Konvergenz nach {max_iter} Iterationen.")
        return x

    # --- Parameterbestimmung ---
    print("=" * 60)
    print("Parameterbestimmung: k1, k2, k3")
    print("=" * 60)
    k_sol = damped_newton(f_eval, Df_eval, k0)
    k1, k2, k3 = k_sol
    print(f"\nGefundene Parameter:")
    print(f"  k1 = {k1:.6f}")
    print(f"  k2 = {k2:.6f}")
    print(f"  k3 = {k3:.6f}")

    # Überprüfung an den Messpunkten
    print("\nÜberprüfung an den Messpunkten:")
    for r, p_meas in data_points:
        p_model = k1 * np.exp(k2 * r) + k3 * r
        print(f"  r={r}: p_gemessen={p_meas}, p_Modell={p_model:.4f}")

    if mode == "fit":
        return k_sol

    # --- Vorhersage: minimaler Radius für Last von 500 N ---
    print("\n" + "=" * 60)
    print(f"Vorhersage: minimaler Radius für Last = {target_force} N")
    print("=" * 60)
    # Gleichgewicht: Last = Druck × Fläche
    # target_force = p(r) · π · r²   =>   g(r) = k1·e^(k2·r) + k3·r - target_force/(π·r²) = 0
    # Bisektion auf [r_search_min, r_search_max]
    def g(r):
        area = np.pi * r**2    # Fläche der runden Scheibe in cm²
        return k1 * np.exp(k2 * r) + k3 * r - target_force / area

    # Prüfe Vorzeichen an Suchbereichsgrenzen
    ga = g(r_search_min)
    gb = g(r_search_max)
    if ga * gb > 0:
        print(f"Vorzeichenwechsel nicht im Intervall [{r_search_min}, {r_search_max}] gefunden.")
        print(f"  g({r_search_min}) = {ga:.4f},  g({r_search_max}) = {gb:.4f}")
        print("  Bitte r_search_min / r_search_max anpassen.")
        return k_sol

    # Bisektion
    a, b = r_search_min, r_search_max
    for _ in range(100):
        mid = (a + b) / 2
        if g(mid) * g(a) <= 0:
            b = mid
        else:
            a = mid
        if abs(b - a) < 1e-8:
            break

    r_star = (a + b) / 2
    p_star = k1 * np.exp(k2 * r_star) + k3 * r_star
    F_star = p_star * np.pi * r_star**2

    print(f"\nMinimaler Radius: r* = {r_star:.4f} cm")
    print(f"Druck bei r*:     p(r*) = {p_star:.4f} N/cm^2")
    print(f"Traglast bei r*:  F = p(r*)*pi*r*^2 = {F_star:.2f} N  (Ziel: {target_force} N)")

    return k_sol

# ============================================================
# PART 4 — Call
# ============================================================
fit_soil_pressure_model_damped_newton(
    data_points, k0, tol, max_iter, k_max,
    target_force, r_search_min, r_search_max,
    mode
)
