# ============================================================
# TOPIC: DGL — System 1. Ordnung mit wählbarem Einschrittverfahren (vektoriell)
# DESCRIPTION:
# Löst ein vektorwertiges AWP z'(t) = f(t, z), z(t0) = z0 mit Euler,
# Mittelpunkt, modifiziertem Euler (Heun) oder klassischem RK4 — alle
# komponentenweise (für DGL höherer Ordnung, die in ein System umgeschrieben
# wurden). Plottet jede Komponente sowie optional die "Beschleunigung"
# (eine Komponente von z'). Im Vergleichsmodus werden zwei Verfahren
# gerechnet und ihre relative Abweichung halblogarithmisch dargestellt.
# USE WHEN:
# Wenn eine DGL 2. (oder höherer) Ordnung als System mit RK4 / mod. Euler
# gelöst und ggf. zwei Verfahren verglichen werden sollen.
# EXAMPLE:
# Rakete: h'' = v_rel·mu/(m_A-mu·t) - g - exp(-h/8000)/(m_A-mu·t)·(h')^2,
# h(0)=h'(0)=0, h=0.1, [0, t_E]; RK4 vs. mod. Euler vergleichen.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PART 1 — Inputs
# ============================================================
# --- Modellkonstanten (1. Stufe der Rakete) ---
v_rel = 2600.0        # relative Ausströmgeschwindigkeit [m/s]
m_A   = 300_000.0     # Anfangsmasse [kg]
m_E   = 80_000.0      # Masse am Ende der Brennphase [kg]
t_E   = 190.0         # Dauer der Brennphase [s]
g     = 9.81          # Fallbeschleunigung [m/s^2]
mu    = (m_A - m_E) / t_E   # Massenstrom dm/dt [kg/s]

# z = [h, v] mit v = h';  z' = [v, h''(t, h, v)]
def f(t, z):
    h, v = z
    accel = v_rel * mu / (m_A - mu * t) - g - np.exp(-h / 8000.0) / (m_A - mu * t) * v**2
    return np.array([v, accel])

z0    = np.array([0.0, 0.0])   # Startvektor [h(0), h'(0)]
t0    = 0.0                    # Startzeit
t_end = t_E                    # Endzeit (Ende Brennphase)
h_step = 0.1                   # Schrittweite [s]

# Beschriftungen der Zustandskomponenten und der "Beschleunigung" (= z'[accel_index])
labels      = ["Höhe h(t) [m]", "Geschwindigkeit h'(t) [m/s]"]
accel_index = 1
accel_label = "Beschleunigung h''(t) [m/s^2]"
plot_acceleration = True
compare_components = [0, 1]    # Komponenten für den Abweichungs-Plot (h und v)

# ============================================================
# PART 2 — Method selection
# ============================================================
# method / compare_method:  "euler" | "midpoint" | "mod_euler" | "rk4"
# mode:
#   "single"  -> nur 'method' rechnen und Komponenten plotten
#   "compare" -> zusätzlich 'compare_method' rechnen und die relative
#                Abweichung beider Lösungen halblogarithmisch plotten
method         = "rk4"
compare_method = "mod_euler"
mode           = "compare"

# ============================================================
# PART 3 — Implementation
# ============================================================
def _step_euler(f, t, z, h):
    return z + h * f(t, z)

def _step_midpoint(f, t, z, h):
    return z + h * f(t + h/2, z + h/2 * f(t, z))

def _step_mod_euler(f, t, z, h):
    k1 = f(t, z)
    k2 = f(t + h, z + h * k1)
    return z + h * (k1 + k2) / 2

def _step_rk4(f, t, z, h):
    k1 = f(t,       z)
    k2 = f(t + h/2, z + h/2 * k1)
    k3 = f(t + h/2, z + h/2 * k2)
    k4 = f(t + h,   z + h * k3)
    return z + h/6 * (k1 + 2*k2 + 2*k3 + k4)

_STEPPERS = {"euler": _step_euler, "midpoint": _step_midpoint,
             "mod_euler": _step_mod_euler, "rk4": _step_rk4}

def _integrate(f, t0, t_end, z0, h_step, method):
    if method not in _STEPPERS:
        raise ValueError(f"Unbekannte Methode: {method!r}")
    n = int(round((t_end - t0) / h_step))
    t = np.array([t0 + i * h_step for i in range(n + 1)])
    Z = np.zeros((n + 1, len(z0)))
    Z[0] = np.asarray(z0, dtype=float)
    step = _STEPPERS[method]
    for i in range(n):
        Z[i + 1] = step(f, t[i], Z[i], h_step)
    return t, Z

def solve_ode_system_with_single_step_methods(mode, method, compare_method, f,
                                              t0, t_end, z0, h_step,
                                              labels, accel_index, accel_label,
                                              plot_acceleration, compare_components):
    t, Z = _integrate(f, t0, t_end, z0, h_step, method)
    n = len(t) - 1
    print("============================================================")
    print(f"System 1. Ordnung — Methode '{method}' (vektoriell)")
    print("============================================================")
    print(f"Schritte: {n},  h = {h_step},  t in [{t0}, {t_end}]")
    for c, lab in enumerate(labels):
        print(f"  Endwert {lab}: {Z[-1, c]:.6f}")

    # Beschleunigung (eine Komponente von z') entlang der Lösung
    accel = np.array([f(t[i], Z[i])[accel_index] for i in range(n + 1)])

    # --- je eine separate Grafik pro Komponente (+ Beschleunigung) ---
    for c, lab in enumerate(labels):
        plt.figure(figsize=(9, 5))
        plt.plot(t, Z[:, c], 'b-', label=f"{lab} ({method})")
        plt.xlabel('t [s]'); plt.ylabel(lab); plt.grid(True); plt.legend()
        plt.title(lab)
        plt.tight_layout()
    if plot_acceleration:
        plt.figure(figsize=(9, 5))
        plt.plot(t, accel, 'b-', label=f"{accel_label} ({method})")
        plt.xlabel('t [s]'); plt.ylabel(accel_label); plt.grid(True); plt.legend()
        plt.title(accel_label)
        plt.tight_layout()

    # --- Vergleichsmodus: relative Abweichung halblogarithmisch ---
    if mode == "compare":
        t2, Z2 = _integrate(f, t0, t_end, z0, h_step, compare_method)
        print(f"\nVergleich '{method}' vs. '{compare_method}': relative Abweichung")
        plt.figure(figsize=(10, 6))
        for c in compare_components:
            ref = Z[:, c]
            with np.errstate(divide='ignore', invalid='ignore'):
                rel = np.where(np.abs(ref) > 0, np.abs(Z2[:, c] - ref) / np.abs(ref), np.nan)
            plt.semilogy(t, rel, label=f"rel. Abw. {labels[c]}")
            finite = rel[np.isfinite(rel)]
            if finite.size:
                print(f"  {labels[c]}: max rel. Abw. = {np.nanmax(rel):.3e}, "
                      f"bei t_end = {rel[-1]:.3e}")
        plt.xlabel('t [s]'); plt.ylabel('relative Abweichung (log)')
        plt.title(f"Relative Abweichung {method} vs. {compare_method}")
        plt.grid(True, which='both'); plt.legend()
        plt.tight_layout()

    plt.show()
    return t, Z

# ============================================================
# PART 4 — Call
# ============================================================
solve_ode_system_with_single_step_methods(mode, method, compare_method, f,
                                          t0, t_end, z0, h_step,
                                          labels, accel_index, accel_label,
                                          plot_acceleration, compare_components)
