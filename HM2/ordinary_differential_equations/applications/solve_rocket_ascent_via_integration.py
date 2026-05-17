# ============================================================
# TOPIC: DGL — Raketenaufstieg via summierter Trapezregel
# DESCRIPTION:
# Aus gegebener Beschleunigungsfunktion a(t) wird v(t) = int_0^t a(s) ds
# und h(t) = int_0^t v(s) ds via summierter Trapezregel berechnet, optional
# mit Vergleich zur analytischen Lösung.
# USE WHEN:
# Wenn aus a(t) sukzessive Geschwindigkeit und Höhe per Quadratur statt
# DGL-Solver gewonnen werden sollen (typisch bei Raketen-/Brems-Problemen).
# EXAMPLE:
# Ariane 4: v_rel = 2600 m/s, m_A = 3e6 kg, m_E = 8e5 kg, t_E = 190 s.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PART 1 — Inputs
# ============================================================
v_rel = 2600.0
m_A   = 3_000_000.0
m_E   = 800_000.0
t_E   = 190.0
g     = 9.81
mu    = (m_A - m_E) / t_E    # konstanter Massenstrom

def a(t):
    return v_rel * mu / (m_A - mu * t) - g

# Analytische Referenzlösungen (auf None setzen, wenn keine verfügbar)
def v_exact(t):
    return v_rel * np.log(m_A / (m_A - mu * t)) - g * t

def h_exact(t):
    return -v_rel * (m_A - mu * t) / mu * np.log(m_A / (m_A - mu * t)) + v_rel * t - 0.5 * g * t**2

n_steps = 2000

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: cumulative summed trapezoidal rule.

# ============================================================
# PART 3 — Implementation
# ============================================================
def integrate_rocket_ascent(a, v_exact, h_exact, t_E, n_steps):
    t = np.linspace(0, t_E, n_steps + 1)
    h_step = t[1] - t[0]
    a_vals = a(t)

    # Kumulative Trapez-Integration
    v_vals = np.zeros_like(t)
    h_vals = np.zeros_like(t)
    for i in range(1, len(t)):
        v_vals[i] = v_vals[i-1] + h_step * (a_vals[i-1] + a_vals[i]) / 2
    for i in range(1, len(t)):
        h_vals[i] = h_vals[i-1] + h_step * (v_vals[i-1] + v_vals[i]) / 2

    print(f"Am Ende der Brennphase (t = {t_E} s):")
    print(f"  v(t_E) = {v_vals[-1]:.2f} m/s")
    print(f"  h(t_E) = {h_vals[-1]:.2f} m")
    print(f"  a(t_E) = {a_vals[-1]:.2f} m/s^2 ({a_vals[-1]/g:.2f} g)")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(t, a_vals, 'r-'); axes[0].set_title("Beschleunigung a(t)")
    axes[0].set_xlabel('t [s]'); axes[0].set_ylabel('a [m/s^2]'); axes[0].grid(True)

    axes[1].plot(t, v_vals, 'b-', label='Trapez')
    if v_exact is not None:
        axes[1].plot(t, v_exact(t), 'r--', label='Exakt')
    axes[1].set_title("Geschwindigkeit v(t)")
    axes[1].set_xlabel('t [s]'); axes[1].set_ylabel('v [m/s]'); axes[1].legend(); axes[1].grid(True)

    axes[2].plot(t, h_vals, 'b-', label='Trapez')
    if h_exact is not None:
        axes[2].plot(t, h_exact(t), 'r--', label='Exakt')
    axes[2].set_title("Höhe h(t)")
    axes[2].set_xlabel('t [s]'); axes[2].set_ylabel('h [m]'); axes[2].legend(); axes[2].grid(True)

    plt.tight_layout(); plt.show()
    return t, v_vals, h_vals

# ============================================================
# PART 4 — Call
# ============================================================
integrate_rocket_ascent(a, v_exact, h_exact, t_E, n_steps)
