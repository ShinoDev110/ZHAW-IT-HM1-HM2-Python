# ============================================================
# TOPIC: DGL-Anwendung — klassisches Runge-Kutta mit Aufprall/Reflexion
# DESCRIPTION:
# Löst ein DGL-System 1. Ordnung z' = f(t, z) mit z = [x, v] (aus einer DGL 2.
# Ordnung) mittels klassischem RK4. Optional wird ein Aufprall am Boden
# modelliert: sobald x < 0 und v < 0, wird die Geschwindigkeit reflektiert
# (v -> -v) und ein Aufprall gezählt. Plottet x(t) und v(t).
# USE WHEN:
# Wenn eine Bewegung mit Reflexion/Abprall simuliert und die Anzahl der
# Aufpralle bestimmt werden soll (Ereignisbehandlung im Integrator).
# EXAMPLE:
# x'' = -0.1·x'·|x'| - 10, x(0)=20, x'(0)=0, h=0.05, t in [0,8]
# -> Anzahl Aufpralle zählen.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PART 1 — Inputs
# ============================================================
def f(t, z):
    # z = [x, v];  x'' = -0.1 v|v| - 10  ->  z' = [v, -0.1 v|v| - 10]
    x, v = z
    return np.array([v, -0.1 * v * abs(v) - 10.0])

z0    = np.array([20.0, 0.0])   # Anfangszustand [x(0), v(0)]
t0    = 0.0                     # Startzeit
t_end = 8.0                     # Endzeit
h     = 0.05                    # Schrittweite

# ============================================================
# PART 2 — Method selection
# ============================================================
# mode:
#   "free"   -> ohne Reflexion (freie Bewegung)
#   "bounce" -> bei x<0 und v<0 wird v reflektiert (v -> -v) und gezählt
mode = "bounce"

# ============================================================
# PART 3 — Implementation
# ============================================================
def _rk4_step(f, t, z, h):
    k1 = f(t,         z)
    k2 = f(t + h/2,   z + h/2 * k1)
    k3 = f(t + h/2,   z + h/2 * k2)
    k4 = f(t + h,     z + h * k3)
    return z + h/6 * (k1 + 2*k2 + 2*k3 + k4)

def solve_ode_with_bounce_events(f, z0, t0, t_end, h, mode):
    n = int(round((t_end - t0) / h))
    ts = np.array([t0 + i * h for i in range(n + 1)])
    Z = np.zeros((n + 1, len(z0)))
    Z[0] = z0
    bounces = 0

    z = z0.astype(float).copy()
    for i in range(n):
        # Ereignisprüfung VOR dem Schritt (gemäss Aufgabenhinweis)
        if mode == "bounce" and z[0] < 0 and z[1] < 0:
            z[1] = -z[1]
            bounces += 1
        z = _rk4_step(f, ts[i], z, h)
        Z[i + 1] = z

    print("============================================================")
    print(f"RK4 mit Ereignisbehandlung — mode = '{mode}'")
    print("============================================================")
    print(f"Schritte: {n},  h = {h},  t in [{t0}, {t_end}]")
    if mode == "bounce":
        print(f"Anzahl Aufpralle (Reflexionen) = {bounces}")
    print(f"Endzustand z({t_end}) = [x={Z[-1,0]:.6f}, v={Z[-1,1]:.6f}]")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(ts, Z[:, 0], 'b-', label='x(t)')
    ax1.axhline(0, color='gray', lw=0.8)
    ax1.set_ylabel('Position x(t)'); ax1.legend(); ax1.grid(True)
    ax1.set_title(f"Bewegung mit Reflexion (Aufpralle: {bounces})" if mode == "bounce"
                  else "Freie Bewegung")
    ax2.plot(ts, Z[:, 1], 'g-', label="v(t) = x'(t)")
    ax2.axhline(0, color='gray', lw=0.8)
    ax2.set_xlabel('t'); ax2.set_ylabel('Geschwindigkeit v(t)'); ax2.legend(); ax2.grid(True)
    plt.tight_layout(); plt.show()
    return ts, Z, bounces

# ============================================================
# PART 4 — Call
# ============================================================
solve_ode_with_bounce_events(f, z0, t0, t_end, h, mode)
