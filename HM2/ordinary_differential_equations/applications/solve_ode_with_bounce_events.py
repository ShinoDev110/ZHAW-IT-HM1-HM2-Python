# ============================================================
# TOPIC: ODE application — classical Runge-Kutta with bounce/reflection events
# DESCRIPTION:
# Solves a first-order ODE system z' = f(t, z) with z = [x, v] (reduced from a
# second-order ODE) using classical RK4. Optionally models a ground bounce:
# whenever x < 0 and v < 0, the velocity is reflected (v -> -v) and the bounce
# is counted. Plots x(t) and v(t).
# USE WHEN:
# When a motion with reflection/bounce is to be simulated and the number of
# bounces is to be determined (event handling inside the integrator).
# EXAMPLE:
# x'' = -0.1·x'·|x'| - 10, x(0)=20, x'(0)=0, h=0.05, t in [0,8]
# -> count the number of bounces.
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

z0    = np.array([20.0, 0.0])   # initial state [x(0), v(0)]
t0    = 0.0                     # start time
t_end = 8.0                     # end time
h     = 0.05                    # step size

# ============================================================
# PART 2 — Method selection
# ============================================================
# mode:
#   "free"   -> without reflection (free motion)
#   "bounce" -> when x<0 and v<0, v is reflected (v -> -v) and counted
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
        # Event check BEFORE the step (per exercise note)
        if mode == "bounce" and z[0] < 0 and z[1] < 0:
            z[1] = -z[1]
            bounces += 1
        z = _rk4_step(f, ts[i], z, h)
        Z[i + 1] = z

    print("============================================================")
    print(f"RK4 with event handling — mode = '{mode}'")
    print("============================================================")
    print(f"Steps: {n},  h = {h},  t in [{t0}, {t_end}]")
    if mode == "bounce":
        print(f"Number of bounces (reflections) = {bounces}")
    print(f"Final state z({t_end}) = [x={Z[-1,0]:.6f}, v={Z[-1,1]:.6f}]")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(ts, Z[:, 0], 'b-', label='x(t)')
    ax1.axhline(0, color='gray', lw=0.8)
    ax1.set_ylabel('Position x(t)'); ax1.legend(); ax1.grid(True)
    ax1.set_title(f"Motion with reflection (bounces: {bounces})" if mode == "bounce"
                  else "Free motion")
    ax2.plot(ts, Z[:, 1], 'g-', label="v(t) = x'(t)")
    ax2.axhline(0, color='gray', lw=0.8)
    ax2.set_xlabel('t'); ax2.set_ylabel('Velocity v(t)'); ax2.legend(); ax2.grid(True)
    plt.tight_layout(); plt.show()
    return ts, Z, bounces

# ============================================================
# PART 4 — Call
# ============================================================
solve_ode_with_bounce_events(f, z0, t0, t_end, h, mode)
