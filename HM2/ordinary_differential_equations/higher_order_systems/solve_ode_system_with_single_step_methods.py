# ============================================================
# TOPIC: ODE — First-order system with selectable single-step method (vector-valued)
# DESCRIPTION:
# Solves a vector-valued IVP z'(t) = f(t, z), z(t0) = z0 with Euler,
# midpoint, modified Euler (Heun), or classical RK4 — all component-wise
# (for higher-order ODEs reduced to a system). Plots each component and
# optionally the "acceleration" (one component of z'). In comparison mode,
# two methods are computed and their relative deviation is shown on a semilog
# plot.
# USE WHEN:
# When a second-order (or higher-order) ODE as a system is to be solved with
# RK4 / modified Euler and two methods are optionally to be compared.
# EXAMPLE:
# Rocket: h'' = v_rel·mu/(m_A-mu·t) - g - exp(-h/8000)/(m_A-mu·t)·(h')^2,
# h(0)=h'(0)=0, h=0.1, [0, t_E]; compare RK4 vs. modified Euler.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PART 1 — Inputs
# ============================================================
# --- Model constants (stage 1 of the rocket) ---
v_rel = 2600.0        # relative exhaust velocity [m/s]
m_A   = 300_000.0     # initial mass [kg]
m_E   = 80_000.0      # mass at end of burn phase [kg]
t_E   = 190.0         # duration of burn phase [s]
g     = 9.81          # gravitational acceleration [m/s^2]
mu    = (m_A - m_E) / t_E   # mass flow rate dm/dt [kg/s]

# z = [h, v] with v = h';  z' = [v, h''(t, h, v)]
def f(t, z):
    h, v = z
    accel = v_rel * mu / (m_A - mu * t) - g - np.exp(-h / 8000.0) / (m_A - mu * t) * v**2
    return np.array([v, accel])

z0    = np.array([0.0, 0.0])   # initial vector [h(0), h'(0)]
t0    = 0.0                    # start time
t_end = t_E                    # end time (end of burn phase)
h_step = 0.1                   # step size [s]

# Labels for the state components and the "acceleration" (= z'[accel_index])
labels      = ["Altitude h(t) [m]", "Velocity h'(t) [m/s]"]
accel_index = 1
accel_label = "Acceleration h''(t) [m/s^2]"
plot_acceleration = True
compare_components = [0, 1]    # components for the deviation plot (h and v)

# ============================================================
# PART 2 — Method selection
# ============================================================
# method / compare_method:  "euler" | "midpoint" | "mod_euler" | "rk4"
# mode:
#   "single"  -> compute only 'method' and plot components
#   "compare" -> additionally compute 'compare_method' and plot the relative
#                deviation of both solutions on a semilog scale
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
        raise ValueError(f"Unknown method: {method!r}")
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
    print(f"First-order system — method '{method}' (vector-valued)")
    print("============================================================")
    print(f"Steps: {n},  h = {h_step},  t in [{t0}, {t_end}]")
    for c, lab in enumerate(labels):
        print(f"  Final value {lab}: {Z[-1, c]:.6f}")

    # Acceleration (one component of z') along the solution
    accel = np.array([f(t[i], Z[i])[accel_index] for i in range(n + 1)])

    # --- one separate figure per component (+ acceleration) ---
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

    # --- comparison mode: relative deviation on semilog scale ---
    if mode == "compare":
        t2, Z2 = _integrate(f, t0, t_end, z0, h_step, compare_method)
        print(f"\nComparison '{method}' vs. '{compare_method}': relative deviation")
        plt.figure(figsize=(10, 6))
        for c in compare_components:
            ref = Z[:, c]
            with np.errstate(divide='ignore', invalid='ignore'):
                rel = np.where(np.abs(ref) > 0, np.abs(Z2[:, c] - ref) / np.abs(ref), np.nan)
            plt.semilogy(t, rel, label=f"rel. deviation {labels[c]}")
            finite = rel[np.isfinite(rel)]
            if finite.size:
                print(f"  {labels[c]}: max rel. deviation = {np.nanmax(rel):.3e}, "
                      f"at t_end = {rel[-1]:.3e}")
        plt.xlabel('t [s]'); plt.ylabel('relative deviation (log)')
        plt.title(f"Relative deviation {method} vs. {compare_method}")
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
