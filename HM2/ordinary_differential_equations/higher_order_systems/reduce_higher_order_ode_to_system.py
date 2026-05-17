# ============================================================
# TOPIC: DGL — Reduktion einer DGL k-ter Ordnung auf System 1. Ordnung
# DESCRIPTION:
# Setzt Hilfsfunktionen z_1 = y, z_2 = y', ..., z_k = y^(k-1) ein und gibt
# das vektorwertige System z' = f(x, z) sowie den Anfangsvektor z(x0) aus.
# Implementiert mit sympy für beliebige DGL k-ter Ordnung.
# USE WHEN:
# Wenn eine DGL höherer Ordnung in ein System 1. Ordnung umgewandelt
# werden soll, um es mit Euler/Mittelpunkt/RK lösen zu können.
# EXAMPLE:
# Reduziere y''' + 5y'' + 8y' + 6y = 10 exp(-x) mit y(0) = 2, y'(0) = 0,
# y''(0) = 0 auf ein System 1. Ordnung.
# ============================================================

import sympy as sp

# ============================================================
# PART 1 — Inputs
# ============================================================
x = sp.symbols('x')
y = sp.Function('y')

# DGL in der Form: höchste Ableitung = rhs(x, y, y', ..., y^(k-1))
# Für y''' = 10 exp(-x) - 5 y'' - 8 y' - 6 y:
k    = 3
rhs  = 10 * sp.exp(-x) - 5 * y(x).diff(x, 2) - 8 * y(x).diff(x) - 6 * y(x)

x0   = 0
y_initial = [2, 0, 0]   # y(0), y'(0), y''(0)

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: introduce z_i = y^(i-1) and rewrite as first-order system.

# ============================================================
# PART 3 — Implementation
# ============================================================
def reduce_higher_order_ode_to_system(rhs, y, x, k, x0, y_initial):
    z = sp.symbols(f'z1:{k+1}')

    # Replace y, y', ..., y^(k-1) in rhs by z_1, ..., z_k
    subs_map = []
    for i in range(k):
        subs_map.append((y(x).diff(x, i), z[i]) if i > 0 else (y(x), z[0]))
    rhs_z = rhs.subs(subs_map)

    f_vec = sp.Matrix([z[i] for i in range(1, k)] + [rhs_z])
    z_vec = sp.Matrix(z)
    z0    = sp.Matrix(y_initial)

    print(f"DGL {k}. Ordnung -> System {k} DGL 1. Ordnung")
    print(f"\nHilfsfunktionen: z = (z_1, ..., z_{k}) = (y, y', ..., y^({k-1}))")
    print(f"\nz' = f(x, z) =")
    sp.pprint(f_vec)
    print(f"\nz(x_0 = {x0}) = z^(0) =")
    sp.pprint(z0)
    return f_vec, z_vec, z0

# ============================================================
# PART 4 — Call
# ============================================================
reduce_higher_order_ode_to_system(rhs, y, x, k, x0, y_initial)
