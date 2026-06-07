# ============================================================
# TOPIC: ODE — Reduction of a k-th order ODE to a first-order system
# DESCRIPTION:
# Introduces auxiliary functions z_1 = y, z_2 = y', ..., z_k = y^(k-1) and
# prints the vector-valued system z' = f(x, z) along with the initial vector
# z(x0). Implemented with sympy for arbitrary k-th order ODEs.
# USE WHEN:
# When a higher-order ODE is to be converted to a first-order system
# so that it can be solved with Euler/midpoint/RK.
# EXAMPLE:
# Reduce y''' + 5y'' + 8y' + 6y = 10 exp(-x) with y(0) = 2, y'(0) = 0,
# y''(0) = 0 to a first-order system.
# ============================================================

import sympy as sp

# ============================================================
# PART 1 — Inputs
# ============================================================
x = sp.symbols('x')
y = sp.Function('y')

# ODE in the form: highest derivative = rhs(x, y, y', ..., y^(k-1))
# For y''' = 10 exp(-x) - 5 y'' - 8 y' - 6 y:
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

    print(f"ODE of order {k} -> system of {k} first-order ODEs")
    print(f"\nAuxiliary functions: z = (z_1, ..., z_{k}) = (y, y', ..., y^({k-1}))")
    print(f"\nz' = f(x, z) =")
    sp.pprint(f_vec)
    print(f"\nz(x_0 = {x0}) = z^(0) =")
    sp.pprint(z0)
    return f_vec, z_vec, z0

# ============================================================
# PART 4 — Call
# ============================================================
reduce_higher_order_ode_to_system(rhs, y, x, k, x0, y_initial)
