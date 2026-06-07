# ============================================================
# TOPIC: Plotting implicit functions — with sympy.plot_implicit
# DESCRIPTION:
# Plots implicit equations f_i(x,y)=0 together in one window using
# sympy.plot_implicit().
# USE WHEN:
# When implicit curves such as hyperbolas or ellipses need to be visualized.
# EXAMPLE:
# First examine the LORAN system graphically with sympy.plot_implicit().
# ============================================================

import sympy as sp

# ============================================================
# PART 1 — Inputs
# ============================================================
x, y = sp.symbols('x y')      # symbolic variables

# List of implicit equations f_i(x, y) = 0
equations = [
    x**2 / 186**2 - y**2 / (300**2 - 186**2) - 1,                 # first implicit equation
    (y - 500)**2 / 279**2 - (x - 300)**2 / (500**2 - 279**2) - 1  # second implicit equation
]

x_range = (x, -2000, 2000)   # x range for the plot
y_range = (y, -2000, 2000)   # y range for the plot

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method: combine all f_i = 0 into one plot window.

# ============================================================
# PART 3 — Implementation
# ============================================================
def plot_implicit_equations(equations, x_range, y_range):
    plots = []
    for f in equations:
        p = sp.plot_implicit(sp.Eq(f, 0), x_range, y_range, show=False)
        plots.append(p)

    # Combine all plots into the first one
    base = plots[0]
    for p in plots[1:]:
        base.append(p[0])
    base.show()

# ============================================================
# PART 4 — Call
# ============================================================
plot_implicit_equations(equations, x_range, y_range)
