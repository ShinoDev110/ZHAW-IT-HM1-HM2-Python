# ZHAW-IT-HM2-Python
A collection of standalone Python scripts covering the full range of exercises that may be derived from the HM2 lecture material.

## Script overview

### `visualization/`
- `plot_wave_equation_wireframe.py` — Plots one or more given wave-equation-style functions as 3D wireframes.
- `visualize_2d_scalar_function.py` — Visualizes a scalar function of two variables as surface, wireframe, and contour plots.
- `plot_implicit_equations.py` — Plots implicit equations with `sympy.plot_implicit()`.

### `differentiation/`
- `calculate_jacobian_matrix.py` — Computes symbolic Jacobian matrices and evaluates them at a point.

### `linearization/`
- `linearize_multivariable_function.py` — Linearizes a multivariable function at a chosen point.

### `newton_methods/`
- `solve_nonlinear_system_newton.py` — Solves a nonlinear system with standard Newton iteration.
- `solve_nonlinear_system_simplified_newton.py` — Solves a nonlinear system with simplified Newton iteration.
- `solve_nonlinear_system_frozen_jacobian_newton.py` — Solves a nonlinear system with a frozen-Jacobian Newton variant.
- `compare_newton_methods.py` — Compares standard, simplified, and frozen-Jacobian Newton variants.
- `solve_newton_with_iteration_norms.py` — Prints residual and step norms during Newton iteration.
- `solve_all_newton_solutions.py` — Runs Newton from multiple start vectors and lists all found solutions.

### `systems_of_equations/`
- `plot_nonlinear_system_contours.py` — Visualizes the zero contours of a nonlinear 2D system.

