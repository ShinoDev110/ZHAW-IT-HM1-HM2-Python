# ZHAW-IT-HM2-Python
A collection of standalone Python scripts covering HM2 exercise topics.
The currently available script category is located under `nonlinear_systems/`, which contains scripts for the numerical solution of nonlinear systems of equations.
The scripts are educational support material for students.
Correctness is not guaranteed, so results should always be verified before they are relied on.

## Script overview

All currently available scripts are organized below `nonlinear_systems/` and grouped by topic into subfolders.

### `nonlinear_systems/newton_methods/`
- `nonlinear_systems/newton_methods/solve_nonlinear_system_newton.py` — Solves a nonlinear system with standard Newton iteration.
- `nonlinear_systems/newton_methods/solve_nonlinear_system_simplified_newton.py` — Solves a nonlinear system with simplified Newton iteration.
- `nonlinear_systems/newton_methods/solve_nonlinear_system_frozen_jacobian_newton.py` — Solves a nonlinear system with a frozen-Jacobian Newton variant.
- `nonlinear_systems/newton_methods/compare_newton_methods.py` — Compares standard, simplified, and frozen-Jacobian Newton variants.
- `nonlinear_systems/newton_methods/solve_newton_with_iteration_norms.py` — Prints residual and step norms during Newton iteration.
- `nonlinear_systems/newton_methods/solve_all_newton_solutions.py` — Runs Newton from multiple start vectors and lists all found solutions.

### `nonlinear_systems/linearization/`
- `nonlinear_systems/linearization/calculate_jacobian_matrix.py` — Computes symbolic Jacobian matrices and evaluates them at a point.
- `nonlinear_systems/linearization/linearize_multivariable_function.py` — Linearizes a multivariable function at a chosen point.

### `nonlinear_systems/visualization/`
- `nonlinear_systems/visualization/plot_implicit_equations.py` — Plots implicit equations with `sympy.plot_implicit()`.
- `nonlinear_systems/visualization/plot_nonlinear_system_contours.py` — Visualizes the zero contours of a nonlinear 2D system.
- `nonlinear_systems/visualization/plot_wave_equation_wireframe.py` — Plots one or more given wave-equation-style functions as 3D wireframes.
- `nonlinear_systems/visualization/visualize_2d_scalar_function.py` — Visualizes a scalar function of two variables as surface, wireframe, and contour plots.

Future HM2 topic categories should be added as new top-level folders next to `nonlinear_systems/`, not inside it.

