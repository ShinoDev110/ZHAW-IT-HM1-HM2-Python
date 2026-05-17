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
- `nonlinear_systems/newton_methods/solve_3d_nonlinear_system_damped_newton.py` — Solves a 3D nonlinear system with the damped Newton method and prints residual/step norms per iteration.
- `nonlinear_systems/newton_methods/fit_soil_pressure_model_damped_newton.py` — Fits the soil-pressure model parameters via damped Newton from measurement points and bisects for the minimum disc radius.

### `nonlinear_systems/linearization/`
- `nonlinear_systems/linearization/calculate_jacobian_matrix.py` — Computes symbolic Jacobian matrices and evaluates them at a point.
- `nonlinear_systems/linearization/linearize_multivariable_function.py` — Linearizes a multivariable function at a chosen point.
- `nonlinear_systems/linearization/compute_partial_derivatives_and_linearize.py` — Computes symbolic first-order partial derivatives at given points and linearizes multivariable functions via the Jacobian.

### `nonlinear_systems/visualization/`
- `nonlinear_systems/visualization/plot_implicit_equations.py` — Plots implicit equations with `sympy.plot_implicit()`.
- `nonlinear_systems/visualization/plot_nonlinear_system_contours.py` — Visualizes the zero contours of a nonlinear 2D system.
- `nonlinear_systems/visualization/plot_wave_equation_wireframe.py` — Plots one or more given wave-equation-style functions as 3D wireframes.
- `nonlinear_systems/visualization/visualize_2d_scalar_function.py` — Visualizes a scalar function of two variables as surface, wireframe, and contour plots.
- `nonlinear_systems/visualization/plot_projectile_range_and_ideal_gas.py` — Plots projectile range and ideal-gas relations as 3D wireframes, surfaces, and contour diagrams.

### `interpolation_and_least_squares/polynomial_interpolation/`
- `interpolation_and_least_squares/polynomial_interpolation/interpolate_value_with_lagrange.py` — Interpolates y-values at given x-points using the Lagrange polynomial formula (own implementation).
- `interpolation_and_least_squares/polynomial_interpolation/interpolate_with_polyfit_polyval.py` — Polynomial interpolation via numpy.polyfit/polyval, with optional mean-centering of x for better conditioning.
- `interpolation_and_least_squares/polynomial_interpolation/compare_lagrange_vs_polyfit.py` — Overlays own Lagrange interpolation and numpy.polyfit for the same data.

### `interpolation_and_least_squares/spline_interpolation/`
- `interpolation_and_least_squares/spline_interpolation/interpolate_with_natural_cubic_spline.py` — Implements the natural cubic spline algorithm from scratch (Kap. 6.2.3).
- `interpolation_and_least_squares/spline_interpolation/interpolate_with_scipy_cubic_spline.py` — Uses scipy.interpolate.CubicSpline with selectable boundary conditions.
- `interpolation_and_least_squares/spline_interpolation/compare_cubic_spline_methods.py` — Compares own spline, scipy spline, and high-degree polynomial interpolation.

### `interpolation_and_least_squares/linear_least_squares/`
- `interpolation_and_least_squares/linear_least_squares/fit_polynomial_with_normal_equations.py` — Polynomial least-squares fit via normal equations, QR-decomposition, or numpy.polyfit, with condition numbers.
- `interpolation_and_least_squares/linear_least_squares/fit_multivariate_linear_regression.py` — Multivariate linear regression with several feature columns plus intercept.
- `interpolation_and_least_squares/linear_least_squares/fit_data_via_log_linearization.py` — Linearizes exponential models by taking logarithms; supports extrapolation.

### `interpolation_and_least_squares/nonlinear_least_squares/`
- `interpolation_and_least_squares/nonlinear_least_squares/fit_with_gauss_newton.py` — Undamped Gauss-Newton iteration with QR-based linear least-squares step.
- `interpolation_and_least_squares/nonlinear_least_squares/fit_with_damped_gauss_newton.py` — Damped Gauss-Newton with step factor delta/2^p.
- `interpolation_and_least_squares/nonlinear_least_squares/compare_gauss_newton_methods.py` — Compares damped vs. undamped Gauss-Newton from multiple start vectors.
- `interpolation_and_least_squares/nonlinear_least_squares/fit_with_scipy_optimize_fmin.py` — Direct minimization of the error functional with scipy.optimize.fmin (Nelder-Mead).

### `numerical_integration/newton_cotes/`
- `numerical_integration/newton_cotes/integrate_with_summed_rectangle_rule.py` — Summed rectangle/midpoint rule for a callable f.
- `numerical_integration/newton_cotes/integrate_with_summed_trapezoidal_rule.py` — Summed trapezoidal rule for equidistant subintervals.
- `numerical_integration/newton_cotes/integrate_with_summed_simpson_rule.py` — Summed Simpson rule with mid-of-subinterval evaluations.
- `numerical_integration/newton_cotes/integrate_with_trapezoidal_rule_non_equidistant.py` — Trapezoidal rule for tabulated data with non-equidistant x-values.

### `numerical_integration/gauss_quadrature/`
- `numerical_integration/gauss_quadrature/integrate_with_gauss_formulas.py` — Gauss quadrature with 1, 2, or 3 nodes (selectable).

### `numerical_integration/romberg_extrapolation/`
- `numerical_integration/romberg_extrapolation/integrate_with_romberg_extrapolation.py` — Builds the full Romberg triangle and returns T(0, m).

### `numerical_integration/error_and_comparison/`
- `numerical_integration/error_and_comparison/estimate_required_step_size.py` — Computes the maximum h (and minimum n) for a given tolerance using the error bounds from Satz 7.1.
- `numerical_integration/error_and_comparison/compare_quadrature_methods.py` — Runs all quadrature methods on the same integral and tabulates errors against scipy.integrate.quad as reference.

### `ordinary_differential_equations/direction_fields/`
- `ordinary_differential_equations/direction_fields/plot_direction_field.py` — Plots the direction field of a first-order ODE y'(x) = f(x, y) using meshgrid + quiver.

### `ordinary_differential_equations/single_step_methods/`
- `ordinary_differential_equations/single_step_methods/solve_ode_with_euler.py` — Classical Euler method (convergence order 1) with iteration log and optional comparison to an exact solution.
- `ordinary_differential_equations/single_step_methods/solve_ode_with_midpoint.py` — Midpoint method (order 2) with half-step slope evaluation.
- `ordinary_differential_equations/single_step_methods/solve_ode_with_modified_euler.py` — Modified Euler / Heun (order 2) averaging predictor and corrector slopes.
- `ordinary_differential_equations/single_step_methods/solve_ode_with_classical_runge_kutta.py` — Classical four-stage Runge-Kutta (order 4).
- `ordinary_differential_equations/single_step_methods/solve_ode_with_custom_runge_kutta.py` — Generic explicit s-stage Runge-Kutta from a user-supplied Butcher tableau (c, A, b).
- `ordinary_differential_equations/single_step_methods/compare_ode_single_step_methods.py` — Runs Euler, midpoint, modified Euler, and RK4 on the same problem and plots solutions plus global error.

### `ordinary_differential_equations/higher_order_systems/`
- `ordinary_differential_equations/higher_order_systems/reduce_higher_order_ode_to_system.py` — Reduces a k-th order ODE to a first-order system using sympy substitution.
- `ordinary_differential_equations/higher_order_systems/solve_ode_system_with_midpoint.py` — Solves a vector-valued first-order system with the midpoint method.

### `ordinary_differential_equations/applications/`
- `ordinary_differential_equations/applications/solve_rocket_ascent_via_integration.py` — Computes v(t) and h(t) from a given a(t) via cumulative summed trapezoidal rule, with optional comparison to analytical solutions.
