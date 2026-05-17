# ZHAW-IT-HM2-Python
A collection of standalone Python scripts covering exercise topics from the HM1 and HM2 lectures.
This repoitory includes all versions of exercises from the script and all exercise-sheets used in the HM1 and HM2 Lecture.
Scripts are grouped into two top-level course folders (`HM1/` and `HM2/`), each split into chapter folders.
Correctness is not guaranteed, so results should always be verified before they are relied on.

Every script follows the same four-part structure (TOPIC header + PART 1 Inputs / PART 2 Method selection / PART 3 Implementation / PART 4 Call). See [AGENTS.md](AGENTS.md) for the exact convention.

## HM1 — Höhere Mathematik 1

### `HM1/machine_numbers/`
- `HM1/machine_numbers/compute_min_max_machine_number.py` — Smallest/largest positive machine number for a given floating-point format M(B, n, e_min, e_max).
- `HM1/machine_numbers/count_machine_numbers.py` — Counts distinct representable machine numbers in a given format.
- `HM1/machine_numbers/compute_machine_precision.py` — Computes the machine epsilon eps = (B/2) · B^(-n).
- `HM1/machine_numbers/round_to_machine_number.py` — Converts a decimal value to an IEEE-style |sign|exponent|mantissa| bit string.
- `HM1/machine_numbers/sum_series_with_rounding.py` — Sums Σ 1/i^2 upwards and downwards under fixed-decimal rounding to show order-dependence.
- `HM1/machine_numbers/approximate_e_with_powers_of_ten.py` — Computes (1 + 1/10^n)^(10^n) and compares to math.e; shows where it stops converging.
- `HM1/machine_numbers/test_precision_loss_at_1_plus_eps.py` — Finds the n at which 1 + 10^-n ceases to be distinguishable from 1.

### `HM1/functions/basic_operations/`
- `HM1/functions/basic_operations/differentiate_and_integrate_symbolic.py` — Symbolic derivative and antiderivative of a sympy function, with point evaluation.
- `HM1/functions/basic_operations/simplify_function.py` — Symbolic simplification of an expression via sympy.simplify.
- `HM1/functions/basic_operations/compute_condition_number_of_function.py` — κ_f(x) = |x f'(x)/f(x)| with analytic, numeric, or grid-sweep method.
- `HM1/functions/basic_operations/evaluate_polynomial_with_horner.py` — Horner-scheme evaluation of p(x), p'(x), and antiderivative P(x) on a grid.

### `HM1/functions/root_finding/`
- `HM1/functions/root_finding/find_root_with_bisection.py` — Classical bisection (interval halving) for f(x) = 0.
- `HM1/functions/root_finding/find_fixed_point.py` — Generic fixed-point iteration x_{k+1} = F(x_k) with tolerance or fixed-iteration stop.
- `HM1/functions/root_finding/find_root_with_newton.py` — Newton method for symbolic f(x), tolerance or fixed-iteration stop.
- `HM1/functions/root_finding/find_root_with_simplified_newton.py` — Simplified Newton with f'(x_0) held constant.
- `HM1/functions/root_finding/find_root_with_secant.py` — Secant method (no derivative) for symbolic f.
- `HM1/functions/root_finding/classify_attractive_repulsive_fixed_point.py` — Classifies a fixed point as attractive/repulsive via |F'(x̄)| at a point or on an interval.
- `HM1/functions/root_finding/estimate_iterations_a_priori.py` — Banach a-priori bound for required number of iterations from Lipschitz constant α and first step.
- `HM1/functions/root_finding/estimate_convergence_order.py` — Estimates p and C in e_{k+1} ≈ C · e_k^p from an iteration history.
- `HM1/functions/root_finding/verify_root_with_sign_change.py` — Sign-change check around an approximation gives hard bound |x* - x̂| ≤ r.

### `HM1/functions/visualization/`
- `HM1/functions/visualization/plot_function_generic.py` — Generic multi-function plotter (linear / log / semilog) for sympy-parsed function strings.
- `HM1/functions/visualization/plot_fixed_point_iteration.py` — Plots F(x) - x and marks fixed points with attractive/repulsive labels.
- `HM1/functions/visualization/plot_newton_iteration.py` — Plots f(x), marks zeros, runs Newton (or simplified Newton) from x0 and shows the iteration path.
- `HM1/functions/visualization/plot_secant_iteration.py` — Plots f(x), marks zeros, runs secant method from (x0, x1).
- `HM1/functions/visualization/plot_polynomial_instability_demo.py` — Demonstrates catastrophic cancellation by comparing (x-2)^7 expanded vs. compact near x = 2.

### `HM1/matrices/basic_operations/`
- `HM1/matrices/basic_operations/invert_matrix.py` — Computes A^-1 via numpy.linalg.inv.
- `HM1/matrices/basic_operations/check_orthogonal_matrix.py` — Tests whether A · A^T ≈ I.
- `HM1/matrices/basic_operations/check_diagonally_dominant.py` — Row- and column-sum criteria for strict diagonal dominance.
- `HM1/matrices/basic_operations/compute_matrix_norm_and_condition.py` — Prints ||A||_1, ||A||_2, ||A||_∞ and (square A) cond_1, cond_2, cond_∞.
- `HM1/matrices/basic_operations/compute_vector_norm_and_condition.py` — Prints ||v||_1, ||v||_2, ||v||_∞ of a vector.

### `HM1/matrices/decompositions/`
- `HM1/matrices/decompositions/decompose_with_lr.py` — LR decomposition without pivoting; also solves Ax = b.
- `HM1/matrices/decompositions/decompose_with_plr.py` — PLR decomposition with row pivoting; also solves Ax = b.
- `HM1/matrices/decompositions/decompose_with_qr.py` — QR via numpy.linalg.qr; also solves Ax = b.
- `HM1/matrices/decompositions/decompose_with_qr_householder.py` — QR via own Householder reflections, with optional step-by-step debug.

### `HM1/matrices/linear_systems/`
- `HM1/matrices/linear_systems/solve_linear_system_with_gauss.py` — Gauss elimination with column pivoting, returns U, det(A), x.
- `HM1/matrices/linear_systems/solve_linear_system_with_lr.py` — Solves Ax = b via LR decomposition (forward + back substitution).
- `HM1/matrices/linear_systems/solve_linear_system_with_plr.py` — Solves Ax = b via PLR with row pivoting.
- `HM1/matrices/linear_systems/solve_linear_system_with_cramer.py` — Cramer's rule (for small systems).
- `HM1/matrices/linear_systems/estimate_arithmetic_complexity.py` — Rule-of-thumb counts (2/3) n^3 and 2 n^2 for Gauss / triangular solves.

### `HM1/matrices/iterative_methods/`
- `HM1/matrices/iterative_methods/solve_with_jacobi.py` — Jacobi iteration with optional a-priori / a-posteriori stopping.
- `HM1/matrices/iterative_methods/solve_with_gauss_seidel.py` — Gauss-Seidel iteration with optional a-priori / a-posteriori stopping.

### `HM1/matrices/error_estimation/`
- `HM1/matrices/error_estimation/estimate_error_general.py` — Bound on rel. error of x for perturbations in BOTH A and b; supports symbolic ε.
- `HM1/matrices/error_estimation/estimate_error_right_side.py` — Bound on rel. error of x when only b is perturbed (A exact); symbolic / explicit / percent / worst-case b̃.

### `HM1/interpolation/`
- `HM1/interpolation/interpolate_with_cubic_polynomial.py` — Builds the unique cubic through 4 (year, value) pairs via Vandermonde and evaluates at a target year.

### `HM1/eigenvalue_problems/`
- `HM1/eigenvalue_problems/compute_complete_eigen_analysis.py` — Full eigen-analysis: eigenvalues, eigenvectors, algebraic and geometric multiplicities with ZSF debug.

### `HM1/eigenvalue_problems/basic_operations/`
- `HM1/eigenvalue_problems/basic_operations/compute_eigenvalues_only.py` — Eigenvalues only via numpy.linalg.eigvals.
- `HM1/eigenvalue_problems/basic_operations/compute_algebraic_multiplicity.py` — Counts algebraic multiplicities after rounding.
- `HM1/eigenvalue_problems/basic_operations/compute_characteristic_polynomial.py` — Symbolic p(λ) = det(A - λI), expanded and factorized.

### `HM1/eigenvalue_problems/spectral_properties/`
- `HM1/eigenvalue_problems/spectral_properties/compute_spectral_radius.py` — ρ(B) for both Jacobi and Gauss-Seidel iteration matrices.
- `HM1/eigenvalue_problems/spectral_properties/compute_spectrum.py` — Prints the full spectrum and its cardinality.

### `HM1/eigenvalue_problems/diagonalization/`
- `HM1/eigenvalue_problems/diagonalization/build_diagonalization_matrix_t.py` — Builds T from eigenvectors (auto) or compares with a given T_given (manual).

### `HM1/eigenvalue_problems/iterative_methods/`
- `HM1/eigenvalue_problems/iterative_methods/compute_eigenvalues_with_qr_iteration.py` — QR iteration (fixed iters or tolerance, optional Rayleigh shift); reads eigenvalues off the (quasi-)upper-triangle.
- `HM1/eigenvalue_problems/iterative_methods/compute_dominant_eigenvalue_with_power_method.py` — Von Mises / power method for the dominant eigenvalue + Rayleigh quotient.

### `HM1/complex_dynamics/`
- `HM1/complex_dynamics/plot_mandelbrot_set.py` — Renders the Mandelbrot set as iterations-until-escape (Z_{n+1} = Z_n^2 + C).

## HM2 — Höhere Mathematik 2

### `HM2/nonlinear_systems/newton_methods/`
- `HM2/nonlinear_systems/newton_methods/solve_nonlinear_system_newton.py` — Solves a nonlinear system with standard Newton iteration.
- `HM2/nonlinear_systems/newton_methods/solve_nonlinear_system_simplified_newton.py` — Solves a nonlinear system with simplified Newton iteration.
- `HM2/nonlinear_systems/newton_methods/solve_nonlinear_system_frozen_jacobian_newton.py` — Solves a nonlinear system with a frozen-Jacobian Newton variant.
- `HM2/nonlinear_systems/newton_methods/compare_newton_methods.py` — Compares standard, simplified, and frozen-Jacobian Newton variants.
- `HM2/nonlinear_systems/newton_methods/solve_newton_with_iteration_norms.py` — Prints residual and step norms during Newton iteration.
- `HM2/nonlinear_systems/newton_methods/solve_all_newton_solutions.py` — Runs Newton from multiple start vectors and lists all found solutions.
- `HM2/nonlinear_systems/newton_methods/solve_3d_nonlinear_system_damped_newton.py` — Solves a 3D nonlinear system with the damped Newton method and prints residual/step norms per iteration.
- `HM2/nonlinear_systems/newton_methods/fit_soil_pressure_model_damped_newton.py` — Fits the soil-pressure model parameters via damped Newton from measurement points and bisects for the minimum disc radius.

### `HM2/nonlinear_systems/linearization/`
- `HM2/nonlinear_systems/linearization/calculate_jacobian_matrix.py` — Computes symbolic Jacobian matrices and evaluates them at a point.
- `HM2/nonlinear_systems/linearization/linearize_multivariable_function.py` — Linearizes a multivariable function at a chosen point.
- `HM2/nonlinear_systems/linearization/compute_partial_derivatives_and_linearize.py` — Computes symbolic first-order partial derivatives at given points and linearizes multivariable functions via the Jacobian.

### `HM2/nonlinear_systems/visualization/`
- `HM2/nonlinear_systems/visualization/plot_implicit_equations.py` — Plots implicit equations with `sympy.plot_implicit()`.
- `HM2/nonlinear_systems/visualization/plot_nonlinear_system_contours.py` — Visualizes the zero contours of a nonlinear 2D system.
- `HM2/nonlinear_systems/visualization/plot_wave_equation_wireframe.py` — Plots one or more given wave-equation-style functions as 3D wireframes.
- `HM2/nonlinear_systems/visualization/visualize_2d_scalar_function.py` — Visualizes a scalar function of two variables as surface, wireframe, and contour plots.
- `HM2/nonlinear_systems/visualization/plot_projectile_range_and_ideal_gas.py` — Plots projectile range and ideal-gas relations as 3D wireframes, surfaces, and contour diagrams.

### `HM2/interpolation_and_least_squares/polynomial_interpolation/`
- `HM2/interpolation_and_least_squares/polynomial_interpolation/interpolate_value_with_lagrange.py` — Interpolates y-values at given x-points using the Lagrange polynomial formula (own implementation).
- `HM2/interpolation_and_least_squares/polynomial_interpolation/interpolate_with_polyfit_polyval.py` — Polynomial interpolation via numpy.polyfit/polyval, with optional mean-centering of x for better conditioning.
- `HM2/interpolation_and_least_squares/polynomial_interpolation/compare_lagrange_vs_polyfit.py` — Overlays own Lagrange interpolation and numpy.polyfit for the same data.

### `HM2/interpolation_and_least_squares/spline_interpolation/`
- `HM2/interpolation_and_least_squares/spline_interpolation/interpolate_with_natural_cubic_spline.py` — Implements the natural cubic spline algorithm from scratch (Kap. 6.2.3).
- `HM2/interpolation_and_least_squares/spline_interpolation/interpolate_with_scipy_cubic_spline.py` — Uses scipy.interpolate.CubicSpline with selectable boundary conditions.
- `HM2/interpolation_and_least_squares/spline_interpolation/compare_cubic_spline_methods.py` — Compares own spline, scipy spline, and high-degree polynomial interpolation.

### `HM2/interpolation_and_least_squares/linear_least_squares/`
- `HM2/interpolation_and_least_squares/linear_least_squares/fit_polynomial_with_normal_equations.py` — Polynomial least-squares fit via normal equations, QR-decomposition, or numpy.polyfit, with condition numbers.
- `HM2/interpolation_and_least_squares/linear_least_squares/fit_multivariate_linear_regression.py` — Multivariate linear regression with several feature columns plus intercept.
- `HM2/interpolation_and_least_squares/linear_least_squares/fit_data_via_log_linearization.py` — Linearizes exponential models by taking logarithms; supports extrapolation.

### `HM2/interpolation_and_least_squares/nonlinear_least_squares/`
- `HM2/interpolation_and_least_squares/nonlinear_least_squares/fit_with_gauss_newton.py` — Undamped Gauss-Newton iteration with QR-based linear least-squares step.
- `HM2/interpolation_and_least_squares/nonlinear_least_squares/fit_with_damped_gauss_newton.py` — Damped Gauss-Newton with step factor delta/2^p.
- `HM2/interpolation_and_least_squares/nonlinear_least_squares/compare_gauss_newton_methods.py` — Compares damped vs. undamped Gauss-Newton from multiple start vectors.
- `HM2/interpolation_and_least_squares/nonlinear_least_squares/fit_with_scipy_optimize_fmin.py` — Direct minimization of the error functional with scipy.optimize.fmin (Nelder-Mead).

### `HM2/numerical_integration/newton_cotes/`
- `HM2/numerical_integration/newton_cotes/integrate_with_summed_rectangle_rule.py` — Summed rectangle/midpoint rule for a callable f.
- `HM2/numerical_integration/newton_cotes/integrate_with_summed_trapezoidal_rule.py` — Summed trapezoidal rule for equidistant subintervals.
- `HM2/numerical_integration/newton_cotes/integrate_with_summed_simpson_rule.py` — Summed Simpson rule with mid-of-subinterval evaluations.
- `HM2/numerical_integration/newton_cotes/integrate_with_trapezoidal_rule_non_equidistant.py` — Trapezoidal rule for tabulated data with non-equidistant x-values.

### `HM2/numerical_integration/gauss_quadrature/`
- `HM2/numerical_integration/gauss_quadrature/integrate_with_gauss_formulas.py` — Gauss quadrature with 1, 2, or 3 nodes (selectable).

### `HM2/numerical_integration/romberg_extrapolation/`
- `HM2/numerical_integration/romberg_extrapolation/integrate_with_romberg_extrapolation.py` — Builds the full Romberg triangle and returns T(0, m).

### `HM2/numerical_integration/error_and_comparison/`
- `HM2/numerical_integration/error_and_comparison/estimate_required_step_size.py` — Computes the maximum h (and minimum n) for a given tolerance using the error bounds from Satz 7.1.
- `HM2/numerical_integration/error_and_comparison/compare_quadrature_methods.py` — Runs all quadrature methods on the same integral and tabulates errors against scipy.integrate.quad as reference.

### `HM2/ordinary_differential_equations/direction_fields/`
- `HM2/ordinary_differential_equations/direction_fields/plot_direction_field.py` — Plots the direction field of a first-order ODE y'(x) = f(x, y) using meshgrid + quiver.

### `HM2/ordinary_differential_equations/single_step_methods/`
- `HM2/ordinary_differential_equations/single_step_methods/solve_ode_with_euler.py` — Classical Euler method (convergence order 1) with iteration log and optional comparison to an exact solution.
- `HM2/ordinary_differential_equations/single_step_methods/solve_ode_with_midpoint.py` — Midpoint method (order 2) with half-step slope evaluation.
- `HM2/ordinary_differential_equations/single_step_methods/solve_ode_with_modified_euler.py` — Modified Euler / Heun (order 2) averaging predictor and corrector slopes.
- `HM2/ordinary_differential_equations/single_step_methods/solve_ode_with_classical_runge_kutta.py` — Classical four-stage Runge-Kutta (order 4).
- `HM2/ordinary_differential_equations/single_step_methods/solve_ode_with_custom_runge_kutta.py` — Generic explicit s-stage Runge-Kutta from a user-supplied Butcher tableau (c, A, b).
- `HM2/ordinary_differential_equations/single_step_methods/compare_ode_single_step_methods.py` — Runs Euler, midpoint, modified Euler, and RK4 on the same problem and plots solutions plus global error.

### `HM2/ordinary_differential_equations/higher_order_systems/`
- `HM2/ordinary_differential_equations/higher_order_systems/reduce_higher_order_ode_to_system.py` — Reduces a k-th order ODE to a first-order system using sympy substitution.
- `HM2/ordinary_differential_equations/higher_order_systems/solve_ode_system_with_midpoint.py` — Solves a vector-valued first-order system with the midpoint method.

### `HM2/ordinary_differential_equations/applications/`
- `HM2/ordinary_differential_equations/applications/solve_rocket_ascent_via_integration.py` — Computes v(t) and h(t) from a given a(t) via cumulative summed trapezoidal rule, with optional comparison to analytical solutions.
