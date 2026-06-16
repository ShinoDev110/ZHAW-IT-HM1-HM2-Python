### `Heads up for Students:`

All the code related exercise series solutions are in here. From my own experience I strongly recommend to try them 
yourself first, otherwise you basically learn nothing... The chance that you have to edit the scripts on the final 
exams (SEP) is very high, and if you don't understand what's going on in the code you will probably not be able to get 
a good grade in the code part of the exam.

# ZHAW-IT-HM1-HM2-Python

A collection of standalone Python scripts covering the exercise topics of the **Höhere
Mathematik 1 and 2** courses (ZHAW Informatik). It includes implementations of the
lecture-script exercises, the exercise sheets, and a number of past-exam tasks.

Each script is self-contained and does not depend on any other file in the repository: set
the inputs at the top of the file, run it, and read the result. Correctness is not
guaranteed, so results should be verified before they are relied on.

## Contributing

Contributions are welcome:

- Fork the repository to use or adapt the scripts.
- Open an issue to report an error or request a missing topic.
- Submit a pull request with fixes, improvements, or additional scripts.

## Repository structure

Scripts are grouped into two top-level course folders, each divided into chapter sub-folders:

- `HM1/` — machine numbers, functions and root finding, matrices and linear systems,
  eigenvalue problems, and complex dynamics.
- `HM2/` — nonlinear systems, interpolation and least squares, numerical integration and
  differentiation, and ordinary differential equations.

Every script follows the same four-part structure:

| Part | Contents |
| --- | --- |
| 0 · TOPIC header | Summary of what the script does, when to use it, and a concrete example. |
| 1 · Inputs | The problem data; normally the only section that needs editing. |
| 2 · Method selection | A selector for the variant, where a problem can be solved in more than one way. |
| 3 · Implementation | The computation. |
| 4 · Call | A single call that runs the script. |

The scripts require `numpy`, `sympy`, `matplotlib`, and `scipy`, and run on a standard
Python interpreter.

> In the tables below, only the file name is shown; the containing folder is given in each
> section heading.

## HM1 — Höhere Mathematik 1

### Machine numbers — `HM1/machine_numbers/`

| Script | What it does |
| --- | --- |
| `compute_min_max_machine_number.py` | Smallest / largest positive machine number for a floating-point format M(B, n, e_min, e_max). |
| `count_machine_numbers.py` | Counts the distinct representable machine numbers in a given format. |
| `compute_machine_precision.py` | Computes the machine epsilon eps = (B/2) · B^(-n). |
| `round_to_machine_number.py` | Converts a decimal value to an IEEE-style sign / exponent / mantissa bit string. |
| `sum_series_with_rounding.py` | Sums Σ 1/i² up and down under fixed-decimal rounding to show order-dependence. |
| `approximate_e_with_powers_of_ten.py` | Computes (1 + 1/10^n)^(10^n), compares to math.e, and shows where it stops converging. |
| `test_precision_loss_at_1_plus_eps.py` | Finds the n at which 1 + 10^-n is no longer distinguishable from 1. |
| `represent_and_round_floating_point.py` | Represents a value in a finite mantissa (base 2 or 10) by rounding vs. truncation and reports the error; an optional second value shows subtraction cancellation. |
| `compare_machine_precision.py` | Compares eps of several formats / bases to decide which machine computes more precisely. |
| `analyze_custom_float_format.py` | Full report (precision, exponent range, smallest / largest number, count) for a custom sign / exponent+bias / mantissa format. |

### Functions — basic operations — `HM1/functions/basic_operations/`

| Script | What it does |
| --- | --- |
| `differentiate_and_integrate_symbolic.py` | Symbolic derivative and antiderivative of a sympy function, with point evaluation. |
| `simplify_function.py` | Symbolic simplification of an expression via sympy.simplify. |
| `compute_condition_number_of_function.py` | The condition number κ_f(x) via analytic, numeric, or grid-sweep evaluation. |
| `evaluate_polynomial_with_horner.py` | Horner-scheme evaluation of p(x), p'(x), and the antiderivative P(x) on a grid. |
| `estimate_error_with_condition_number.py` | Error propagation through κ_f(x): input→output error, target output→max input error, and an actual-vs-predicted comparison. |

### Functions — root finding — `HM1/functions/root_finding/`

| Script | What it does |
| --- | --- |
| `find_root_with_bisection.py` | Classical bisection (interval halving) for f(x) = 0. |
| `find_fixed_point.py` | Generic fixed-point iteration x_{k+1} = F(x_k) with a tolerance or fixed-iteration stop. |
| `find_root_with_newton.py` | Newton's method for a symbolic f(x), tolerance or fixed-iteration stop. |
| `find_root_with_simplified_newton.py` | Simplified Newton with f'(x_0) held constant. |
| `find_root_with_secant.py` | Secant method (no derivative needed) for a symbolic f. |
| `classify_attractive_repulsive_fixed_point.py` | Classifies a fixed point as attractive / repulsive from the size of F'(x̄), at a point or over an interval. |
| `estimate_iterations_a_priori.py` | Banach a-priori bound for the number of iterations, from the Lipschitz constant α and the first step. |
| `estimate_convergence_order.py` | Estimates p and C in e_{k+1} ≈ C · e_k^p from an iteration history. |
| `verify_root_with_sign_change.py` | A sign-change check around an approximation gives a hard error bound on the distance to the true root. |
| `estimate_error_a_posteriori_fixed_point.py` | Scalar Banach a-posteriori error bound from the last step size and the Lipschitz constant α. |

### Functions — visualization — `HM1/functions/visualization/`

| Script | What it does |
| --- | --- |
| `plot_function_generic.py` | Generic multi-function plotter (linear / log / semilog) for sympy-parsed function strings. |
| `plot_fixed_point_iteration.py` | Plots F(x) − x and marks fixed points with attractive / repulsive labels. |
| `plot_newton_iteration.py` | Plots f(x), marks its zeros, runs Newton (or simplified Newton) from x0, and shows the iteration path. |
| `plot_secant_iteration.py` | Plots f(x), marks its zeros, and runs the secant method from (x0, x1). |
| `plot_polynomial_instability_demo.py` | Demonstrates catastrophic cancellation by comparing (x−2)^7 expanded vs. compact near x = 2. |
| `plot_condition_number_of_function.py` | Semilog plot of κ_f(x) with a threshold line; reports where the function is well-conditioned and the limit behavior. |

### Matrices — basic operations — `HM1/matrices/basic_operations/`

| Script | What it does |
| --- | --- |
| `invert_matrix.py` | Computes A⁻¹ via numpy.linalg.inv. |
| `check_orthogonal_matrix.py` | Tests whether A · Aᵀ ≈ I. |
| `check_diagonally_dominant.py` | Row- and column-sum criteria for strict diagonal dominance. |
| `compute_matrix_norm_and_condition.py` | Prints the 1-, 2-, and ∞-norms of A and, for square A, the matching condition numbers. |
| `compute_vector_norm_and_condition.py` | Prints the 1-, 2-, and ∞-norms of a vector. |

### Matrices — decompositions — `HM1/matrices/decompositions/`

| Script | What it does |
| --- | --- |
| `decompose_with_lr.py` | LR decomposition without pivoting; also solves Ax = b. |
| `decompose_with_plr.py` | PLR decomposition with row pivoting; also solves Ax = b. |
| `decompose_with_qr.py` | QR via numpy.linalg.qr; also solves Ax = b. |
| `decompose_with_qr_householder.py` | QR via hand-written Householder reflections, with optional step-by-step debug. |

### Matrices — linear systems — `HM1/matrices/linear_systems/`

| Script | What it does |
| --- | --- |
| `solve_linear_system_with_gauss.py` | Gauss elimination with column pivoting; returns U, det(A), and x. |
| `solve_linear_system_with_lr.py` | Solves Ax = b via LR decomposition (forward + back substitution). |
| `solve_linear_system_with_plr.py` | Solves Ax = b via PLR with row pivoting. |
| `solve_linear_system_with_cramer.py` | Cramer's rule (for small systems). |
| `estimate_arithmetic_complexity.py` | Rule-of-thumb operation counts (2/3)n³ and 2n² for Gauss / triangular solves. |

### Matrices — iterative solvers — `HM1/matrices/iterative_methods/`

| Script | What it does |
| --- | --- |
| `solve_with_jacobi.py` | Jacobi iteration with optional a-priori / a-posteriori stopping. |
| `solve_with_gauss_seidel.py` | Gauss-Seidel iteration with optional a-priori / a-posteriori stopping. |
| `solve_with_relaxation.py` | Relaxation / JOR variant of Jacobi with a parameter ω; compares its convergence to plain Jacobi. |

### Matrices — error estimation — `HM1/matrices/error_estimation/`

| Script | What it does |
| --- | --- |
| `estimate_error_general.py` | Bound on the relative error of x for perturbations in both A and b; supports a symbolic ε. |
| `estimate_error_right_side.py` | Bound on the relative error of x when only b is perturbed (A exact): symbolic / explicit / percent / worst-case. |

### Interpolation — `HM1/interpolation/`

| Script | What it does |
| --- | --- |
| `interpolate_with_cubic_polynomial.py` | Builds the unique cubic through four (year, value) pairs via a Vandermonde system and evaluates it at a target year. |

### Eigenvalue problems — full analysis — `HM1/eigenvalue_problems/`

| Script | What it does |
| --- | --- |
| `compute_complete_eigen_analysis.py` | Full eigen-analysis: eigenvalues, eigenvectors, and algebraic + geometric multiplicities, with row-echelon debug output. |

### Eigenvalue problems — basics — `HM1/eigenvalue_problems/basic_operations/`

| Script | What it does |
| --- | --- |
| `compute_eigenvalues_only.py` | Eigenvalues only, via numpy.linalg.eigvals. |
| `compute_algebraic_multiplicity.py` | Counts algebraic multiplicities after rounding. |
| `compute_characteristic_polynomial.py` | Symbolic p(λ) = det(A − λI), expanded and factorized. |
| `find_parameter_for_repeated_eigenvalue.py` | Solves (via the characteristic-polynomial discriminant) for the parameter giving a repeated eigenvalue, and checks diagonalizability. |

### Eigenvalue problems — spectral properties — `HM1/eigenvalue_problems/spectral_properties/`

| Script | What it does |
| --- | --- |
| `compute_spectral_radius.py` | Spectral radius ρ(B) for both the Jacobi and Gauss-Seidel iteration matrices. |
| `compute_spectrum.py` | Prints the full spectrum and its cardinality. |

### Eigenvalue problems — diagonalization — `HM1/eigenvalue_problems/diagonalization/`

| Script | What it does |
| --- | --- |
| `build_diagonalization_matrix_t.py` | Builds T from the eigenvectors (auto) or checks a given T (manual). |

### Eigenvalue problems — iterative methods — `HM1/eigenvalue_problems/iterative_methods/`

| Script | What it does |
| --- | --- |
| `compute_eigenvalues_with_qr_iteration.py` | QR iteration (fixed iters or tolerance, optional Rayleigh shift); reads eigenvalues off the (quasi-)upper triangle. |
| `compute_dominant_eigenvalue_with_power_method.py` | Von-Mises / power method for the dominant eigenvalue, plus the Rayleigh quotient. |

### Eigenvalue problems — visualization — `HM1/eigenvalue_problems/visualization/`

| Script | What it does |
| --- | --- |
| `plot_power_method_convergence.py` | Runs the von-Mises iteration and plots the eigenvalue error vs. iteration (semilog) to show convergence speed. |

### Complex dynamics — `HM1/complex_dynamics/`

| Script | What it does |
| --- | --- |
| `plot_mandelbrot_set.py` | Renders the Mandelbrot set as iterations-until-escape (Z_{n+1} = Z_n² + C). |

## HM2 — Höhere Mathematik 2

### Nonlinear systems — Newton methods — `HM2/nonlinear_systems/newton_methods/`

| Script | What it does |
| --- | --- |
| `solve_nonlinear_system_newton.py` | Solves a nonlinear system with the standard Newton iteration. |
| `solve_nonlinear_system_simplified_newton.py` | Solves a nonlinear system with the simplified Newton iteration. |
| `solve_nonlinear_system_frozen_jacobian_newton.py` | Solves a nonlinear system with a frozen-Jacobian Newton variant. |
| `compare_newton_methods.py` | Compares the standard, simplified, and frozen-Jacobian Newton variants. |
| `solve_newton_with_iteration_norms.py` | Prints the residual and step norms during Newton iteration. |
| `solve_all_newton_solutions.py` | Runs Newton from several start vectors and lists all solutions found. |
| `solve_3d_nonlinear_system_damped_newton.py` | Solves a 3D nonlinear system with damped Newton and prints residual / step norms per iteration. |
| `fit_soil_pressure_model_damped_newton.py` | Fits the soil-pressure model via damped Newton from measurement points, then bisects for the minimum disc radius. |
| `minimize_function_with_newton.py` | Minimizes a scalar g(x, y, …) by solving ∇g = 0 with Newton for systems; confirms the minimum via the Hessian eigenvalues. |

### Nonlinear systems — linearization — `HM2/nonlinear_systems/linearization/`

| Script | What it does |
| --- | --- |
| `calculate_jacobian_matrix.py` | Computes symbolic Jacobian matrices and evaluates them at a point. |
| `linearize_multivariable_function.py` | Linearizes a multivariable function at a chosen point. |
| `compute_partial_derivatives_and_linearize.py` | Computes symbolic first-order partial derivatives and linearizes multivariable functions via the Jacobian. |

### Nonlinear systems — visualization — `HM2/nonlinear_systems/visualization/`

| Script | What it does |
| --- | --- |
| `plot_implicit_equations.py` | Plots implicit equations with sympy.plot_implicit(). |
| `plot_nonlinear_system_contours.py` | Visualizes the zero contours of a nonlinear 2D system. |
| `plot_wave_equation_wireframe.py` | Plots one or more wave-equation-style functions as 3D wireframes. |
| `visualize_2d_scalar_function.py` | Visualizes a scalar function of two variables as surface, wireframe, and contour plots. |
| `plot_projectile_range_and_ideal_gas.py` | Plots projectile range and ideal-gas relations as 3D wireframes, surfaces, and contours. |

### Interpolation & least squares — polynomial interpolation — `HM2/interpolation_and_least_squares/polynomial_interpolation/`

| Script | What it does |
| --- | --- |
| `interpolate_value_with_lagrange.py` | Interpolates y-values at given x-points using the Lagrange formula (own implementation). |
| `interpolate_with_polyfit_polyval.py` | Polynomial interpolation via numpy.polyfit / polyval, with optional mean-centering for better conditioning. |
| `compare_lagrange_vs_polyfit.py` | Overlays the own Lagrange interpolation and numpy.polyfit for the same data. |

### Interpolation & least squares — splines — `HM2/interpolation_and_least_squares/spline_interpolation/`

| Script | What it does |
| --- | --- |
| `interpolate_with_natural_cubic_spline.py` | Natural cubic spline implemented from scratch (Ch. 6.2.3). |
| `interpolate_with_scipy_cubic_spline.py` | Uses scipy.interpolate.CubicSpline with selectable boundary conditions. |
| `compare_cubic_spline_methods.py` | Compares the own spline, the scipy spline, and a high-degree polynomial. |
| `evaluate_spline_derivatives.py` | Evaluates a natural cubic spline's value, velocity (S′), and acceleration (S″) at a point, and plots S′(t). |

### Interpolation & least squares — linear least squares — `HM2/interpolation_and_least_squares/linear_least_squares/`

| Script | What it does |
| --- | --- |
| `fit_polynomial_with_normal_equations.py` | Polynomial least-squares fit via normal equations, QR, or numpy.polyfit, with condition numbers. |
| `fit_multivariate_linear_regression.py` | Multivariate linear regression with several feature columns plus an intercept. |
| `fit_data_via_log_linearization.py` | Linearizes exponential models by taking logarithms; supports extrapolation. |
| `fit_with_linear_basis_functions.py` | General linear least squares with arbitrary basis functions (e.g. 1/x, x, x²); optional reciprocal / log y-transform; reports the error functional. |

### Interpolation & least squares — nonlinear least squares — `HM2/interpolation_and_least_squares/nonlinear_least_squares/`

| Script | What it does |
| --- | --- |
| `fit_with_gauss_newton.py` | Undamped Gauss-Newton with a QR-based linear least-squares step. |
| `fit_with_damped_gauss_newton.py` | Damped Gauss-Newton with step factor δ/2^p. |
| `compare_gauss_newton_methods.py` | Compares damped vs. undamped Gauss-Newton from several start vectors. |
| `fit_with_scipy_optimize_fmin.py` | Direct minimization of the error functional with scipy.optimize.fmin (Nelder-Mead). |

### Numerical differentiation — `HM2/numerical_differentiation/`

| Script | What it does |
| --- | --- |
| `differentiate_numerically_with_extrapolation.py` | Forward / backward / central difference plus Richardson ("h-algorithm") extrapolation table for f′(x₀). |

### Numerical integration — Newton-Cotes — `HM2/numerical_integration/newton_cotes/`

| Script | What it does |
| --- | --- |
| `integrate_with_summed_rectangle_rule.py` | Summed rectangle / midpoint rule for a callable f. |
| `integrate_with_summed_trapezoidal_rule.py` | Summed trapezoidal rule for equidistant subintervals. |
| `integrate_with_summed_simpson_rule.py` | Summed Simpson rule with mid-of-subinterval evaluations. |
| `integrate_with_trapezoidal_rule_non_equidistant.py` | Trapezoidal rule for tabulated data with non-equidistant x-values. |

### Numerical integration — Gauss quadrature — `HM2/numerical_integration/gauss_quadrature/`

| Script | What it does |
| --- | --- |
| `integrate_with_gauss_formulas.py` | Gauss quadrature with 1, 2, or 3 nodes (selectable). |

### Numerical integration — Romberg — `HM2/numerical_integration/romberg_extrapolation/`

| Script | What it does |
| --- | --- |
| `integrate_with_romberg_extrapolation.py` | Builds the full Romberg triangle and returns T(0, m). |
| `integrate_romberg_from_data.py` | Romberg extrapolation built straight from a table of equidistant samples (no callable f), using strides 2ᵏ. |

### Numerical integration — error & comparison — `HM2/numerical_integration/error_and_comparison/`

| Script | What it does |
| --- | --- |
| `estimate_required_step_size.py` | Computes the largest h (and smallest n) for a given tolerance using the error bounds from Thm. 7.1. |
| `compare_quadrature_methods.py` | Runs every quadrature method on the same integral and tabulates errors against scipy.integrate.quad. |

### ODEs — direction fields — `HM2/ordinary_differential_equations/direction_fields/`

| Script | What it does |
| --- | --- |
| `plot_direction_field.py` | Plots the direction field of a first-order ODE y′ = f(x, y) using meshgrid + quiver. |

### ODEs — single-step methods — `HM2/ordinary_differential_equations/single_step_methods/`

| Script | What it does |
| --- | --- |
| `solve_ode_with_euler.py` | Classical Euler method (order 1) with an iteration log and optional exact-solution comparison. |
| `solve_ode_with_midpoint.py` | Midpoint method (order 2) with a half-step slope evaluation. |
| `solve_ode_with_modified_euler.py` | Modified Euler / Heun (order 2), averaging the predictor and corrector slopes. |
| `solve_ode_with_classical_runge_kutta.py` | Classical four-stage Runge-Kutta (order 4). |
| `solve_ode_with_custom_runge_kutta.py` | Generic explicit s-stage Runge-Kutta from a user-supplied Butcher tableau (c, A, b). |
| `compare_ode_single_step_methods.py` | Runs Euler, midpoint, modified Euler, and RK4 on the same problem and plots the solutions plus the global error. |

### ODEs — error analysis — `HM2/ordinary_differential_equations/error_analysis/`

| Script | What it does |
| --- | --- |
| `determine_ode_convergence_order.py` | Runs a single-step method (Euler / midpoint / RK4 / custom Butcher) at several step sizes, plots the endpoint error vs. h (log-log), and infers the integer convergence order p. |

### ODEs — higher-order systems — `HM2/ordinary_differential_equations/higher_order_systems/`

| Script | What it does |
| --- | --- |
| `reduce_higher_order_ode_to_system.py` | Reduces a k-th order ODE to a first-order system via sympy substitution. |
| `solve_ode_system_with_midpoint.py` | Solves a vector-valued first-order system with the midpoint method. |
| `solve_ode_system_with_single_step_methods.py` | Vector / system solver with a selectable single-step method (Euler / midpoint / mod-Euler / RK4); plots each component plus the acceleration, and in compare mode the relative deviation of two methods (semilog). |

### ODEs — applications — `HM2/ordinary_differential_equations/applications/`

| Script | What it does |
| --- | --- |
| `solve_rocket_ascent_via_integration.py` | Computes v(t) and h(t) from a given a(t) via a cumulative trapezoidal sum, with optional comparison to the analytical solution. |
| `solve_ode_with_bounce_events.py` | Classical RK4 on a 2nd-order system with event-based velocity reflection (reflects when x < 0); counts the bounces. |
