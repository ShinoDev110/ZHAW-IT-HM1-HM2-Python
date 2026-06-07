# ============================================================
# TOPIC: Error estimation — perturbed matrix A AND perturbed right-hand side b
# DESCRIPTION:
# Compares the solution x of Ax = b with the solution x̃ of the perturbed
# system Ã x̃ = b̃, computes the observed absolute/relative error
# and an upper error bound via cond(A)/(1 - cond(A)·relA)·(relA + relb).
# Also supports symbolic perturbations for b via sympy.
# USE WHEN:
# When the sensitivity of the linear system solution to combined perturbations
# in the matrix AND the right-hand side should be quantified.
# EXAMPLE:
# A=[[1,0,2],[0,1,0],[1,0,1]], b=[1,1,0]; Ã = A + 1e-7.
# ============================================================

import numpy as np
import numpy.linalg as lin
import sympy as sp

np.set_printoptions(precision=6, suppress=True)

# ============================================================
# PART 1 — Inputs
# ============================================================
A = np.array([[1.0, 0.0, 2.0],
              [0.0, 1.0, 0.0],
              [1.0, 0.0, 1.0]])

b = np.array([1.0, 1.0, 0.0])

norm      = np.inf
max_rel_x = 0.01    # maximum tolerable relative error in x
debug     = True

# Options for perturbed matrix A:
perturbation_A = 1e-7
A_tilde        = A + perturbation_A                              # simple additive perturbation

# Options for perturbed right-hand side b̃ (numerical):
b_tilde = np.array([1.0, -0.2, 0.0])

# Options for symbolic b̃ (uncomment if used):
# eps = sp.Symbol("eps", real=True)
# v = np.array([0.0, 0.0, 1.0], dtype=object)
# b_tilde_sym = b.astype(object) + eps * v
b_tilde_sym = None
eps_sym     = None

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here. Both actual and estimated error are always
# computed (for b_tilde_sym additionally the eps solution).

# ============================================================
# PART 3 — Implementation
# ============================================================
def _convert_norm(norm):
    return "inf" if norm == np.inf else str(norm)

def _is_sympy_obj(x):
    return isinstance(x, sp.Basic)

def _contains_sympy(arr):
    a = np.asarray(arr, dtype=object).ravel()
    return any(_is_sympy_obj(v) for v in a)

def _sympy_vector_norm(v, norm):
    v = sp.Matrix(list(np.asarray(v, dtype=object).ravel()))
    comps = [sp.Abs(v[i, 0]) for i in range(v.rows)]
    if norm == np.inf: return sp.Max(*comps)
    if norm == 1:      return sp.Add(*comps)
    if norm == 2:      return sp.sqrt(sp.Add(*[v[i, 0] ** 2 for i in range(v.rows)]))
    if isinstance(norm, int) and norm >= 1:
        return (sp.Add(*[comps[i] ** norm for i in range(len(comps))])) ** (sp.Rational(1, norm))
    raise ValueError("Unsupported norm.")

def _vector_norm(v, norm):
    if _contains_sympy(v):
        return _sympy_vector_norm(v, norm)
    return float(lin.norm(np.asarray(v, dtype=float), ord=norm))

def _as_1d(v, name="b"):
    v = np.asarray(v, dtype=object)
    if v.ndim == 2:
        if v.shape[1] == 1: v = v[:, 0]
        elif v.shape[0] == 1: v = v[0, :]
        else: raise ValueError(f"{name} must be 1D or (n,1) or (1,n). Got shape {v.shape}.")
    elif v.ndim != 1:
        raise ValueError(f"{name} must be 1D. Got ndim={v.ndim}.")
    return v

def _subs_to_float_array(arr, subs):
    a = np.asarray(arr, dtype=object).ravel()
    out = []
    for v in a:
        if _is_sympy_obj(v):
            out.append(float(sp.N(v.subs(subs))))
        else:
            out.append(float(v))
    return np.array(out, dtype=float).reshape(np.asarray(arr).shape)

def _solve(A, b):
    return lin.solve(np.asarray(A, dtype=float), np.asarray(b, dtype=float))

def _abs_err(A, A_t, b, b_t, norm):
    x  = _solve(A, b)
    xt = _solve(A_t, b_t)
    return float(lin.norm(x - xt, ord=norm)), x, xt

def _rel_err(A, A_t, b, b_t, norm):
    abs_e, x, xt = _abs_err(A, A_t, b, b_t, norm)
    return abs_e / float(lin.norm(x, ord=norm)), x, xt

def _A_rel_error(A, A_t, norm):
    return float(lin.norm(np.asarray(A) - np.asarray(A_t), ord=norm)) / float(lin.norm(np.asarray(A), ord=norm))

def _b_rel_error(b, b_t, norm):
    b_obj   = _as_1d(b, "b").astype(object)
    b_t_obj = _as_1d(b_t, "b_tilde").astype(object)
    return _vector_norm(b_t_obj - b_obj, norm) / _vector_norm(b_obj, norm)

def _bound(A, A_t, b, b_t, norm, debug=False):
    ns = _convert_norm(norm)
    A = np.asarray(A, dtype=float)
    condA = float(lin.cond(A, p=norm))
    relA  = _A_rel_error(A, A_t, norm)
    relb  = _b_rel_error(b, b_t, norm)
    t = condA * relA
    if debug:
        print(f"cond(A)_{ns} = {condA}, relA = {relA}, relb = {relb}, condA*relA = {t}")
    if t >= 1:
        if debug: print("-> error NOT bounded (condA*relA >= 1)")
        return None
    bound = (condA / (1 - t)) * (relA + relb)
    if debug: print(f"-> error bound rel_x <= {bound}\n")
    return bound

def estimate_error_general(A, A_tilde, b, b_tilde, b_tilde_sym, eps_sym,
                           max_rel_x, norm, debug=False):
    # symbolic b̃ -> eps solution
    if b_tilde_sym is not None and eps_sym is not None:
        ns = _convert_norm(norm)
        condA = float(lin.cond(np.asarray(A, dtype=float), p=norm))
        relA = _A_rel_error(A, A_tilde, norm)
        t = condA * relA
        if t >= 1:
            raise ValueError("condA * relA >= 1, bound not defined.")
        b_norm = float(lin.norm(np.asarray(b, dtype=float), ord=norm))
        relb_max = (max_rel_x * (1 - t) / condA) - relA
        if relb_max <= 0:
            raise ValueError("relb_max <= 0; max_rel_x too small for the given A_tilde.")
        eps_max = relb_max * b_norm
        delta_b = b.astype(object) - b_tilde_sym
        delta_norm_expr = _vector_norm(delta_b, norm)
        sol = sp.solve_univariate_inequality(delta_norm_expr <= eps_max, eps_sym)
        print("---------- Symbolic epsilon mode ----------")
        print("b_tilde(eps) =", b_tilde_sym)
        print(f"||b - b_tilde||_{ns} =", delta_norm_expr)
        print("Constraint: ||b-b~|| <=", eps_max)
        print("eps solution set:", sol, "\n")
        b_tilde = _subs_to_float_array(b_tilde_sym, {eps_sym: float(eps_max)})

    # numerical errors
    abs_e, x, xt = _abs_err(A, A_tilde, b, b_tilde, norm)
    rel_e        = abs_e / float(lin.norm(x, ord=norm))
    if debug:
        ns = _convert_norm(norm)
        print(f"x  = {x}")
        print(f"x̃  = {xt}")
        print(f"||x̃ - x||_{ns} = {abs_e}")
        print(f"||x̃ - x||_{ns} / ||x||_{ns} = {rel_e}\n")

    # upper bound
    _bound(A, A_tilde, b, b_tilde, norm, debug)
    return abs_e, rel_e

# ============================================================
# PART 4 — Call
# ============================================================
estimate_error_general(A, A_tilde, b, b_tilde, b_tilde_sym, eps_sym,
                       max_rel_x, norm, debug)
