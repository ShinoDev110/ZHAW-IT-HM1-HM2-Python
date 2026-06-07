# ============================================================
# TOPIC: Error estimation — only perturbed right-hand side b
# DESCRIPTION:
# Compares x = A^-1 b and x̃ = A^-1 b̃ (A exact). Provides
#   - actual absolute/relative error
#   - upper bounds ||A^-1||·||Deltab||  and  cond(A)·||Deltab||/||b||
#   - max eps so that ||x-x̃||/||x|| <= max_rel_error
# incl. symbolic eps solution with sympy.
# USE WHEN:
# When only the right-hand side (measurement data) is noisy, but A is exact.
# EXAMPLE:
# A = [[240,120,80],[60,180,170],[60,90,500]], b = [3080, 4070, 5030],
# b̃ = b + eps·[0,0,1] (symbolic).
# ============================================================

import numpy as np
import sympy as sp

np.set_printoptions(precision=6, suppress=True)

# ============================================================
# PART 1 — Inputs
# ============================================================
A = np.array([[240.0, 120.0,  80.0],
              [ 60.0, 180.0, 170.0],
              [ 60.0,  90.0, 500.0]])

b = np.array([3080.0,
              4070.0,
              5030.0])

norm          = np.inf
max_rel_error = 0.01
debug         = True

# ============================================================
# PART 2 — Method selection
# ============================================================
# block:
#   "symbolic"      -> b̃ = b + eps v (sympy), solves eps inequality
#   "explicit"      -> b̃ given explicitly (see b_tilde below)
#   "percent"       -> b̃ = b · b_factor (e.g. 1.05)
#   "worst_case"    -> b̃ = b + (eps_max/||v||) · v  (worst case in direction v)
block = "symbolic"

# numerical (for "explicit" / "percent" / "worst_case"):
b_tilde_explicit = np.array([6.0, -1.0, 0.0])     # only for "explicit"
b_factor         = 1.05                            # only for "percent"
v_direction      = np.array([0.0, 0.0, 1.0])       # only for "worst_case"

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
    return float(np.linalg.norm(np.asarray(v, dtype=float), ord=norm))

def _solve(A, b):
    return np.linalg.solve(np.asarray(A, dtype=float), np.asarray(b, dtype=float))

def _actual_errors(A, b, b_t, norm, debug=False):
    ns = _convert_norm(norm)
    x  = _solve(A, b)
    xt = _solve(A, b_t)
    abs_e = float(np.linalg.norm(x - xt, ord=norm))
    rel_e = abs_e / float(np.linalg.norm(x, ord=norm))
    if debug:
        print("---------- Actual errors ----------")
        print(f"x  = {x}")
        print(f"x̃  = {xt}")
        print(f"||x̃ - x||_{ns} = {abs_e}")
        print(f"||x̃ - x||_{ns} / ||x||_{ns} = {rel_e}\n")
    return abs_e, rel_e

def _estimated_bounds(A, b, b_t, norm, debug=False):
    ns = _convert_norm(norm)
    A_inv      = np.linalg.inv(np.asarray(A, dtype=float))
    A_inv_norm = float(np.linalg.norm(A_inv, ord=norm))
    delta_b      = np.asarray(b, dtype=object) - np.asarray(b_t, dtype=object)
    delta_b_norm = _vector_norm(delta_b, norm)
    b_norm       = _vector_norm(b, norm)
    condA = float(np.linalg.cond(np.asarray(A, dtype=float), p=norm))
    abs_bound = A_inv_norm * delta_b_norm
    rel_bound = condA * (delta_b_norm / b_norm)
    if debug:
        print("---------- Estimated bounds ----------")
        print(f"||A^-1||_{ns} = {A_inv_norm}")
        print(f"||b - b̃||_{ns} = {delta_b_norm}")
        print(f"Bound abs: ||x - x̃|| <= ||A^-1||·||Deltab|| = {abs_bound}")
        print(f"Bound rel: ||x - x̃||/||x|| <= cond(A)·||Deltab||/||b|| = {rel_bound}\n")
    return abs_bound, rel_bound

def _max_eps(A, b, max_rel, norm, debug=False):
    ns = _convert_norm(norm)
    b_norm = float(np.linalg.norm(np.asarray(b, dtype=float), ord=norm))
    condA  = float(np.linalg.cond(np.asarray(A, dtype=float), p=norm))
    eps_max = (max_rel * b_norm) / condA
    if debug:
        print("---------- Max epsilon (||b - b̃|| bound) ----------")
        print(f"cond(A)_{ns} = {condA}")
        print(f"||b||_{ns}   = {b_norm}")
        print(f"eps_max     = {eps_max}\n")
    return eps_max

def estimate_error_right_side(block, A, b, norm, max_rel,
                              b_tilde_explicit, b_factor, v_direction, debug=False):
    if block == "symbolic":
        eps = sp.Symbol("eps", real=True)
        v   = np.array(v_direction, dtype=object)
        b_tilde = b.astype(object) + eps * v
        delta_b = b.astype(object) - b_tilde
        eps_max = _max_eps(A, b, max_rel, norm, debug)
        delta_norm_expr = _vector_norm(delta_b, norm)
        print("b_tilde =", b_tilde)
        print(f"||b - b̃||_{_convert_norm(norm)} =", delta_norm_expr)
        print("Constraint: ||b - b̃|| <=", eps_max)
        print("eps solution set:", sp.solve_univariate_inequality(delta_norm_expr <= eps_max, eps), "\n")
        return

    if block == "explicit":
        b_tilde = b_tilde_explicit
    elif block == "percent":
        b_tilde = b * b_factor
    elif block == "worst_case":
        eps_max = _max_eps(A, b, max_rel, norm, debug)
        v = np.asarray(v_direction, dtype=float)
        b_tilde = b + (eps_max / float(np.linalg.norm(v, ord=norm))) * v
    else:
        raise ValueError(f"Unknown method: {block!r}")

    _actual_errors(A, b, b_tilde, norm, debug)
    _estimated_bounds(A, b, b_tilde, norm, debug)

# ============================================================
# PART 4 — Call
# ============================================================
estimate_error_right_side(block, A, b, norm, max_rel_error,
                          b_tilde_explicit, b_factor, v_direction, debug)
