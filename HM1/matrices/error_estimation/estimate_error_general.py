# ============================================================
# TOPIC: Fehlerabschätzung — gestörte Matrix A UND gestörte rechte Seite b
# DESCRIPTION:
# Vergleicht die Lösung x von Ax = b mit der Lösung x̃ des gestörten
# Systems Ã x̃ = b̃, berechnet beobachteten absoluten/relativen Fehler
# und obere Fehlerschranke via cond(A)/(1 - cond(A)·relA)·(relA + relb).
# Unterstützt zusätzlich symbolische Störungen für b über sympy.
# USE WHEN:
# Wenn die Sensitivität der LGS-Lösung gegenüber gemeinsamen Störungen
# in Matrix UND rechter Seite quantifiziert werden soll.
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
max_rel_x = 0.01    # maximal tolerierbarer relativer Fehler in x
debug     = True

# Optionen für gestörte Matrix Ã:
stoerung_A = 1e-7
A_tilde    = A + stoerung_A                              # einfache additive Störung

# Optionen für gestörte rechte Seite b̃ (numerisch):
b_tilde = np.array([1.0, -0.2, 0.0])

# Optionen für symbolisches b̃ (auskommentieren falls genutzt):
# eps = sp.Symbol("eps", real=True)
# v = np.array([0.0, 0.0, 1.0], dtype=object)
# b_tilde_sym = b.astype(object) + eps * v
b_tilde_sym = None
eps_sym     = None

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here. Es werden immer tatsächlicher und geschätzter
# Fehler berechnet (für b_tilde_sym zusätzlich die eps-Lösung).

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

def _A_rel_fehler(A, A_t, norm):
    return float(lin.norm(np.asarray(A) - np.asarray(A_t), ord=norm)) / float(lin.norm(np.asarray(A), ord=norm))

def _b_rel_fehler(b, b_t, norm):
    b_obj   = _as_1d(b, "b").astype(object)
    b_t_obj = _as_1d(b_t, "b_tilde").astype(object)
    return _vector_norm(b_t_obj - b_obj, norm) / _vector_norm(b_obj, norm)

def _bound(A, A_t, b, b_t, norm, debug=False):
    ns = _convert_norm(norm)
    A = np.asarray(A, dtype=float)
    condA = float(lin.cond(A, p=norm))
    relA  = _A_rel_fehler(A, A_t, norm)
    relb  = _b_rel_fehler(b, b_t, norm)
    t = condA * relA
    if debug:
        print(f"cond(A)_{ns} = {condA}, relA = {relA}, relb = {relb}, condA*relA = {t}")
    if t >= 1:
        if debug: print("-> Fehler NICHT begrenzt (condA*relA >= 1)")
        return None
    bound = (condA / (1 - t)) * (relA + relb)
    if debug: print(f"-> Fehlerschranke rel_x <= {bound}\n")
    return bound

def estimate_error_general(A, A_tilde, b, b_tilde, b_tilde_sym, eps_sym,
                           max_rel_x, norm, debug=False):
    # symbolisches b̃ -> eps-Lösung
    if b_tilde_sym is not None and eps_sym is not None:
        ns = _convert_norm(norm)
        condA = float(lin.cond(np.asarray(A, dtype=float), p=norm))
        relA = _A_rel_fehler(A, A_tilde, norm)
        t = condA * relA
        if t >= 1:
            raise ValueError("condA * relA >= 1, Schranke nicht definiert.")
        b_norm = float(lin.norm(np.asarray(b, dtype=float), ord=norm))
        relb_max = (max_rel_x * (1 - t) / condA) - relA
        if relb_max <= 0:
            raise ValueError("relb_max <= 0; max_rel_x zu klein für gegebene Ã.")
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

    # numerische Fehler
    abs_e, x, xt = _abs_err(A, A_tilde, b, b_tilde, norm)
    rel_e        = abs_e / float(lin.norm(x, ord=norm))
    if debug:
        ns = _convert_norm(norm)
        print(f"x  = {x}")
        print(f"x̃  = {xt}")
        print(f"||x̃ - x||_{ns} = {abs_e}")
        print(f"||x̃ - x||_{ns} / ||x||_{ns} = {rel_e}\n")

    # obere Schranke
    _bound(A, A_tilde, b, b_tilde, norm, debug)
    return abs_e, rel_e

# ============================================================
# PART 4 — Call
# ============================================================
estimate_error_general(A, A_tilde, b, b_tilde, b_tilde_sym, eps_sym,
                       max_rel_x, norm, debug)
