# ============================================================
# TOPIC: Functions — error propagation via the condition number
# DESCRIPTION:
# Uses the condition number K_f(x) = |x f'(x)/f(x)| and the relation
# (rel. error of f) ~= K_f(x) · (rel. error of x) in three directions:
#   forward  -> estimate output error from input error
#   backward -> max. allowable input error for a target output error
#   actual   -> compute actual rel. input/output error from x0, x̃0
#               and compare with K_f(x0) (is K a good approximation?)
# USE WHEN:
# When assessing how accurate an input value must be or how strongly
# an input error affects the function value.
# EXAMPLE:
# f(x) = x^2 sin(x), x0 = pi/3: max. absolute error of x0 so that the
# relative error of f(x0) is at most 10% (mode = "backward").
# ============================================================

import sympy as sp

# ============================================================
# PART 1 — Inputs
# ============================================================
function         = "x**2 * sin(x)"  # f(x) as string (sympy syntax)
x0               = "pi/3"           # evaluation point (number or expression, e.g. "pi/3")
x0_tilde         = "0.9991"         # error-affected value x̃0 (mode = "actual" only)
input_rel        = 0.05             # rel. input error (mode = "forward" only)
target_output_rel = 0.10            # desired max. rel. output error ("backward" only)

# ============================================================
# PART 2 — Method selection
# ============================================================
# mode:
#   "forward"  -> output error ~= K · input_rel
#   "backward" -> max. input error = target_output_rel / K (abs. = · |x0|)
#   "actual"   -> actual rel. errors from x0 and x0_tilde, comparison with K
mode = "backward"

# ============================================================
# PART 3 — Implementation
# ============================================================
def _build_symbols(function):
    x = sp.Symbol("x")
    f = sp.sympify(function, locals={"x": x})
    df = sp.diff(f, x)
    K = sp.Abs(x * df / f)
    return x, f, df, K

def _num(expr_or_value, x, x_val):
    return float(sp.sympify(expr_or_value).subs(x, x_val).evalf())

def estimate_error_with_condition_number(mode, function, x0, x0_tilde,
                                         input_rel, target_output_rel):
    x, f, df, K = _build_symbols(function)
    x0_val = sp.sympify(x0).evalf()

    print("============================================================")
    print("Error propagation via the condition number")
    print("============================================================")
    print(f"f(x)   = {f}")
    print(f"f'(x)  = {sp.simplify(df)}")
    print(f"K_f(x) = |x f'(x)/f(x)| = {K}")
    fx0 = float(f.subs(x, x0_val).evalf())
    K0 = float(K.subs(x, x0_val).evalf())
    print(f"x0     = {x0} = {float(x0_val):.6g}")
    print(f"f(x0)  = {fx0:.6g}")
    print(f"K_f(x0) = {K0:.6g}\n")

    if mode == "forward":
        out_rel = K0 * input_rel
        print(f"-- forward (rel. input error = {input_rel:.6g})")
        print(f"rel. output error ~= K · input_rel = {out_rel:.6g}  ({out_rel*100:.4g} %)")
        print(f"abs. output error ~= {out_rel * abs(fx0):.6g}")
        return out_rel

    if mode == "backward":
        if K0 == 0:
            print("K_f(x0) = 0 -> arbitrarily large input error is permissible.")
            return float("inf")
        in_rel_max = target_output_rel / K0
        in_abs_max = in_rel_max * abs(float(x0_val))
        print(f"-- backward (target: rel. output error <= {target_output_rel:.6g})")
        print(f"max. rel. input error = target/K = {in_rel_max:.6g}  ({in_rel_max*100:.4g} %)")
        print(f"max. abs. input error = rel · |x0| = {in_abs_max:.6g}")
        return in_rel_max, in_abs_max

    if mode == "actual":
        xt_val = sp.sympify(x0_tilde).evalf()
        fxt = float(f.subs(x, xt_val).evalf())
        in_rel = abs(float(xt_val) - float(x0_val)) / abs(float(x0_val))
        out_rel = abs(fxt - fx0) / abs(fx0)
        quotient = out_rel / in_rel if in_rel != 0 else float("inf")
        print(f"-- actual (x̃0 = {x0_tilde} = {float(xt_val):.10g})")
        print(f"f(x̃0) = {fxt:.10g}")
        print(f"actual rel. input error  = |x̃0-x0|/|x0|     = {in_rel:.6g}")
        print(f"actual rel. output error = |f(x̃0)-f(x0)|/|f(x0)| = {out_rel:.6g}")
        print(f"quotient (output/input)  = {quotient:.6g}")
        print(f"condition number K_f(x0) = {K0:.6g}")
        close = abs(quotient - K0) <= 0.05 * max(abs(K0), 1e-30)
        print(f"=> K_f(x0) is {'a realistic' if close else 'only a rough'} "
              f"approximation of the error propagation.")
        return in_rel, out_rel, quotient, K0

    raise ValueError(f"Unknown mode: {mode!r}")

# ============================================================
# PART 4 — Call
# ============================================================
estimate_error_with_condition_number(mode, function, x0, x0_tilde,
                                     input_rel, target_output_rel)
