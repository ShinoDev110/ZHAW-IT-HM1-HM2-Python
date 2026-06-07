# ============================================================
# TOPIC: Funktionen — Fehlerfortpflanzung über die Konditionszahl
# DESCRIPTION:
# Nutzt die Konditionszahl K_f(x) = |x f'(x)/f(x)| und die Beziehung
# (rel. Fehler von f) ~= K_f(x) · (rel. Fehler von x) für drei Richtungen:
#   forward  -> aus Eingabefehler den Ausgabefehler schätzen
#   backward -> max. zulässigen Eingabefehler für einen Zielausgabefehler
#   actual   -> tatsächlichen rel. Ein-/Ausgabefehler aus x0, x̃0 berechnen
#               und mit K_f(x0) vergleichen (ist K eine gute Näherung?)
# USE WHEN:
# Wenn beurteilt werden soll, wie genau ein Eingabewert sein muss bzw. wie
# stark sich ein Eingabefehler auf den Funktionswert auswirkt.
# EXAMPLE:
# f(x) = x^2 sin(x), x0 = pi/3: max. absoluter Fehler von x0, damit der
# relative Fehler von f(x0) höchstens 10% beträgt (mode = "backward").
# ============================================================

import sympy as sp

# ============================================================
# PART 1 — Inputs
# ============================================================
funktion         = "x**2 * sin(x)"  # f(x) als String (sympy-Syntax)
x0               = "pi/3"           # Auswertungsstelle (Zahl oder Ausdruck, z.B. "pi/3")
x0_tilde         = "0.9991"         # fehlerbehafteter Wert x̃0 (nur mode = "actual")
input_rel        = 0.05             # rel. Eingabefehler (nur mode = "forward")
target_output_rel = 0.10            # gewünschter max. rel. Ausgabefehler (nur "backward")

# ============================================================
# PART 2 — Method selection
# ============================================================
# mode:
#   "forward"  -> Ausgabefehler ~= K · input_rel
#   "backward" -> max. Eingabefehler = target_output_rel / K (abs. = · |x0|)
#   "actual"   -> tatsächliche rel. Fehler aus x0 und x0_tilde, Vergleich mit K
mode = "backward"

# ============================================================
# PART 3 — Implementation
# ============================================================
def _build_symbols(funktion):
    x = sp.Symbol("x")
    f = sp.sympify(funktion, locals={"x": x})
    df = sp.diff(f, x)
    K = sp.Abs(x * df / f)
    return x, f, df, K

def _num(expr_or_value, x, x_val):
    return float(sp.sympify(expr_or_value).subs(x, x_val).evalf())

def estimate_error_with_condition_number(mode, funktion, x0, x0_tilde,
                                         input_rel, target_output_rel):
    x, f, df, K = _build_symbols(funktion)
    x0_val = sp.sympify(x0).evalf()

    print("============================================================")
    print("Fehlerfortpflanzung über die Konditionszahl")
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
        print(f"-- forward (rel. Eingabefehler = {input_rel:.6g})")
        print(f"rel. Ausgabefehler ~= K · input_rel = {out_rel:.6g}  ({out_rel*100:.4g} %)")
        print(f"abs. Ausgabefehler ~= {out_rel * abs(fx0):.6g}")
        return out_rel

    if mode == "backward":
        if K0 == 0:
            print("K_f(x0) = 0 -> beliebig grosser Eingabefehler zulässig.")
            return float("inf")
        in_rel_max = target_output_rel / K0
        in_abs_max = in_rel_max * abs(float(x0_val))
        print(f"-- backward (Ziel: rel. Ausgabefehler <= {target_output_rel:.6g})")
        print(f"max. rel. Eingabefehler = target/K = {in_rel_max:.6g}  ({in_rel_max*100:.4g} %)")
        print(f"max. abs. Eingabefehler = rel · |x0| = {in_abs_max:.6g}")
        return in_rel_max, in_abs_max

    if mode == "actual":
        xt_val = sp.sympify(x0_tilde).evalf()
        fxt = float(f.subs(x, xt_val).evalf())
        in_rel = abs(float(xt_val) - float(x0_val)) / abs(float(x0_val))
        out_rel = abs(fxt - fx0) / abs(fx0)
        quotient = out_rel / in_rel if in_rel != 0 else float("inf")
        print(f"-- actual (x̃0 = {x0_tilde} = {float(xt_val):.10g})")
        print(f"f(x̃0) = {fxt:.10g}")
        print(f"tatsächlicher rel. Eingabefehler  = |x̃0-x0|/|x0|     = {in_rel:.6g}")
        print(f"tatsächlicher rel. Ausgabefehler  = |f(x̃0)-f(x0)|/|f(x0)| = {out_rel:.6g}")
        print(f"Quotient (Ausgabe/Eingabe)        = {quotient:.6g}")
        print(f"Konditionszahl K_f(x0)            = {K0:.6g}")
        nahe = abs(quotient - K0) <= 0.05 * max(abs(K0), 1e-30)
        print(f"=> K_f(x0) ist {'eine realistische' if nahe else 'NUR eine grobe'} "
              f"Näherung der Fehlerfortpflanzung.")
        return in_rel, out_rel, quotient, K0

    raise ValueError(f"Unbekannter mode: {mode!r}")

# ============================================================
# PART 4 — Call
# ============================================================
estimate_error_with_condition_number(mode, funktion, x0, x0_tilde,
                                     input_rel, target_output_rel)
