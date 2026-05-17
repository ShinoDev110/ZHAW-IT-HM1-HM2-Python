# ============================================================
# TOPIC: Nullstellenverfahren — Fehlerschranke per Vorzeichenwechsel
# DESCRIPTION:
# Prüft, ob im Intervall [x* - r, x* + r] um eine Näherung x* ein
# Vorzeichenwechsel von f vorliegt. Wenn ja, gilt (unter Stetigkeit)
# |x_wahr - x*| ≤ r.
# USE WHEN:
# Wenn nach Newton/Sekanten/etc. eine harte Fehlerschranke ohne
# Ableitungsinformation benötigt wird.
# EXAMPLE:
# f(x) = e^(x^2) + x^-3 - 10, x* ~= 1.65, r = 0.01.
# ============================================================

import math

# ============================================================
# PART 1 — Inputs
# ============================================================
def f(x):
    return math.exp(x * x) + x ** -3 - 10.0

x_approx = 1.65
radius   = 0.01

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here.

# ============================================================
# PART 3 — Implementation
# ============================================================
def verify_root_with_sign_change(f, x_approx, radius):
    links  = x_approx - radius
    rechts = x_approx + radius
    f_l    = f(links)
    f_r    = f(rechts)
    ok     = f_l * f_r <= 0.0
    print(f"f({links}) = {f_l}")
    print(f"f({rechts}) = {f_r}")
    if ok:
        print(f"-> Vorzeichenwechsel: |x_wahr - {x_approx}| <= {radius}")
    else:
        print(f"-> KEIN Vorzeichenwechsel im Intervall [{links}, {rechts}]")
    return ok

# ============================================================
# PART 4 — Call
# ============================================================
verify_root_with_sign_change(f, x_approx, radius)
