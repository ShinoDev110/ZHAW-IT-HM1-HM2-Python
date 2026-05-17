# ============================================================
# TOPIC: Nullstellenverfahren — Konvergenzordnung aus Iterationsfolge
# DESCRIPTION:
# Schätzt aus einer Iterationsfolge {x_k} die Konvergenzordnung p
# und die Konstante C im Modell e_{k+1} ~= C · e_k^p, mit
# e_k = |x_k - x*|. Falls x* unbekannt, wird x* ~= x_letzte verwendet.
# USE WHEN:
# Wenn empirisch unterschieden werden soll, ob ein Verfahren linear,
# quadratisch oder kubisch konvergiert.
# EXAMPLE:
# Newton-Folge xs = [0.5, 0.8, 0.93, 0.991, 0.99988, 1.0] -> p ~= 2.
# ============================================================

import math

# ============================================================
# PART 1 — Inputs
# ============================================================
folge_x         = [0.5, 0.8, 0.93, 0.991, 0.99988, 1.0]
exakte_loesung  = 1.0  # auf None setzen, falls unbekannt -> dann wird letztes x verwendet

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here.

# ============================================================
# PART 3 — Implementation
# ============================================================
def estimate_convergence_order(folge_x, exakte_loesung=None):
    if len(folge_x) < 3:
        print("Folge zu kurz (mind. 3 Werte nötig).")
        return None, None

    if exakte_loesung is None:
        exakte_loesung = folge_x[-1]

    fehler = [abs(x - exakte_loesung) for x in folge_x]
    fehler = [e for e in fehler if e > 0]
    if len(fehler) < 3:
        print("Zu wenige nicht-null Fehler.")
        return None, None

    p_werte = []
    for k in range(1, len(fehler) - 1):
        e_km1, e_k, e_kp1 = fehler[k - 1], fehler[k], fehler[k + 1]
        nenner = math.log(e_k / e_km1)
        zaehler = math.log(e_kp1 / e_k)
        if nenner != 0:
            p_werte.append(zaehler / nenner)

    if not p_werte:
        print("Konnte p nicht schätzen.")
        return None, None

    p_geschaetzt = sum(p_werte) / len(p_werte)
    e_k, e_kp1 = fehler[-2], fehler[-1]
    C_geschaetzt = e_kp1 / (e_k ** p_geschaetzt) if e_k != 0 else None

    print(f"Geschätzte Konvergenzordnung p ~= {p_geschaetzt}")
    print(f"Geschätzte Konstante         C ~= {C_geschaetzt}")
    return p_geschaetzt, C_geschaetzt

# ============================================================
# PART 4 — Call
# ============================================================
estimate_convergence_order(folge_x, exakte_loesung)
