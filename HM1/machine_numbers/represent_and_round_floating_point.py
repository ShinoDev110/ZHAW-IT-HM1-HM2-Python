# ============================================================
# TOPIC: Maschinenzahlen — Runden vs. Abschneiden + Auslöschung
# DESCRIPTION:
# Stellt eine reelle Zahl als normalisierte Gleitpunktzahl 0.m1..mn · B^e
# mit n Mantissenstellen dar, einmal durch Abschneiden und einmal durch
# Runden, und gibt jeweils absoluten und relativen Fehler an. Optional wird
# für zwei Zahlen die Differenz p - q gebildet, um Auslöschung sichtbar zu
# machen (relativer Fehler der Differenz für beide Rundungsarten).
# USE WHEN:
# Wenn der Darstellungsfehler einer Zahl in endlicher Mantissenlänge gesucht
# ist oder gezeigt werden soll, dass Runden bei Subtraktion besser ist.
# EXAMPLE:
# p = 9890.9, q = 9890.1, Basis 10, n = 4 -> p-q (abschneiden) = 0 (100% Fehler),
# p-q (runden) = 1 (25% Fehler). Alternativ: x = sqrt(3), Basis 2, n = 5.
# ============================================================

from math import floor, log

# ============================================================
# PART 1 — Inputs
# ============================================================
value_1 = 9890.9   # erste Zahl p
value_2 = 9890.1   # zweite Zahl q (nur für mode = "pair")
basis   = 10       # Basis B des Gleitpunktsystems (2 oder 10 typisch)
n_mant  = 4        # Anzahl Mantissenstellen n der normalisierten Darstellung

# Alternatives Beispiel (Aufgabe 2014.1b): sqrt(3) in 5 Binärstellen
# from math import sqrt; value_1 = sqrt(3); basis = 2; n_mant = 5; mode = "single"

# ============================================================
# PART 2 — Method selection
# ============================================================
# mode:
#   "single" -> nur value_1 darstellen (Runden vs. Abschneiden + Fehler)
#   "pair"   -> zusätzlich Differenz value_1 - value_2 (Auslöschung zeigen)
mode = "pair"

# ============================================================
# PART 3 — Implementation
# ============================================================
def _to_base_digits(m_int, base, n):
    # n Ziffern (höchstwertig zuerst) der Zahl m_int zur Basis base
    digits = []
    x = m_int
    for _ in range(n):
        digits.append(x % base)
        x //= base
    digits.reverse()
    return digits

def _digit_str(digits):
    return "".join(str(d) for d in digits)

def _normalize(value, base):
    # value = sign * mantisse * base^e  mit  1/base <= mantisse < 1
    if value == 0:
        return 0, 0.0, 0
    sign = -1 if value < 0 else 1
    a = abs(value)
    e = floor(log(a) / log(base)) + 1
    mant = a / base ** e
    while mant >= 1.0:            # Rundungsschutz an den Grenzen
        mant /= base; e += 1
    while mant < 1.0 / base:
        mant *= base; e -= 1
    return sign, mant, e

def _represent(value, base, n, rounding):
    # gibt (dargestellter Wert, sign, Mantissenziffern, Exponent) zurück
    sign, mant, e = _normalize(value, base)
    if sign == 0:
        return 0.0, 0, [0] * n, 0
    scaled = mant * base ** n
    if rounding == "round":
        m_int = floor(scaled + 0.5)
    else:  # "truncate"
        m_int = floor(scaled)
    if m_int >= base ** n:        # Übertrag, z.B. 0.999 -> 1.000
        m_int //= base
        e += 1
    rep_value = sign * m_int * base ** (e - n)
    digits = _to_base_digits(m_int, base, n)
    return rep_value, sign, digits, e

def _print_value_report(name, value, base, n):
    print(f"--- {name} = {value} (exakt) ---")
    results = {}
    for rounding in ("truncate", "round"):
        rep, sign, digits, e = _represent(value, base, n, rounding)
        abs_err = abs(rep - value)
        rel_err = abs_err / abs(value) if value != 0 else 0.0
        vz = "-" if sign < 0 else ""
        label = "Abschneiden" if rounding == "truncate" else "Runden     "
        print(f"  {label}: {vz}0.{_digit_str(digits)} · {base}^{e} = {rep}")
        print(f"               abs. Fehler = {abs_err:.6g}, rel. Fehler = {rel_err:.6g}")
        results[rounding] = rep
    print()
    return results

def represent_and_round_floating_point(mode, value_1, value_2, base, n):
    print("============================================================")
    print(f"Normalisierte Darstellung 0.m1..m{n} · {base}^e  (n = {n}, Basis {base})")
    print("============================================================\n")

    rep1 = _print_value_report("p (value_1)", value_1, base, n)
    if mode == "single":
        return rep1

    if mode != "pair":
        raise ValueError(f"Unbekannter mode: {mode!r}")

    rep2 = _print_value_report("q (value_2)", value_2, base, n)

    exact_diff = value_1 - value_2
    print("--- Differenz p - q (Auslöschung) ---")
    print(f"  exakt:        p - q = {exact_diff}")
    best = None
    for rounding, label in (("truncate", "Abschneiden"), ("round", "Runden")):
        diff = rep1[rounding] - rep2[rounding]
        rel = abs(diff - exact_diff) / abs(exact_diff) if exact_diff != 0 else float("inf")
        print(f"  {label:>11}: p - q = {diff}, rel. Fehler = {rel:.6g}")
        if best is None or rel < best[1]:
            best = (label, rel)
    print(f"\nFazit: '{best[0]}' liefert den kleineren relativen Fehler der Differenz")
    print("(Runden ist hier besser, da Abschneiden zu stärkerer Auslöschung führt).")
    return rep1, rep2

# ============================================================
# PART 4 — Call
# ============================================================
represent_and_round_floating_point(mode, value_1, value_2, basis, n_mant)
