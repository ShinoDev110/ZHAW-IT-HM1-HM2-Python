# ============================================================
# TOPIC: Maschinenzahlen — vollständige Analyse eines Bit-Formats mit Bias
# DESCRIPTION:
# Analysiert ein frei definiertes Gleitkommaformat (Vorzeichen | Exponent
# mit Bias | Mantisse) und liefert in einem Report: Maschinengenauigkeit,
# grössten/kleinsten Exponenten (unter Berücksichtigung des Bias), kleinste/
# grösste positive Maschinenzahl (Wert + Bitmuster) und die Anzahl
# verschiedener Maschinenzahlen (inkl. 0). Annahme: normalisierte Mantisse
# 0.m1..mn mit m1 != 0 (ohne Hidden Bit) bzw. 1.m1..mn (mit Hidden Bit).
# USE WHEN:
# Wenn ein eigener Standard (z.B. "IDDD-643") komplett charakterisiert
# werden soll, statt die Teilwerte einzeln zu berechnen.
# EXAMPLE:
# IDDD-643: 1 Vorzeichenbit, 5 Exponentenbits (Bias 15), 10 Mantissenbits,
# kein Hidden Bit -> eps = 2^-10, e in [-15, 16], x_min = 2^-16,
# x_max = 65472, 32769 verschiedene Maschinenzahlen.
# ============================================================

# ============================================================
# PART 1 — Inputs
# ============================================================
basis          = 2     # Basis B (2 = Dual)
vorzeichen_bit = True  # eigenes Vorzeichenbit vorhanden?
exponent_bits  = 5     # Anzahl Exponentenstellen
bias           = 15    # Bias des Exponenten (gespeicherter Exp - bias = echter Exp)
mantisse_bits  = 10    # Anzahl Mantissenstellen n
hidden_bit     = False # True = 1.m1..mn (IEEE-artig), False = 0.m1..mn mit m1!=0

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here. Alle Kennzahlen werden immer berechnet.
# (Annahme: keine reservierten Exponenten für Inf/NaN — voller Bereich.)

# ============================================================
# PART 3 — Implementation
# ============================================================
def _digits(value, base, width):
    # value als Stellenfolge zur Basis base mit fester Breite (höchstwertig zuerst)
    ds = []
    x = value
    for _ in range(width):
        ds.append(x % base)
        x //= base
    ds.reverse()
    return "".join(str(d) for d in ds)

def analyze_custom_float_format(base, sign_bit, exp_bits, bias, m_bits, hidden_bit):
    print("============================================================")
    print("Analyse eines eigenen Gleitkommaformats")
    print("============================================================")
    total_bits = (1 if sign_bit else 0) + exp_bits + m_bits
    print(f"Format: {'1 Vorzeichen | ' if sign_bit else ''}{exp_bits} Exponent (Bias {bias}) "
          f"| {m_bits} Mantisse  ({total_bits} Bits, Basis {base})")
    print(f"Mantisse: {'1.m1..mn (Hidden Bit)' if hidden_bit else '0.m1..mn mit m1 != 0'}\n")

    # --- Exponentenbereich (gespeicherter Exponent 0 .. base^exp_bits - 1) ---
    e_min = 0 - bias
    e_max = (base ** exp_bits - 1) - bias
    n_exp = e_max - e_min + 1
    print(f"kleinster Exponent e_min = 0 - {bias}                 = {e_min}")
    print(f"grösster  Exponent e_max = ({base}^{exp_bits} - 1) - {bias} = {e_max}")
    print(f"Anzahl Exponenten        = {n_exp}\n")

    # --- Maschinengenauigkeit ---
    if hidden_bit:
        eps = base ** (-m_bits)
        print(f"Maschinengenauigkeit eps = B^(-n) = {base}^(-{m_bits}) = {eps:.6g}\n")
    else:
        eps = (base / 2) * base ** (-m_bits)
        print(f"Maschinengenauigkeit eps = (B/2)·B^(-n) = {eps:.6g}\n")

    # --- kleinste / grösste positive Maschinenzahl ---
    if hidden_bit:
        x_min = base ** e_min                                   # Mantisse 1.0
        x_max = (base - base ** (1 - m_bits)) * base ** e_max   # Mantisse (base - ulp)
        mant_min_digits = "0" * m_bits
        mant_max_digits = str(base - 1) * m_bits
    else:
        x_min = base ** (e_min - 1)                             # Mantisse 0.10..0
        x_max = (1 - base ** (-m_bits)) * base ** e_max         # Mantisse 0.11..1
        mant_min_digits = "1" + "0" * (m_bits - 1)
        mant_max_digits = "1" * m_bits

    sign_str = "0|" if sign_bit else ""
    exp_min_field = _digits(0, base, exp_bits)
    exp_max_field = _digits(base ** exp_bits - 1, base, exp_bits)
    print("kleinste positive Maschinenzahl:")
    print(f"   Bitmuster |{sign_str}{exp_min_field}|{mant_min_digits}|  = {x_min:.6g}")
    print("grösste positive Maschinenzahl:")
    print(f"   Bitmuster |{sign_str}{exp_max_field}|{mant_max_digits}|  = {x_max:.6g}\n")

    # --- Anzahl verschiedener Maschinenzahlen (inkl. 0) ---
    if hidden_bit:
        mant_count = base ** m_bits                 # alle m1..mn frei (führende 1 implizit)
    else:
        mant_count = (base - 1) * base ** (m_bits - 1)   # m1 != 0
    sign_factor = 2 if sign_bit else 1
    total = sign_factor * mant_count * n_exp + 1
    print(f"Anzahl Mantissen (normalisiert) = {mant_count}")
    print(f"Anzahl verschiedener Maschinenzahlen (inkl. 0):")
    print(f"   {sign_factor} · {mant_count} · {n_exp} + 1 = {total}")
    return {"eps": eps, "e_min": e_min, "e_max": e_max,
            "x_min": x_min, "x_max": x_max, "count": total}

# ============================================================
# PART 4 — Call
# ============================================================
analyze_custom_float_format(basis, vorzeichen_bit, exponent_bits, bias,
                            mantisse_bits, hidden_bit)
