# ============================================================
# TOPIC: Maschinenzahlen — Maschinengenauigkeit zweier Rechner vergleichen
# DESCRIPTION:
# Berechnet für mehrere Gleitkommaformate die Maschinengenauigkeit
# eps = (B/2) · B^(-n) und bestimmt, welcher Rechner genauer rechnet
# (kleineres eps). Erlaubt unterschiedliche Basen (binär, dezimal, hex, ...).
# USE WHEN:
# Wenn zwei Rechner mit verschiedenen Basen/Mantissenlängen verglichen
# werden sollen ("Welche Maschine rechnet genauer?").
# EXAMPLE:
# 46-stellige Binärarithmetik vs. 14-stellige Dezimalarithmetik
# -> eps = 2^-46 ~= 1.42e-14 < 5·10^-14, also rechnet die Binärmaschine genauer.
# ============================================================

# ============================================================
# PART 1 — Inputs
# ============================================================
# Jede Maschine: Name, Basis B, Anzahl Mantissenstellen n.
maschinen = [
    {"name": "Binär (46-stellig)",   "base": 2,  "mantisse": 46},
    {"name": "Dezimal (14-stellig)", "base": 10, "mantisse": 14},
]

# Weiteres Beispiel (Aufgabe 2020_2.1c): 5-stellig binär vs. 2-stellig hex
# maschinen = [
#     {"name": "Binär (5-stellig)", "base": 2,  "mantisse": 5},
#     {"name": "Hex (2-stellig)",   "base": 16, "mantisse": 2},
# ]

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here: eps = (B/2) · B^(-n) für jede Maschine, dann Vergleich.

# ============================================================
# PART 3 — Implementation
# ============================================================
def _eps(base, mantisse):
    return (base / 2) * base ** (-mantisse)

def compare_machine_precision(maschinen):
    print("============================================================")
    print("Vergleich der Maschinengenauigkeit  eps = (B/2)·B^(-n)")
    print("============================================================")
    results = []
    for m in maschinen:
        eps = _eps(m["base"], m["mantisse"])
        results.append((m["name"], m["base"], m["mantisse"], eps))
        print(f"  {m['name']:<24}: B={m['base']:<3} n={m['mantisse']:<3} "
              f"-> eps = {eps:.6g}")

    eps_values = [r[3] for r in results]
    eps_min = min(eps_values)
    genaueste = [r[0] for r in results if r[3] == eps_min]

    print("\nKleinstes eps => genaueste Maschine.")
    if len(genaueste) == len(results):
        print("=> Alle Maschinen rechnen gleich genau (eps identisch).")
    elif len(genaueste) > 1:
        print(f"=> Gleich genau (kleinstes eps): {', '.join(genaueste)}")
    else:
        print(f"=> Genaueste Maschine: {genaueste[0]} (eps = {eps_min:.6g})")
    return results

# ============================================================
# PART 4 — Call
# ============================================================
compare_machine_precision(maschinen)
