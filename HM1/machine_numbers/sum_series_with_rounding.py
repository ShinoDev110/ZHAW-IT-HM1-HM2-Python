# ============================================================
# TOPIC: Maschinenzahlen — Summation in beide Richtungen mit Rundung
# DESCRIPTION:
# Summiert die Reihe Σ 1/i^2 einmal aufwärts (i = 1..n) und einmal
# abwärts (i = n..1) und rundet jede Zwischensumme auf eine feste
# Anzahl Nachkommastellen. Zeigt, wie Summationsreihenfolge das
# Ergebnis bei Gleitkommazahlen beeinflusst.
# USE WHEN:
# Wenn der Einfluss von Rundungsfehlern und Summationsreihenfolge
# demonstriert werden soll (klassisches Skript-Beispiel).
# EXAMPLE:
# n = 300 Glieder, gerundet auf 5 Nachkommastellen.
# ============================================================

# ============================================================
# PART 1 — Inputs
# ============================================================
anzahl_terme     = 300  # n = Obergrenze
rundungsstellen  = 5    # Nachkommastellen je Zwischensumme

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here. Beide Richtungen (aufwärts UND abwärts) werden
# immer berechnet und verglichen.

# ============================================================
# PART 3 — Implementation
# ============================================================
def sum_series_with_rounding(anzahl_terme, rundungsstellen):
    def folgenglied(idx):
        return 1.0 / (idx ** 2)

    summe = 0.0
    for i in range(1, anzahl_terme + 1):
        summe += round(folgenglied(i), rundungsstellen)
        summe = round(summe, rundungsstellen)
    summe_aufwaerts = summe

    summe = 0.0
    for i in range(anzahl_terme, 0, -1):
        summe += round(folgenglied(i), rundungsstellen)
        summe = round(summe, rundungsstellen)
    summe_abwaerts = summe

    print(f"Summe aufwärts (i=1..{anzahl_terme}):  {summe_aufwaerts}")
    print(f"Summe abwärts  (i={anzahl_terme}..1):  {summe_abwaerts}")
    print(f"Differenz:                              {summe_aufwaerts - summe_abwaerts}")
    return summe_aufwaerts, summe_abwaerts

# ============================================================
# PART 4 — Call
# ============================================================
sum_series_with_rounding(anzahl_terme, rundungsstellen)
