# ============================================================
# TOPIC: Eigenwerte — Parameter für mehrfachen Eigenwert + Diagonalisierbarkeit
# DESCRIPTION:
# Bestimmt über das charakteristische Polynom den Parameterwert, für den eine
# parameterabhängige Matrix A(c) einen Eigenwert mit algebraischer Vielfachheit
# >= 2 besitzt (Diskriminante des char. Polynoms = 0). Für jede Lösung wird
# geprüft, ob A(c) diagonalisierbar ist (geometrische == algebraische
# Vielfachheit für jeden Eigenwert).
# USE WHEN:
# Wenn ein Parameter so gewählt werden soll, dass ein Doppel-Eigenwert
# entsteht, und entschieden werden muss, ob die Matrix dann diagonalisierbar
# ist.
# EXAMPLE:
# A = [[30, c], [-13, 4]] -> c = 13 ergibt Doppel-Eigenwert lambda = 17;
# geometrische Vielfachheit 1 < 2 => NICHT diagonalisierbar.
# ============================================================

import sympy as sp

# ============================================================
# PART 1 — Inputs
# ============================================================
c = sp.Symbol("c")                 # Parameter-Symbol

A = sp.Matrix([[30, c],            # parameterabhängige Matrix A(c)
               [-13, 4]])

parameter = c                      # nach diesem Symbol wird aufgelöst

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here: Diskriminante(char. Polynom) = 0 nach parameter lösen,
# danach für jede Lösung die Diagonalisierbarkeit prüfen.

# ============================================================
# PART 3 — Implementation
# ============================================================
def _diagonalizability_report(A_val):
    n = A_val.shape[0]
    eigen = A_val.eigenvals()  # {Eigenwert: algebraische Vielfachheit}
    diagbar = True
    for ew, alg in eigen.items():
        geo = n - (A_val - ew * sp.eye(n)).rank()
        print(f"   Eigenwert lambda = {sp.nsimplify(ew)}: "
              f"algebraische Vielfachheit = {alg}, geometrische Vielfachheit = {geo}")
        if geo != alg:
            diagbar = False
    return diagbar

def find_parameter_for_repeated_eigenvalue(A, parameter):
    lam = sp.Symbol("lambda")
    n = A.shape[0]
    p = sp.expand((A - lam * sp.eye(n)).det())
    print("============================================================")
    print("Parameter für mehrfachen Eigenwert")
    print("============================================================")
    print("A =")
    sp.pprint(A)
    print(f"\ncharakteristisches Polynom p(lambda) = det(A - lambda I) =")
    print(f"   {sp.collect(p, lam)}")

    disc = sp.discriminant(p, lam)
    print(f"\nDiskriminante bzgl. lambda = {sp.expand(disc)}")
    loesungen = sp.solve(sp.Eq(disc, 0), parameter)
    print(f"Lösung(en) für {parameter} (Doppel-Eigenwert): {loesungen}\n")

    for val in loesungen:
        print(f"--- {parameter} = {val} ---")
        A_val = A.subs(parameter, val)
        print("A =")
        sp.pprint(A_val)
        diagbar = _diagonalizability_report(A_val)
        print(f"   => A ist {'diagonalisierbar' if diagbar else 'NICHT diagonalisierbar'}.\n")
    return loesungen

# ============================================================
# PART 4 — Call
# ============================================================
find_parameter_for_repeated_eigenvalue(A, parameter)
