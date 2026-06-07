# ============================================================
# TOPIC: Eigenvalues — parameter for repeated eigenvalue + diagonalizability
# DESCRIPTION:
# Determines via the characteristic polynomial the parameter value for which a
# parameter-dependent matrix A(c) has an eigenvalue with algebraic multiplicity
# >= 2 (discriminant of char. polynomial = 0). For each solution it is
# checked whether A(c) is diagonalizable (geometric == algebraic
# multiplicity for every eigenvalue).
# USE WHEN:
# When a parameter is to be chosen such that a double eigenvalue
# arises, and it must be decided whether the matrix is then diagonalizable.
# EXAMPLE:
# A = [[30, c], [-13, 4]] -> c = 13 yields double eigenvalue lambda = 17;
# geometric multiplicity 1 < 2 => NOT diagonalizable.
# ============================================================

import sympy as sp

# ============================================================
# PART 1 — Inputs
# ============================================================
c = sp.Symbol("c")                 # parameter symbol

A = sp.Matrix([[30, c],            # parameter-dependent matrix A(c)
               [-13, 4]])

parameter = c                      # solve for this symbol

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here: discriminant(char. polynomial) = 0, solve for parameter,
# then check diagonalizability for each solution.

# ============================================================
# PART 3 — Implementation
# ============================================================
def _diagonalizability_report(A_val):
    n = A_val.shape[0]
    eigen = A_val.eigenvals()  # {eigenvalue: algebraic multiplicity}
    diagonalizable = True
    for ew, alg in eigen.items():
        geo = n - (A_val - ew * sp.eye(n)).rank()
        print(f"   Eigenvalue lambda = {sp.nsimplify(ew)}: "
              f"algebraic multiplicity = {alg}, geometric multiplicity = {geo}")
        if geo != alg:
            diagonalizable = False
    return diagonalizable

def find_parameter_for_repeated_eigenvalue(A, parameter):
    lam = sp.Symbol("lambda")
    n = A.shape[0]
    p = sp.expand((A - lam * sp.eye(n)).det())
    print("============================================================")
    print("Parameter for repeated eigenvalue")
    print("============================================================")
    print("A =")
    sp.pprint(A)
    print(f"\ncharacteristic polynomial p(lambda) = det(A - lambda I) =")
    print(f"   {sp.collect(p, lam)}")

    disc = sp.discriminant(p, lam)
    print(f"\nDiscriminant with respect to lambda = {sp.expand(disc)}")
    solutions = sp.solve(sp.Eq(disc, 0), parameter)
    print(f"Solution(s) for {parameter} (double eigenvalue): {solutions}\n")

    for val in solutions:
        print(f"--- {parameter} = {val} ---")
        A_val = A.subs(parameter, val)
        print("A =")
        sp.pprint(A_val)
        diagonalizable = _diagonalizability_report(A_val)
        print(f"   => A is {'diagonalizable' if diagonalizable else 'NOT diagonalizable'}.\n")
    return solutions

# ============================================================
# PART 4 — Call
# ============================================================
find_parameter_for_repeated_eigenvalue(A, parameter)
