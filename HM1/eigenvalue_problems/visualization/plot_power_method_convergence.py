# ============================================================
# TOPIC: Visualisierung — Konvergenz der von-Mises / Power-Methode
# DESCRIPTION:
# Führt die Vektoriteration v_{k+1} = A v_k / ||A v_k|| aus, schätzt in jedem
# Schritt den dominanten Eigenwert (Rayleigh-Quotient oder Norm-Verhältnis)
# und stellt den absoluten Fehler |lambda_k - lambda_ref| über der Iterationszahl
# halblogarithmisch dar. lambda_ref wird per numpy.linalg.eig bestimmt
# (oder fest vorgegeben).
# USE WHEN:
# Wenn die Konvergenzgeschwindigkeit der Power-Methode visualisiert werden
# soll (typische Aufgabe: "stellen Sie den abs. Fehler halblog. dar").
# EXAMPLE:
# A = [[13,-4],[30,-9]] (Eigenwerte 1 und 3), v0 = [1,0], 40 Iterationen.
# ============================================================

import numpy as np
import numpy.linalg as lin
from matplotlib import pyplot as plt

np.set_printoptions(precision=6, suppress=True)

# ============================================================
# PART 1 — Inputs
# ============================================================
A = np.array([[13.0, -4.0],
              [30.0, -9.0]], dtype=np.float64)   # Matrix

v0 = np.array([1.0, 0.0], dtype=np.float64)        # Startvektor

iters      = 40      # Anzahl Iterationen
lambda_ref = None    # Referenz-Eigenwert (None -> exakt via numpy.linalg.eig)

# ============================================================
# PART 2 — Method selection
# ============================================================
# lambda_estimate:
#   "rayleigh" -> lambda_k = v^T A v / (v^T v)        (vorzeichenbehaftet)
#   "ratio"    -> lambda_k = ||A v_k|| / ||v_k||      (liefert |lambda|)
lambda_estimate = "rayleigh"

# ============================================================
# PART 3 — Implementation
# ============================================================
def _rayleigh(A, v):
    return float((v.T @ (A @ v)) / (v.T @ v))

def _reference_eigenvalue(A, lambda_ref):
    if lambda_ref is not None:
        return float(lambda_ref)
    vals = lin.eigvals(np.asarray(A, dtype=np.float64))
    dominant = vals[np.argmax(np.abs(vals))]
    return float(dominant.real)

def _run_power_method(A, v0, iters, estimate):
    A = np.asarray(A, dtype=np.float64)
    v = np.asarray(v0, dtype=np.float64).reshape(-1)
    v = v / lin.norm(v, ord=2)
    lambdas = []
    for _ in range(iters):
        Av = A @ v
        if estimate == "ratio":
            lam = lin.norm(Av, ord=2) / lin.norm(v, ord=2)
        elif estimate == "rayleigh":
            lam = _rayleigh(A, v)
        else:
            raise ValueError(f"Unbekannte lambda_estimate: {estimate!r}")
        lambdas.append(lam)
        v = Av / lin.norm(Av, ord=2)
    return np.array(lambdas, dtype=np.float64), v

def plot_power_method_convergence(A, v0, iters, lambda_ref, lambda_estimate):
    lam_ref = _reference_eigenvalue(A, lambda_ref)
    print("============================================================")
    print("Konvergenz der Power-Methode")
    print("============================================================")
    print("A =\n", A)
    print(f"Referenz-Eigenwert lambda_ref = {lam_ref}\n")

    lambdas, v_end = _run_power_method(A, v0, iters, lambda_estimate)
    errors = np.abs(lambdas - lam_ref)
    ks = np.arange(1, iters + 1)

    print(f"{'k':>3} | {'lambda_k':>14} | {'|lambda_k - lambda_ref|':>22}")
    print("-" * 48)
    for k, lam, err in zip(ks, lambdas, errors):
        print(f"{k:>3} | {lam:>14.10f} | {err:>22.3e}")
    print(f"\nletzter Eigenwert  lambda ~= {lambdas[-1]:.12g}")
    print(f"letzter Eigenvektor v ~= {v_end}")

    # ---- Plot (halblogarithmisch) ----
    err_plot = np.where(errors > 0, errors, np.nan)  # log kann keine 0
    plt.figure()
    plt.semilogy(ks, err_plot, marker="o", linestyle="-", label="|lambda_k - lambda_ref|")
    plt.xlabel("Iteration k")
    plt.ylabel("absoluter Fehler des Eigenwertes")
    plt.title("Konvergenz der von-Mises-Iteration (halblogarithmisch)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return lambdas, errors

# ============================================================
# PART 4 — Call
# ============================================================
plot_power_method_convergence(A, v0, iters, lambda_ref, lambda_estimate)
