# ============================================================
# TOPIC: Vektoren — Vektornormen ||v||_1, ||v||_2, ||v||_inf
# DESCRIPTION:
# Druckt für einen gegebenen Vektor die drei üblichen Normen
# (1-Norm, 2-Norm, Maximumsnorm).
# USE WHEN:
# Wenn ein Vektor (z.B. rechte Seite b oder Lösung x) in
# unterschiedlichen Normen verglichen werden soll.
# EXAMPLE:
# v = [3080, 4070, 5030].
# ============================================================

import numpy as np
import numpy.linalg as lin

np.set_printoptions(precision=6, suppress=True)

# ============================================================
# PART 1 — Inputs
# ============================================================
v = np.array([3080.0,
              4070.0,
              5030.0])

# ============================================================
# PART 2 — Method selection
# ============================================================
# Only one method here. Es werden immer ||v||_1, ||v||_2 und ||v||_inf ausgegeben.

# ============================================================
# PART 3 — Implementation
# ============================================================
def compute_vector_norm_and_condition(v):
    v = np.asarray(v, dtype=float)
    if v.ndim == 2:
        if v.shape[1] == 1:
            v = v[:, 0]
        elif v.shape[0] == 1:
            v = v[0, :]
        else:
            raise ValueError(f"v must be 1D or (n,1) or (1,n). Got shape {v.shape}.")
    elif v.ndim != 1:
        raise ValueError(f"v must be 1D. Got shape {v.shape}.")
    print("Vector v:")
    print(v)
    print("------------------------------------------------------------")
    for name, ord_val in [("1", 1), ("2", 2), ("inf", np.inf)]:
        print(f"||v||_{name:>2} = {lin.norm(v, ord=ord_val)}")

# ============================================================
# PART 4 — Call
# ============================================================
compute_vector_norm_and_condition(v)
