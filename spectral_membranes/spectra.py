
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def smallest_eigenpairs(operator, k: int):
    n = operator.shape[0]
    k_eff = max(2, min(k, n - 1))
    evals, evecs = spla.eigsh(operator, k=k_eff, which="SM")
    order = np.argsort(evals)
    return np.asarray(evals[order], dtype=float), np.asarray(evecs[:, order], dtype=float)

def generalized_smallest_eigenpairs(C, M, k: int):
    n = C.shape[0]
    k_eff = max(2, min(k, n - 1))
    evals, evecs = spla.eigsh(C, M=M, k=k_eff, which="SM")
    order = np.argsort(evals)
    return np.asarray(evals[order], dtype=float), np.asarray(evecs[:, order], dtype=float)
