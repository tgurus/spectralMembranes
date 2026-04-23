"""
Eigensolvers for graph and cotangent Laplacian operators.
Uses shift-invert mode by default for much faster convergence.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def smallest_eigenpairs(operator, k: int):
    """Compute k smallest eigenpairs of a sparse symmetric operator.
    Uses shift-invert (sigma=1e-8) for fast convergence.
    """
    n = operator.shape[0]
    k_eff = max(2, min(k, n - 1))
    try:
        evals, evecs = spla.eigsh(operator, k=k_eff, sigma=1e-8, which="LM")
    except Exception:
        evals, evecs = spla.eigsh(operator, k=k_eff, which="SM")
    order = np.argsort(evals)
    return np.asarray(evals[order], dtype=float), np.asarray(evecs[:, order], dtype=float)


def generalized_smallest_eigenpairs(C, M, k: int):
    """Compute k smallest generalized eigenpairs C x = mu M x.
    Uses shift-invert (sigma=1e-8) for fast convergence.
    """
    n = C.shape[0]
    k_eff = max(2, min(k, n - 1))
    try:
        evals, evecs = spla.eigsh(C, M=M, k=k_eff, sigma=1e-8, which="LM")
    except Exception:
        evals, evecs = spla.eigsh(C, M=M, k=k_eff, which="SM")
    order = np.argsort(evals)
    return np.asarray(evals[order], dtype=float), np.asarray(evecs[:, order], dtype=float)
