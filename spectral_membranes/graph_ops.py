
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from .mesh import unique_edges
from .types import Mesh

def adjacency_matrix(mesh: Mesh, weighted: bool = False) -> sp.csr_matrix:
    edges = unique_edges(mesh)
    n = len(mesh.vertices)
    if weighted:
        diffs = mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]]
        lengths = np.linalg.norm(diffs, axis=1)
        weights = 1.0 / np.maximum(lengths, 1e-12)
    else:
        weights = np.ones(len(edges), dtype=float)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([weights, weights])
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))

def normalized_laplacian(W: sp.csr_matrix) -> sp.csr_matrix:
    d = np.asarray(W.sum(axis=1)).ravel()
    dinv_sqrt = np.zeros_like(d)
    mask = d > 0
    dinv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
    D_inv_sqrt = sp.diags(dinv_sqrt)
    I = sp.eye(W.shape[0], format="csr")
    return I - D_inv_sqrt @ W @ D_inv_sqrt
