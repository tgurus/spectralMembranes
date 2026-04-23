
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from .mesh import vertex_areas
from .types import Mesh

def _cotangent(u: np.ndarray, v: np.ndarray) -> float:
    cross = np.linalg.norm(np.cross(u, v))
    if cross <= 1e-15:
        return 0.0
    return float(np.dot(u, v) / cross)

def cotan_laplacian(mesh: Mesh):
    V = mesh.vertices
    F = mesh.faces
    n = len(V)
    weights: dict[tuple[int, int], float] = {}

    for tri in F:
        i, j, k = tri.tolist()
        vi, vj, vk = V[i], V[j], V[k]
        cot_k = _cotangent(vi - vk, vj - vk)
        cot_i = _cotangent(vj - vi, vk - vi)
        cot_j = _cotangent(vi - vj, vk - vj)
        for a, b, c in [(i, j, cot_k), (j, k, cot_i), (i, k, cot_j)]:
            key = tuple(sorted((a, b)))
            weights[key] = weights.get(key, 0.0) + 0.5 * c

    rows, cols, data = [], [], []
    diag = np.zeros(n, dtype=float)
    for (i, j), w in weights.items():
        rows.extend([i, j]); cols.extend([j, i]); data.extend([-w, -w])
        diag[i] += w; diag[j] += w
    rows.extend(range(n)); cols.extend(range(n)); data.extend(diag.tolist())
    C = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    M = sp.diags(np.maximum(vertex_areas(mesh), 1e-12))
    return C, M
