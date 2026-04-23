
from __future__ import annotations
import numpy as np
from spectral_membranes.types import Mesh

def make_necked_tube(n_theta: int = 28, n_z: int = 48, radius: float = 1.0, length: float = 4.0, neck_depth: float = 0.35, neck_width: float = 0.8) -> Mesh:
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    z = np.linspace(-length / 2.0, length / 2.0, n_z)
    verts = []
    for zi in z:
        profile = 1.0 - neck_depth * np.exp(-0.5 * (zi / max(neck_width, 1e-3)) ** 2)
        r = radius * profile
        for th in theta:
            verts.append([r * np.cos(th), r * np.sin(th), zi])
    verts = np.asarray(verts, dtype=float)

    def idx(i_theta: int, i_z: int) -> int:
        return i_z * n_theta + (i_theta % n_theta)

    faces = []
    for iz in range(n_z - 1):
        for it in range(n_theta):
            a = idx(it, iz)
            b = idx(it + 1, iz)
            c = idx(it, iz + 1)
            d = idx(it + 1, iz + 1)
            faces.append([a, b, c])
            faces.append([b, d, c])
    return Mesh(vertices=verts, faces=np.asarray(faces, dtype=int))
