
from __future__ import annotations
import numpy as np
from .mesh import edge_lengths, boundary_vertices, surface_area
from .types import Mesh

def quality_control(mesh: Mesh) -> dict:
    e = edge_lengths(mesh)
    bmask = boundary_vertices(mesh)
    return {
        "n_vertices": int(len(mesh.vertices)),
        "n_faces": int(len(mesh.faces)),
        "median_edge_length": float(np.median(e)) if len(e) else 0.0,
        "mean_edge_length": float(np.mean(e)) if len(e) else 0.0,
        "boundary_fraction": float(np.mean(bmask)) if len(bmask) else 0.0,
        "surface_area": surface_area(mesh),
    }

def remesh_to_target_edge_length(mesh: Mesh, target_h: float) -> Mesh:
    # Lightweight stub: return unchanged mesh
    return mesh
