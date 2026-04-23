
from __future__ import annotations
import numpy as np
from .types import Mesh

def unique_edges(mesh: Mesh) -> np.ndarray:
    tri_edges = np.vstack([
        mesh.faces[:, [0, 1]],
        mesh.faces[:, [1, 2]],
        mesh.faces[:, [0, 2]],
    ])
    tri_edges = np.sort(tri_edges, axis=1)
    return np.unique(tri_edges, axis=0)

def edge_lengths(mesh: Mesh) -> np.ndarray:
    edges = unique_edges(mesh)
    diffs = mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]]
    return np.linalg.norm(diffs, axis=1)

def vertex_areas(mesh: Mesh) -> np.ndarray:
    verts = mesh.vertices
    faces = mesh.faces
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    areas = np.zeros(len(verts), dtype=float)
    for i in range(3):
        np.add.at(areas, faces[:, i], face_areas / 3.0)
    return areas

def surface_area(mesh: Mesh) -> float:
    verts = mesh.vertices
    faces = mesh.faces
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum())

def boundary_edges(mesh: Mesh) -> np.ndarray:
    tri_edges = np.vstack([
        np.sort(mesh.faces[:, [0, 1]], axis=1),
        np.sort(mesh.faces[:, [1, 2]], axis=1),
        np.sort(mesh.faces[:, [0, 2]], axis=1),
    ])
    uniq, counts = np.unique(tri_edges, axis=0, return_counts=True)
    return uniq[counts == 1]

def boundary_vertices(mesh: Mesh) -> np.ndarray:
    mask = np.zeros(len(mesh.vertices), dtype=bool)
    b_edges = boundary_edges(mesh)
    if len(b_edges):
        mask[b_edges.ravel()] = True
    return mask
