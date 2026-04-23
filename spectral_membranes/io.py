
from __future__ import annotations
import csv
import os
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
from .types import Mesh

def load_mesh(path: str) -> Mesh:
    ext = Path(path).suffix.lower()
    if ext == ".npz":
        data = np.load(path)
        return Mesh(vertices=np.asarray(data["vertices"], dtype=float), faces=np.asarray(data["faces"], dtype=int))
    if ext == ".obj":
        return _load_obj(path)
    if ext == ".ply":
        return _load_ply_ascii(path)
    if ext == ".off":
        return _load_off(path)
    raise ValueError(f"Unsupported mesh format: {ext}")

def save_mesh_npz(mesh: Mesh, path: str) -> None:
    np.savez(path, vertices=mesh.vertices, faces=mesh.faces)

def save_mesh_obj(mesh: Mesh, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for v in mesh.vertices:
            handle.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in mesh.faces:
            a, b, c = face + 1
            handle.write(f"f {a} {b} {c}\n")

def save_feature_table_csv(rows: Sequence[Mapping[str, object]], path: str) -> None:
    if not rows:
        raise ValueError("rows must not be empty")
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key); fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def _load_obj(path: str) -> Mesh:
    vertices, faces = [], []
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                _, x, y, z, *_ = line.split()
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                parts = line.split()[1:]
                idx = [int(part.split("/")[0]) - 1 for part in parts[:3]]
                faces.append(idx)
    return Mesh(vertices=np.asarray(vertices, dtype=float), faces=np.asarray(faces, dtype=int))

def _load_ply_ascii(path: str) -> Mesh:
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    n_vertices = n_faces = None
    header_end = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("element vertex"):
            n_vertices = int(s.split()[-1])
        elif s.startswith("element face"):
            n_faces = int(s.split()[-1])
        elif s == "end_header":
            header_end = i; break
    if header_end is None or n_vertices is None or n_faces is None:
        raise ValueError("Invalid ASCII PLY")
    start = header_end + 1
    vertices = np.asarray([[float(x) for x in lines[start + i].split()[:3]] for i in range(n_vertices)], dtype=float)
    faces = []
    for i in range(n_faces):
        parts = lines[start + n_vertices + i].split()
        n = int(parts[0]); idx = [int(x) for x in parts[1:1+n]]
        for j in range(1, n - 1):
            faces.append([idx[0], idx[j], idx[j+1]])
    return Mesh(vertices=vertices, faces=np.asarray(faces, dtype=int))

def _load_off(path: str) -> Mesh:
    with open(path, "r", encoding="utf-8") as handle:
        lines = [ln.strip() for ln in handle if ln.strip() and not ln.startswith("#")]
    if lines[0] != "OFF":
        raise ValueError("Invalid OFF")
    n_vertices, n_faces, _ = map(int, lines[1].split()[:3])
    vertices = np.asarray([[float(x) for x in lines[2+i].split()[:3]] for i in range(n_vertices)], dtype=float)
    faces = []
    for i in range(n_faces):
        parts = lines[2+n_vertices+i].split()
        n = int(parts[0]); idx = [int(x) for x in parts[1:1+n]]
        for j in range(1, n - 1):
            faces.append([idx[0], idx[j], idx[j+1]])
    return Mesh(vertices=vertices, faces=np.asarray(faces, dtype=int))
