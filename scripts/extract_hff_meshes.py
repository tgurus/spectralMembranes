
"""Extract embedded mesh surfaces from EMDB-SFF HFF files to PLY.

Usage:
    python extract_hff_meshes.py /path/to/file_or_dir --outdir extracted_meshes
"""
from __future__ import annotations
import argparse
import base64
import os
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

def decode_mesh_component(group):
    data = group['data'][()]
    endianness = group['endianness'][()].decode() if isinstance(group['endianness'][()], (bytes, bytearray)) else str(group['endianness'][()])
    mode = group['mode'][()].decode() if isinstance(group['mode'][()], (bytes, bytearray)) else str(group['mode'][()])
    if mode == 'float32':
        dt = np.dtype('<f4' if endianness == 'little' else '>f4')
    elif mode == 'uint32':
        dt = np.dtype('<u4' if endianness == 'little' else '>u4')
    else:
        raise ValueError(f"Unsupported mesh payload mode: {mode}")
    return np.frombuffer(base64.b64decode(data), dtype=dt)

def write_ply(path: Path, verts: np.ndarray, tris: np.ndarray) -> None:
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(tris)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        np.savetxt(f, verts, fmt="%.6f %.6f %.6f")
        np.savetxt(f, np.c_[np.full(len(tris), 3), tris.astype(int)], fmt="%d %d %d %d")

def extract_hff(hff_path: Path, outdir: Path):
    records = []
    with h5py.File(hff_path, 'r') as f:
        for sid in f['segment_list'].keys():
            grp = f['segment_list'][sid]
            name = grp['biological_annotation/name'][()]
            name = name.decode() if isinstance(name, (bytes, bytearray)) else str(name)
            mesh_grp = grp['mesh_list']['0']
            verts = decode_mesh_component(mesh_grp['vertices']).reshape(-1, 3)
            tris = decode_mesh_component(mesh_grp['triangles']).reshape(-1, 3)
            ply_name = f"{hff_path.stem}__{name}.ply"
            write_ply(outdir / ply_name, verts, tris)
            records.append({
                'source_hff': hff_path.name,
                'segment_id': sid,
                'segment_name': name,
                'n_vertices': len(verts),
                'n_faces': len(tris),
                'ply_file': ply_name,
            })
    return records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='HFF file or directory containing HFF files')
    parser.add_argument('--outdir', default='hff_meshes_out')
    args = parser.parse_args()
    inpath = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    meshdir = outdir / 'ply_meshes'
    meshdir.mkdir(exist_ok=True)

    hffs = [inpath] if inpath.is_file() else sorted(inpath.glob('*.hff'))
    rows = []
    for hp in hffs:
        rows.extend(extract_hff(hp, meshdir))
    pd.DataFrame(rows).to_csv(outdir / 'mesh_inventory.csv', index=False)
    print(f"Extracted {len(rows)} meshes to {meshdir}")

if __name__ == '__main__':
    main()
