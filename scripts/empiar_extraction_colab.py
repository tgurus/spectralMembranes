##############################################################################
# SPECTRAL MEMBRANES — EMPIAR-11370 FULL MESH EXTRACTION PIPELINE
# Paste this entire cell into Google Colab and run.
#
# Prerequisites:
#   Upload EMPIAR-11370-MembraneSurfaceMesh-ALL.zip to Colab
#   (or adjust INPUT_ZIP path below)
#
# Outputs:
#   empiar_all_meshes.zip — ready to upload back to Claude
#   Contains: PLY meshes, mesh_inventory.csv, condition_map.csv
##############################################################################

# --- Config ---
INPUT_ZIP = "EMPIAR-11370-MembraneSurfaceMesh-ALL.zip"  # adjust if needed
WORK_DIR = "/content/empiar_extraction"
OUT_DIR = f"{WORK_DIR}/empiar_all_meshes"
PLY_DIR = f"{OUT_DIR}/ply_meshes"

# --- Install dependencies ---
!pip install -q h5py numpy pandas

import os, base64, zipfile, shutil
import numpy as np
import pandas as pd
import h5py

os.makedirs(PLY_DIR, exist_ok=True)

# --- Unzip the HFF archive ---
print("Extracting HFF archive...")
with zipfile.ZipFile(INPUT_ZIP, 'r') as z:
    z.extractall(WORK_DIR)

# Find the HFF directory
HFF_DIR = None
for root, dirs, files in os.walk(WORK_DIR):
    if any(f.endswith('.hff') for f in files):
        HFF_DIR = root
        break

if HFF_DIR is None:
    raise FileNotFoundError("No .hff files found in the extracted archive!")

hff_files = sorted([f for f in os.listdir(HFF_DIR) if f.endswith('.hff')])
print(f"Found {len(hff_files)} HFF files in {HFF_DIR}")
for f in hff_files:
    print(f"  {f}")

# --- Condition mapping (from naming convention) ---
# TE = Thapsigargin + Elongated
# TF = Thapsigargin + Fragmented
# UE = Untreated/Vehicle + Elongated
# UF = Untreated/Vehicle + Fragmented

def parse_condition(tomogram_id):
    """Parse condition and morphology from tomogram ID."""
    prefix = ''.join(c for c in tomogram_id if c.isalpha())
    mapping = {
        'TE': ('thapsigargin', 'elongated'),
        'TF': ('thapsigargin', 'fragmented'),
        'UE': ('vehicle', 'elongated'),
        'UF': ('vehicle', 'fragmented'),
    }
    if prefix in mapping:
        return mapping[prefix]
    return ('unknown', 'unknown')

# --- HFF mesh extraction functions ---

def decode_mesh_component(group):
    """Decode base64-encoded mesh data from HFF."""
    data = group['data'][()]
    endianness = group['endianness'][()]
    if isinstance(endianness, (bytes, bytearray)):
        endianness = endianness.decode()
    else:
        endianness = str(endianness)

    mode = group['mode'][()]
    if isinstance(mode, (bytes, bytearray)):
        mode = mode.decode()
    else:
        mode = str(mode)

    if mode == 'float32':
        dt = np.dtype('<f4' if endianness == 'little' else '>f4')
    elif mode == 'uint32':
        dt = np.dtype('<u4' if endianness == 'little' else '>u4')
    else:
        raise ValueError(f"Unsupported mesh payload mode: {mode}")

    return np.frombuffer(base64.b64decode(data), dtype=dt)


def write_ply(path, verts, tris):
    """Write a PLY file (ASCII format)."""
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(tris)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for v in verts:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in tris:
            f.write(f"3 {int(t[0])} {int(t[1])} {int(t[2])}\n")


def compute_boundary_stats(verts, tris):
    """Compute boundary fraction from triangle mesh."""
    edges = np.vstack([
        np.sort(tris[:, [0, 1]], axis=1),
        np.sort(tris[:, [1, 2]], axis=1),
        np.sort(tris[:, [0, 2]], axis=1),
    ])
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = uniq[counts == 1]
    n_boundary_edges = len(boundary_edges)

    if n_boundary_edges == 0:
        return 0.0, 0

    boundary_verts = set(boundary_edges.ravel())
    boundary_fraction = len(boundary_verts) / len(verts) if len(verts) > 0 else 0.0
    return boundary_fraction, n_boundary_edges


def extract_hff(hff_path, outdir):
    """Extract all mesh segments from one HFF file."""
    tomogram_id = os.path.splitext(os.path.basename(hff_path))[0]
    condition, morphology = parse_condition(tomogram_id)
    records = []

    with h5py.File(hff_path, 'r') as f:
        if 'segment_list' not in f:
            print(f"  WARNING: No segment_list in {hff_path}")
            return records

        for sid in sorted(f['segment_list'].keys()):
            grp = f['segment_list'][sid]

            # Get segment name
            try:
                name = grp['biological_annotation/name'][()]
                if isinstance(name, (bytes, bytearray)):
                    name = name.decode()
                else:
                    name = str(name)
            except KeyError:
                name = f"segment_{sid}"

            # Determine membrane type from name
            name_lower = name.lower()
            if 'imm' in name_lower or 'inner' in name_lower:
                segment_type = 'IMM'
            elif 'omm' in name_lower or 'outer' in name_lower:
                segment_type = 'OMM'
            elif 'er' in name_lower or 'endoplasmic' in name_lower:
                segment_type = 'ER'
            else:
                segment_type = 'other'

            # Extract mesh
            try:
                mesh_grp = grp['mesh_list']['0']
                verts = decode_mesh_component(mesh_grp['vertices']).reshape(-1, 3)
                tris = decode_mesh_component(mesh_grp['triangles']).reshape(-1, 3)
            except Exception as e:
                print(f"  WARNING: Could not extract mesh for {name} in {tomogram_id}: {e}")
                continue

            # Compute boundary stats
            bf, n_bedges = compute_boundary_stats(verts, tris)

            # Write PLY
            ply_name = f"{tomogram_id}__{name}.ply"
            # Sanitize filename
            ply_name = ply_name.replace(' ', '_').replace('/', '_')
            write_ply(os.path.join(outdir, ply_name), verts, tris)

            records.append({
                'source_hff': os.path.basename(hff_path),
                'tomogram_id': tomogram_id,
                'condition': condition,
                'morphology': morphology,
                'segment_id': sid,
                'segment_name': name,
                'segment_type': segment_type,
                'n_vertices': len(verts),
                'n_faces': len(tris),
                'boundary_fraction': round(bf, 6),
                'boundary_edges': n_bedges,
                'ply_file': ply_name,
            })

    return records


# --- Run extraction on all HFF files ---
print("\n" + "="*60)
print("EXTRACTING MESHES FROM ALL HFF FILES")
print("="*60)

all_records = []
for hff_name in hff_files:
    hff_path = os.path.join(HFF_DIR, hff_name)
    tomo_id = os.path.splitext(hff_name)[0]
    cond, morph = parse_condition(tomo_id)
    print(f"\n{tomo_id} ({cond}/{morph}):")

    records = extract_hff(hff_path, PLY_DIR)
    for r in records:
        print(f"  {r['segment_name']:40s} {r['segment_type']:4s} "
              f"v={r['n_vertices']:6d} f={r['n_faces']:6d} bf={r['boundary_fraction']:.3f}")
    all_records.extend(records)

# --- Save mesh inventory ---
inventory_df = pd.DataFrame(all_records)
inventory_path = os.path.join(OUT_DIR, 'mesh_inventory.csv')
inventory_df.to_csv(inventory_path, index=False)

# --- Save condition map ---
condition_rows = []
for hff_name in hff_files:
    tomo_id = os.path.splitext(hff_name)[0]
    cond, morph = parse_condition(tomo_id)
    condition_rows.append({
        'tomogram': tomo_id,
        'condition': cond,
        'morphology': morph,
    })
condition_df = pd.DataFrame(condition_rows)
condition_path = os.path.join(OUT_DIR, 'condition_map.csv')
condition_df.to_csv(condition_path, index=False)

# --- Summary ---
print("\n" + "="*60)
print("EXTRACTION COMPLETE")
print("="*60)

n_ply = len([f for f in os.listdir(PLY_DIR) if f.endswith('.ply')])
print(f"\nTotal PLY meshes extracted: {n_ply}")
print(f"Mesh inventory: {inventory_path}")
print(f"Condition map: {condition_path}")

print("\n--- Condition/Morphology breakdown ---")
summary = inventory_df.groupby(['condition', 'morphology', 'segment_type']).size().reset_index(name='count')
print(summary.to_string(index=False))

print("\n--- Tomogram counts ---")
tomo_summary = condition_df.groupby(['condition', 'morphology']).size().reset_index(name='n_tomograms')
print(tomo_summary.to_string(index=False))

# Count paired mitochondria (have both IMM and OMM)
print("\n--- Paired IMM/OMM mitochondria per condition ---")
for cond in ['vehicle', 'thapsigargin']:
    sub = inventory_df[inventory_df['condition'] == cond]
    imm_mitos = set()
    omm_mitos = set()
    for _, row in sub.iterrows():
        name = row['segment_name']
        if row['segment_type'] == 'IMM':
            # Extract mito identifier (everything before _IMM)
            mito_key = f"{row['tomogram_id']}__{name.rsplit('_IMM', 1)[0] if '_IMM' in name else name}"
            imm_mitos.add(mito_key)
        elif row['segment_type'] == 'OMM':
            mito_key = f"{row['tomogram_id']}__{name.rsplit('_OMM', 1)[0] if '_OMM' in name else name}"
            omm_mitos.add(mito_key)
    paired = imm_mitos & omm_mitos
    print(f"  {cond}: {len(imm_mitos)} IMM, {len(omm_mitos)} OMM, {len(paired)} paired")

# --- Package for download ---
print("\n--- Packaging for download ---")
zip_path = "/content/empiar_all_meshes.zip"

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    # Add CSVs
    zf.write(inventory_path, 'empiar_all_meshes/mesh_inventory.csv')
    zf.write(condition_path, 'empiar_all_meshes/condition_map.csv')

    # Add all PLY files
    for ply_file in sorted(os.listdir(PLY_DIR)):
        if ply_file.endswith('.ply'):
            zf.write(os.path.join(PLY_DIR, ply_file),
                     f'empiar_all_meshes/ply_meshes/{ply_file}')

zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print(f"\nOutput: {zip_path} ({zip_size_mb:.1f} MB)")
print("Download this file and upload it to Claude.")

# Auto-download in Colab
try:
    from google.colab import files
    files.download(zip_path)
    print("Download started automatically.")
except ImportError:
    print("Not in Colab — download the zip manually from the file browser.")

print("\n" + "="*60)
print("DONE. Upload empiar_all_meshes.zip to Claude for Phase 2 analysis.")
print("="*60)
