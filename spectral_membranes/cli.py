
from __future__ import annotations
import argparse, os
from .batch import process_directory, process_surface_morphometrics_project

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch spectral morphometrics for membrane meshes")
    parser.add_argument("input_dir")
    parser.add_argument("output_csv")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--per-mesh-dir", default=None)
    parser.add_argument("--adapter", choices=("default", "surface_morphometrics"), default="default")
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    if args.adapter == "surface_morphometrics":
        rows = process_surface_morphometrics_project(args.input_dir, args.output_csv, k=args.k, weighted=args.weighted, manifest_path=args.manifest, per_mesh_dir=args.per_mesh_dir)
    else:
        rows = process_directory(args.input_dir, args.output_csv, k=args.k, weighted=args.weighted, per_mesh_dir=args.per_mesh_dir)
    print(f"Processed {len(rows)} meshes")
    print(f"Wrote CSV to {os.path.abspath(args.output_csv)}")
