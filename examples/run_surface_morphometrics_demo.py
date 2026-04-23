
from __future__ import annotations
import csv, json
from pathlib import Path
from spectral_membranes.batch import process_surface_morphometrics_project
from spectral_membranes.io import save_mesh_obj
from spectral_membranes.visualize import plot_group_distributions
from examples.generate_synthetic_neck import make_necked_tube

def main():
    outdir = Path("example_output/surface_morphometrics_demo")
    project = outdir / "project"
    per_mesh_dir = outdir / "per_mesh"
    per_mesh_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("control", "mito", "tomo_001", "membrane_A", 0.20),
        ("control", "mito", "tomo_002", "membrane_B", 0.28),
        ("stress", "mito", "tomo_007", "membrane_C", 0.48),
        ("stress", "er", "tomo_009", "membrane_D", 0.58),
    ]
    rows = []
    for idx, (condition, organelle, tomo, stem, neck) in enumerate(specs, start=1):
        mesh_dir = project / condition / organelle / tomo
        mesh_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = mesh_dir / f"{stem}.obj"
        save_mesh_obj(make_necked_tube(n_theta=24, n_z=36, neck_depth=neck), str(mesh_path))
        (mesh_dir / f"{stem}.json").write_text(json.dumps({"sample_id": f"S{idx:03d}", "neck_strength_seed": neck}, indent=2), encoding="utf-8")
        rows.append({"relative_path": str(mesh_path.relative_to(project)), "experiment_batch": "demo_batch"})
    manifest_path = project / "surface_morphometrics_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    output_csv = outdir / "surface_morphometrics_summary.csv"
    data_rows = process_surface_morphometrics_project(str(project), str(output_csv), per_mesh_dir=str(per_mesh_dir))
    plot_group_distributions(data_rows, "lambda2", "condition", str(outdir / "lambda2_by_condition.png"))
    summary = {"n_meshes": len(data_rows), "output_csv": str(output_csv), "columns": list(data_rows[0].keys())}
    (outdir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
