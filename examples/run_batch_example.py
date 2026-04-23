
from __future__ import annotations
import json
from pathlib import Path
from spectral_membranes.batch import process_directory
from spectral_membranes.io import save_mesh_obj
from spectral_membranes.visualize import plot_group_distributions
from examples.generate_synthetic_neck import make_necked_tube

def main():
    outdir = Path("example_output/batch_demo")
    meshdir = outdir / "input_meshes"
    per_mesh_dir = outdir / "per_mesh"
    meshdir.mkdir(parents=True, exist_ok=True)
    per_mesh_dir.mkdir(parents=True, exist_ok=True)
    specs = [("control", 0.15), ("control", 0.25), ("stress", 0.45), ("stress", 0.60)]
    for idx, (condition, neck_depth) in enumerate(specs, start=1):
        mesh = make_necked_tube(n_theta=24, n_z=36, neck_depth=neck_depth, neck_width=0.8 if condition == "stress" else 1.0)
        name = f"organelle=mito__condition={condition}__rep={idx:02d}__neck={neck_depth:.2f}.obj"
        save_mesh_obj(mesh, str(meshdir / name))
    csv_path = outdir / "batch_summary.csv"
    rows = process_directory(str(meshdir), str(csv_path), k=16, per_mesh_dir=str(per_mesh_dir))
    plot_group_distributions(rows, "lambda2", "condition", str(outdir / "lambda2_by_condition.png"))
    summary = {"n_meshes": len(rows), "conditions": sorted({row.get("condition", "unknown") for row in rows}), "csv_path": str(csv_path)}
    (outdir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
