
import csv, json
from pathlib import Path
from examples.generate_synthetic_neck import make_necked_tube
from spectral_membranes.batch import process_surface_morphometrics_project
from spectral_membranes.io import save_mesh_obj

def test_surface_morphometrics_adapter(tmp_path: Path):
    project = tmp_path / "project"
    mesh_dir = project / "stress" / "mito" / "tomo_001"
    mesh_dir.mkdir(parents=True)
    save_mesh_obj(make_necked_tube(n_theta=16, n_z=24, neck_depth=0.35), str(mesh_dir / "membrane_A.obj"))
    (mesh_dir / "membrane_A.json").write_text(json.dumps({"sample_id": "S001"}), encoding="utf-8")
    manifest = project / "surface_morphometrics_manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "experiment"])
        writer.writeheader()
        writer.writerow({"relative_path": "stress/mito/tomo_001/membrane_A.obj", "experiment": "demo"})
    output_csv = tmp_path / "summary.csv"
    rows = process_surface_morphometrics_project(str(project), str(output_csv), k=8, per_mesh_dir=str(tmp_path / "per_mesh"))
    assert len(rows) == 1
    with open(output_csv, newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["condition"] == "stress"
    assert row["organelle"] == "mito"
    assert row["tomogram"] == "tomo_001"
    assert row["sample_id"] == "S001"
    assert row["experiment"] == "demo"
