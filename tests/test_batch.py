
import csv
from pathlib import Path
from examples.generate_synthetic_neck import make_necked_tube
from spectral_membranes.batch import process_directory
from spectral_membranes.io import save_mesh_obj

def test_batch_directory(tmp_path: Path):
    input_dir = tmp_path / "meshes"
    input_dir.mkdir()
    save_mesh_obj(make_necked_tube(n_theta=16, n_z=24, neck_depth=0.15), str(input_dir / "organelle=mito__condition=control__rep=01.obj"))
    save_mesh_obj(make_necked_tube(n_theta=16, n_z=24, neck_depth=0.55), str(input_dir / "organelle=mito__condition=stress__rep=01.obj"))
    output_csv = tmp_path / "summary.csv"
    rows = process_directory(str(input_dir), str(output_csv), k=8)
    assert len(rows) == 2
    with open(output_csv, newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 2
    assert "lambda2" in csv_rows[0]
    assert "condition" in csv_rows[0]
