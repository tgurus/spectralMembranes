
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Callable, Iterable
import numpy as np
from .adapters import SurfaceMorphometricsAdapter, parse_key_value_stem
from .io import load_mesh, save_feature_table_csv
from .pipeline import run_dual_operator_pipeline

def parse_filename_metadata(path: str) -> dict:
    return parse_key_value_stem(Path(path).stem)

def summarize_feature_set(graph_features, cotan_evals: np.ndarray | None = None, mesh_id: str | None = None, source_path: str | None = None, metadata: dict | None = None) -> dict:
    tau = np.asarray(graph_features.heat_trace_tau, dtype=float)
    ht = np.asarray(graph_features.heat_trace_values, dtype=float)
    hs = np.asarray(graph_features.spectral_entropy, dtype=float)
    ds = np.asarray(graph_features.spectral_dimension, dtype=float)

    def triad(arr):
        if arr.size == 0:
            return (np.nan, np.nan, np.nan)
        i = arr.size // 2
        return float(arr[0]), float(arr[i]), float(arr[-1])

    row = {
        "mesh_id": mesh_id,
        "source_path": source_path,
        "lambda2": graph_features.lambda2,
        "lambda3": graph_features.lambda3,
        "lambda2_lambda3_ratio": float(graph_features.lambda2 / graph_features.lambda3) if graph_features.lambda2 not in (None, 0) and graph_features.lambda3 not in (None, 0) else np.nan,
        "fiedler_ipr": graph_features.fiedler_ipr,
        "conductance_min": graph_features.conductance_min,
    }
    for prefix, arr in [("tau", tau), ("heat_trace", ht), ("spectral_entropy", hs), ("spectral_dimension", ds)]:
        first, mid, last = triad(arr)
        row[f"{prefix}_first"] = first
        row[f"{prefix}_mid"] = mid
        row[f"{prefix}_last"] = last
    for key, value in graph_features.extra.items():
        if key != "fiedler_vector":
            row[key] = value
    if cotan_evals is not None:
        cotan_evals = np.asarray(cotan_evals, dtype=float)
        row["cotan_mu2"] = float(cotan_evals[1]) if cotan_evals.size > 1 else np.nan
        row["cotan_mu3"] = float(cotan_evals[2]) if cotan_evals.size > 2 else np.nan
    if metadata:
        for key, value in metadata.items():
            if key not in row:
                row[key] = value
    return row

def process_mesh_path(path: str, k: int = 50, weighted: bool = False, metadata_parser: Callable[[str], dict] | None = parse_filename_metadata, output_dir: str | None = None) -> dict:
    mesh = load_mesh(path)
    results = run_dual_operator_pipeline(mesh, k=k, weighted=weighted)
    metadata = metadata_parser(path) if metadata_parser else {}
    row = summarize_feature_set(results["graph"], cotan_evals=results["cotan_evals"], mesh_id=Path(path).stem, source_path=str(path), metadata=metadata)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        stem = Path(path).stem
        serializable = {}
        for k0, v0 in row.items():
            if isinstance(v0, (np.floating, np.integer)):
                serializable[k0] = float(v0)
            else:
                serializable[k0] = v0
        with open(os.path.join(output_dir, f"{stem}.json"), "w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2)
        np.savez(
            os.path.join(output_dir, f"{stem}_graph_arrays.npz"),
            heat_trace_tau=results["graph"].heat_trace_tau,
            heat_trace_values=results["graph"].heat_trace_values,
            spectral_entropy=results["graph"].spectral_entropy,
            spectral_dimension=results["graph"].spectral_dimension,
            fiedler_vector=results["graph"].extra["fiedler_vector"],
            cotan_evals=results["cotan_evals"],
        )
    return row

def collect_mesh_paths(input_dir: str, exts: Iterable[str] = (".npz", ".obj", ".ply", ".off")) -> list[str]:
    matches = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in exts:
                matches.append(os.path.join(root, name))
    return sorted(matches)

def process_directory(input_dir: str, output_csv: str, k: int = 50, weighted: bool = False, metadata_parser: Callable[[str], dict] | None = parse_filename_metadata, per_mesh_dir: str | None = None) -> list[dict]:
    paths = collect_mesh_paths(input_dir)
    if not paths:
        raise ValueError(f"No supported mesh files found under {input_dir}")
    rows = [process_mesh_path(path, k=k, weighted=weighted, metadata_parser=metadata_parser, output_dir=per_mesh_dir) for path in paths]
    save_feature_table_csv(rows, output_csv)
    return rows

def process_surface_morphometrics_project(project_root: str, output_csv: str, k: int = 50, weighted: bool = False, manifest_path: str | None = None, per_mesh_dir: str | None = None) -> list[dict]:
    adapter = SurfaceMorphometricsAdapter(project_root=project_root, manifest_path=manifest_path)
    paths = adapter.collect_mesh_paths()
    if not paths:
        raise ValueError(f"No supported mesh files found under {project_root}")
    rows = [process_mesh_path(path, k=k, weighted=weighted, metadata_parser=adapter.parse_metadata, output_dir=per_mesh_dir) for path in paths]
    save_feature_table_csv(rows, output_csv)
    return rows
