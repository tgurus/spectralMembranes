
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from spectral_membranes.io import save_mesh_npz
from spectral_membranes.pipeline import run_dual_operator_pipeline
from spectral_membranes.visualize import plot_heat_trace, plot_mesh_scalar
from examples.generate_synthetic_neck import make_necked_tube

def main():
    outdir = Path("example_output")
    outdir.mkdir(exist_ok=True)
    mesh = make_necked_tube()
    save_mesh_npz(mesh, str(outdir / "synthetic_neck_mesh.npz"))
    results = run_dual_operator_pipeline(mesh, k=24)
    graph = results["graph"]
    fiedler = graph.extra["fiedler_vector"]
    summary = {
        "lambda2": graph.lambda2,
        "lambda3": graph.lambda3,
        "fiedler_ipr": graph.fiedler_ipr,
        "conductance_min": graph.conductance_min,
        "boundary_fraction": graph.extra["boundary_fraction"],
        "surface_area": graph.extra["surface_area"],
        "median_edge_length": graph.extra["median_edge_length"],
        "n_vertices": graph.extra["n_vertices"],
        "n_faces": graph.extra["n_faces"],
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_mesh_scalar(mesh, fiedler, str(outdir / "fiedler_vector.png"), title="Synthetic neck mesh: Fiedler vector")
    plot_heat_trace(graph.heat_trace_tau, graph.heat_trace_values, str(outdir / "heat_trace.png"))
    np.savez(outdir / "graph_features.npz", heat_trace_tau=graph.heat_trace_tau, heat_trace_values=graph.heat_trace_values, spectral_entropy=graph.spectral_entropy, spectral_dimension=graph.spectral_dimension, fiedler_vector=fiedler)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
