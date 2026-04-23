
from __future__ import annotations
import numpy as np
from .features import (
    algebraic_connectivity,
    fiedler_conductance_min,
    heat_trace,
    inverse_participation_ratio,
    spectral_dimension,
    spectral_entropy,
)
from .graph_ops import adjacency_matrix, normalized_laplacian
from .cotan_ops import cotan_laplacian
from .preprocess import quality_control
from .spectra import generalized_smallest_eigenpairs, smallest_eigenpairs
from .types import FeatureSet, Mesh

def default_tau_grid(median_edge_length: float, n: int = 20) -> np.ndarray:
    h = max(float(median_edge_length), 1e-6)
    return np.geomspace(0.5 * h * h, 50.0 * h * h, n)

def run_graph_pipeline(mesh: Mesh, k: int = 50, weighted: bool = False) -> FeatureSet:
    qc = quality_control(mesh)
    W = adjacency_matrix(mesh, weighted=weighted)
    Lsym = normalized_laplacian(W)
    evals, evecs = smallest_eigenpairs(Lsym, k=k)
    tau = default_tau_grid(qc["median_edge_length"])
    ht = heat_trace(evals, tau)
    hs = spectral_entropy(evals, tau)
    ds = spectral_dimension(tau, ht)
    fiedler = evecs[:, 1] if evecs.shape[1] > 1 else np.zeros(len(mesh.vertices))
    return FeatureSet(
        lambda2=algebraic_connectivity(evals),
        lambda3=float(evals[2]) if len(evals) > 2 else None,
        fiedler_ipr=inverse_participation_ratio(fiedler),
        conductance_min=fiedler_conductance_min(fiedler, W),
        heat_trace_tau=tau,
        heat_trace_values=ht,
        spectral_entropy=hs,
        spectral_dimension=ds,
        extra={**qc, "weighted": bool(weighted), "fiedler_vector": fiedler},
    )

def run_dual_operator_pipeline(mesh: Mesh, k: int = 50, weighted: bool = False) -> dict:
    graph = run_graph_pipeline(mesh, k=k, weighted=weighted)
    C, M = cotan_laplacian(mesh)
    mu, phi = generalized_smallest_eigenpairs(C, M, k=k)
    return {"graph": graph, "cotan_evals": mu, "cotan_evecs": phi}
