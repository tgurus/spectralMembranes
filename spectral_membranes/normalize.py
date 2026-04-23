"""
Vertex-count normalization for spectral features.

The normalized graph Laplacian's λ₂ scales as ~1/n for meshes of
the same geometry at different densities. This module provides
normalization functions that remove mesh-density artifacts while
preserving geometry-dependent bottleneck information.

Empirically verified scaling (necked_tube, n=264 to n=7680):
  - Raw λ₂: varies by factor of 31x
  - n·λ₂: CV = 2.6% (effectively constant)

Usage:
    from spectral_membranes.normalize import normalized_lambda2
    nl2 = normalized_lambda2(features, n_vertices)

    # Or for batch comparison:
    from spectral_membranes.normalize import normalize_feature_table
    df = normalize_feature_table(rows)
"""

from __future__ import annotations
import numpy as np
from .types import FeatureSet


def normalized_lambda2(
    features: FeatureSet,
    n_vertices: int,
) -> float | None:
    """
    Compute n·λ₂ normalization.

    For the symmetric normalized graph Laplacian, λ₂ scales
    approximately as 1/n for meshes of the same underlying
    geometry at different discretization densities. Multiplying
    by n removes this scaling.

    The resulting quantity preserves biological signal: meshes
    with genuine bottlenecks have lower n·λ₂ than uniformly
    connected meshes, regardless of vertex count.
    """
    if features.lambda2 is None:
        return None
    return float(n_vertices * features.lambda2)


def normalized_lambda3(
    features: FeatureSet,
    n_vertices: int,
) -> float | None:
    """Same normalization for λ₃."""
    if features.lambda3 is None:
        return None
    return float(n_vertices * features.lambda3)


def normalized_spectral_gap(
    features: FeatureSet,
) -> float | None:
    """
    λ₂/λ₃ ratio — already density-independent.

    Both eigenvalues scale as 1/n, so the ratio cancels
    the density dependence without any normalization.
    Useful for detecting near-degenerate modes (multiple
    competing bottlenecks).
    """
    if features.lambda2 is None or features.lambda3 is None:
        return None
    if features.lambda3 == 0:
        return None
    return float(features.lambda2 / features.lambda3)


def normalize_feature_row(
    row: dict,
    n_vertices_key: str = "n_vertices",
) -> dict:
    """
    Add normalized columns to a feature row dict.

    Expects the row to contain 'lambda2', 'lambda3', and
    an n_vertices column. Adds 'nlambda2', 'nlambda3',
    and 'spectral_gap_ratio' columns.
    """
    n = row.get(n_vertices_key)
    l2 = row.get("lambda2")
    l3 = row.get("lambda3")

    out = dict(row)

    if n is not None and l2 is not None and not np.isnan(l2):
        out["nlambda2"] = float(n * l2)
    else:
        out["nlambda2"] = np.nan

    if n is not None and l3 is not None and not np.isnan(l3):
        out["nlambda3"] = float(n * l3)
    else:
        out["nlambda3"] = np.nan

    if l2 is not None and l3 is not None and l3 != 0:
        if not (np.isnan(l2) or np.isnan(l3)):
            out["spectral_gap_ratio"] = float(l2 / l3)
        else:
            out["spectral_gap_ratio"] = np.nan
    else:
        out["spectral_gap_ratio"] = np.nan

    return out


def normalize_feature_table(
    rows: list[dict],
    n_vertices_key: str = "n_vertices",
) -> list[dict]:
    """Add normalized columns to every row in a feature table."""
    return [normalize_feature_row(r, n_vertices_key) for r in rows]


def verify_scaling(
    lambda2_values: list[float],
    n_vertices_values: list[int],
) -> dict:
    """
    Check whether the 1/n scaling holds for a set of measurements.

    Returns the log-log slope (should be ~-1.0 for valid scaling),
    the R² of the fit, and the CV of n·λ₂ (should be <0.10 for
    good normalization).
    """
    l2 = np.array(lambda2_values, dtype=float)
    nv = np.array(n_vertices_values, dtype=float)

    mask = (l2 > 0) & (nv > 0) & ~np.isnan(l2)
    if mask.sum() < 3:
        return {"slope": np.nan, "r_squared": np.nan,
                "nlambda2_cv": np.nan, "n_points": int(mask.sum())}

    log_n = np.log(nv[mask])
    log_l2 = np.log(l2[mask])

    # Linear fit in log-log space
    A = np.vstack([log_n, np.ones(mask.sum())]).T
    coeffs, residuals, _, _ = np.linalg.lstsq(A, log_l2, rcond=None)
    slope = float(coeffs[0])

    # R²
    ss_res = float(np.sum((log_l2 - A @ coeffs) ** 2))
    ss_tot = float(np.sum((log_l2 - log_l2.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # CV of n·λ₂
    nl2 = nv[mask] * l2[mask]
    cv = float(np.std(nl2) / np.mean(nl2)) if np.mean(nl2) > 0 else np.nan

    return {
        "slope": slope,
        "r_squared": r2,
        "nlambda2_cv": cv,
        "nlambda2_mean": float(np.mean(nl2)),
        "nlambda2_std": float(np.std(nl2)),
        "n_points": int(mask.sum()),
    }
