
from __future__ import annotations
import numpy as np

def algebraic_connectivity(evals: np.ndarray) -> float | None:
    return float(evals[1]) if len(evals) > 1 else None

def inverse_participation_ratio(vec: np.ndarray, weights: np.ndarray | None = None) -> float | None:
    if vec.size == 0:
        return None
    if weights is None:
        denom = float(np.sum(vec ** 2) ** 2)
        return float(np.sum(vec ** 4) / denom) if denom > 0 else None
    denom = float(np.sum(weights * vec ** 2) ** 2)
    return float(np.sum(weights * vec ** 4) / denom) if denom > 0 else None

def conductance_from_partition(mask: np.ndarray, W) -> float:
    degrees = np.asarray(W.sum(axis=1)).ravel()
    cut = float(W[mask][:, ~mask].sum())
    vol_s = float(degrees[mask].sum())
    vol_c = float(degrees[~mask].sum())
    denom = min(vol_s, vol_c)
    return cut / denom if denom > 0 else np.inf

def fiedler_conductance_min(fiedler_vec: np.ndarray, W) -> float | None:
    if fiedler_vec.size == 0:
        return None
    quantiles = np.linspace(0.05, 0.95, 19)
    thresholds = np.unique(np.quantile(fiedler_vec, quantiles))
    values = []
    for t in thresholds:
        mask = fiedler_vec >= t
        if mask.any() and (~mask).any():
            values.append(conductance_from_partition(mask, W))
    return float(min(values)) if values else None

def heat_trace(evals: np.ndarray, tau: np.ndarray) -> np.ndarray:
    return np.asarray([np.exp(-t * evals).sum() for t in tau], dtype=float)

def spectral_entropy(evals: np.ndarray, tau: np.ndarray) -> np.ndarray:
    out = []
    for t in tau:
        p = np.exp(-t * evals)
        p /= p.sum()
        out.append(-(p * np.log(p + 1e-15)).sum())
    return np.asarray(out, dtype=float)

def spectral_dimension(tau: np.ndarray, heat_vals: np.ndarray) -> np.ndarray:
    return -2.0 * np.gradient(np.log(heat_vals + 1e-15), np.log(tau + 1e-15))
