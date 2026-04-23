
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np

@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray

@dataclass
class FeatureSet:
    lambda2: float | None
    lambda3: float | None
    fiedler_ipr: float | None
    conductance_min: float | None
    heat_trace_tau: np.ndarray
    heat_trace_values: np.ndarray
    spectral_entropy: np.ndarray
    spectral_dimension: np.ndarray
    extra: dict[str, Any] = field(default_factory=dict)

@dataclass
class SpectralResult:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    operator_name: str
    metadata: dict[str, Any] = field(default_factory=dict)
