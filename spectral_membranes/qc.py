"""
Automated quality-control flags for spectral membrane analysis.

Evaluates mesh quality, boundary contamination, and spectral
reliability. Returns structured QC reports with severity-graded
flags for batch filtering, statistical stratification, and
transparent reporting.

Severity levels:
  PASS — no concern
  NOTE — minor; report but don't exclude
  WARN — moderate; interpret with caution
  FAIL — severe; exclude from primary analysis
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any
import numpy as np
from .types import Mesh, FeatureSet
from .mesh import (
    edge_lengths, boundary_vertices, boundary_edges,
    surface_area, unique_edges,
)


class Severity(IntEnum):
    PASS = 0
    NOTE = 1
    WARN = 2
    FAIL = 3


@dataclass
class QCFlag:
    name: str
    severity: Severity
    value: float | None
    threshold: float | None
    message: str

    def __str__(self):
        return f"[{self.severity.name}] {self.name}: {self.message}"


@dataclass
class QCReport:
    mesh_id: str
    flags: list[QCFlag] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def max_severity(self) -> Severity:
        return Severity(max((f.severity for f in self.flags), default=0))

    @property
    def passed(self) -> bool:
        return self.max_severity < Severity.FAIL

    @property
    def warnings(self) -> list[QCFlag]:
        return [f for f in self.flags if f.severity >= Severity.WARN]

    def summary_dict(self) -> dict:
        d = {"mesh_id": self.mesh_id, "qc_max_severity": self.max_severity.name,
             "qc_passed": self.passed, "qc_n_warnings": len(self.warnings)}
        d.update(self.metrics)
        for flag in self.flags:
            d[f"qc_{flag.name}"] = flag.severity.name
        return d

    def __str__(self):
        lines = [f"QC: {self.mesh_id} — {self.max_severity.name}"]
        for f in self.flags:
            if f.severity > Severity.PASS:
                lines.append(f"  {f}")
        return "\n".join(lines) if len(lines) > 1 else lines[0]


@dataclass
class QCThresholds:
    boundary_fraction_note: float = 0.05
    boundary_fraction_warn: float = 0.15
    boundary_fraction_fail: float = 0.30
    min_vertices_warn: int = 100
    min_vertices_fail: int = 30
    min_angle_warn: float = 5.0       # degrees
    min_angle_fail: float = 1.0
    degenerate_fraction_warn: float = 0.01
    degenerate_fraction_fail: float = 0.05
    edge_cv_warn: float = 1.0
    edge_cv_fail: float = 2.0
    max_components: int = 1
    lambda2_near_zero_warn: float = 1e-8
    fiedler_ipr_degenerate_warn: float = 0.5
    spectral_gap_ratio_warn: float = 0.95


DEFAULT_THRESHOLDS = QCThresholds()


def _triangle_min_angles(mesh: Mesh) -> np.ndarray:
    v, f = mesh.vertices, mesh.faces
    min_angles = np.full(len(f), 180.0)
    for c in range(3):
        u = v[f[:, (c + 1) % 3]] - v[f[:, c]]
        w = v[f[:, (c + 2) % 3]] - v[f[:, c]]
        cos_a = np.sum(u * w, axis=1) / (
            np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1) + 1e-15)
        ang = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
        min_angles = np.minimum(min_angles, ang)
    return min_angles


def _face_areas(mesh: Mesh) -> np.ndarray:
    v, f = mesh.vertices, mesh.faces
    return 0.5 * np.linalg.norm(
        np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)


def _connected_components(mesh: Mesh) -> int:
    n = len(mesh.vertices)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in unique_edges(mesh):
        ra, rb = find(int(e[0])), find(int(e[1]))
        if ra != rb:
            parent[ra] = rb

    used = set(mesh.faces.ravel())
    return len(set(find(v) for v in used))


def check_mesh_quality(
    mesh: Mesh,
    mesh_id: str = "unknown",
    thresholds: QCThresholds | None = None,
) -> QCReport:
    """Pre-pipeline QC: mesh geometry checks."""
    th = thresholds or DEFAULT_THRESHOLDS
    rpt = QCReport(mesh_id=mesh_id)
    fl, mt = rpt.flags, rpt.metrics
    n_v, n_f = len(mesh.vertices), len(mesh.faces)
    mt["n_vertices"], mt["n_faces"] = n_v, n_f

    # Mesh size
    if n_v < th.min_vertices_fail:
        fl.append(QCFlag("mesh_size", Severity.FAIL, n_v, th.min_vertices_fail,
                         f"{n_v} vertices — too small for reliable spectra"))
    elif n_v < th.min_vertices_warn:
        fl.append(QCFlag("mesh_size", Severity.WARN, n_v, th.min_vertices_warn,
                         f"{n_v} vertices — spectral features may be unstable"))
    else:
        fl.append(QCFlag("mesh_size", Severity.PASS, n_v, None, f"{n_v} vertices"))

    # Boundary fraction
    bmask = boundary_vertices(mesh)
    bf = float(np.mean(bmask)) if len(bmask) else 0.0
    b_edges = boundary_edges(mesh)
    bl = 0.0
    if len(b_edges):
        bl = float(np.linalg.norm(
            mesh.vertices[b_edges[:, 0]] - mesh.vertices[b_edges[:, 1]], axis=1).sum())
    mt["boundary_fraction"] = bf
    mt["boundary_length"] = bl

    if bf > th.boundary_fraction_fail:
        fl.append(QCFlag("boundary", Severity.FAIL, bf, th.boundary_fraction_fail,
                         f"Boundary {bf:.1%} — heavily truncated"))
    elif bf > th.boundary_fraction_warn:
        fl.append(QCFlag("boundary", Severity.WARN, bf, th.boundary_fraction_warn,
                         f"Boundary {bf:.1%} — interpret λ₂ with caution"))
    elif bf > th.boundary_fraction_note:
        fl.append(QCFlag("boundary", Severity.NOTE, bf, th.boundary_fraction_note,
                         f"Boundary {bf:.1%}"))
    else:
        fl.append(QCFlag("boundary", Severity.PASS, bf, None, f"Boundary {bf:.1%}"))

    # Triangle quality
    min_angles = _triangle_min_angles(mesh)
    global_min = float(min_angles.min()) if len(min_angles) else 0.0
    mt["min_triangle_angle"] = global_min
    mt["median_min_angle"] = float(np.median(min_angles)) if len(min_angles) else 0.0

    areas = _face_areas(mesh)
    degen = float((areas < 1e-12).mean()) if len(areas) else 0.0
    mt["degenerate_fraction"] = degen

    if degen > th.degenerate_fraction_fail:
        fl.append(QCFlag("degenerate_faces", Severity.FAIL, degen,
                         th.degenerate_fraction_fail,
                         f"{degen:.1%} degenerate — cotangent operator will fail"))
    elif degen > th.degenerate_fraction_warn:
        fl.append(QCFlag("degenerate_faces", Severity.WARN, degen,
                         th.degenerate_fraction_warn,
                         f"{degen:.1%} degenerate faces"))
    else:
        fl.append(QCFlag("degenerate_faces", Severity.PASS, degen, None, "OK"))

    if global_min < th.min_angle_fail:
        fl.append(QCFlag("triangle_angles", Severity.FAIL, global_min,
                         th.min_angle_fail, f"Min angle {global_min:.1f}°"))
    elif global_min < th.min_angle_warn:
        fl.append(QCFlag("triangle_angles", Severity.WARN, global_min,
                         th.min_angle_warn, f"Min angle {global_min:.1f}°"))
    else:
        fl.append(QCFlag("triangle_angles", Severity.PASS, global_min, None,
                         f"Min angle {global_min:.1f}°"))

    # Edge uniformity
    el = edge_lengths(mesh)
    if len(el):
        cv = float(np.std(el) / (np.mean(el) + 1e-15))
        mt["edge_length_cv"] = cv
        mt["median_edge_length"] = float(np.median(el))
        if cv > th.edge_cv_fail:
            fl.append(QCFlag("edge_uniformity", Severity.FAIL, cv, th.edge_cv_fail,
                             f"Edge CV {cv:.2f}"))
        elif cv > th.edge_cv_warn:
            fl.append(QCFlag("edge_uniformity", Severity.WARN, cv, th.edge_cv_warn,
                             f"Edge CV {cv:.2f} — consider remeshing"))
        else:
            fl.append(QCFlag("edge_uniformity", Severity.PASS, cv, None,
                             f"Edge CV {cv:.2f}"))

    # Connected components
    nc = _connected_components(mesh)
    mt["n_components"] = nc
    if nc > th.max_components:
        fl.append(QCFlag("connectivity", Severity.WARN, nc, th.max_components,
                         f"{nc} components — spectra reflect largest only"))
    else:
        fl.append(QCFlag("connectivity", Severity.PASS, nc, None, f"{nc} component(s)"))

    mt["surface_area"] = surface_area(mesh)
    return rpt


def check_spectral_quality(
    features: FeatureSet,
    mesh_id: str = "unknown",
    thresholds: QCThresholds | None = None,
) -> QCReport:
    """Post-pipeline QC: spectral feature checks."""
    th = thresholds or DEFAULT_THRESHOLDS
    rpt = QCReport(mesh_id=mesh_id)
    fl, mt = rpt.flags, rpt.metrics

    l2, l3 = features.lambda2, features.lambda3

    if l2 is not None:
        mt["lambda2"] = l2
        if l2 < th.lambda2_near_zero_warn:
            fl.append(QCFlag("lambda2_near_zero", Severity.WARN, l2,
                             th.lambda2_near_zero_warn,
                             f"λ₂ = {l2:.2e} — nearly disconnected"))
        else:
            fl.append(QCFlag("lambda2_near_zero", Severity.PASS, l2, None,
                             f"λ₂ = {l2:.6f}"))

    if l2 is not None and l3 is not None and l3 > 0:
        ratio = l2 / l3
        mt["spectral_gap_ratio"] = ratio
        if ratio > th.spectral_gap_ratio_warn:
            fl.append(QCFlag("spectral_gap", Severity.WARN, ratio,
                             th.spectral_gap_ratio_warn,
                             f"λ₂/λ₃ = {ratio:.3f} — near-degenerate"))
        else:
            fl.append(QCFlag("spectral_gap", Severity.PASS, ratio, None,
                             f"λ₂/λ₃ = {ratio:.3f}"))

    ipr = features.fiedler_ipr
    if ipr is not None:
        mt["fiedler_ipr"] = ipr
        if ipr > th.fiedler_ipr_degenerate_warn:
            fl.append(QCFlag("fiedler_localization", Severity.WARN, ipr,
                             th.fiedler_ipr_degenerate_warn,
                             f"IPR {ipr:.4f} — Fiedler nearly uniform"))
        else:
            fl.append(QCFlag("fiedler_localization", Severity.PASS, ipr, None,
                             f"IPR {ipr:.4f}"))

    return rpt


def full_qc(
    mesh: Mesh,
    features: FeatureSet,
    mesh_id: str = "unknown",
    thresholds: QCThresholds | None = None,
) -> QCReport:
    """Mesh-geometry + spectral QC, merged into one report."""
    m = check_mesh_quality(mesh, mesh_id, thresholds)
    s = check_spectral_quality(features, mesh_id, thresholds)
    combined = QCReport(mesh_id=mesh_id)
    combined.flags = m.flags + s.flags
    combined.metrics = {**m.metrics, **s.metrics}
    return combined
