"""
Sensitivity analysis framework for spectral membrane descriptors.

Tests robustness to: vertex subsampling, normal perturbation,
boundary cropping, and bootstrap face resampling. Produces
structured reports with CV, worst-case change, and per-feature
reliability verdicts.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import json, csv
from typing import Any
import numpy as np
from .types import Mesh, FeatureSet
from .pipeline import run_graph_pipeline
from .mesh import boundary_vertices, unique_edges


# ── Perturbation functions ──

def subsample_vertices(mesh: Mesh, keep_frac: float, seed: int = 0) -> Mesh:
    rng = np.random.RandomState(seed)
    bmask = boundary_vertices(mesh)
    interior = np.where(~bmask)[0]
    n_keep = max(1, int(len(interior) * keep_frac))
    kept_int = rng.choice(interior, size=n_keep, replace=False)
    kept = np.sort(np.concatenate([np.where(bmask)[0], kept_int]))
    idx_map = np.full(len(mesh.vertices), -1, dtype=int)
    idx_map[kept] = np.arange(len(kept))
    new_f = [idx_map[f] for f in mesh.faces if np.all(idx_map[f] >= 0)]
    faces = np.array(new_f, dtype=int) if new_f else np.empty((0, 3), dtype=int)
    return Mesh(vertices=mesh.vertices[kept], faces=faces)


def perturb_normals(mesh: Mesh, sigma: float, seed: int = 0) -> Mesh:
    rng = np.random.RandomState(seed)
    v, f = mesh.vertices, mesh.faces
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-15
    vn = np.zeros_like(v)
    for i in range(3):
        np.add.at(vn, f[:, i], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-15
    edges = unique_edges(mesh)
    med_h = float(np.median(np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)))
    disp = rng.normal(0, sigma * med_h, (len(v), 1)) * vn
    return Mesh(vertices=v + disp, faces=mesh.faces.copy())


def crop_boundary_ring(mesh: Mesh, n_rings: int = 1) -> Mesh:
    m = Mesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy())
    for _ in range(n_rings):
        bmask = boundary_vertices(m)
        interior = np.where(~bmask)[0]
        if len(interior) < 10:
            break
        idx_map = np.full(len(m.vertices), -1, dtype=int)
        idx_map[interior] = np.arange(len(interior))
        new_f = [idx_map[f] for f in m.faces if np.all(idx_map[f] >= 0)]
        if not new_f:
            break
        m = Mesh(vertices=m.vertices[interior], faces=np.array(new_f, dtype=int))
    return m


def bootstrap_faces(mesh: Mesh, seed: int = 0) -> Mesh:
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(mesh.faces), size=len(mesh.faces), replace=True)
    return Mesh(vertices=mesh.vertices.copy(), faces=mesh.faces[idx])


# ── Feature extraction ──

def _key_features(feat: FeatureSet) -> dict[str, float]:
    ht = np.asarray(feat.heat_trace_values, dtype=float)
    se = np.asarray(feat.spectral_entropy, dtype=float)
    sd = np.asarray(feat.spectral_dimension, dtype=float)
    return {
        "lambda2": feat.lambda2 if feat.lambda2 is not None else np.nan,
        "lambda3": feat.lambda3 if feat.lambda3 is not None else np.nan,
        "fiedler_ipr": feat.fiedler_ipr if feat.fiedler_ipr is not None else np.nan,
        "conductance_min": feat.conductance_min if feat.conductance_min is not None else np.nan,
        "heat_trace_mid": float(ht[len(ht)//2]) if len(ht) > 0 else np.nan,
        "spectral_entropy_mid": float(se[len(se)//2]) if len(se) > 0 else np.nan,
        "spectral_dimension_mid": float(sd[len(sd)//2]) if len(sd) > 0 else np.nan,
    }


# ── Results dataclasses ──

@dataclass
class PerturbationResult:
    perturbation_type: str
    levels: list
    level_label: str
    feature_table: list[dict[str, float]]
    reference: dict[str, float]

    def cv(self) -> dict[str, float]:
        out = {}
        for feat in self.reference:
            vals = [r[feat] for r in self.feature_table if not np.isnan(r.get(feat, np.nan))]
            if len(vals) > 1:
                a = np.array(vals)
                out[feat] = float(np.std(a) / (np.abs(np.mean(a)) + 1e-15))
            else:
                out[feat] = np.nan
        return out

    def max_relative_change(self) -> dict[str, float]:
        out = {}
        for feat in self.reference:
            ref = self.reference[feat]
            if np.isnan(ref) or abs(ref) < 1e-15:
                out[feat] = np.nan
                continue
            changes = [abs(r.get(feat, np.nan) - ref) / abs(ref)
                       for r in self.feature_table
                       if not np.isnan(r.get(feat, np.nan))]
            out[feat] = float(max(changes)) if changes else np.nan
        return out


@dataclass
class SensitivityReport:
    mesh_id: str
    results: dict[str, PerturbationResult] = field(default_factory=dict)
    reference_features: dict[str, float] = field(default_factory=dict)

    def reliability(self) -> dict[str, str]:
        all_cv: dict[str, list[float]] = {}
        for r in self.results.values():
            for feat, val in r.cv().items():
                all_cv.setdefault(feat, []).append(val)
        verdicts = {}
        for feat, vals in all_cv.items():
            v = [x for x in vals if not np.isnan(x)]
            if not v:
                verdicts[feat] = "UNKNOWN"
            elif max(v) < 0.05:
                verdicts[feat] = "STABLE"
            elif max(v) < 0.20:
                verdicts[feat] = "MODERATE"
            else:
                verdicts[feat] = "UNSTABLE"
        return verdicts

    def save_csv(self, path: str):
        rows = []
        for pt, r in self.results.items():
            for row in r.feature_table:
                flat = {"mesh_id": self.mesh_id, "perturbation": pt}
                flat.update(row)
                rows.append(flat)
        if not rows:
            return
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    def save_summary(self, path: str):
        def _clean(d):
            return {k: (float(v) if not np.isnan(v) else None) for k, v in d.items()}
        summary = {
            "mesh_id": self.mesh_id,
            "reference": _clean(self.reference_features),
            "cv": {pt: _clean(r.cv()) for pt, r in self.results.items()},
            "worst_case": {pt: _clean(r.max_relative_change()) for pt, r in self.results.items()},
            "reliability": self.reliability(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def print_summary(self):
        print(f"\nSensitivity: {self.mesh_id}")
        print("=" * 60)
        rel = self.reliability()
        cvs = {pt: r.cv() for pt, r in self.results.items()}
        ptypes = list(self.results.keys())
        print(f"{'Feature':<25s} {'Verdict':<12s}", end="")
        for pt in ptypes:
            print(f" {pt[:12]:<12s}", end="")
        print()
        print("-" * (37 + 13 * len(ptypes)))
        for feat in rel:
            v = rel[feat]
            tag = "✓" if v == "STABLE" else ("~" if v == "MODERATE" else "✗")
            print(f"{feat:<25s} {tag} {v:<10s}", end="")
            for pt in ptypes:
                c = cvs.get(pt, {}).get(feat, np.nan)
                print(f" {c:>10.4f}  " if not np.isnan(c) else f" {'—':>10s}  ", end="")
            print()


# ── Individual analyses ──

def analyze_subsampling(mesh, fracs=None, n_rep=3, k=50):
    fracs = fracs or [0.9, 0.75, 0.6, 0.5, 0.4]
    ref = _key_features(run_graph_pipeline(mesh, k=k))
    rows, lvls = [], []
    for fr in fracs:
        for rep in range(n_rep):
            try:
                m2 = subsample_vertices(mesh, fr, seed=rep*100+int(fr*1000))
                if len(m2.faces) < 10: continue
                row = _key_features(run_graph_pipeline(m2, k=min(k, len(m2.vertices)-2)))
                row.update(keep_fraction=fr, repeat=rep, n_verts=len(m2.vertices))
                rows.append(row); lvls.append(fr)
            except Exception: continue
    return PerturbationResult("subsampling", sorted(set(lvls)), "keep_fraction", rows, ref)


def analyze_normal_noise(mesh, sigmas=None, n_rep=3, k=50):
    sigmas = sigmas or [0.01, 0.03, 0.05, 0.1, 0.2]
    ref = _key_features(run_graph_pipeline(mesh, k=k))
    rows = []
    for s in sigmas:
        for rep in range(n_rep):
            try:
                m2 = perturb_normals(mesh, s, seed=rep*100+int(s*10000))
                row = _key_features(run_graph_pipeline(m2, k=k))
                row.update(sigma=s, repeat=rep)
                rows.append(row)
            except Exception: continue
    return PerturbationResult("normal_noise", sigmas, "sigma", rows, ref)


def analyze_boundary_crop(mesh, max_rings=3, k=50):
    ref = _key_features(run_graph_pipeline(mesh, k=k))
    rows, lvls = [], []
    for nr in range(1, max_rings+1):
        try:
            m2 = crop_boundary_ring(mesh, n_rings=nr)
            if len(m2.faces) < 10: continue
            row = _key_features(run_graph_pipeline(m2, k=min(k, len(m2.vertices)-2)))
            row.update(rings_removed=nr, n_verts=len(m2.vertices))
            rows.append(row); lvls.append(nr)
        except Exception: continue
    return PerturbationResult("boundary_crop", lvls, "rings_removed", rows, ref)


def analyze_face_bootstrap(mesh, n_boots=8, k=50):
    ref = _key_features(run_graph_pipeline(mesh, k=k))
    rows = []
    for b in range(n_boots):
        try:
            m2 = bootstrap_faces(mesh, seed=b)
            row = _key_features(run_graph_pipeline(m2, k=k))
            row["boot_idx"] = b
            rows.append(row)
        except Exception: continue
    return PerturbationResult("face_bootstrap", list(range(n_boots)), "boot_idx", rows, ref)


# ── Full suite ──

def run_sensitivity_suite(
    mesh: Mesh,
    mesh_id: str = "unknown",
    subsample_fractions=None,
    noise_sigmas=None,
    crop_max_rings=3,
    n_bootstraps=8,
    n_repeats=3,
    k=50,
    verbose=True,
) -> SensitivityReport:
    report = SensitivityReport(mesh_id=mesh_id)
    report.reference_features = _key_features(run_graph_pipeline(mesh, k=k))

    if verbose: print(f"Sensitivity suite: {mesh_id}")
    if verbose: print("  [1/4] Subsampling...")
    report.results["subsampling"] = analyze_subsampling(mesh, subsample_fractions, n_repeats, k)
    if verbose: print("  [2/4] Normal noise...")
    report.results["normal_noise"] = analyze_normal_noise(mesh, noise_sigmas, n_repeats, k)
    if verbose: print("  [3/4] Boundary crop...")
    report.results["boundary_crop"] = analyze_boundary_crop(mesh, crop_max_rings, k)
    if verbose: print("  [4/4] Face bootstrap...")
    report.results["face_bootstrap"] = analyze_face_bootstrap(mesh, n_bootstraps, k)

    if verbose: report.print_summary()
    return report
