
from __future__ import annotations
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

def _coerce(value):
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        if any(ch in text for ch in [".", "e", "E"]):
            return float(text)
        return int(text)
    except ValueError:
        return text

def parse_key_value_stem(stem: str) -> dict:
    meta = {}
    for token in stem.split("__"):
        if "=" in token:
            key, value = token.split("=", 1)
            if key.strip():
                meta[key.strip()] = _coerce(value)
    return meta

def _read_manifest_rows(path: Path) -> list[dict]:
    ext = path.suffix.lower()
    if ext == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if ext == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [dict(row) for row in data]
        if isinstance(data, dict):
            return [dict(row) for row in data.get("rows", [data])]
    raise ValueError(f"Unsupported manifest format: {path}")

@dataclass
class SurfaceMorphometricsAdapter:
    project_root: str
    manifest_path: str | None = None
    mesh_extensions: tuple[str, ...] = (".npz", ".obj", ".ply", ".off")
    ignore_dirs: tuple[str, ...] = ("per_mesh", "example_output", "__pycache__", ".git", ".pytest_cache")
    _manifest_lookup: dict[str, dict] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.project_root = str(Path(self.project_root).resolve())
        manifest = self._resolve_manifest()
        if manifest is not None:
            self._manifest_lookup = self._index_manifest(_read_manifest_rows(manifest))

    def _resolve_manifest(self) -> Path | None:
        if self.manifest_path:
            path = Path(self.manifest_path)
            if not path.is_absolute():
                path = Path(self.project_root) / path
            return path.resolve()
        root = Path(self.project_root)
        for name in ("surface_morphometrics_manifest.csv", "surface_morphometrics_manifest.json", "manifest.csv", "manifest.json", "metadata.csv", "metadata.json"):
            candidate = root / name
            if candidate.exists():
                return candidate.resolve()
        return None

    def _index_manifest(self, rows: Iterable[Mapping[str, object]]) -> dict[str, dict]:
        lookup = {}
        for row in rows:
            clean = {str(k): row[k] for k in row.keys()}
            for field in ("relative_path", "mesh_id", "stem", "filename"):
                value = clean.get(field)
                if value:
                    lookup[str(value)] = clean
        return lookup

    def collect_mesh_paths(self) -> list[str]:
        root = Path(self.project_root)
        out = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in self.mesh_extensions:
                continue
            rel_parts = path.relative_to(root).parts
            if any(part in self.ignore_dirs for part in rel_parts):
                continue
            out.append(str(path))
        return sorted(out)

    def relative_path(self, path: str) -> str:
        return str(Path(path).resolve().relative_to(Path(self.project_root)))

    def parse_metadata(self, path: str) -> dict:
        path_obj = Path(path).resolve()
        rel = self.relative_path(str(path_obj))
        meta = {}
        meta.update(self._infer_from_path(rel))
        meta.update(parse_key_value_stem(path_obj.stem))
        manifest = self._manifest_lookup.get(rel) or self._manifest_lookup.get(path_obj.stem) or self._manifest_lookup.get(path_obj.name)
        if manifest:
            meta.update({k: _coerce(v) for k, v in manifest.items()})
        sidecar = path_obj.with_suffix(".json")
        if sidecar.exists():
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    meta.update(data)
            except json.JSONDecodeError:
                pass
        meta.setdefault("relative_path", rel)
        meta.setdefault("filename", path_obj.name)
        meta.setdefault("stem", path_obj.stem)
        return meta

    def _infer_from_path(self, rel: str) -> dict:
        parts = Path(rel).parts[:-1]
        meta = {}
        if len(parts) >= 1:
            meta.setdefault("collection", parts[0])
        labels = [p.lower() for p in parts]
        for raw, low in zip(parts, labels):
            if low.startswith("tomo") or low.startswith("tomogram"):
                meta.setdefault("tomogram", raw)
            elif low.startswith("cell"):
                meta.setdefault("cell", raw)
            elif low in {"mito", "mitochondria", "er", "golgi", "lysosome", "nucleus"}:
                meta.setdefault("organelle", raw)
            elif low in {"control", "stress", "treated", "untreated", "vehicle", "drug"}:
                meta.setdefault("condition", raw)
        if len(parts) >= 3:
            meta.setdefault("condition", parts[0])
            meta.setdefault("organelle", parts[1])
            meta.setdefault("tomogram", parts[2])
        elif len(parts) == 2:
            meta.setdefault("organelle", parts[0])
            meta.setdefault("tomogram", parts[1])
        return meta

def make_surface_morphometrics_metadata_parser(project_root: str, manifest_path: str | None = None):
    adapter = SurfaceMorphometricsAdapter(project_root=project_root, manifest_path=manifest_path)
    return adapter.parse_metadata
