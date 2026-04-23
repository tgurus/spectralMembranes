# Spectral Membranes

Spectral connectivity descriptors for cryo-ET membrane surface meshes. Extends the [Surface Morphometrics](https://github.com/GrotjahnLab/surface_morphometrics) pipeline with graph Laplacian and cotangent Laplace–Beltrami operators that capture global membrane connectivity—complementary to existing local curvature and thickness measurements.

## Paper

> **Spectral connectivity descriptors separate inner from outer mitochondrial membranes in cryo-electron tomography**
> John Reimer Morales (2026)

## Key Findings

- **Cotangent operator separates IMM from OMM** in 22/31 vehicle-treated and 13/15 thapsigargin-treated paired mitochondria
- **Spectral-only features match local curvature features** in balanced accuracy (83.3%) for membrane-type classification
- **Cross-condition generalization**: the IMM < OMM hierarchy is preserved across vehicle and thapsigargin conditions
- **153 meshes** from 34 tomograms in a 2×2 factorial design (EMPIAR-11370)

## Installation

```bash
pip install -e .
```

For HFF mesh extraction (requires h5py):
```bash
pip install -e ".[extraction]"
```

## Quick Start

```python
from spectral_membranes.io import load_mesh
from spectral_membranes.extract_lcc import extract_lcc
from spectral_membranes.pipeline import run_dual_operator_pipeline

# Load and extract largest connected component
mesh = load_mesh("membrane.ply")
lcc_info = extract_lcc(mesh)
lcc_mesh = lcc_info["lcc_mesh"]

# Run dual-operator spectral analysis
result = run_dual_operator_pipeline(lcc_mesh, k=6)

# Graph spectral features
graph = result["graph"]
print(f"nλ₂ = {len(lcc_mesh.vertices) * graph.lambda2:.4f}")
print(f"Gap ratio = {graph.lambda2 / graph.lambda3:.4f}")

# Cotangent spectral features
from spectral_membranes.mesh import surface_area
A = surface_area(lcc_mesh)
mu2 = result["cotan_evals"][1]
print(f"Aμ₂ = {A * mu2:.4f}")
```

## Pipeline

The analysis pipeline operates on PLY mesh files (as produced by Surface Morphometrics):

1. **Mesh loading** — PLY, OBJ, OFF, NPZ formats
2. **Connected-component extraction** — Required because deposited cryo-ET meshes are often fragmented (mean 17.5 components per IMM)
3. **Graph Laplacian spectral analysis** — Normalized symmetric Laplacian, shift-invert eigensolver
4. **Cotangent Laplace–Beltrami analysis** — Vectorized assembly (37× faster than loop-based), shift-invert generalized eigensolver
5. **Local curvature features** — Mean curvature from cotangent Laplacian of position
6. **Quality control** — 10 automated checks per mesh with severity grading

### Extracting Meshes from EMPIAR-11370

The deposited HFF files can be extracted to PLY using:

```bash
python scripts/extract_hff_meshes.py /path/to/Membrane_surface_mesh/ --outdir meshes
```

Or use the Colab pipeline for one-click extraction of all 34 tomograms:
```
scripts/empiar_extraction_colab.py
```

## Dataset

EMPIAR-11370 contains 34 tomograms in a 2×2 factorial design:

| | Elongated | Fragmented |
|---|---|---|
| **Vehicle** | UE1–UE10 (10 tomos) | UF1–UF6 (6 tomos) |
| **Thapsigargin** | TE2–TE14 (13 tomos) | TF1–TF6 (5 tomos) |

Total: 153 meshes, 64 paired IMM/OMM mitochondria.

## Dependencies

- NumPy, SciPy, Matplotlib (core)
- h5py, pandas (for HFF extraction only)

## License

GPL-3.0

## Citation

If you use this software, please cite:

```bibtex
@article{morales2026spectral,
  title={Spectral connectivity descriptors separate inner from outer
         mitochondrial membranes in cryo-electron tomography},
  author={Morales, John Reimer},
  year={2026}
}
```

Also cite the Surface Morphometrics pipeline:

```bibtex
@article{barad2023surface,
  title={Quantifying organellar ultrastructure in cryo-electron tomography
         using a surface morphometrics pipeline},
  author={Barad, Benjamin A and Medina, Michaela and Fuentes, Daniel
          and Wiseman, R Luke and Grotjahn, Danielle A},
  journal={Journal of Cell Biology},
  volume={222},
  number={4},
  pages={e202204093},
  year={2023}
}
```
