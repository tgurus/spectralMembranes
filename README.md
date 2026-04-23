# Spectral Membranes

**Graph Laplacian and cotangent Laplace–Beltrami spectral descriptors for cryo-ET membrane surface meshes.**

This package extends the [Surface Morphometrics](https://github.com/GrotjahnLab/surface_morphometrics) pipeline with spectral connectivity descriptors that capture global membrane architecture — bottleneck structure, connectivity stiffness, and multiscale organization — complementary to local curvature and thickness measurements.

## Citation

> Reimer Morales, J. (2026). Spectral connectivity descriptors separate inner from outer mitochondrial membranes in cryo-electron tomography. *Submitted to Journal of Cell Biology.*

## Key Result

In matched within-mitochondrion comparisons on 15 paired IMM/OMM surfaces from EMPIAR-11370, the geometry-aware cotangent operator separated inner from outer mitochondrial membranes in all 15 pairs (Wilcoxon *p* = 6.1 × 10⁻⁵), while the normalized graph operator separated 11 of 15 (*p* = 0.018).

## What It Computes

- **Graph Laplacian**: algebraic connectivity (λ₂), Fiedler vector, spectral gap, inverse participation ratio, conductance
- **Cotangent Laplace–Beltrami**: geometry-aware eigenvalues (μ₂), area-normalized A·μ₂
- **Multiscale**: heat trace, spectral entropy, spectral dimension
- **Quality control**: 10 automated mesh and spectral checks
- **Sensitivity**: face dropout, vertex perturbation, boundary crop analysis
- **Normalization**: n·λ₂ vertex-count normalization for cross-mesh comparison

## Installation

```bash
pip install numpy scipy matplotlib
git clone https://github.com/[your-username]/spectral-membranes.git
cd spectral-membranes
```

## Quick Start

```python
from spectral_membranes import load_mesh, run_dual_operator_pipeline, full_qc
from spectral_membranes.normalize import normalized_lambda2

# Load a Surface Morphometrics .ply mesh
mesh = load_mesh("path/to/membrane.ply")

# Run both operators
result = run_dual_operator_pipeline(mesh)
graph = result["graph"]
print(f"λ₂ = {graph.lambda2:.4e}")
print(f"μ₂ = {result['cotan_evals'][1]:.4e}")

# QC
qc = full_qc(mesh, graph, mesh_id="my_membrane")
print(f"QC: {'PASS' if qc.passed else 'ISSUES'}")
```

## Data

The analyses in the paper use publicly deposited cryo-ET membrane meshes from:

- **EMPIAR-11370** — Barad et al. (2023), *J. Cell Biol.* 222, e202204093

## Dependencies

- NumPy
- SciPy
- Matplotlib (visualization only)

## License

MIT
