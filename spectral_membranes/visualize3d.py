"""
3D mesh visualization with scalar field overlays.

Uses matplotlib Poly3DCollection for rendering triangulated surfaces
with per-vertex scalar coloring. Designed for publication-quality
Fiedler-vector overlays on cryo-ET membrane meshes.

Key functions:
  - plot_mesh_3d: render a mesh with scalar coloring from a single viewpoint
  - plot_mesh_multiview: 2x2 panel with four viewpoints
  - plot_fiedler_overlay: convenience wrapper for Fiedler vector display
  - plot_comparison_panel: side-by-side comparison of multiple meshes
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .types import Mesh


def _face_scalars(mesh: Mesh, vertex_values: np.ndarray) -> np.ndarray:
    """Average per-vertex values to per-face for coloring."""
    return vertex_values[mesh.faces].mean(axis=1)


def _make_poly_collection(
    mesh: Mesh,
    face_colors: np.ndarray,
    cmap: str = "RdBu_r",
    alpha: float = 0.95,
    edgecolor: str = "none",
    linewidth: float = 0.0,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[Poly3DCollection, matplotlib.cm.ScalarMappable]:
    """Build a Poly3DCollection with face coloring from a scalar array."""
    polys = mesh.vertices[mesh.faces]

    if vmin is None:
        vmin = float(np.nanmin(face_colors))
    if vmax is None:
        vmax = float(np.nanmax(face_colors))

    # Symmetric normalization for diverging colormaps
    if cmap in ("RdBu_r", "RdBu", "coolwarm", "bwr", "seismic", "PiYG"):
        abs_max = max(abs(vmin), abs(vmax))
        norm = Normalize(vmin=-abs_max, vmax=abs_max)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    rgba = mapper.to_rgba(face_colors)

    pc = Poly3DCollection(polys, alpha=alpha)
    pc.set_facecolor(rgba)
    if edgecolor != "none":
        pc.set_edgecolor(edgecolor)
        pc.set_linewidth(linewidth)
    else:
        pc.set_edgecolor("none")

    return pc, mapper


def _set_equal_aspect(ax, mesh: Mesh, pad: float = 0.1):
    """Force equal aspect ratio on 3D axes."""
    v = mesh.vertices
    ranges = np.ptp(v, axis=0)
    max_range = ranges.max() * (1 + pad)
    centers = v.mean(axis=0)
    for i, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        setter(centers[i] - max_range / 2, centers[i] + max_range / 2)


def plot_mesh_3d(
    mesh: Mesh,
    values: np.ndarray,
    outpath: str,
    title: str = "",
    cmap: str = "RdBu_r",
    cbar_label: str = "",
    elev: float = 25.0,
    azim: float = -60.0,
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 200,
    alpha: float = 0.95,
    show_edges: bool = False,
    bg_color: str = "white",
):
    """
    Render a 3D mesh surface colored by per-vertex scalar values.

    Parameters
    ----------
    mesh : Mesh
        Triangulated surface.
    values : ndarray
        Per-vertex scalar values (len = n_vertices).
    outpath : str
        Output image path.
    title : str
        Figure title.
    cmap : str
        Colormap name. 'RdBu_r' is default (diverging, good for Fiedler).
    cbar_label : str
        Colorbar label.
    elev, azim : float
        Camera elevation and azimuth in degrees.
    """
    face_vals = _face_scalars(mesh, values)

    fig = plt.figure(figsize=figsize, facecolor=bg_color)
    ax = fig.add_subplot(111, projection="3d", facecolor=bg_color)

    edge_kw = {"edgecolor": "#333333", "linewidth": 0.15} if show_edges else {"edgecolor": "none"}
    pc, mapper = _make_poly_collection(mesh, face_vals, cmap=cmap, alpha=alpha, **edge_kw)
    ax.add_collection3d(pc)
    _set_equal_aspect(ax, mesh)

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x", fontsize=8, labelpad=-2)
    ax.set_ylabel("y", fontsize=8, labelpad=-2)
    ax.set_zlabel("z", fontsize=8, labelpad=-2)
    ax.tick_params(labelsize=6, pad=0)

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    # Colorbar
    cbar = fig.colorbar(mapper, ax=ax, shrink=0.6, pad=0.08, aspect=20)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    # Clean up panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("lightgray")
    ax.yaxis.pane.set_edgecolor("lightgray")
    ax.zaxis.pane.set_edgecolor("lightgray")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, facecolor=bg_color, bbox_inches="tight")
    plt.close(fig)


def plot_mesh_multiview(
    mesh: Mesh,
    values: np.ndarray,
    outpath: str,
    title: str = "",
    cmap: str = "RdBu_r",
    cbar_label: str = "",
    dpi: int = 200,
    views: list[tuple[float, float]] | None = None,
    view_labels: list[str] | None = None,
):
    """
    2x2 panel showing the mesh from four viewpoints.

    Default views: front, side, top, perspective.
    """
    if views is None:
        views = [(0, -90), (0, 0), (90, 0), (25, -60)]
    if view_labels is None:
        view_labels = ["Front (xz)", "Side (yz)", "Top (xy)", "Perspective"]

    face_vals = _face_scalars(mesh, values)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={"projection": "3d"},
                              facecolor="white")
    axes = axes.ravel()

    pc_ref = None
    mapper = None
    for i, (ax, (elev, azim), label) in enumerate(zip(axes, views, view_labels)):
        edge_kw = {"edgecolor": "none"}
        pc, mapper = _make_poly_collection(mesh, face_vals, cmap=cmap, **edge_kw)
        ax.add_collection3d(pc)
        _set_equal_aspect(ax, mesh)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=5, pad=0)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("lightgray")
        ax.yaxis.pane.set_edgecolor("lightgray")
        ax.zaxis.pane.set_edgecolor("lightgray")
        ax.grid(True, alpha=0.15)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    # Shared colorbar
    if mapper is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(mapper, cax=cbar_ax)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=9)
        cbar.ax.tick_params(labelsize=7)

    fig.subplots_adjust(wspace=0.05, hspace=0.15, right=0.90)
    fig.savefig(outpath, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_fiedler_overlay(
    mesh: Mesh,
    fiedler_vector: np.ndarray,
    outpath: str,
    title: str = "Fiedler vector overlay",
    multiview: bool = True,
    dpi: int = 200,
):
    """
    Convenience wrapper for Fiedler-vector visualization.

    Uses RdBu_r colormap (red/blue diverging) with symmetric
    normalization so zero = white. The Fiedler zero-crossing
    shows the soft partition boundary.
    """
    if multiview:
        plot_mesh_multiview(
            mesh, fiedler_vector, outpath,
            title=title,
            cmap="RdBu_r",
            cbar_label="Fiedler value (u₂)",
            dpi=dpi,
        )
    else:
        plot_mesh_3d(
            mesh, fiedler_vector, outpath,
            title=title,
            cmap="RdBu_r",
            cbar_label="Fiedler value (u₂)",
            dpi=dpi,
        )


def plot_comparison_panel(
    meshes: dict[str, Mesh],
    fiedler_vectors: dict[str, np.ndarray],
    lambda2_values: dict[str, float],
    outpath: str,
    title: str = "Fiedler vector comparison",
    elev: float = 15.0,
    azim: float = -60.0,
    dpi: int = 200,
):
    """
    Side-by-side comparison of multiple meshes with Fiedler overlays.

    Each panel shows one mesh labeled with its name and λ₂ value.
    """
    names = list(meshes.keys())
    n = len(names)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                              subplot_kw={"projection": "3d"}, facecolor="white",
                              squeeze=False)

    # Find global vmin/vmax across all Fiedler vectors for consistent coloring
    all_vals = np.concatenate([v for v in fiedler_vectors.values()])
    abs_max = max(abs(all_vals.min()), abs(all_vals.max()))

    for idx, name in enumerate(names):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        mesh = meshes[name]
        fv = fiedler_vectors[name]
        l2 = lambda2_values[name]

        face_vals = _face_scalars(mesh, fv)
        pc, mapper = _make_poly_collection(
            mesh, face_vals, cmap="RdBu_r",
            vmin=-abs_max, vmax=abs_max,
            edgecolor="none",
        )
        ax.add_collection3d(pc)
        _set_equal_aspect(ax, mesh)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{name}\nλ₂ = {l2:.4f}", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=5, pad=0)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("lightgray")
        ax.yaxis.pane.set_edgecolor("lightgray")
        ax.zaxis.pane.set_edgecolor("lightgray")
        ax.grid(True, alpha=0.15)

    # Hide unused panels
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    # Shared colorbar
    norm = Normalize(vmin=-abs_max, vmax=abs_max)
    sm = cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Fiedler value (u₂)", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    fig.subplots_adjust(wspace=0.08, hspace=0.25, right=0.90)
    fig.savefig(outpath, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
