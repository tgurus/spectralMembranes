
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from collections import defaultdict
from .types import Mesh

def plot_mesh_scalar(mesh: Mesh, values: np.ndarray, outpath: str, title: str = ""):
    fig, ax = plt.subplots(figsize=(6, 5))
    tri = Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1], triangles=mesh.faces)
    tpc = ax.tripcolor(tri, values, shading="gouraud")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(tpc, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)

def plot_heat_trace(tau, heat_vals, outpath: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(tau, heat_vals)
    ax.set_xlabel("tau")
    ax.set_ylabel("heat trace")
    ax.set_title("Heat trace")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)

def plot_group_distributions(rows, feature_name: str, group_name: str, outpath: str):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_name, "unknown"))].append(float(row[feature_name]))
    labels = list(groups.keys())
    values = [groups[label] for label in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(values, tick_labels=labels)
    ax.set_xlabel(group_name)
    ax.set_ylabel(feature_name)
    ax.set_title(f"{feature_name} by {group_name}")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
