"""
Extract the largest connected component from a mesh.
Returns a new Mesh with reindexed vertices/faces, plus mapping info.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import sys
# removed sys.path hack for package install
from .types import Mesh
from .mesh import unique_edges


def extract_lcc(mesh: Mesh) -> dict:
    """Extract the largest connected component from a mesh.
    
    Returns dict with:
        lcc_mesh: Mesh object for the LCC
        lcc_vertices_orig: original vertex indices in LCC
        n_components: total number of connected components
        lcc_fraction: fraction of vertices in LCC
        component_labels: per-vertex component labels
        component_sizes: list of (component_id, n_vertices) sorted by size
    """
    n = len(mesh.vertices)
    edges = unique_edges(mesh)
    
    # Build adjacency for scipy
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(len(row), dtype=np.float32)
    adj = csr_matrix((data, (row, col)), shape=(n, n))
    
    n_components, labels = connected_components(adj, directed=False)
    
    # Find LCC
    used_verts = set(mesh.faces.ravel())
    # Count only vertices that appear in faces
    comp_sizes = {}
    for v in used_verts:
        c = labels[v]
        comp_sizes[c] = comp_sizes.get(c, 0) + 1
    
    lcc_label = max(comp_sizes, key=comp_sizes.get)
    lcc_mask = (labels == lcc_label)
    
    # Extract LCC submesh
    lcc_vert_indices = np.where(lcc_mask)[0]
    old_to_new = np.full(n, -1, dtype=np.int64)
    old_to_new[lcc_vert_indices] = np.arange(len(lcc_vert_indices))
    
    # Keep only faces where all 3 vertices are in LCC
    face_mask = np.all(lcc_mask[mesh.faces], axis=1)
    lcc_faces = old_to_new[mesh.faces[face_mask]]
    lcc_verts = mesh.vertices[lcc_vert_indices]
    
    # Component size summary
    comp_summary = sorted(comp_sizes.items(), key=lambda x: -x[1])
    
    return {
        'lcc_mesh': Mesh(vertices=lcc_verts, faces=lcc_faces),
        'lcc_vertices_orig': lcc_vert_indices,
        'n_components': n_components,
        'lcc_fraction': len(lcc_vert_indices) / n if n > 0 else 0.0,
        'lcc_n_vertices': len(lcc_vert_indices),
        'lcc_n_faces': int(face_mask.sum()),
        'component_labels': labels,
        'component_sizes': comp_summary,
    }


def extract_top_k_components(mesh: Mesh, k: int = 3) -> list:
    """Extract top-k largest connected components as separate Mesh objects."""
    n = len(mesh.vertices)
    edges = unique_edges(mesh)
    
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(len(row), dtype=np.float32)
    adj = csr_matrix((data, (row, col)), shape=(n, n))
    
    n_components, labels = connected_components(adj, directed=False)
    
    used_verts = set(mesh.faces.ravel())
    comp_sizes = {}
    for v in used_verts:
        c = labels[v]
        comp_sizes[c] = comp_sizes.get(c, 0) + 1
    
    top_comps = sorted(comp_sizes.items(), key=lambda x: -x[1])[:k]
    
    results = []
    for comp_label, comp_size in top_comps:
        mask = (labels == comp_label)
        vert_idx = np.where(mask)[0]
        old_to_new = np.full(n, -1, dtype=np.int64)
        old_to_new[vert_idx] = np.arange(len(vert_idx))
        
        face_mask = np.all(mask[mesh.faces], axis=1)
        sub_faces = old_to_new[mesh.faces[face_mask]]
        sub_verts = mesh.vertices[vert_idx]
        
        results.append({
            'mesh': Mesh(vertices=sub_verts, faces=sub_faces),
            'n_vertices': len(vert_idx),
            'n_faces': int(face_mask.sum()),
            'component_label': comp_label,
        })
    
    return results
