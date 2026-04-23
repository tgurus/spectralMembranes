"""
Synthetic mesh generators for testing spectral membrane descriptors.

Each generator produces a Mesh (vertices, faces) representing a
biologically motivated 3D surface. The geometries are designed to
test whether the spectral pipeline correctly detects bottlenecks,
branching, and multiscale organization.

Generators:
  - necked_tube: cylinder with a single Gaussian constriction
                 (models a crista junction or ER tubule neck)
  - double_neck_tube: two constrictions at different depths
                      (models multiple crista junctions along a tube)
  - branching_tube: Y-junction where one tube branches into two
                    (models ER branching or mitochondrial network fork)
  - crista_sheet: flattened disc connected to a cylindrical stalk
                  through a narrow junction (models crista body + junction)
  - smooth_tube: uniform cylinder with no constriction (negative control)

All surfaces are open (not capped), matching cryo-ET Surface
Morphometrics conventions where membranes are open meshes.
"""

from __future__ import annotations
import numpy as np
from .types import Mesh


def _tube_ring(theta: np.ndarray, radius: float, z: float) -> np.ndarray:
    """Generate a ring of vertices at height z with given radius."""
    return np.column_stack([
        radius * np.cos(theta),
        radius * np.sin(theta),
        np.full_like(theta, z),
    ])


def _stitch_rings(n_theta: int, n_rings: int) -> np.ndarray:
    """Triangulate adjacent rings of vertices into faces."""
    faces = []
    for iz in range(n_rings - 1):
        for it in range(n_theta):
            a = iz * n_theta + it
            b = iz * n_theta + (it + 1) % n_theta
            c = (iz + 1) * n_theta + it
            d = (iz + 1) * n_theta + (it + 1) % n_theta
            faces.append([a, b, c])
            faces.append([b, d, c])
    return np.array(faces, dtype=int)


def smooth_tube(
    n_theta: int = 32,
    n_z: int = 48,
    radius: float = 0.5,
    length: float = 4.0,
) -> Mesh:
    """Uniform cylinder — negative control with no constriction."""
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    z_vals = np.linspace(-length / 2, length / 2, n_z)
    verts = np.vstack([_tube_ring(theta, radius, z) for z in z_vals])
    faces = _stitch_rings(n_theta, n_z)
    return Mesh(vertices=verts, faces=faces)


def necked_tube(
    n_theta: int = 32,
    n_z: int = 60,
    radius: float = 0.5,
    length: float = 5.0,
    neck_depth: float = 0.55,
    neck_width: float = 0.6,
    neck_position: float = 0.0,
) -> Mesh:
    """
    Cylinder with a single Gaussian constriction.

    Models a crista junction or constricted ER tubule.
    neck_depth: fraction of radius removed at the neck (0-1).
    neck_width: Gaussian sigma of the constriction profile.
    neck_position: z-coordinate of the neck center.
    """
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    z_vals = np.linspace(-length / 2, length / 2, n_z)

    verts = []
    for z in z_vals:
        profile = 1.0 - neck_depth * np.exp(
            -0.5 * ((z - neck_position) / max(neck_width, 1e-6)) ** 2
        )
        r = radius * max(profile, 0.05)  # floor to prevent degenerate triangles
        verts.append(_tube_ring(theta, r, z))

    return Mesh(
        vertices=np.vstack(verts),
        faces=_stitch_rings(n_theta, n_z),
    )


def double_neck_tube(
    n_theta: int = 32,
    n_z: int = 80,
    radius: float = 0.5,
    length: float = 7.0,
    neck_depth_1: float = 0.55,
    neck_depth_2: float = 0.35,
    neck_width: float = 0.5,
    separation: float = 2.5,
) -> Mesh:
    """
    Cylinder with two Gaussian constrictions at different depths.

    Models multiple crista junctions along a tubular membrane.
    The Fiedler vector should localize at the deeper constriction;
    λ₃ should pick up the secondary constriction.
    """
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    z_vals = np.linspace(-length / 2, length / 2, n_z)

    pos1 = -separation / 2
    pos2 = separation / 2

    verts = []
    for z in z_vals:
        g1 = neck_depth_1 * np.exp(-0.5 * ((z - pos1) / max(neck_width, 1e-6)) ** 2)
        g2 = neck_depth_2 * np.exp(-0.5 * ((z - pos2) / max(neck_width, 1e-6)) ** 2)
        profile = 1.0 - g1 - g2
        r = radius * max(profile, 0.05)
        verts.append(_tube_ring(theta, r, z))

    return Mesh(
        vertices=np.vstack(verts),
        faces=_stitch_rings(n_theta, n_z),
    )


def branching_tube(
    n_theta: int = 24,
    n_z_trunk: int = 30,
    n_z_branch: int = 25,
    radius: float = 0.4,
    trunk_length: float = 3.0,
    branch_length: float = 2.5,
    branch_angle: float = 35.0,
    junction_taper: float = 0.7,
) -> Mesh:
    """
    Y-junction: a trunk tube that branches into two daughter tubes.

    Models ER branching or mitochondrial network forks. The junction
    region creates a natural bottleneck that λ₂ and the Fiedler
    vector should detect.

    The two branches diverge at ±branch_angle degrees from the
    trunk axis (z-axis).
    """
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    angle_rad = np.radians(branch_angle)

    # --- Trunk: from z = -trunk_length to z = 0 ---
    z_trunk = np.linspace(-trunk_length, 0, n_z_trunk)
    trunk_verts = []
    for z in z_trunk:
        # Slight taper toward junction
        taper = 1.0 - (1.0 - junction_taper) * max(0, z / 0.01) if z > -0.5 else 1.0
        r = radius * max(taper, junction_taper)
        trunk_verts.append(_tube_ring(theta, r, z))
    trunk_verts = np.vstack(trunk_verts)
    trunk_faces = _stitch_rings(n_theta, n_z_trunk)

    # --- Branch A: diverging at +angle ---
    def branch_ring(t_param, direction_sign):
        """Generate vertices along a branch diverging from origin."""
        z_local = t_param * branch_length
        # Center of the branch ring moves away from trunk axis
        cx = direction_sign * z_local * np.sin(angle_rad)
        cy = 0.0
        cz = z_local * np.cos(angle_rad)
        # Radius expands back to full after junction
        expand = junction_taper + (1.0 - junction_taper) * min(1.0, t_param * 3)
        r = radius * expand

        # Rotate the ring to face along the branch direction
        # Branch direction: (sin(angle)*sign, 0, cos(angle))
        ring = _tube_ring(theta, r, 0.0)
        # Rotate around y-axis by direction_sign * angle
        ca, sa = np.cos(direction_sign * angle_rad), np.sin(direction_sign * angle_rad)
        rot = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
        ring = ring @ rot.T
        ring[:, 0] += cx
        ring[:, 1] += cy
        ring[:, 2] += cz
        return ring

    t_branch = np.linspace(0.01, 1.0, n_z_branch)

    branch_a_verts = np.vstack([branch_ring(t, +1) for t in t_branch])
    branch_b_verts = np.vstack([branch_ring(t, -1) for t in t_branch])

    # Offset face indices for branches
    n_trunk_v = len(trunk_verts)
    n_branch_v = len(branch_a_verts)

    branch_a_faces = _stitch_rings(n_theta, n_z_branch) + n_trunk_v
    branch_b_faces = _stitch_rings(n_theta, n_z_branch) + n_trunk_v + n_branch_v

    # Connect trunk top ring to each branch bottom ring
    trunk_top_start = (n_z_trunk - 1) * n_theta
    branch_a_start = n_trunk_v
    branch_b_start = n_trunk_v + n_branch_v

    junction_faces_a = []
    junction_faces_b = []
    for it in range(n_theta):
        a = trunk_top_start + it
        b = trunk_top_start + (it + 1) % n_theta
        ca = branch_a_start + it
        da = branch_a_start + (it + 1) % n_theta
        junction_faces_a.append([a, b, ca])
        junction_faces_a.append([b, da, ca])

        cb = branch_b_start + it
        db = branch_b_start + (it + 1) % n_theta
        junction_faces_b.append([a, b, cb])
        junction_faces_b.append([b, db, cb])

    all_verts = np.vstack([trunk_verts, branch_a_verts, branch_b_verts])
    all_faces = np.vstack([
        trunk_faces,
        branch_a_faces,
        branch_b_faces,
        np.array(junction_faces_a, dtype=int),
        np.array(junction_faces_b, dtype=int),
    ])

    return Mesh(vertices=all_verts, faces=all_faces)


def crista_sheet(
    n_radial: int = 20,
    n_theta: int = 36,
    n_z_stalk: int = 20,
    sheet_radius: float = 1.5,
    stalk_radius: float = 0.2,
    stalk_length: float = 1.5,
    junction_width: float = 0.3,
) -> Mesh:
    """
    Flattened disc (crista body) connected to a cylindrical stalk
    (crista junction) through a narrow neck.

    Models the characteristic crista morphology: a broad lamellar
    body connected to the inner boundary membrane through a
    constricted junction. The Fiedler vector should strongly
    localize at the junction.
    """
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

    # --- Stalk: cylinder from z = 0 to z = stalk_length ---
    z_stalk = np.linspace(0, stalk_length, n_z_stalk)
    stalk_verts = []
    for z in z_stalk:
        r = stalk_radius
        stalk_verts.append(_tube_ring(theta, r, z))
    stalk_verts = np.vstack(stalk_verts)
    stalk_faces = _stitch_rings(n_theta, n_z_stalk)

    # --- Junction transition: stalk top to disc edge ---
    # Flare from stalk_radius to sheet_radius over a few rings
    n_junction = 8
    t_junction = np.linspace(0, 1, n_junction + 1)[1:]  # skip 0 (= stalk top)
    junction_verts = []
    for t in t_junction:
        # Sigmoid flare
        s = 1.0 / (1.0 + np.exp(-10 * (t - 0.5)))
        r = stalk_radius + (sheet_radius - stalk_radius) * s
        z = stalk_length + junction_width * t
        junction_verts.append(_tube_ring(theta, r, z))
    junction_verts = np.vstack(junction_verts)

    n_stalk_v = len(stalk_verts)
    junction_faces = _stitch_rings(n_theta, n_junction) + n_stalk_v

    # Connect stalk top to junction bottom
    stalk_top = (n_z_stalk - 1) * n_theta
    junc_bottom = n_stalk_v
    connect_faces = []
    for it in range(n_theta):
        a = stalk_top + it
        b = stalk_top + (it + 1) % n_theta
        c = junc_bottom + it
        d = junc_bottom + (it + 1) % n_theta
        connect_faces.append([a, b, c])
        connect_faces.append([b, d, c])

    # --- Disc cap: radial rings from the junction top edge inward ---
    disc_z = stalk_length + junction_width
    n_disc_rings = n_radial
    disc_verts = []
    r_outer = sheet_radius
    for i_ring in range(1, n_disc_rings):
        # Rings from outer edge inward, with slight z-variation for
        # a more natural lamellar shape (slight curvature)
        t = i_ring / n_disc_rings
        r = r_outer * (1.0 - t)
        z_offset = 0.05 * np.sin(np.pi * t)  # gentle dome
        if r < 0.01:
            r = 0.01
        disc_verts.append(_tube_ring(theta, r, disc_z + z_offset))

    disc_verts = np.vstack(disc_verts) if disc_verts else np.empty((0, 3))

    # Stitch junction top to first disc ring
    n_junc_v = len(junction_verts)
    disc_start = n_stalk_v + n_junc_v
    junc_top = n_stalk_v + (n_junction - 1) * n_theta

    connect_disc = []
    for it in range(n_theta):
        a = junc_top + it
        b = junc_top + (it + 1) % n_theta
        c = disc_start + it
        d = disc_start + (it + 1) % n_theta
        connect_disc.append([a, b, c])
        connect_disc.append([b, d, c])

    # Stitch disc rings
    n_disc_v = len(disc_verts)
    n_disc_actual = n_disc_rings - 1
    disc_faces = _stitch_rings(n_theta, n_disc_actual) + disc_start if n_disc_actual > 1 else np.empty((0, 3), dtype=int)

    # Assemble
    all_verts = np.vstack([stalk_verts, junction_verts, disc_verts])
    face_blocks = [stalk_faces, junction_faces]
    if len(connect_faces):
        face_blocks.append(np.array(connect_faces, dtype=int))
    if len(connect_disc):
        face_blocks.append(np.array(connect_disc, dtype=int))
    if len(disc_faces):
        face_blocks.append(disc_faces)
    all_faces = np.vstack(face_blocks)

    return Mesh(vertices=all_verts, faces=all_faces)


# --- Convenience catalog ---
CATALOG = {
    "smooth_tube": smooth_tube,
    "necked_tube": necked_tube,
    "double_neck": double_neck_tube,
    "branching": branching_tube,
    "crista_sheet": crista_sheet,
}


def generate_catalog(save_dir: str | None = None) -> dict[str, Mesh]:
    """Generate all synthetic meshes. Optionally save as .npz files."""
    import os
    meshes = {}
    for name, fn in CATALOG.items():
        meshes[name] = fn()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"{name}.npz")
            np.savez(path, vertices=meshes[name].vertices, faces=meshes[name].faces)
    return meshes
