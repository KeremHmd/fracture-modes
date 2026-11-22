"""Numpy-only mesh metrics (edges, normals, dent depth, smoothness).

All functions work on plain arrays so they can be unit-tested without the
fracture stack. Definitions:

* ``mesh_edges(F)``: unique undirected edges of a triangle mesh.
* ``vertex_normals(V, F)``: area-weighted vertex normals, unit length.
* ``blockiness(disp, V, edges, patch)``: how much of the displacement
  variation sits on the seams between cage-node patches.

  For every fine edge (i, j) the gradient energy is
  ``g_ij = |d_i - d_j|^2 / |x_i - x_j|^2`` (squared directional derivative of
  the displacement along the edge, rest positions). ``patch[i]`` is the
  nearest cage node of fine vertex i, so the edges split into *seam* edges
  (``patch[i] != patch[j]``) and *interior* edges (same patch). With
  ``G_seam`` and ``G_int`` the mean energies of the two classes,

      blockiness = G_seam / (G_seam + G_int)          in [0, 1]

  A nearest-node lift gives every vertex of a patch the same displacement,
  so ``G_int = 0`` and blockiness is exactly 1 (all variation is in steps at
  the seams). A smooth field varies about equally along every edge and
  scores about 0.5. Returns ``None`` when the displacement is (numerically)
  zero or a class has no edges.
* ``dent_metrics(...)``: dent depth along the rest surface normal at the hit
  point and the offset between the hit point and the deepest dent.
* ``dent_volume(...)``: inward displacement integrated over the surface, and
  the area- and depth-weighted dent centre (its offset from the hit point is a
  steadier location measure than the single deepest vertex on a flat floor).
"""

import numpy as np


def mesh_edges(F):
    F = np.asarray(F, dtype=np.int64)
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e.sort(axis=1)
    return np.unique(e, axis=0)


def vertex_normals(V, F):
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=np.int64)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])  # length = 2 * area
    n = np.zeros_like(V)
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(ln, 1e-300)


def edge_gradient_energy(disp, V, edges):
    disp = np.asarray(disp, dtype=float)
    V = np.asarray(V, dtype=float)
    i, j = edges[:, 0], edges[:, 1]
    l2 = np.sum((V[i] - V[j]) ** 2, axis=1)
    dd = np.sum((disp[i] - disp[j]) ** 2, axis=1)
    return dd / np.maximum(l2, 1e-300)


def blockiness(disp, V, edges, patch, tol=1e-12):
    """Seam share of the displacement-gradient energy (see module docstring)."""
    disp = np.asarray(disp, dtype=float)
    if disp.size == 0 or float(np.abs(disp).max()) <= tol:
        return None
    g = edge_gradient_energy(disp, V, edges)
    patch = np.asarray(patch)
    seam = patch[edges[:, 0]] != patch[edges[:, 1]]
    if not seam.any() or seam.all():
        return None
    gs, gi = float(g[seam].mean()), float(g[~seam].mean())
    if gs + gi <= 0.0:
        return None
    return gs / (gs + gi)


def dent_metrics(hit_point, rest_V, disp, normals, radius=None):
    """Dent depth and location for one surface.

    ``rest_V`` are the rest fine vertices (same frame as ``hit_point``),
    ``disp`` their displacement, ``normals`` outward rest normals.

    * ``depth_at_hit``: ``-(d . n)`` averaged (inverse-distance weights) over
      the fine vertices within ``radius`` of the hit point's nearest vertex on
      the rest surface, or that vertex alone when ``radius`` is None.
      Positive = pushed inward.
    * ``max_depth``: the largest ``-(d . n)`` over all vertices.
    * ``max_location``: rest position of that vertex.
    * ``offset``: distance from the hit point to ``max_location``.
    * ``nearest_distance``: how far the hit point is from the rest surface.
    """
    rest_V = np.asarray(rest_V, dtype=float)
    disp = np.asarray(disp, dtype=float)
    normals = np.asarray(normals, dtype=float)
    p = np.asarray(hit_point, dtype=float)
    inward = -np.sum(disp * normals, axis=1)
    dist = np.linalg.norm(rest_V - p, axis=1)
    k = int(np.argmin(dist))
    if radius is None or radius <= 0:
        depth = float(inward[k])
    else:
        near = np.linalg.norm(rest_V - rest_V[k], axis=1) <= radius
        w = 1.0 / (np.linalg.norm(rest_V[near] - rest_V[k], axis=1) + 1e-3 * radius)
        depth = float(np.sum(w * inward[near]) / np.sum(w))
    m = int(np.argmax(inward))
    return dict(depth_at_hit=depth, max_depth=float(inward[m]),
                max_location=rest_V[m].tolist(), offset=float(np.linalg.norm(rest_V[m] - p)),
                nearest_distance=float(dist[k]))


def vertex_areas(V, F):
    """Barycentric vertex areas (one third of the incident triangle areas)."""
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=np.int64)
    a = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    out = np.zeros(len(V))
    for k in range(3):
        np.add.at(out, F[:, k], a / 3.0)
    return out


def dent_volume(hit_point, rest_V, disp, normals, areas):
    """Dent volume and dent centre.

    * ``volume``: ``sum_i max(-(d_i . n_i), 0) * area_i`` (inward displacement
      integrated over the rest surface). It does not depend on how finely the
      surface is sampled, so it compares lifts fairly where the peak does not.
    * ``centroid``: rest position averaged with weights ``max(-(d . n), 0) * area``.
    * ``centroid_offset``: distance from the hit point to that centre.
    """
    inward = np.clip(-np.sum(np.asarray(disp) * np.asarray(normals), axis=1), 0.0, None)
    w = inward * np.asarray(areas, dtype=float)
    vol = float(w.sum())
    if vol <= 0.0:
        return dict(volume=0.0, centroid=None, centroid_offset=None)
    c = (np.asarray(rest_V, dtype=float) * w[:, None]).sum(axis=0) / vol
    return dict(volume=vol, centroid=c.tolist(),
                centroid_offset=float(np.linalg.norm(c - np.asarray(hit_point, dtype=float))))


def norm_stats(x):
    """``(max, mean)`` of the row norms of ``x`` (0, 0 for an empty array)."""
    x = np.asarray(x, dtype=float).reshape(-1, 3)
    if x.shape[0] == 0:
        return 0.0, 0.0
    n = np.linalg.norm(x, axis=1)
    return float(n.max()), float(n.mean())
