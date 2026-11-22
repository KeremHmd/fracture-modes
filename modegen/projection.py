"""Fracture projection helpers for responders.

``project_force_field`` mirrors ``fracture_modes.impact_projection`` but
accepts an arbitrary per-vertex impact profile instead of the built-in
"one-hot within 0.05 of the contact point". The profile is passed through
the same precomputed wave-equation filter (``wave_piece_lsqr``), so a
unit-peak profile lives on the same threshold scale as the baseline.

``restrict_separation`` re-merges components that do not satisfy a predicate
(e.g. "touches the puncture channel") into the main body, which is how
localised damage plugins keep far-away pieces attached.
"""

import numpy as np


def force_to_profile(force, gain=1.0):
    """Unit-peak, component-major impact vector from a (V, 3) force field."""
    F = np.asarray(force, dtype=float)
    peak = np.linalg.norm(F, axis=1).max()
    if peak <= 0.0:
        return np.zeros(F.size)
    return (float(gain) * F / peak).reshape(-1, order="F")


def _finish_labels(modes, n_pieces, piece_labels):
    modes.n_pieces_after_impact = int(n_pieces)
    modes.piece_labels_after_impact = np.asarray(piece_labels, dtype=int)
    modes.tet_labels_after_impact = modes.piece_to_tet_matrix @ modes.piece_labels_after_impact
    if getattr(modes, "fine_vertices", None) is not None:
        modes.fine_vertex_labels_after_impact = (
            modes.piece_to_fine_vertices_matrix @ modes.piece_labels_after_impact
        )
    modes.impact_projected = True


def project_force_field(modes, force, threshold=10.0, gain=1.0,
                        use_wave=True, project_on_modes=False):
    from scipy.sparse import csr_matrix, kron, eye
    from scipy.sparse.csgraph import connected_components

    dim = modes.modes.shape[0] // modes.elements.shape[0]
    F = np.asarray(force, dtype=float)
    if dim == 1:
        F = np.linalg.norm(F, axis=1, keepdims=True)
    vec = force_to_profile(F, gain)
    modes.impact = vec
    if use_wave and getattr(modes, "wave_piece_lsqr", None) is not None:
        piece_impact = modes.wave_piece_lsqr.T @ vec
    else:
        piece_impact = kron(eye(dim), modes.tet_to_piece_matrix @ modes.massmatrix) @ vec
    piece_impact = np.asarray(piece_impact).reshape(-1)
    if project_on_modes:
        proj = np.zeros(piece_impact.shape[0])
        for k in range(modes.modes.shape[1]):
            pk = modes.piece_modes[:, k]
            proj = proj + (piece_impact @ (modes.piece_massmatrix @ pk)) * pk
        piece_impact = proj
    modes.piece_impact = piece_impact
    modes.projected_impact = piece_impact

    P = modes.precomputed_num_pieces
    pi = np.reshape(piece_impact, (-1, dim), order="F")
    nb = modes.piece_neighbors
    dist = np.linalg.norm(pi[nb[:, 0]] - pi[nb[:, 1]], axis=1)
    keep = nb[dist < threshold]
    adj = csr_matrix((np.ones(keep.shape[0]), (keep[:, 0], keep[:, 1])),
                     shape=(P, P), dtype=int)
    n, labels = connected_components(adj, directed=False)
    _finish_labels(modes, n, labels)
    return n


def piece_centroids(modes):
    """Centroid of every precomputed piece (from its tets), local frame."""
    tc = np.asarray(modes.vertices)[np.asarray(modes.elements)].mean(axis=1)
    P2T = modes.piece_to_tet_matrix
    counts = np.asarray(P2T.sum(axis=0)).reshape(-1)
    sums = np.asarray(P2T.T @ tc)
    return sums / np.maximum(counts, 1)[:, None]


def restrict_separation(modes, keep_piece_mask):
    """Merge every component containing no ``keep_piece_mask`` piece into the
    largest such component (the body). Returns the new component count."""
    labels = np.asarray(modes.piece_labels_after_impact, dtype=int).copy()
    keep = np.asarray(keep_piece_mask, dtype=bool)
    comps = np.unique(labels)
    if comps.size <= 1:
        return int(comps.size)
    sizes = {c: int((labels == c).sum()) for c in comps}
    active = {c for c in comps if keep[labels == c].any()}
    passive = [c for c in comps if c not in active]
    if not passive:
        return int(comps.size)
    body = max(comps, key=lambda c: sizes[c])
    for c in passive:
        labels[labels == c] = body
    _, relabeled = np.unique(labels, return_inverse=True)
    n = int(relabeled.max()) + 1
    _finish_labels(modes, n, relabeled)
    return n
