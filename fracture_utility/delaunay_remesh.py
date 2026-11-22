"""Geometry-preserving Delaunay remeshing restricted to cap-face edges"""

from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix



def _build_edge_adjacency(F):
    """Return ``{(min_v, max_v): [face_indices]}`` from a face list."""
    edges = defaultdict(list)
    for fi in range(len(F)):
        face = F[fi]
        for k in range(3):
            a = int(face[k])
            b = int(face[(k + 1) % 3])
            key = (a, b) if a < b else (b, a)
            edges[key].append(fi)
    return edges


def _triangle_normal(V, face):
    a = V[face[0]]
    b = V[face[1]]
    c = V[face[2]]
    n = np.cross(b - a, c - a)
    return n / (np.linalg.norm(n) + 1e-12)


def _opposite_angle(V, v_idx, p_idx, q_idx):
    """Angle at vertex v in triangle (p, q, v), opposite to edge (p, q)."""
    vp = V[p_idx] - V[v_idx]
    vq = V[q_idx] - V[v_idx]
    cos_a = np.dot(vp, vq) / (
        (np.linalg.norm(vp) + 1e-12) * (np.linalg.norm(vq) + 1e-12)
    )
    cos_a = float(np.clip(cos_a, -1.0, 1.0))
    return float(np.arccos(cos_a))


def _opposite_vertices(F, edge_key, faces_of_edge):
    """Vertex opposite the edge in each adjacent face."""
    p, q = edge_key
    out = []
    for fi in faces_of_edge:
        for v in F[fi]:
            vi = int(v)
            if vi != p and vi != q:
                out.append(vi)
                break
    return out


def _is_nld(V, F, edge_key, faces_of_edge):
    """Edge is NLD iff the sum of opposite angles exceeds pi."""
    if len(faces_of_edge) != 2:
        return False
    opps = _opposite_vertices(F, edge_key, faces_of_edge)
    if len(opps) != 2:
        return False
    p, q = edge_key
    u, vv = opps
    return _opposite_angle(V, u, p, q) + _opposite_angle(V, vv, p, q) > np.pi


def _is_planar_edge(V, F, faces_of_edge, dihedral_tol_rad):
    """Two adjacent faces count as coplanar if their normals are aligned."""
    if len(faces_of_edge) != 2:
        return False
    n1 = _triangle_normal(V, F[faces_of_edge[0]])
    n2 = _triangle_normal(V, F[faces_of_edge[1]])
    
    return abs(float(np.dot(n1, n2))) > np.cos(dihedral_tol_rad)


def _edge_geometric_key(V, edge_key, decimals=8):
    """Position-based key. Independent of vertex indexing -> deterministic."""
    p, q = edge_key
    pp = tuple(np.round(V[p], decimals))
    qq = tuple(np.round(V[q], decimals))
    return (pp, qq) if pp <= qq else (qq, pp)




def _flip_edge(F, f1_idx, f2_idx, p, q, u, vv):
    """Replace edge (p,q) shared by F[f1_idx], F[f2_idx] with edge (u,vv)."""
    new_f1 = list(F[f1_idx])
    new_f1[new_f1.index(q)] = vv
    new_f2 = list(F[f2_idx])
    new_f2[new_f2.index(p)] = u
    F[f1_idx] = new_f1
    F[f2_idx] = new_f2


def _split_vertex_position(V, p_idx, q_idx, delta):
    """Compute the power-of-two split vertex per Dyer Eq. (1).

    Choose ``s`` along edge ``(p, q)`` such that ``|p - s| = 2^k * delta``
    for some integer ``k`` and ``|s - m|`` is minimised, where ``m`` is
    the midpoint of the edge.
    """
    P = V[p_idx]
    Q = V[q_idx]
    L = float(np.linalg.norm(Q - P))
    if L < 1e-12 or delta <= 0:
        return None
    target = L / 2.0
    log2_ratio = float(np.log2(target / delta))
    k_lo = int(np.floor(log2_ratio))
    candidates = [k_lo, k_lo + 1]
    best_k = min(candidates, key=lambda k: abs((2.0 ** k) * delta - target))
    split_d = (2.0 ** best_k) * delta
    if split_d <= 0 or split_d >= L:
        # Fall back to midpoint if the power-of-2 location is degenerate
        split_d = L * 0.5
    t = split_d / L
    return P + t * (Q - P)


def _do_split(V, F, cap_face_set, edge_key, faces_of_e, pivot, other,
              is_original_vertex, vertex_piece_labels, delta):
    """Insert a new vertex on the edge; replace the two adjacent faces with four."""
    new_pos = _split_vertex_position(V, pivot, other, delta)
    if new_pos is None:
        return V, False, vertex_piece_labels
    f1_idx, f2_idx = faces_of_e
    f1 = [int(x) for x in F[f1_idx]]
    f2 = [int(x) for x in F[f2_idx]]
    p, q = edge_key
    u = next((x for x in f1 if x != p and x != q), None)
    vv = next((x for x in f2 if x != p and x != q), None)
    if u is None or vv is None:
        return V, False, vertex_piece_labels

    # Append the new vertex
    s = V.shape[0]
    V_new = np.vstack([V, new_pos.reshape(1, 3)])
    is_original_vertex.append(False)

    # Inherit piece label from either endpoint
    if vertex_piece_labels is not None:
        new_label = int(vertex_piece_labels[p])
        vertex_piece_labels = np.concatenate(
            [vertex_piece_labels, np.array([new_label], dtype=vertex_piece_labels.dtype)]
        )

    new_f1a = list(f1)
    new_f1a[new_f1a.index(q)] = s
    new_f1b = list(f1)
    new_f1b[new_f1b.index(p)] = s
    new_f2a = list(f2)
    new_f2a[new_f2a.index(q)] = s
    new_f2b = list(f2)
    new_f2b[new_f2b.index(p)] = s

    F[f1_idx] = new_f1a
    F[f2_idx] = new_f2a
    F.append(new_f1b)
    F.append(new_f2b)
    # The two new faces inherit cap status from their parents.
    if f1_idx in cap_face_set:
        cap_face_set.add(len(F) - 2)
    if f2_idx in cap_face_set:
        cap_face_set.add(len(F) - 1)
    return V_new, True, vertex_piece_labels


def delaunay_remesh_cap_faces(V, F, cap_face_indices,
                              vertex_piece_labels=None,
                              original_vertex_mask=None,
                              max_iterations=2000,
                              dihedral_tol_deg=5.0,
                              split_delta=None,
                              clean_surface_edges=True):
    """Apply Algorithm 1 of Dyer et al. 2007 restricted to cap-face edges"""
    V = np.asarray(V, dtype=float).copy()
    F = [list(int(x) for x in face) for face in np.asarray(F)]
    cap_set = set(int(c) for c in cap_face_indices)
    face_status = [0] * len(F)

    is_orig = list(bool(b) for b in (
        original_vertex_mask
        if original_vertex_mask is not None
        else np.ones(len(V), dtype=bool)
    ))
    vpl = (
        np.asarray(vertex_piece_labels).copy()
        if vertex_piece_labels is not None
        else None
    )
    dihedral_tol_rad = float(np.deg2rad(dihedral_tol_deg))

    # Auto-pick split delta from the shortest cap edge
    if split_delta is None:
        edges_dict_init = _build_edge_adjacency(F)
        min_len = float("inf")
        for key, faces_of in edges_dict_init.items():
            if not all(f in cap_set for f in faces_of):
                continue
            L = float(np.linalg.norm(V[key[0]] - V[key[1]]))
            if 0.0 < L < min_len:
                min_len = L
        if min_len == float("inf"):
            split_delta = 1.0
        else:
            split_delta = max(min_len * 0.01, 1e-9)

    for _ in range(max_iterations):
        edges_dict = _build_edge_adjacency(F)
        # Collect NLD edges, classified as 'cap'
        nld_edges = []
        for key, faces_of in edges_dict.items():
            if len(faces_of) != 2:
                continue
            f1, f2 = faces_of
            c1 = f1 in cap_set
            c2 = f2 in cap_set
            if c1 and c2:
                kind = "cap"
            elif (not c1) and (not c2):
                if not clean_surface_edges:
                    continue
                kind = "surface"
            else:
                # Cap-surface boundary edge.
                continue
            if _is_nld(V, F, key, faces_of):
                nld_edges.append((key, kind))
        if not nld_edges:
            break
        nld_edges.sort(key=lambda item: _edge_geometric_key(V, item[0]))

        progressed = False
        for edge_key, kind in nld_edges:
            faces_of = edges_dict.get(edge_key, [])
            if len(faces_of) != 2:
                continue
            f1, f2 = faces_of
            c1 = f1 in cap_set
            c2 = f2 in cap_set
            if kind == "cap" and not (c1 and c2):
                continue
            if kind == "surface" and (c1 or c2):
                continue

            planar = _is_planar_edge(V, F, faces_of, dihedral_tol_rad)
            p, q = edge_key

            # Try flip if planar.
            flipped = False
            if planar:
                f1_idx, f2_idx = faces_of
                u = next(
                    (int(x) for x in F[f1_idx] if int(x) != p and int(x) != q),
                    None,
                )
                vv = next(
                    (int(x) for x in F[f2_idx] if int(x) != p and int(x) != q),
                    None,
                )
                if u is not None and vv is not None:
                    new_key = (u, vv) if u < vv else (vv, u)
                    if new_key not in edges_dict:
                        _flip_edge(F, f1_idx, f2_idx, p, q, u, vv)
                        face_status[f1_idx] |= 1
                        face_status[f2_idx] |= 1
                        flipped = True
            if flipped:
                progressed = True
                break

            # Splits are only allowed for cap-cap edges. Surface edges never split as that would add a vertex on the image's silhouette and alter the input shape.
            if kind == "surface":
                continue

            # Cap-cap physical edge: split per Dyer Eq. (1).
            p_is_orig = p < len(is_orig) and is_orig[p]
            q_is_orig = q < len(is_orig) and is_orig[q]
            if p_is_orig and not q_is_orig:
                pivot, other = p, q
            elif q_is_orig and not p_is_orig:
                pivot, other = q, p
            else:
                pivot, other = (p, q) if p < q else (q, p)
            f1_idx, f2_idx = faces_of
            faces_count_before = len(F)
            V, ok, vpl = _do_split(
                V, F, cap_set, edge_key, faces_of,
                pivot, other, is_orig, vpl, split_delta,
            )
            if ok:
                face_status[f1_idx] |= 2
                face_status[f2_idx] |= 2
                for new_face_idx in range(faces_count_before, len(F)):
                    if new_face_idx >= len(face_status):
                        face_status.append(2)
                    else:
                        face_status[new_face_idx] |= 2
                progressed = True
                break

        if not progressed:
            break

    cap_face_idx_new = np.fromiter(sorted(cap_set), dtype=np.int64)
    F_new = np.asarray(F, dtype=np.int64)
    # Make sure face_status matches F_new's length
    if len(face_status) < F_new.shape[0]:
        face_status.extend([0] * (F_new.shape[0] - len(face_status)))
    elif len(face_status) > F_new.shape[0]:
        face_status = face_status[: F_new.shape[0]]
    return V, F_new, cap_face_idx_new, vpl, np.asarray(face_status, dtype=np.int8)



def apply_delaunay_to_fracture_modes(modes, v_orig, f_orig, **kwargs):
    """In-place Delaunay remesh on a fracture mode"""

    # Lazy import to avoid circular dependency
    from .cap_detail import cap_face_indices

    if getattr(modes, "fine_vertices", None) is None:
        return
    if getattr(modes, "fine_triangles", None) is None:
        return

    cap_idx = cap_face_indices(
        modes.fine_vertices, modes.fine_triangles, v_orig, f_orig
    )
    if len(cap_idx) == 0:
        return

    # Extract per-vertex piece labels from the current sparse matrix.
    P2V = modes.piece_to_fine_vertices_matrix
    vertex_piece_labels = np.asarray(P2V.argmax(axis=1)).reshape(-1).astype(np.int64)

    V_new, F_new, _cap_new, labels_new, face_status = delaunay_remesh_cap_faces(
        modes.fine_vertices,
        modes.fine_triangles,
        cap_idx,
        vertex_piece_labels=vertex_piece_labels,
        **kwargs,
    )

    modes.fine_vertices = V_new
    modes.fine_triangles = F_new
    # Attach the per-face modification status so callers can visualise which triangles were flipped or split.
    modes.delaunay_face_status = face_status

    n_verts = V_new.shape[0]
    n_pieces = int(P2V.shape[1])
    rows = np.arange(n_verts, dtype=np.int64)
    cols = labels_new.astype(np.int64)
    cols = np.clip(cols, 0, max(0, n_pieces - 1))
    modes.piece_to_fine_vertices_matrix = csr_matrix(
        (np.ones(n_verts, dtype=int), (rows, cols)),
        shape=(n_verts, n_pieces),
        dtype=int,
    )

    if hasattr(modes, "piece_labels"):
        try:
            new_fine_labels = modes.piece_to_fine_vertices_matrix @ modes.piece_labels
            modes.fine_labels = np.asarray(new_fine_labels).astype(int)
        except Exception:
            pass
