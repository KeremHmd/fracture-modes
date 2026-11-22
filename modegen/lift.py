"""Cage-to-fine displacement lifts (numpy only; scipy optional for speed).

The frozen ``ActiveInstance.piece_deformed_fine_vertices`` lifts the cage
displacement ``cum`` (one vector per cage vertex) to the fine render mesh by
**nearest cage node**: every fine vertex copies the displacement of its
closest cage vertex. Fine vertices that share a nearest node move rigidly
together, so dents look like flat terraces with steps between them.

Three lifts (``runtime.render.lift``):

* ``nearest`` (default, the frozen behaviour).
* ``barycentric``: see below.
* ``surface``: each fine vertex follows the closest point on the cage
  *surface* (barycentric weights on a cage boundary triangle). It is as
  smooth as ``barycentric``, but a fine surface that lies inside the cage
  follows the cage surface instead of averaging it with interior cage
  nodes, so a dent keeps its depth.

The **barycentric** lift maps each fine vertex to the cage tet that contains
it (barycentric weights, precomputed per node). Fine vertices outside every
tet use the closest point on the cage surface (and its boundary tet). The displacement
is then the weighted sum of the four tet corners, which is continuous
(piecewise linear) across the cage. An optional light Laplacian smoothing
(``smooth_iterations``) can be applied on the fine mesh afterwards.

Data stored on a node (computed lazily if missing, so old trees still load):

* ``node.modegen_bary_tets``    (F,) int   containing / nearest tet per fine vertex
* ``node.modegen_bary_weights`` (F, 4)     clamped, renormalised barycentric weights
* ``node.modegen_surf_corners`` (F, 3) int  closest cage boundary triangle per fine vertex
* ``node.modegen_surf_weights`` (F, 3)      barycentric weights of the closest point

Smoothing note: on fragment meshes with coincident crack vertices, smoothing
can move the copies apart, so it is off by default and not used by any
shipped profile.
"""

import numpy as np

LIFTS = ("nearest", "barycentric", "surface")


def _tet_inverses(cage_V, tets):
    v0 = cage_V[tets[:, 0]]
    M = np.stack([cage_V[tets[:, 1]] - v0, cage_V[tets[:, 2]] - v0, cage_V[tets[:, 3]] - v0], axis=2)
    det = np.linalg.det(M)
    ok = np.abs(det) > 1e-18
    inv = np.zeros_like(M)
    inv[ok] = np.linalg.inv(M[ok])
    return v0, inv, ok


def _bary_all(P, v0, inv):
    """Barycentric coordinates of points P (C,3) in every tet: (C, T, 4)."""
    rel = P[:, None, :] - v0[None, :, :]                     # C,T,3
    l123 = np.einsum("tij,ctj->cti", inv, rel)                # C,T,3
    return np.concatenate([1.0 - l123.sum(axis=2, keepdims=True), l123], axis=2)


def _candidates(cage_V, tets, P, k):
    """Indices (C, k) of the k tets with the nearest centroids (None = all tets)."""
    if k is None or k >= len(tets):
        return None
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None
    cent = cage_V[tets].mean(axis=1)
    _, idx = cKDTree(cent).query(P, k=k)
    return np.asarray(idx, dtype=np.int64).reshape(len(P), k)


def barycentric_map(cage_V, tets, fine_V, chunk=128, candidates=32, inside_tol=1e-9):
    """Map fine vertices to cage tets.

    Returns ``(tet_index (F,), weights (F,4), inside (F,) bool)``. Weights are
    clamped to >= 0 and renormalised (exact barycentrics for inside points).
    ``candidates`` limits the search to the tets with the nearest centroids
    (scipy only); points not found inside them are searched against all tets.
    """
    cage_V = np.asarray(cage_V, dtype=float)
    tets = np.asarray(tets, dtype=np.int64)
    fine_V = np.asarray(fine_V, dtype=float)
    v0, inv, ok = _tet_inverses(cage_V, tets)
    F = len(fine_V)
    tet_idx = np.zeros(F, dtype=np.int64)
    weights = np.zeros((F, 4))
    inside = np.zeros(F, dtype=bool)
    cand = _candidates(cage_V, tets, fine_V, candidates)

    todo = []
    for s in range(0, F, chunk):
        P = fine_V[s:s + chunk]
        if cand is not None:
            c = cand[s:s + chunk]                                       # C,k
            rel = P[:, None, :] - v0[c]                                  # C,k,3
            l123 = np.einsum("ckij,ckj->cki", inv[c], rel)
            B = np.concatenate([1.0 - l123.sum(axis=2, keepdims=True), l123], axis=2)
            score = np.where(ok[c], B.min(axis=2), -np.inf)
            best = np.argmax(score, axis=1)
            rows = np.arange(len(P))
            good = score[rows, best] >= -inside_tol
            g = np.nonzero(good)[0]
            tet_idx[s + g] = c[g, best[g]]
            weights[s + g] = B[g, best[g]]
            inside[s + g] = True
            todo.extend((s + np.nonzero(~good)[0]).tolist())
        else:
            todo.extend(range(s, s + len(P)))

    todo = np.asarray(todo, dtype=np.int64)
    outside = []
    for s in range(0, len(todo), chunk):
        ids = todo[s:s + chunk]
        P = fine_V[ids]
        B = _bary_all(P, v0, inv)                                        # C,T,4
        score = np.where(ok[None, :], B.min(axis=2), -np.inf)
        best = np.argmax(score, axis=1)
        rows = np.arange(len(ids))
        is_in = score[rows, best] >= -inside_tol
        tet_idx[ids] = best
        weights[ids] = B[rows, best]
        inside[ids] = is_in
        out = np.nonzero(~is_in)[0]
        if len(out):
            outside.extend(ids[out].tolist())

    if outside:
        # Outside every tet: use the closest point on the cage surface, which
        # lies in a boundary tet, and that tet's barycentrics of the point.
        outside = np.asarray(outside, dtype=np.int64)
        c, w = surface_map(cage_V, tets, fine_V[outside])
        Q = np.einsum("fk,fkj->fj", w, cage_V[c])
        B = _bary_all(Q, v0, inv)
        score = np.where(ok[None, :], B.min(axis=2), -np.inf)
        best = np.argmax(score, axis=1)
        tet_idx[outside] = best
        weights[outside] = B[np.arange(len(outside)), best]

    weights = np.clip(weights, 0.0, None)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-300)
    return tet_idx, weights, inside


def boundary_faces(tets):
    """Triangles that belong to exactly one tet (the cage surface), (B, 3)."""
    tets = np.asarray(tets, dtype=np.int64)
    f = np.vstack([tets[:, [1, 2, 3]], tets[:, [0, 3, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 1]]])
    key = np.sort(f, axis=1)
    _, idx, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
    return f[idx[cnt == 1]]


def closest_point_triangles(P, A, B, C):
    """Barycentric weights (N, 3) of the closest points of P (N,3) on triangles
    (A, B, C) (each (N,3)); Ericson, Real-Time Collision Detection, 5.1.5."""
    ab, ac, ap = B - A, C - A, P - A
    d1, d2 = np.sum(ab * ap, 1), np.sum(ac * ap, 1)
    bp = P - B
    d3, d4 = np.sum(ab * bp, 1), np.sum(ac * bp, 1)
    cp = P - C
    d5, d6 = np.sum(ab * cp, 1), np.sum(ac * cp, 1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    eps = 1e-300
    denom = np.maximum(va + vb + vc, eps)
    v = vb / denom
    w = vc / denom
    W = np.stack([1 - v - w, v, w], axis=1)                       # interior
    def put(mask, a, b, c):
        W[mask] = np.stack([a[mask], b[mask], c[mask]], axis=1)
    z = np.zeros(len(P))
    o = np.ones(len(P))
    # edge BC
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t = (d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), eps)
    put(m, z, 1 - t, t)
    # edge AC
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = d2 / np.maximum(d2 - d6, eps)
    put(m, 1 - t, z, t)
    # edge AB
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = d1 / np.maximum(d1 - d3, eps)
    put(m, 1 - t, t, z)
    # vertices
    put((d6 >= 0) & (d5 <= d6), z, z, o)
    put((d3 >= 0) & (d4 <= d3), z, o, z)
    put((d1 <= 0) & (d2 <= 0), o, z, z)
    return W


def surface_map(cage_V, tets, fine_V, candidates=24, chunk=256):
    """Closest point on the cage surface for every fine vertex.

    Returns ``(corners (F,3), weights (F,3))``: the surface triangle's vertex
    indices and the barycentric weights of the closest point."""
    cage_V = np.asarray(cage_V, dtype=float)
    fine_V = np.asarray(fine_V, dtype=float)
    tris = boundary_faces(tets)
    cent = cage_V[tris].mean(axis=1)
    k = min(int(candidates), len(tris))
    try:
        from scipy.spatial import cKDTree
        cand = np.asarray(cKDTree(cent).query(fine_V, k=k)[1], dtype=np.int64).reshape(len(fine_V), k)
    except ImportError:
        cand = None
    corners = np.zeros((len(fine_V), 3), dtype=np.int64)
    weights = np.zeros((len(fine_V), 3))
    for s in range(0, len(fine_V), chunk):
        P = fine_V[s:s + chunk]
        c = cand[s:s + chunk] if cand is not None else np.broadcast_to(np.arange(len(tris)), (len(P), len(tris)))
        n, kk = c.shape
        T = tris[c.reshape(-1)]                                      # n*kk,3
        PP = np.repeat(P, kk, axis=0)
        W = closest_point_triangles(PP, cage_V[T[:, 0]], cage_V[T[:, 1]], cage_V[T[:, 2]])
        Q = np.einsum("nk,nkj->nj", W, cage_V[T])
        d = np.sum((Q - PP) ** 2, axis=1).reshape(n, kk)
        b = np.argmin(d, axis=1)
        sel = np.arange(n) * kk + b
        corners[s:s + n] = T[sel]
        weights[s:s + n] = W[sel]
    weights = np.clip(weights, 0.0, None)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-300)
    return corners, weights


def node_surface(node):
    """Surface-projection lift data of a node, computed and cached on first use."""
    c = getattr(node, "modegen_surf_corners", None)
    w = getattr(node, "modegen_surf_weights", None)
    fv = getattr(node, "fine_vertices", None)
    if fv is None:
        return None
    if c is None or w is None or len(c) != len(fv):
        c, w = surface_map(node.vertices, node.elements, fv)
        node.modegen_surf_corners = c
        node.modegen_surf_weights = w
    return c, w


def lift_corners(cum, corners, weights):
    """Fine displacement = sum_k w_k * cum[corners[:, k]]."""
    return np.einsum("fk,fkj->fj", weights, np.asarray(cum, dtype=float)[corners])


def nearest_map(cage_V, fine_V, chunk=512):
    """Nearest cage vertex per fine vertex (same result as the frozen cKDTree query)."""
    cage_V = np.asarray(cage_V, dtype=float)
    fine_V = np.asarray(fine_V, dtype=float)
    try:
        from scipy.spatial import cKDTree
        return np.asarray(cKDTree(cage_V).query(fine_V)[1], dtype=np.int64)
    except ImportError:
        out = np.empty(len(fine_V), dtype=np.int64)
        for s in range(0, len(fine_V), chunk):
            d = np.sum((fine_V[s:s + chunk, None, :] - cage_V[None]) ** 2, axis=2)
            out[s:s + chunk] = np.argmin(d, axis=1)
        return out


def lift_barycentric(cum, tets, tet_idx, weights):
    """Fine displacement = sum_k w_k * cum[tets[tet_idx, k]]."""
    cum = np.asarray(cum, dtype=float)
    corners = np.asarray(tets, dtype=np.int64)[tet_idx]                    # F,4
    return np.einsum("fk,fkj->fj", weights, cum[corners])


def uniform_neighbours(F, n):
    """Edge list (both directions) and degree for the umbrella operator."""
    from .visualqa import mesh_edges
    e = mesh_edges(F)
    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    deg = np.bincount(src, minlength=n).astype(float)
    return src, dst, deg


def smooth_displacement(disp, F, iterations=0, weight=0.5, neighbours=None):
    """``iterations`` steps of ``d <- (1-w) d + w * mean(neighbours)`` on the fine mesh."""
    disp = np.asarray(disp, dtype=float)
    if iterations <= 0 or disp.size == 0:
        return disp
    n = len(disp)
    src, dst, deg = neighbours if neighbours is not None else uniform_neighbours(F, n)
    d = disp.copy()
    for _ in range(int(iterations)):
        acc = np.zeros_like(d)
        np.add.at(acc, src, d[dst])
        mean = np.where(deg[:, None] > 0, acc / np.maximum(deg, 1.0)[:, None], d)
        d = (1.0 - weight) * d + weight * mean
    return d


def node_barycentric(node):
    """Barycentric lift data of a node, computed and cached on first use."""
    ti = getattr(node, "modegen_bary_tets", None)
    w = getattr(node, "modegen_bary_weights", None)
    fv = getattr(node, "fine_vertices", None)
    if fv is None:
        return None
    if ti is None or w is None or len(ti) != len(fv):
        ti, w, _inside = barycentric_map(node.vertices, node.elements, fv)
        node.modegen_bary_tets = ti
        node.modegen_bary_weights = w
    return ti, w


def make_lift(spec):
    """Normalise a runtime ``render`` dict to ``(lift, smooth_iterations, smooth_weight)``."""
    spec = dict(spec or {})
    lift = spec.get("lift", "nearest")
    if lift not in LIFTS:
        raise ValueError(f"runtime.render.lift must be one of {LIFTS}, got {lift!r}")
    return lift, int(spec.get("smooth_iterations", 0)), float(spec.get("smooth_weight", 0.5))
