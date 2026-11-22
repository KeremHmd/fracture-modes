"""Numerics for example-based plasticity (Jones, Thuerey, Shinar, Bargteil,
"Example-based Plastic Deformation of Rigid Bodies", ACM TOG 35(4), SIGGRAPH 2016).

Equation numbers below are the paper's.

Unlike the other modules in ``modegen/solvers``, this one is numpy-only at
import time and at runtime, so the runtime numerics are unit-testable without
the geometry stack. scipy (Dijkstra, KD-tree) and igl (optional harmonic
weights) are imported lazily, and only by precompute helpers.

Conventions
-----------
* ``D``: example displacement fields, shape ``(K, V, 3)``. ``D[e, i]`` is vertex i's
  displacement in example e+1 (example 0 is the rest pose, which has zero displacement).
* ``E``: example weights, shape ``(V, K + 1)``. Column 0 is the rest pose; rows are barycentric.
* ``J``: per-vertex Jacobian ``dx_i / dE_i``, shape ``(V, 3, K + 1)``. With linear
  blending it is constant: ``J[i, :, 0] = 0`` and ``J[i, :, e] = D[e - 1, i]``.
* Plastic ``modes`` are interleaved (index ``3 * v + axis``), like every plastic basis.
"""

import heapq
import math

import numpy as np

JT_EPS = 1.0e-8   # Eq. 8: de_i = 0 when ||J_i^T dx_i|| <= 1e-8 (value given in the paper)

#: Use scipy's Dijkstra when available. The pure-Python fallback gives the
#: same distances; the numpy-only unit tests switch scipy off.
USE_SCIPY = True


# =====================================================================
# Runtime numerics (Eqs. 4, 8, 10, 11-14) - numpy only
# =====================================================================

def cubic_kernel(x):
    """Eq. 12: phi(x) = 2x^3 - 3x^2 + 1 for 0 <= x < 1, else 0 (|x| is used)."""
    x = np.abs(np.asarray(x, dtype=float))
    xc = np.minimum(x, 1.0)                       # also keeps inf distances finite
    out = 2.0 * xc ** 3 - 3.0 * xc ** 2 + 1.0
    return np.where(x < 1.0, out, 0.0)


def initial_weights(num_vertices, num_examples):
    """All weight on the rest pose (paper Sec. 3): rows ``[1, 0, ..., 0]``."""
    E = np.zeros((int(num_vertices), int(num_examples) + 1))
    E[:, 0] = 1.0
    return E


def project_barycentric(E):
    """Clamp negatives to 0 and rescale each row to sum to 1 (paper, after Eq. 13).
    A row that is all zero after clamping falls back to the rest pose."""
    E = np.clip(np.asarray(E, dtype=float), 0.0, None)
    s = E.sum(axis=1)
    dead = s <= 1e-300
    if dead.any():
        E[dead] = 0.0
        E[dead, 0] = 1.0
        s[dead] = 1.0
    return E / s[:, None]


def jacobian(D):
    """Constant per-vertex Jacobian (V, 3, K+1) of linear example blending."""
    D = np.asarray(D, dtype=float)
    K, V, _ = D.shape
    J = np.zeros((V, 3, K + 1))
    J[:, :, 1:] = np.transpose(D, (1, 2, 0))
    return J


def blend_displacement(E, D):
    """Skinned displacement (V, 3) for weights E: ``sum_e E_ie D_e(i)``.

    This is Eq. 2 (linear blending of the per-bone transforms), which equals
    linear blending of the example displacements, because LBS is linear in T.
    """
    E = np.asarray(E, dtype=float)
    return np.einsum("vk,kvc->vc", E[:, 1:], np.asarray(D, dtype=float))


def desired_displacement(impulses, total_mass, dt, beta, max_step=None):
    """Eq. 4: dx_i = (dt / m) * max(|j_i| - beta, 0) * j_i / |j_i|.

    ``impulses`` (V, 3) are per-vertex impulses, ``total_mass`` is the body
    mass m, and ``beta`` is the yield threshold (an impulse, scalar or (V,)).
    ``max_step`` optionally caps each |dx_i| (a modegen safety cap).
    """
    j = np.asarray(impulses, dtype=float)
    mag = np.linalg.norm(j, axis=1)
    excess = np.maximum(mag - np.asarray(beta, dtype=float), 0.0)
    scale = np.zeros_like(mag)
    ok = mag > 0.0
    scale[ok] = (float(dt) / float(total_mass)) * excess[ok] / mag[ok]
    dx = scale[:, None] * j
    if max_step is not None:
        n = np.linalg.norm(dx, axis=1)
        over = n > float(max_step)
        dx[over] *= (float(max_step) / n[over])[:, None]
    return dx


def weight_step(J, dx, alpha, step="normalized", eps=JT_EPS):
    """Per-vertex example-weight change ``de`` (V, K+1).

    * ``step="normalized"`` (Eq. 8): ``alpha |dx_i| J_i^T dx_i / |J_i^T dx_i|``, and 0 where
      ``|J_i^T dx_i| <= eps``.
    * ``step="natural"`` (Eq. 10): ``alpha J_i^T dx_i``.
    """
    g = np.einsum("vck,vc->vk", np.asarray(J, dtype=float), np.asarray(dx, dtype=float))
    if step == "natural":
        return float(alpha) * g
    if step != "normalized":
        raise ValueError(f"unknown step {step!r}")
    n = np.linalg.norm(g, axis=1)
    dxn = np.linalg.norm(dx, axis=1)
    de = np.zeros_like(g)
    ok = n > eps
    de[ok] = (float(alpha) * dxn[ok] / n[ok])[:, None] * g[ok]
    return de


def propagate(de, indptr, indices, dist, gamma):
    """Eqs. 11-12: ``dE_j += sum_i phi(d_ij / gamma) de_i``.

    The sum runs over every vertex i with a nonzero ``de_i``. ``(indptr,
    indices, dist)`` is a CSR table: row i lists the vertices within the
    precomputed radius of i with their geodesic distance, i itself
    (distance 0) included. Returns dE (V, K+1).
    """
    de = np.asarray(de, dtype=float)
    out = np.zeros_like(de)
    src = np.flatnonzero(np.abs(de).max(axis=1) > 0.0)
    if src.size == 0 or gamma <= 0.0:
        return out
    starts, ends = indptr[src], indptr[src + 1]
    lens = ends - starts
    total = int(lens.sum())
    if total == 0:
        return out
    pos = np.repeat(ends - lens.cumsum(), lens) + np.arange(total)
    rows = np.repeat(src, lens)
    w = cubic_kernel(np.asarray(dist)[pos] / float(gamma))
    keep = w > 0.0
    np.add.at(out, np.asarray(indices)[pos][keep], w[keep, None] * de[rows[keep]])
    return out


def iterations_for(lam, tol):
    """Smallest n with ``lam ** n <= tol`` (at least 1): the fraction of the
    pending update still unapplied after n steps of Eq. 13."""
    lam = float(lam)
    if lam <= 0.0:
        return 1
    if lam >= 1.0:
        raise ValueError("lambda must be < 1")
    return max(1, int(math.ceil(math.log(float(tol)) / math.log(lam))))


def apply_schedule(E, dE, lam, iterations, clamp=True):
    """Eq. 13 repeated ``iterations`` times: ``E += (1 - lam) dE; dE *= lam``.
    With ``clamp`` the rows are made barycentric after every step.

    Without clamping, ``E_n = E_0 + (1 - lam^n) dE_0`` (Eq. 14 in the limit).
    Returns ``(E, dE)``, where dE is the residual still pending.
    """
    E = np.array(E, dtype=float, copy=True)
    dE = np.array(dE, dtype=float, copy=True)
    lam = float(lam)
    for _ in range(int(iterations)):
        E += (1.0 - lam) * dE
        dE *= lam
        if clamp:
            E = project_barycentric(E)
    return E, dE


def reconstruct_weights(cum, D):
    """Least-squares example weights reproducing a displacement field ``cum``.

    Used when a piece has ``cum`` but no stored weights. Per vertex it solves
    ``min |J_i e - cum_i|`` over the non-rest examples, clamps to >= 0, scales
    the row down if it sums past 1, and puts the remainder on the rest pose.
    """
    D = np.asarray(D, dtype=float)
    K, V, _ = D.shape
    A = np.transpose(D, (1, 2, 0))                       # (V, 3, K)
    e = np.einsum("vkc,vc->vk", np.linalg.pinv(A, rcond=1e-10), np.asarray(cum, dtype=float))
    e = np.clip(e, 0.0, None)
    s = e.sum(axis=1)
    over = s > 1.0
    e[over] /= s[over, None]
    E = np.zeros((V, K + 1))
    E[:, 1:] = e
    E[:, 0] = 1.0 - e.sum(axis=1)
    return project_barycentric(E)


# =====================================================================
# Precompute helpers - numpy (+ lazy scipy / igl)
# =====================================================================

def match_rows(reference, points, tol=0.0, chunk=64):
    """Index of each point's coincident row in ``reference`` (numpy only).

    Tree children copy their cage vertices verbatim from the parent, so an
    exact byte match is tried first. Points that do not match exactly fall back to
    the nearest row, which must lie within ``tol``. Returns None if any
    point has no reference row within ``tol``.
    """
    R = np.ascontiguousarray(reference, dtype=np.float64)
    P = np.ascontiguousarray(points, dtype=np.float64)
    lookup = {R[i].tobytes(): i for i in range(R.shape[0] - 1, -1, -1)}
    rows = np.array([lookup.get(P[i].tobytes(), -1) for i in range(P.shape[0])], dtype=np.int64)
    miss = np.flatnonzero(rows < 0)
    for c0 in range(0, len(miss), chunk):
        idx = miss[c0:c0 + chunk]
        d = np.linalg.norm(P[idx, None, :] - R[None, :, :], axis=2)
        k = np.argmin(d, axis=1)
        if np.any(d[np.arange(len(idx)), k] > tol):
            return None
        rows[idx] = k
    return rows


def bbox_diag(points):
    p = np.asarray(points, dtype=float)
    return float(np.linalg.norm(p.max(axis=0) - p.min(axis=0)))


def tet_edges(elements):
    """Unique undirected edges (n, 2) of a tet mesh."""
    T = np.asarray(elements, dtype=np.int64)
    pairs = np.concatenate([T[:, [a, b]] for a, b in
                            ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))])
    pairs.sort(axis=1)
    return np.unique(pairs, axis=0)


def _adjacency(num_vertices, edges, lengths):
    adj = [[] for _ in range(num_vertices)]
    for (a, b), w in zip(edges.tolist(), lengths.tolist()):
        adj[a].append((b, w))
        adj[b].append((a, w))
    return adj


def _dijkstra_py(adj, source, radius):
    dist = {source: 0.0}
    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, math.inf):
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd <= radius and nd < dist.get(v, math.inf):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def geodesic_table(vertices, elements, radius=math.inf, sources=None, use_scipy=None):
    """Dijkstra distances on the tet-edge graph (edge length = Euclidean).

    Returns a CSR table ``(indptr, indices, dist)`` with one row per source
    (default: every vertex). A row lists every vertex within ``radius``, the
    source included, sorted by index.
    """
    V = np.asarray(vertices, dtype=float)
    n = V.shape[0]
    edges = tet_edges(elements)
    lengths = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
    src = np.arange(n) if sources is None else np.asarray(sources, dtype=np.int64)
    rows_idx, rows_dist = [], []
    done = False
    if USE_SCIPY if use_scipy is None else use_scipy:
        try:
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import dijkstra
            g = coo_matrix((np.concatenate([lengths, lengths]),
                            (np.concatenate([edges[:, 0], edges[:, 1]]),
                             np.concatenate([edges[:, 1], edges[:, 0]]))), shape=(n, n)).tocsr()
            lim = np.inf if not np.isfinite(radius) else float(radius)
            for c0 in range(0, len(src), 256):
                chunk = src[c0:c0 + 256]
                d = dijkstra(g, directed=False, indices=chunk, limit=lim)
                for row in d:
                    idx = np.flatnonzero(np.isfinite(row) & (row <= lim))
                    rows_idx.append(idx)
                    rows_dist.append(row[idx])
            done = True
        except ImportError:
            pass
    if not done:
        adj = _adjacency(n, edges, lengths)
        for s in src.tolist():
            dd = _dijkstra_py(adj, s, radius)
            idx = np.array(sorted(dd), dtype=np.int64)
            rows_idx.append(idx)
            rows_dist.append(np.array([dd[i] for i in idx.tolist()], dtype=float))
    indptr = np.zeros(len(rows_idx) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(r) for r in rows_idx])
    indices = np.concatenate(rows_idx) if rows_idx else np.zeros(0, dtype=np.int64)
    dist = np.concatenate(rows_dist) if rows_dist else np.zeros(0)
    return indptr, indices.astype(np.int64), dist.astype(float)


def dense_rows(indptr, indices, dist, num_vertices, fill=np.inf):
    """CSR distance table -> dense (rows, V) array (``fill`` where unreachable)."""
    out = np.full((len(indptr) - 1, int(num_vertices)), fill, dtype=float)
    for r in range(len(indptr) - 1):
        a, b = indptr[r], indptr[r + 1]
        out[r, indices[a:b]] = dist[a:b]
    return out


def surface_faces(elements):
    """Boundary triangles (F, 3) of a tet mesh, oriented with outward normals
    (so ``cross(b - a, c - a)`` points away from the tet's fourth vertex)."""
    T = np.asarray(elements, dtype=np.int64)
    local = np.array([[1, 2, 3, 0], [0, 3, 2, 1], [0, 1, 3, 2], [0, 2, 1, 3]])
    faces = T[:, local[:, :3]].reshape(-1, 3)
    opp = T[:, local[:, 3]].reshape(-1)
    key = np.sort(faces, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    once = cnt[inv.reshape(-1)] == 1
    return faces[once], opp[once]


def outward_vertex_normals(vertices, elements):
    """Area-weighted outward normals on boundary vertices (zero inside) and
    the boolean mask of surface vertices."""
    V = np.asarray(vertices, dtype=float)
    F, opp = surface_faces(elements)
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    flip = np.einsum("ij,ij->i", n, V[opp] - V[F[:, 0]]) > 0.0   # points toward the inside
    n[flip] *= -1.0
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], n)
    on = np.zeros(V.shape[0], dtype=bool)
    on[F.reshape(-1)] = True
    ln = np.linalg.norm(N, axis=1)
    N[ln > 0] /= ln[ln > 0, None]
    return N, on


def farthest_point_sampling(points, k, start):
    """``k`` indices into ``points``, chosen greedily (deterministic given ``start``)."""
    P = np.asarray(points, dtype=float)
    k = int(min(k, P.shape[0]))
    chosen = [int(start)]
    d = np.linalg.norm(P - P[start], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(P - P[nxt], axis=1))
    return np.array(chosen, dtype=np.int64)


def principal_axes(points):
    """Centroid, axes (rows: longest, middle, shortest) and extents along them."""
    P = np.asarray(points, dtype=float)
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    axes = vt.copy()
    for r in range(3):                       # deterministic signs
        big = int(np.argmax(np.abs(axes[r])))
        if axes[r, big] < 0:
            axes[r] *= -1.0
    axes[2] = np.cross(axes[0], axes[1])     # right-handed
    proj = (P - c) @ axes.T
    return c, axes, proj.max(axis=0) - proj.min(axis=0)


def rotation_matrix(axis, angle):
    """Rotation about ``axis`` by ``angle`` radians (Rodrigues)."""
    a = np.asarray(axis, dtype=float)
    a = a / max(np.linalg.norm(a), 1e-300)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def affine(R=None, pivot=None, translate=None):
    """3x4 transform of ``x' = R (x - p) + p + t``."""
    R = np.eye(3) if R is None else np.asarray(R, dtype=float)
    p = np.zeros(3) if pivot is None else np.asarray(pivot, dtype=float)
    t = np.zeros(3) if translate is None else np.asarray(translate, dtype=float)
    T = np.zeros((3, 4))
    T[:, :3] = R
    T[:, 3] = p - R @ p + t
    return T


def apply_affine(T, X):
    return np.asarray(X, dtype=float) @ T[:, :3].T + T[:, 3]


def identity_transforms(num_bones):
    return np.repeat(affine()[None], int(num_bones), axis=0)


# ---------------------------------------------------------- presets

def preset_dent(bone_positions, bone_normals, depth):
    """One example per bone: bone b translates by ``-depth * n_b`` (inward)."""
    B = len(bone_positions)
    out = []
    for b in range(B):
        T = identity_transforms(B)
        T[b] = affine(translate=-float(depth) * np.asarray(bone_normals[b], dtype=float))
        out.append((f"dent_{b}", T))
    return out


def bend_map(points, centre, axes, kappa):
    """Bend points in the (a1, a3) plane with curvature ``kappa`` (radians per
    unit length along a1), bending toward +a3. Returns (positions, rotations)."""
    P = np.asarray(points, dtype=float)
    a1, a2, a3 = axes
    s = (P - centre) @ a1
    n = (P - centre) @ a3
    w = (P - centre) @ a2
    if abs(kappa) < 1e-12:
        return P.copy(), np.repeat(np.eye(3)[None], len(P), axis=0)
    R = 1.0 / kappa
    phi = s * kappa
    pos = (centre[None] + R * a3[None] + w[:, None] * a2[None]
           + (R - n)[:, None] * (np.sin(phi)[:, None] * a1[None] - np.cos(phi)[:, None] * a3[None]))
    rots = []
    for ph in phi:
        Rm = rotation_matrix(a2, -ph)       # maps a1 -> cos a1 + sin a3
        rots.append(Rm)
    return pos, np.array(rots)


def preset_bend(bone_positions, centre, axes, length, angle_deg):
    """Two examples (+/- a3): the beam bends by ``angle_deg`` over ``length``."""
    out = []
    for sign, tag in ((1.0, "bend_pos"), (-1.0, "bend_neg")):
        kappa = sign * math.radians(float(angle_deg)) / max(float(length), 1e-12)
        pos, rots = bend_map(bone_positions, centre, axes, kappa)
        T = np.stack([affine(R=rots[b], pivot=bone_positions[b],
                             translate=pos[b] - bone_positions[b])
                      for b in range(len(bone_positions))])
        out.append((tag, T))
    return out


def preset_twist(bone_positions, centre, axes, length, angle_deg):
    """Two examples (+/-): rotation about the a1 axis through the centroid by
    ``angle_deg * s_b / length``."""
    a1 = axes[0]
    out = []
    for sign, tag in ((1.0, "twist_pos"), (-1.0, "twist_neg")):
        Ts = []
        for p in bone_positions:
            s = float((np.asarray(p) - centre) @ a1)
            ang = sign * math.radians(float(angle_deg)) * s / max(float(length), 1e-12)
            Ts.append(affine(R=rotation_matrix(a1, ang), pivot=centre))
        out.append((tag, np.stack(Ts)))
    return out


def preset_squash(num_bones, centre, axis, amount):
    """One example: volume-preserving squash ``1 - amount`` along ``axis``."""
    a = np.asarray(axis, dtype=float)
    a /= max(np.linalg.norm(a), 1e-300)
    q = 1.0 - float(amount)
    S = (1.0 / math.sqrt(q)) * np.eye(3) + (q - 1.0 / math.sqrt(q)) * np.outer(a, a)
    T = np.repeat(affine(R=S, pivot=centre)[None], int(num_bones), axis=0)
    return [("squash", T)]


# ---------------------------------------------------------- authored files

EXAMPLE_FORMAT = "modegen.exemplar/1"


def parse_example_file(data, root_vertices):
    """Authored example JSON (already loaded as a dict) -> bones and transforms.

    Returns ``(bone_names, bone_positions (B,3), example_names, transforms (E,B,3,4))``.
    Format: see the README (example-based plasticity).
    """
    fmt = data.get("format", EXAMPLE_FORMAT)
    if fmt != EXAMPLE_FORMAT:
        raise ValueError(f"example file format {fmt!r} is not {EXAMPLE_FORMAT!r}")
    units = data.get("units", "local")
    if units not in ("local", "bbox"):
        raise ValueError(f"units must be 'local' or 'bbox', got {units!r}")
    U = np.asarray(root_vertices, dtype=float)
    lo, hi = U.min(axis=0), U.max(axis=0)
    diag = float(np.linalg.norm(hi - lo))

    def point(p):
        p = np.asarray(p, dtype=float).reshape(3)
        return lo + p * (hi - lo) if units == "bbox" else p

    def vector(t):
        t = np.asarray(t, dtype=float).reshape(3)
        return t * diag if units == "bbox" else t

    bones = data.get("bones") or []
    if not bones:
        raise ValueError("example file has no bones")
    names = [str(b.get("name", f"bone{i}")) for i, b in enumerate(bones)]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate bone names in {names}")
    pos = np.stack([point(b["position"]) for b in bones])
    ex_names, ex_T = [], []
    for k, ex in enumerate(data.get("examples") or []):
        T = identity_transforms(len(names))
        for bname, spec in (ex.get("transforms") or {}).items():
            if bname not in names:
                raise ValueError(f"example {ex.get('name', k)!r}: unknown bone {bname!r}")
            b = names.index(bname)
            if "matrix" in spec:
                R = np.asarray(spec["matrix"], dtype=float).reshape(3, 3)
            elif "rotate" in spec:
                r = spec["rotate"]
                R = rotation_matrix(r["axis"], math.radians(float(r["degrees"])))
            else:
                R = np.eye(3)
            pivot = point(spec["pivot"]) if "pivot" in spec else pos[b]
            t = vector(spec.get("translate", (0.0, 0.0, 0.0)))
            T[b] = affine(R=R, pivot=pivot, translate=t)
        ex_names.append(str(ex.get("name", f"example{k + 1}")))
        ex_T.append(T)
    if not ex_T:
        raise ValueError("example file has no examples")
    return names, pos, ex_names, np.stack(ex_T)


# ---------------------------------------------------------- skinning

def geodesic_gaussian_weights(bone_dist, sigma, fallback_dist=None):
    """Skinning weights (V, B) from bone-to-vertex distances (B, V).

    ``W_ib ∝ exp(-(d_ib / sigma)^2)``, normalised per row. Vertices that no
    bone reaches (infinite distance, e.g. a disconnected component) use
    ``fallback_dist`` (Euclidean, (B, V)) instead.
    """
    d = np.array(bone_dist, dtype=float, copy=True)
    if fallback_dist is not None:
        bad = ~np.isfinite(d).any(axis=0)
        d[:, bad] = np.asarray(fallback_dist, dtype=float)[:, bad]
    d = d.T                                                   # (V, B)
    finite = np.isfinite(d)
    d2 = np.where(finite, (d / float(sigma)) ** 2, np.inf)
    d2 -= np.min(d2, axis=1, keepdims=True)                   # same ratios, no underflow
    w = np.where(finite, np.exp(-d2), 0.0)
    return w / w.sum(axis=1, keepdims=True)


def harmonic_bone_weights(vertices, elements, anchors, k=2):
    """``igl.harmonic_weights`` (k=2: biharmonic) with one point handle per bone,
    clamped to [0, 1] and row-normalised (BBW without its constraints)."""
    import igl

    V = np.ascontiguousarray(vertices, dtype=np.float64)
    T = np.ascontiguousarray(elements, dtype=np.int32)
    b = np.ascontiguousarray(anchors, dtype=np.int32)
    bc = np.eye(len(b), dtype=np.float64)
    W = np.asarray(igl.harmonic_weights(V, T, b, bc, int(k)), dtype=float)
    W = np.clip(W, 0.0, 1.0)
    s = W.sum(axis=1, keepdims=True)
    s[s <= 0] = 1.0
    return W / s


def example_displacements(W, transforms, U):
    """D (E, V, 3): ``sum_b W_ib (T_be u_i) - u_i`` for every example e."""
    U = np.asarray(U, dtype=float)
    E_, B = transforms.shape[:2]
    D = np.zeros((E_, U.shape[0], 3))
    for e in range(E_):
        X = np.zeros_like(U)
        for b in range(B):
            X += W[:, b, None] * apply_affine(transforms[e, b], U)
        D[e] = X - U
    return D


def lumped_vertex_mass(vertices, elements, rho=None):
    """Per-vertex lumped mass ``rho * vol / 4`` from each incident tet."""
    V = np.asarray(vertices, dtype=float)
    T = np.asarray(elements, dtype=np.int64)
    vol = tet_volumes(V, T)
    rho = np.full(T.shape[0], 1.0e3) if rho is None else np.asarray(rho, dtype=float)
    m = np.zeros(V.shape[0])
    for k in range(4):
        np.add.at(m, T[:, k], rho * vol / 4.0)
    return m


def tet_volumes(V, T):
    a, b, c, d = (V[T[:, k]] for k in range(4))
    return np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))) / 6.0


def tet_strain(V, T, u):
    """Per-tet linear strain (T, 6): xx, yy, zz, xy, yz, xz (tensor components)."""
    V = np.asarray(V, dtype=float)
    T = np.asarray(T, dtype=np.int64)
    u = np.asarray(u, dtype=float).reshape(-1, 3)
    Dm = np.stack([V[T[:, k]] - V[T[:, 0]] for k in (1, 2, 3)], axis=2)    # (T,3,3)
    Du = np.stack([u[T[:, k]] - u[T[:, 0]] for k in (1, 2, 3)], axis=2)
    G = Du @ np.linalg.inv(Dm)                                            # grad u
    S = 0.5 * (G + np.transpose(G, (0, 2, 1)))
    return np.stack([S[:, 0, 0], S[:, 1, 1], S[:, 2, 2],
                     S[:, 0, 1], S[:, 1, 2], S[:, 0, 2]], axis=1)


def mass_orthonormal_basis(D, vertex_mass, rel_tol=1e-8):
    """Interleaved (3V, k) basis of span{D_e}, M-orthonormal for the lumped mass."""
    D = np.asarray(D, dtype=float)
    E_, V, _ = D.shape
    m3 = np.repeat(np.asarray(vertex_mass, dtype=float), 3)
    sq = np.sqrt(m3)
    A = D.reshape(E_, 3 * V).T * sq[:, None]
    if not np.any(A):
        return np.zeros((3 * V, 0))
    Uu, s, _ = np.linalg.svd(A, full_matrices=False)
    keep = s > rel_tol * s.max()
    return Uu[:, keep] / sq[:, None]


# =====================================================================
# The per-node bundle (pickled with the tree)
# =====================================================================

class ExemplarPlasticModes:
    """Precomputed example-based plasticity data for one tree node.

    It satisfies the plastic contract (``modes``, ``_lumped_mass_diagonal``,
    ``mode_displacement``, ``mode_strain_magnitude``, ``mode_energies``), so the
    other plastic responders also run on it. ``exemplar.impulse_blend`` uses
    the example data directly.
    """

    def __init__(self, nodes, elements, bone_names, bone_positions, skin_weights,
                 example_names, example_transforms, example_displacements,
                 vertex_mass, root_vertex_indices, reference_diag, example_scale,
                 propagation_radius, neighbourhoods, sigma_y=None, generator_params=None):
        self.nodes = np.asarray(nodes, dtype=float)
        self.elements = np.asarray(elements, dtype=np.int64)
        self.num_vertices = self.nodes.shape[0]
        self.num_tets = self.elements.shape[0]
        self.bone_names = list(bone_names)
        self.bone_positions = np.asarray(bone_positions, dtype=float)
        self.skin_weights = np.asarray(skin_weights, dtype=float)
        self.example_names = list(example_names)
        self.example_transforms = np.asarray(example_transforms, dtype=float)
        self.example_displacements = np.asarray(example_displacements, dtype=float)
        self.vertex_mass = np.asarray(vertex_mass, dtype=float)
        self.total_mass = float(self.vertex_mass.sum())
        self.root_vertex_indices = np.asarray(root_vertex_indices, dtype=np.int64)
        self.reference_diag = float(reference_diag)
        self.example_scale = float(example_scale)
        self.propagation_radius = float(propagation_radius)
        self.nbr_indptr, self.nbr_indices, self.nbr_dist = neighbourhoods
        self.generator_params = dict(generator_params or {})
        self.modes = mass_orthonormal_basis(self.example_displacements, self.vertex_mass)
        sy = (np.full(self.num_tets, 1.0) if sigma_y is None
              else np.asarray(sigma_y, dtype=float))
        vol = tet_volumes(self.nodes, self.elements)
        self.mode_energies = np.array([float(np.sum(sy * vol * self.mode_strain_magnitude(k)))
                                       for k in range(self.modes.shape[1])])

    @property
    def num_examples(self):
        return self.example_displacements.shape[0]

    def jacobian(self):
        return jacobian(self.example_displacements)

    def _lumped_mass_diagonal(self):
        return np.repeat(self.vertex_mass, 3)

    def mode_displacement(self, mode_index, amplitude=1.0):
        return (self.modes[:, mode_index] * amplitude).reshape(self.num_vertices, 3)

    def mode_strain_magnitude(self, mode_index):
        u = self.modes[:, mode_index].reshape(-1, 3)
        return np.linalg.norm(tet_strain(self.nodes, self.elements, u), axis=1)

    def example_displacement(self, name_or_index):
        k = (self.example_names.index(name_or_index) if isinstance(name_or_index, str)
             else int(name_or_index))
        return self.example_displacements[k]

    def __repr__(self):
        return (f"ExemplarPlasticModes(V={self.num_vertices}, T={self.num_tets}, "
                f"bones={len(self.bone_names)}, examples={self.num_examples}, "
                f"modes={self.modes.shape[1]})")
