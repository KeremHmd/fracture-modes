"""Cap-face detail synthesis for fractured meshes"""

import numpy as np
import igl


def cap_face_indices(verts, faces, v_orig, f_orig, eps_rel=0.005):
    """Indices of carved faces classified as cap"""
    bbox_diag = float(np.linalg.norm(v_orig.max(axis=0) - v_orig.min(axis=0)))
    eps = eps_rel * bbox_diag
    centroids = verts[faces].mean(axis=1)
    d2, _, _ = igl.point_mesh_squared_distance(centroids, v_orig, f_orig)
    return np.where(d2 > eps ** 2)[0]


def cap_vertex_normals(verts, faces, cap_idx):
    """Per-vertex normals defined by the cap faces only"""
    cap_faces = faces[cap_idx]
    e1 = verts[cap_faces[:, 1]] - verts[cap_faces[:, 0]]
    e2 = verts[cap_faces[:, 2]] - verts[cap_faces[:, 0]]
    face_normals = np.cross(e1, e2)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-12
    face_normals = face_normals / norms
    vertex_normals = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vertex_normals, cap_faces[:, k], face_normals)
    vnorms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals = np.where(vnorms > 1e-12, vertex_normals / vnorms, 0.0)
    return vertex_normals


def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def value_noise_3d(points, frequency=20.0, seed=0, octaves=4,
                   persistence=0.5, grid_size=64):
    """3D fractal value noise sampled"""
    rng = np.random.RandomState(int(seed))
    grid = rng.uniform(-1.0, 1.0, (grid_size, grid_size, grid_size))

    result = np.zeros(points.shape[0])
    amp = 1.0
    norm_sum = 0.0
    for o in range(octaves):
        freq = frequency * (2.0 ** o)
        scaled = points * freq
        i = np.floor(scaled).astype(int)
        f = scaled - i
        i0 = i % grid_size
        i1 = (i + 1) % grid_size

        c000 = grid[i0[:, 0], i0[:, 1], i0[:, 2]]
        c100 = grid[i1[:, 0], i0[:, 1], i0[:, 2]]
        c010 = grid[i0[:, 0], i1[:, 1], i0[:, 2]]
        c110 = grid[i1[:, 0], i1[:, 1], i0[:, 2]]
        c001 = grid[i0[:, 0], i0[:, 1], i1[:, 2]]
        c101 = grid[i1[:, 0], i0[:, 1], i1[:, 2]]
        c011 = grid[i0[:, 0], i1[:, 1], i1[:, 2]]
        c111 = grid[i1[:, 0], i1[:, 1], i1[:, 2]]

        fx = _smoothstep(f[:, 0])
        fy = _smoothstep(f[:, 1])
        fz = _smoothstep(f[:, 2])

        c00 = c000 * (1 - fx) + c100 * fx
        c10 = c010 * (1 - fx) + c110 * fx
        c01 = c001 * (1 - fx) + c101 * fx
        c11 = c011 * (1 - fx) + c111 * fx
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy
        c = c0 * (1 - fz) + c1 * fz

        result += amp * c
        norm_sum += amp
        amp *= persistence

    return result / norm_sum if norm_sum > 0 else result


def displace_cap_faces(verts, faces, v_orig, f_orig,
                       amplitude=0.01, frequency=20.0,
                       seed=0, octaves=4,
                       cap_idx=None):
    """Return a new vertex array with cap-face vertices displaced by noise"""
    if cap_idx is None:
        cap_idx = cap_face_indices(verts, faces, v_orig, f_orig)
    if len(cap_idx) == 0:
        return verts.copy()

    cap_faces = faces[cap_idx]

    surface_mask = np.ones(len(faces), dtype=bool)
    surface_mask[cap_idx] = False
    surface_faces = faces[surface_mask]
    used_by_surface = np.zeros(verts.shape[0], dtype=bool)
    if len(surface_faces) > 0:
        used_by_surface[surface_faces.ravel()] = True
    used_by_cap = np.zeros(verts.shape[0], dtype=bool)
    used_by_cap[cap_faces.ravel()] = True
    cap_only = used_by_cap & ~used_by_surface

    normals = cap_vertex_normals(verts, faces, cap_idx)
    noise = value_noise_3d(verts, frequency=frequency, seed=seed,
                           octaves=octaves)
    new_verts = verts.copy()
    disp = (amplitude * noise[:, None]) * normals
    new_verts[cap_only] += disp[cap_only]
    return new_verts
