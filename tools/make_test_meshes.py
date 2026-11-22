# Generate simple, watertight test meshes (numpy only) into data/.
#
# Usage:
#   python tools\make_test_meshes.py [--wall W H D] [--pillar W H D] [--edge L]
#
# Writes data/wall.obj (thick wall slab) and data/pillar.obj (tall column).
# Boxes are evenly tessellated so the fine surface carves and dents smoothly.

import argparse
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def box_mesh(size, edge):
    sx, sy, sz = size
    n = [max(2, int(round(s / edge))) for s in size]
    verts, faces = [], []

    def grid(origin, u, v, nu, nv):
        base = len(verts)
        for j in range(nv + 1):
            for i in range(nu + 1):
                verts.append(origin + u * (i / nu) + v * (j / nv))
        for j in range(nv):
            for i in range(nu):
                a = base + j * (nu + 1) + i
                b, c, d = a + 1, a + nu + 2, a + nu + 1
                faces.append((a, b, c))
                faces.append((a, c, d))

    o = np.array([-sx / 2, -sy / 2, -sz / 2])
    X, Y, Z = np.array([sx, 0, 0.]), np.array([0, sy, 0.]), np.array([0, 0, sz])
    grid(o, Y, X, n[1], n[0])               # z = min
    grid(o + Z, X, Y, n[0], n[1])           # z = max
    grid(o, X, Z, n[0], n[2])               # y = min
    grid(o + Y, Z, X, n[2], n[0])           # y = max
    grid(o, Z, Y, n[2], n[1])               # x = min
    grid(o + X, Y, Z, n[1], n[2])           # x = max

    V = np.array(verts)
    key = np.round(V / (edge * 1e-3)).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    V = V[first]
    F = inv.reshape(-1)[np.array(faces)]
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    if np.einsum("ij,ij->i", a, np.cross(b, c)).sum() < 0:
        F = F[:, [0, 2, 1]]
    return V, F


def write_obj(path, V, F):
    with open(path, "w") as f:
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in F + 1:
            f.write(f"f {t[0]} {t[1]} {t[2]}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wall", type=float, nargs=3, default=[2.0, 1.2, 0.3])
    ap.add_argument("--pillar", type=float, nargs=3, default=[0.4, 2.0, 0.4])
    ap.add_argument("--edge", type=float, default=0.025)
    import _common  # noqa: E402  (path setup + version string; numpy only)
    _common.add_version_arg(ap)
    args = ap.parse_args()
    for name, size in (("wall", args.wall), ("pillar", args.pillar)):
        V, F = box_mesh(size, args.edge)
        path = os.path.join(ROOT, "data", f"{name}.obj")
        write_obj(path, V, F)
        E = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
        _, cnt = np.unique(E, axis=0, return_counts=True)
        vol = np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6
        print(f"{path}: {len(V)} verts, {len(F)} faces, watertight={bool(np.all(cnt == 2))}, "
              f"volume={vol:.4f} (expected {np.prod(size):.4f})")
