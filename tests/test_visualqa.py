"""Numpy-only tests for modegen/visualqa.py (snapshot metric helpers)."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from modegen import visualqa as vq  # noqa: E402


def grid(n=11):
    xs = np.linspace(0, 1, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    V = np.c_[X.ravel(), Y.ravel(), np.zeros(n * n)]
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
            F += [[a, b, c], [a, c, d]]
    return V, np.array(F)


def test_edges_and_normals():
    V, F = grid(3)
    assert len(vq.mesh_edges(F)) == 16
    n = vq.vertex_normals(V, F)
    assert np.allclose(np.abs(n[:, 2]), 1.0)


def test_blockiness_step_vs_smooth():
    V, F = grid()
    E = vq.mesh_edges(F)
    patch = (V[:, 0] * 4).astype(int) * 10 + (V[:, 1] * 4).astype(int)
    stepped = np.zeros_like(V)
    stepped[:, 2] = patch.astype(float)          # constant per patch
    assert vq.blockiness(stepped, V, E, patch) == 1.0
    smooth = np.zeros_like(V)
    smooth[:, 2] = V[:, 0] + V[:, 1]
    b = vq.blockiness(smooth, V, E, patch)
    assert 0.3 < b < 0.7, b
    assert vq.blockiness(np.zeros_like(V), V, E, patch) is None


def test_dent_metrics():
    V, F = grid()
    N = np.tile([0.0, 0.0, 1.0], (len(V), 1))
    D = np.zeros_like(V)
    r = np.linalg.norm(V[:, :2] - [0.3, 0.6], axis=1)
    D[:, 2] = -0.05 * np.exp(-(r / 0.15) ** 2)
    m = vq.dent_metrics([0.3, 0.6, 0.0], V, D, N)
    assert abs(m["depth_at_hit"] - 0.05) < 1e-9
    assert abs(m["max_depth"] - 0.05) < 1e-9 and m["offset"] < 1e-9
    m2 = vq.dent_metrics([0.7, 0.6, 0.0], V, D, N, radius=0.05)
    assert m2["depth_at_hit"] < 0.01 and abs(m2["offset"] - 0.4) < 1e-9
    assert vq.norm_stats(D)[0] == m["max_depth"]
    A = vq.vertex_areas(V, F)
    assert abs(A.sum() - 1.0) < 1e-12
    dv = vq.dent_volume([0.3, 0.6, 0.0], V, D, N, A)
    # Gaussian volume 0.05 * pi * 0.15^2, slightly cut by the grid edge.
    assert 0.8 * 0.05 * np.pi * 0.15 ** 2 < dv["volume"] < 1.1 * 0.05 * np.pi * 0.15 ** 2
    assert dv["centroid_offset"] < 0.02
    assert vq.dent_volume([0, 0, 0], V, 0 * D, N, A)["volume"] == 0.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)}/{len(tests)} passed")
