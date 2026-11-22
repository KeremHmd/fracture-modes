"""Numpy-only tests for the cage-to-fine lifts (modegen/lift.py) and their runtime use."""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from modegen import lift  # noqa: E402


def cube_cage():
    """Unit cube split into 6 tets around the 0-6 diagonal (plus its 8 corners)."""
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    T = np.array([[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
                  [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]])
    return V, T


def fine_points(n=200, seed=0, lo=0.05, hi=0.95):
    return np.random.default_rng(seed).uniform(lo, hi, (n, 3))


def test_barycentric_reproduces_linear_fields():
    V, T = cube_cage()
    P = fine_points()
    ti, w, inside = lift.barycentric_map(V, T, P, candidates=None)
    assert inside.all()
    assert np.allclose(np.einsum("fk,fkj->fj", w, V[T[ti]]), P)
    A = np.array([[0.1, 0.0, 0.3], [0.0, -0.2, 0.0], [0.05, 0.0, 0.0]])
    cum = V @ A.T + [0.01, 0.02, 0.0]
    assert np.allclose(lift.lift_barycentric(cum, T, ti, w), P @ A.T + [0.01, 0.02, 0.0])


def test_barycentric_outside_points_clamp_to_the_cage():
    V, T = cube_cage()
    P = np.array([[1.2, 0.5, 0.5], [-0.3, -0.3, 0.2]])
    ti, w, inside = lift.barycentric_map(V, T, P, candidates=None)
    assert not inside.any()
    assert (w >= 0).all() and np.allclose(w.sum(axis=1), 1.0)
    Q = np.einsum("fk,fkj->fj", w, V[T[ti]])
    assert np.allclose(Q[0], [1.0, 0.5, 0.5], atol=1e-9)


def test_surface_lift_follows_the_cage_surface():
    V, T = cube_cage()
    assert len(lift.boundary_faces(T)) == 12
    P = np.array([[0.5, 0.5, 0.9], [0.3, 0.2, 0.95], [0.1, 0.5, 0.5]])
    c, w = lift.surface_map(V, T, P)
    Q = np.einsum("fk,fkj->fj", w, V[c])
    assert np.allclose(Q[0], [0.5, 0.5, 1.0])
    assert np.allclose(Q[1], [0.3, 0.2, 1.0])
    assert np.allclose(Q[2], [0.0, 0.5, 0.5])
    # A field that is linear on the top face is reproduced at the projected point.
    cum = np.zeros_like(V)
    cum[:, 2] = -0.1 * V[:, 0] * V[:, 2]          # top face: -0.1 x, bottom: 0
    d = lift.lift_corners(cum, c, w)
    assert np.isclose(d[0, 2], -0.05) and np.isclose(d[1, 2], -0.03)


def test_closest_point_regions():
    A, B, C = np.zeros((1, 3)), np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]])
    for p, q in (([0.2, 0.2, 1], [0.2, 0.2, 0]), ([-1, -1, 0], [0, 0, 0]), ([2, -1, 0], [1, 0, 0]),
                 ([0.5, -1, 0], [0.5, 0, 0]), ([1, 1, 0], [0.5, 0.5, 0]), ([-1, 0.5, 0], [0, 0.5, 0])):
        W = lift.closest_point_triangles(np.array([p], float), A, B, C)
        assert np.allclose(W[0] @ np.vstack([A, B, C]), q), (p, W)


def test_smoothing_keeps_constants_and_flattens_steps():
    F = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4], [3, 5, 4]])
    d = np.tile([0.0, 0.0, -0.02], (6, 1))
    assert np.allclose(lift.smooth_displacement(d, F, iterations=5), d)
    step = np.zeros((6, 3))
    step[:3, 2] = -0.02
    s = lift.smooth_displacement(step, F, iterations=3)
    assert np.ptp(s[:, 2]) < np.ptp(step[:, 2])
    assert lift.smooth_displacement(step, F, iterations=0) is not None


def test_make_lift_validates():
    assert lift.make_lift(None) == ("nearest", 0, 0.5)
    assert lift.make_lift({"lift": "surface", "smooth_iterations": 2})[:2] == ("surface", 2)
    try:
        lift.make_lift({"lift": "cubic"})
    except ValueError:
        return
    raise AssertionError("unknown lift accepted")


def _piece():
    V, T = cube_cage()
    fv = fine_points(50, seed=3)
    return SimpleNamespace(vertices=V, elements=T, fine_vertices=fv,
                           fine_triangles=np.array([[0, 1, 2]]), plastic_modes=object(),
                           materials=None, children=[], is_atomic=True)


def _world(render):
    from modegen.runtime import PluginWorld
    rt = SimpleNamespace(world=dict(gravity=[0, 0, 0], ground_y=None, restitution=0.3,
                                    impact_kick=0.5, poisson_ratio=0.3),
                         render=lift.make_lift(render))
    return PluginWorld(rt)


def test_runtime_nearest_is_the_frozen_instance():
    from fracture_utility.world_state import ActiveInstance
    w = _world(None)
    inst = w.spawn(_piece())
    assert type(inst) is ActiveInstance


def test_runtime_lifts_follow_the_option():
    from modegen.runtime import LiftedInstance
    p = _piece()
    cum = p.vertices @ np.diag([0.0, 0.0, -0.05])            # linear: d_z = -0.05 z
    for name in ("barycentric", "surface", "nearest"):
        w = _world({"lift": name, "smooth_iterations": 1 if name == "nearest" else 0})
        inst = w.spawn(p)
        assert isinstance(inst, LiftedInstance)
        inst.plastic_states[id(p)]["cum"] = cum.copy()
        d = inst.piece_deformed_fine_vertices(p) - p.fine_vertices
        if name == "barycentric":
            assert np.allclose(d[:, 2], -0.05 * p.fine_vertices[:, 2])
        else:
            assert np.all(d[:, 2] <= 0) and np.abs(d).max() <= 0.05 + 1e-12


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)}/{len(tests)} passed")
