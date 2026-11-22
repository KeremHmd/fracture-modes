"""numpy-only tests for the modegen framework and runtime plugins.

Run with ``python tests/test_framework.py`` (or pytest). They never import
fracture_utility, so they run without scipy / igl / MOSEK; the end-to-end
check on the real stack is ``tools/modegen_selftest.py``.
"""

import os
import sys
import types

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import modegen  # noqa: E402
from modegen import (Decision, ImpactContext, ImpactEvent, ImpactModel, Param,  # noqa: E402
                     PluginError, PluginRegistry, Profile, get_registry)
from modegen.interfaces import distribute_force  # noqa: E402

REG = get_registry()


# ----------------------------------------------------------------- fakes

def grid_cloud(n=9, extent=0.5):
    g = np.linspace(-extent, extent, n)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


class FakeMaterials:
    def __init__(self, T, sigma_y=1e8, sigma_c=3e8, cls=2):
        self.num_tets = T
        self.sigma_y = np.full(T, sigma_y)
        self.sigma_c = np.full(T, sigma_c)
        self.material_class = np.full(T, cls, dtype=np.int8)


class FakePlastic:
    def __init__(self, V, k, rng):
        m = rng.normal(size=(3 * V, k))
        self.modes = m / np.linalg.norm(m, axis=0)

    def _lumped_mass_diagonal(self):
        return np.ones(self.modes.shape[0])


def make_ctx(vertices, point, direction=(0, -1, 0), magnitude=1e6, k=8, extra=None, seed=0):
    rng = np.random.default_rng(seed)
    V = vertices.shape[0]
    piece = types.SimpleNamespace(vertices=vertices, elements=np.zeros((1, 4), int),
                                  materials=FakeMaterials(1), plastic_modes=FakePlastic(V, k, rng),
                                  depth=0)
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    world = types.SimpleNamespace(time=0.0, active_instances=[])
    return ImpactContext(world=world, event=ImpactEvent(point=np.asarray(point, float), direction=d,
                                                        magnitude=magnitude),
                         instance=None, piece=piece, point=np.asarray(point, float), direction=d,
                         magnitude=magnitude, rest_vertices=vertices, vertices=vertices.copy(),
                         elements=piece.elements, materials=piece.materials,
                         plastic_state=dict(cum=np.zeros((V, 3)), fired=[]), extra=dict(extra or {}))


def analysis(sigma_vm, sigma_y=1e8, sigma_c=3e8, cls="elastic-plastic", u=None):
    return dict(sigma_vm_max=sigma_vm, sigma_y_local=sigma_y, sigma_c_local=sigma_c,
                excess_y=sigma_vm - sigma_y, excess_c=sigma_vm - sigma_c,
                material_class=cls, u_flat=u)


# ----------------------------------------------------------------- registry / profiles

def test_builtins_and_user_plugins_discovered():
    assert not REG.load_errors, REG.load_errors
    assert "realistic.sellan_socp" in REG.names("fracture.generator")
    assert "artistic.voronoi_shards" in REG.names("fracture.generator")
    assert "example.ring" in REG.names("impact")
    for cat in REG.categories():
        assert REG.names(cat), f"no plugin in category {cat}"


def test_registry_errors_and_override():
    reg = PluginRegistry()

    @reg.register("impact", "t.a")
    class A(ImpactModel):
        PARAMS = {"r": Param(1.0)}

        def force_field(self, ctx):
            return np.zeros_like(ctx.vertices)

    try:
        reg.register("impact", "t.a")(A)
        raise AssertionError("duplicate registration should fail")
    except PluginError:
        pass
    reg.register("impact", "t.a", override=True)(A)
    try:
        reg.create("impact", {"plugin": "t.a", "params": {"rr": 2}})
        raise AssertionError("unknown param should fail")
    except PluginError as e:
        assert "suggestions" in str(e)
    try:
        reg.get("impact", "t.b")
        raise AssertionError("unknown plugin should fail")
    except PluginError as e:
        assert "t.a" in str(e)

    class Old(ImpactModel):
        API_VERSION = "0.9"

    try:
        reg.register("impact", "t.old")(Old)
        raise AssertionError("API mismatch should fail")
    except PluginError:
        pass
    try:
        reg.register("router", "t.wrong")(A)
        raise AssertionError("wrong base class should fail")
    except PluginError:
        pass


def test_all_profiles_validate():
    names = Profile.available()
    assert {"realistic", "blunt_trauma", "puncture", "slash", "exploding_wall"} <= set(names)
    for n in names:
        assert Profile.load(n).validate(REG) == [], n


def test_profile_merge_and_overrides():
    p = Profile.load("exploding_wall")
    assert p.get("precompute.plastic") is None
    assert p.get("precompute.fracture.plugin") == "artistic.voronoi_shards"
    assert "weights" not in p.get("precompute.fracture.params")
    assert p.get("precompute.elastic.params.poisson_ratio") == 0.3
    p.apply_overrides(["precompute.fracture.params.num_cells=12", "runtime.world.ground_y=-0.5",
                       "runtime.default_tool=shot"])
    assert p.get("precompute.fracture.params.num_cells") == 12
    rt = p.build_runtime(REG)
    assert rt.default_tool == "shot" and rt.world["ground_y"] == -0.5
    q = Profile.load("puncture")
    gen = q.build_precompute(REG).fracture
    assert gen.init.NAME == "puncture.local_seeds"
    assert gen.weights.NAME == "realistic.uniform_faces"


# ----------------------------------------------------------------- impact models

def test_impact_models_conserve_total_force_and_locality():
    V = grid_cloud()
    point = np.array([0.0, 0.5, 0.0])
    spreads = {}
    for name in REG.names("impact"):
        model = REG.create("impact", name)
        ctx = make_ctx(V, point)
        F = model.force_field(ctx)
        assert F.shape == V.shape, name
        assert np.allclose(F.sum(axis=0), [0.0, -1e6, 0.0], rtol=1e-6, atol=1e-3), name
        w = np.linalg.norm(F, axis=1)
        rel = V - point
        lateral = np.linalg.norm(rel - (rel @ ctx.direction)[:, None] * ctx.direction, axis=1)
        spreads[name] = float((w * lateral).sum() / w.sum())
    assert spreads["puncture.spike"] < 0.5 * spreads["bludgeon.blunt_dome"]


def test_blade_is_elongated_along_axis():
    V = grid_cloud(11)
    model = REG.create("impact", {"plugin": "slash.blade_line", "params": {"blade_axis": [1, 0, 0]}})
    F = model.force_field(make_ctx(V, [0.0, 0.5, 0.0]))
    w = np.linalg.norm(F, axis=1)
    ext = lambda ax: float((w * np.abs(V[:, ax])).sum() / w.sum())  # noqa: E731
    assert ext(0) > 3.0 * ext(2)


def test_distribute_force_zero_weights_fallback():
    F = distribute_force(np.zeros(4), [1, 0, 0], 2.0)
    assert np.allclose(F.sum(axis=0), [2.0, 0.0, 0.0])


# ----------------------------------------------------------------- routers

def test_stress_gate_table():
    r = REG.create("router", "realistic.stress_gate")
    ctx = make_ctx(grid_cloud(3), [0, 0, 0])
    assert r.decide(ctx, None).branch == "fracture"
    assert r.decide(ctx, analysis(5e8)).branch == "fracture"
    d = r.decide(ctx, analysis(2e8))
    assert d.branch == "plastic" and d.fallback == "fracture"
    assert r.decide(ctx, analysis(5e7)).branch == "none"
    assert r.decide(ctx, analysis(2e8, cls="brittle")).branch == "none"
    assert r.decide(ctx, analysis(9e9, cls="rigid")).branch == "none"


def test_blunt_gate_and_trigger_happy():
    ctx = make_ctx(grid_cloud(3), [0, 0, 0])
    b = REG.create("router", "bludgeon.blunt_gate")
    assert b.decide(ctx, analysis(4e8)).branch == "plastic"
    assert b.decide(ctx, analysis(6e8)).branch == "fracture"
    t = REG.create("router", {"plugin": "artistic.trigger_happy",
                              "params": {"plastic_enabled": False}})
    assert t.decide(ctx, analysis(1.0e8)).branch == "fracture"
    assert t.decide(ctx, analysis(5.0e7)).branch == "none"


# ----------------------------------------------------------------- plastic responders

def test_single_mode_fires_until_exhausted():
    V = grid_cloud(5)
    k = 4
    resp = REG.create("plastic.response", "realistic.single_mode")
    ctx = make_ctx(V, [0, 0.5, 0], k=k)
    u = np.random.default_rng(1).normal(size=3 * V.shape[0])
    seen = []
    for _ in range(k):
        out = resp.respond(ctx, None, analysis(2e8, u=u), Decision("plastic"))
        assert out["fired"]
        seen.append(out["winner"])
    assert sorted(seen) == list(range(k))
    out = resp.respond(ctx, None, analysis(2e8, u=u), Decision("plastic"))
    assert not out["fired"] and out["exhausted"]
    assert np.abs(ctx.plastic_state["cum"]).max() > 0


def test_local_dent_is_local_inward_and_saturates():
    V = grid_cloud(9)
    point = np.array([0.0, 0.5, 0.0])
    resp = REG.create("plastic.response", {"plugin": "bludgeon.local_dent",
                                           "params": {"radius": 0.2, "depth": 0.05,
                                                      "max_depth": 0.1}})
    ctx = make_ctx(V, point, k=10)
    u = np.random.default_rng(2).normal(size=3 * V.shape[0])
    a = analysis(3e8, u=u)
    out = resp.respond(ctx, None, a, Decision("plastic"))
    assert out["fired"]
    cum = ctx.plastic_state["cum"]
    far = np.linalg.norm(V - point, axis=1) > 0.2
    assert np.abs(cum[far]).max() == 0.0
    near = np.argmin(np.linalg.norm(V - point, axis=1))
    assert cum[near] @ ctx.direction >= 0.0
    exhausted = False
    for _ in range(20):
        out = resp.respond(ctx, None, a, Decision("plastic"))
        if not out["fired"]:
            exhausted = out.get("exhausted", False)
            break
    assert exhausted
    diag = np.linalg.norm(V.max(0) - V.min(0))
    assert np.linalg.norm(cum, axis=1).max() <= 0.1 * diag + 1e-9


# ----------------------------------------------------------------- precompute hooks

def test_local_seeds_vectors():
    V = grid_cloud(6)
    init = REG.create("fracture.init", "puncture.local_seeds")
    out = init.vectors(V, None, 5, 3)
    assert out.shape == (3 * V.shape[0], 5)
    n = V.shape[0]
    for i in range(5):
        comp = i % 3
        assert np.isclose(np.abs(out[:, i]).sum(), np.abs(out[comp * n:(comp + 1) * n, i]).sum())
        assert np.isclose(out[comp * n:(comp + 1) * n, i].max(), 1.0)


def test_grain_weighting():
    verts = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [1, 0, 0], [0, -1, 0]], float)
    elements = np.array([[0, 1, 2, 3], [0, 1, 2, 4], [0, 2, 3, 5]])
    g = REG.create("fracture.weights", {"plugin": "slash.grain",
                                        "params": {"grain_normal": [1, 0, 0], "strength": 0.9}})
    g.prepare(verts, elements, None)
    assert np.isclose(g.multiplier(0, 1), 0.1)
    assert np.isclose(g.multiplier(0, 2), 1.0)


# ----------------------------------------------------------------- post / projection

class FakeInstance:
    def __init__(self, c):
        self.c = np.asarray(c, float)
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)

    def world_aabb(self):
        return self.c - 0.1, self.c + 0.1


def test_explosive_kick_and_event_log():
    kick = REG.create("post", "artistic.explosive_kick")
    log = REG.create("post", "common.event_log")
    ctx = make_ctx(grid_cloud(3), [0, 0, 0])
    insts = [FakeInstance([1, 0, 0]), FakeInstance([-1, 0, 0])]
    res = modegen.ImpactResult("fracture", "x", instances=insts, tool="t")
    kick.on_fracture(ctx, res)
    assert insts[0].velocity[0] > 0 and insts[1].velocity[0] < 0
    assert np.linalg.norm(insts[0].angular_velocity) > 0
    log.on_fracture(ctx, res)
    log.on_none(ctx, modegen.ImpactResult("none", "below_yield", tool="t"))
    assert [e["branch"] for e in log.events] == ["fracture", "none"]


def test_restrict_separation():
    from modegen.projection import force_to_profile, restrict_separation

    modes = types.SimpleNamespace(
        piece_labels_after_impact=np.array([0, 0, 1, 2, 3]),
        piece_to_tet_matrix=np.eye(5), fine_vertices=None,
    )
    n = restrict_separation(modes, np.array([False, False, True, False, False]))
    assert n == 2
    assert modes.piece_labels_after_impact[2] != modes.piece_labels_after_impact[0]
    assert len(set(modes.piece_labels_after_impact[[0, 1, 3, 4]])) == 1
    prof = force_to_profile(np.array([[0, -2.0, 0], [0, -1.0, 0]]))
    assert np.isclose(np.abs(prof).max(), 1.0) and prof.shape == (6,)


# ----------------------------------------------------------------- crush (modal, bounded)

class FakeModalPlastic:
    """Orthonormal basis (M = identity) with known per-mode strain/energy."""

    def __init__(self, V, k, rng):
        q, _ = np.linalg.qr(rng.normal(size=(3 * V, k)))
        self.modes = q
        self._peak = rng.uniform(0.5, 2.0, size=k)
        self.mode_energies = rng.uniform(0.5, 2.0, size=k)

    def _lumped_mass_diagonal(self):
        return np.ones(self.modes.shape[0])

    def mode_strain_magnitude(self, k):
        return np.array([self._peak[k], 0.1 * self._peak[k]])


def test_dent_only_router_never_fractures():
    r = REG.create("router", "crush.dent_only")
    ctx = make_ctx(grid_cloud(3), [0, 0, 0])
    for vm in (5e7, 2e8, 5e8, 1e12):
        assert r.decide(ctx, analysis(vm)).branch in ("plastic", "none")
    assert r.decide(ctx, analysis(1e12)).fallback is None


def test_modal_dent_is_modal_bounded_and_converges():
    V = grid_cloud(7)
    k = 12
    ctx = make_ctx(V, [0.0, 0.5, 0.0])
    pm = FakeModalPlastic(V.shape[0], k, np.random.default_rng(3))
    ctx.piece.plastic_modes = pm
    resp = REG.create("plastic.response", {"plugin": "crush.modal_dent",
                                           "params": {"depth": 0.2, "ultimate_strain": 0.02,
                                                      "work_budget": 0.5, "hardening": 1.0}})
    u = np.random.default_rng(4).normal(size=3 * V.shape[0])
    a = analysis(3e8, u=u)
    caps, energies, capacity = resp._limits(pm)
    w_max = 0.5 * capacity
    works, final = [], False
    for _ in range(300):
        out = resp.respond(ctx, None, a, Decision("plastic"))
        if not out["fired"]:
            final = out["exhausted"] and out["reason"] == "final_state"
            break
        works.append(out["work_used"])
        alpha = ctx.plastic_state["alpha"]
        assert np.all(np.abs(alpha) <= caps + 1e-12)
        assert energies @ np.abs(alpha) <= w_max * (1 + 1e-9)
        assert np.allclose(ctx.plastic_state["cum"].reshape(-1), pm.modes @ alpha)
    assert final, "never reached a final state"
    assert all(b >= a_ - 1e-12 for a_, b in zip(works, works[1:]))
    before = ctx.plastic_state["cum"].copy()
    out = resp.respond(ctx, None, a, Decision("plastic"))
    assert not out["fired"] and np.array_equal(before, ctx.plastic_state["cum"])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failed += 1
            import traceback
            print(f"FAIL {fn.__name__}: {exc!r}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
