"""numpy-only tests for example-based plasticity (``exemplar.*``).

Run with ``python tests/test_exemplar.py`` or ``python tests/run_all.py``.
scipy is switched off (``USE_SCIPY = False``) so the pure-Python Dijkstra is
used; nothing here imports fracture_utility, scipy or igl.
"""

import json
import math
import os
import sys
import types

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from modegen import Decision, get_registry  # noqa: E402
from modegen import statecarry  # noqa: E402
from modegen.interfaces import distribute_force  # noqa: E402
from modegen.solvers import exemplar as X  # noqa: E402

X.USE_SCIPY = False
REG = get_registry()


# ----------------------------------------------------------------- fixtures

def box_tets(n=(6, 4, 3), size=(1.2, 0.8, 0.4)):
    """Structured tet mesh of a box (6 tets per cell), centred at the origin."""
    nx, ny, nz = n
    g = [np.linspace(-s / 2, s / 2, k + 1) for s, k in zip(size, n)]
    X_, Y_, Z_ = np.meshgrid(*g, indexing="ij")
    V = np.stack([X_.ravel(), Y_.ravel(), Z_.ravel()], axis=1)
    idx = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k  # noqa: E731
    T = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c = [idx(i + a, j + b, k + d) for a in (0, 1) for b in (0, 1) for d in (0, 1)]
                # c index = 4a + 2b + d ; Kuhn split along the 0-7 diagonal
                for p in ((0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7), (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7)):
                    T.append([c[q] for q in p])
    return V, np.array(T, dtype=np.int64)


def make_bundle(V, T, params=None, sigma_y=None):
    gen = REG.create("plastic.generator", {"plugin": "exemplar.skinned_examples",
                                           "params": dict(params or {})})
    rig = gen.build_rig(V, T)
    radius = float(gen.propagation_radius) * rig.diag
    pm = X.ExemplarPlasticModes(
        nodes=V, elements=T, bone_names=rig.bone_names, bone_positions=rig.bone_pos,
        skin_weights=rig.W, example_names=rig.ex_names, example_transforms=rig.ex_T,
        example_displacements=rig.D, vertex_mass=X.lumped_vertex_mass(V, T),
        root_vertex_indices=np.arange(len(V)), reference_diag=rig.diag,
        example_scale=rig.scale, propagation_radius=radius,
        neighbourhoods=X.geodesic_table(V, T, radius=radius), sigma_y=sigma_y)
    return gen, rig, pm


V_BOX, T_BOX = box_tets()
GEN, RIG, PM = make_bundle(V_BOX, T_BOX, dict(num_bones=6, presets=["dent"]))


def ctx_for(pm, state, point, direction, magnitude=5e6, radius=0.3):
    V = pm.nodes
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    cur = V + state["cum"]
    dist = np.linalg.norm(cur - point, axis=1)
    force = distribute_force(np.clip(1 - (dist / radius) ** 2, 0, None) ** 1.5, d, magnitude)
    piece = types.SimpleNamespace(plastic_modes=pm, vertices=V)
    ctx = types.SimpleNamespace(piece=piece, plastic_state=state, rest_vertices=V,
                                vertices=cur, point=np.asarray(point, float), direction=d)
    return ctx, force


def fresh_state(pm):
    return dict(cum=np.zeros((pm.num_vertices, 3)), fired=[])


YIELD = dict(sigma_y_local=1.0, sigma_vm_max=2.0)
TOP = np.array([0.0, 0.4, 0.0])          # centre of the box's top (y = +0.4) face
DOWN = np.array([0.0, -1.0, 0.0])


def responder(**params):
    return REG.create("plastic.response", {"plugin": "exemplar.impulse_blend", "params": params})


# ----------------------------------------------------------------- kernel / steps

def test_cubic_kernel():
    phi = X.cubic_kernel
    assert phi(0.0) == 1.0 and phi(1.0) == 0.0 and phi(1.5) == 0.0 and phi(-0.5) == phi(0.5)
    x = np.linspace(0, 1, 101)
    y = phi(x)
    assert np.all(np.diff(y) <= 1e-15), "monotone decreasing on [0, 1]"
    h = 1e-6
    assert abs((phi(h) - phi(0.0)) / h) < 1e-5            # phi'(0) = 0
    assert abs((phi(1.0) - phi(1.0 - h)) / h) < 1e-5      # phi'(1) = 0
    assert np.isclose(phi(0.5), 0.5)


def test_eq8_step_direction_and_size():
    rng = np.random.default_rng(1)
    V, K = 7, 4
    J = rng.normal(size=(V, 3, K + 1))
    J[:, :, 0] = 0.0
    dx = rng.normal(size=(V, 3))
    alpha = 0.7
    de = X.weight_step(J, dx, alpha)
    g = np.einsum("vck,vc->vk", J, dx)
    for i in range(V):
        assert np.isclose(np.linalg.norm(de[i]), alpha * np.linalg.norm(dx[i]))
        cos = de[i] @ g[i] / (np.linalg.norm(de[i]) * np.linalg.norm(g[i]))
        assert np.isclose(cos, 1.0)
        assert de[i, 0] == 0.0                              # rest column has J = 0
    J0 = J.copy()
    J0[2] *= 1e-12                                          # |J^T dx| <= 1e-8
    assert np.all(X.weight_step(J0, dx, alpha)[2] == 0.0)
    assert np.allclose(X.weight_step(J, dx, alpha, step="natural"), alpha * g)   # Eq. 10


def test_eq4_and_yield_threshold():
    j = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]])
    dx = X.desired_displacement(j, total_mass=2.0, dt=0.1, beta=1.0)
    assert np.allclose(dx[0], (0.1 / 2.0) * (5.0 - 1.0) * np.array([0.6, 0.8, 0.0]))
    assert np.all(dx[1] == 0.0) and np.all(dx[2] == 0.0)    # below beta / zero impulse
    capped = X.desired_displacement(j, 2.0, 0.1, 1.0, max_step=0.05)
    assert np.isclose(np.linalg.norm(capped[0]), 0.05)
    # responder: a hit exactly at yield (sigma_vm == sigma_y) changes nothing
    st = fresh_state(PM)
    ctx, f = ctx_for(PM, st, TOP, DOWN)
    out = responder().respond(ctx, f, dict(sigma_y_local=1.0, sigma_vm_max=1.0), Decision("plastic"))
    assert not out["fired"] and out["reason"] == "below_beta" and not out.get("exhausted")
    assert np.all(st["cum"] == 0.0) and "exemplar_E" not in st
    # absolute beta above every impulse -> no change either
    out = responder(beta_mode="absolute", beta=1e12).respond(ctx, f, YIELD, Decision("plastic"))
    assert out["reason"] == "below_beta"


def test_propagation_matches_definition():
    ip, ix, dd = X.geodesic_table(V_BOX, T_BOX, radius=0.5)
    rng = np.random.default_rng(2)
    de = np.zeros((len(V_BOX), 3))
    src = rng.choice(len(V_BOX), size=5, replace=False)
    de[src] = rng.normal(size=(5, 3))
    gamma = 0.3
    got = X.propagate(de, ip, ix, dd, gamma)
    dense = X.dense_rows(ip, ix, dd, len(V_BOX))
    want = np.zeros_like(de)
    for i in src:
        want += X.cubic_kernel(dense[i] / gamma)[:, None] * de[i][None, :]
    assert np.allclose(got, want)
    assert np.allclose(got[src], de[src] + (want[src] - de[src]))   # phi(0) = 1 term included


def test_geodesic_table_fallback_is_shortest_path():
    ip, ix, dd = X.geodesic_table(V_BOX, T_BOX)
    D = X.dense_rows(ip, ix, dd, len(V_BOX))
    assert np.allclose(D, D.T) and np.all(np.diag(D) == 0.0)
    E = X.tet_edges(T_BOX)
    L = np.linalg.norm(V_BOX[E[:, 0]] - V_BOX[E[:, 1]], axis=1)
    assert np.all(D[E[:, 0], E[:, 1]] <= L + 1e-12)                  # edges are upper bounds
    eu = np.linalg.norm(V_BOX[:, None] - V_BOX[None], axis=2)
    assert np.all(D >= eu - 1e-12)                                   # never shorter than straight
    ip2, ix2, dd2 = X.geodesic_table(V_BOX, T_BOX, radius=0.25)
    D2 = X.dense_rows(ip2, ix2, dd2, len(V_BOX))
    m = np.isfinite(D2)
    assert np.allclose(D2[m], D[m]) and np.all(D[~m] > 0.25 - 1e-12)


# ----------------------------------------------------------------- schedule / barycentric

def test_lambda_schedule_converges_to_full_update():
    rng = np.random.default_rng(3)
    E0 = X.project_barycentric(rng.random((10, 4)))
    dE = 1e-3 * rng.normal(size=(10, 4))
    for lam in (0.0, 0.3, 0.5, 0.9):
        for n in (1, 3, 10):
            E, rest = X.apply_schedule(E0, dE, lam, n, clamp=False)
            assert np.allclose(E, E0 + (1 - lam ** n) * dE)
            assert np.allclose(E - E0 + rest, dE)            # applied + pending = total
        n = X.iterations_for(lam, 1e-6)
        E, rest = X.apply_schedule(E0, dE, lam, n, clamp=False)
        assert np.abs(E - (E0 + dE)).max() <= 1e-6 * np.abs(dE).max() + 1e-15
        assert lam ** n <= 1e-6
    # clamped version still converges to the clamped full update when nothing clamps
    E, _ = X.apply_schedule(E0, dE - dE.mean(axis=1, keepdims=True), 0.5, 40, clamp=True)
    assert np.allclose(E, X.project_barycentric(E0 + dE - dE.mean(axis=1, keepdims=True)), atol=1e-9)


def test_barycentric_rows_stay_valid():
    rng = np.random.default_rng(4)
    E = X.project_barycentric(rng.random((50, 6)))
    for _ in range(20):
        E, _ = X.apply_schedule(E, 10.0 * rng.normal(size=E.shape), 0.5, 3)
        assert np.all(E >= 0.0) and np.allclose(E.sum(axis=1), 1.0)
    assert np.allclose(X.project_barycentric(-np.ones((2, 3))), [[1, 0, 0], [1, 0, 0]])
    st = fresh_state(PM)
    resp = responder(alpha=50.0, max_step=1.0)
    for k in range(6):
        p = V_BOX[[k * 17 % len(V_BOX)]][0]
        ctx, f = ctx_for(PM, st, p, -p if np.linalg.norm(p) > 0 else DOWN, magnitude=1e9)
        resp.respond(ctx, f, YIELD, Decision("plastic"))
        E = st.get("exemplar_E")
        if E is not None:
            assert np.all(E >= 0.0) and np.allclose(E.sum(axis=1), 1.0)


def test_reconstruct_weights_from_cum():
    rng = np.random.default_rng(5)
    D = rng.normal(size=(2, 9, 3))
    E = X.project_barycentric(rng.random((9, 3)))
    cum = X.blend_displacement(E, D)
    E2 = X.reconstruct_weights(cum, D)
    assert np.allclose(X.blend_displacement(E2, D), cum)
    assert np.allclose(E2, E)


# ----------------------------------------------------------------- presets / file

def test_dent_preset_moves_only_its_bone():
    pos = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    nrm = np.array([[0.0, 1, 0], [0.0, 0, 1]])
    ex = X.preset_dent(pos, nrm, 0.1)
    assert [n for n, _ in ex] == ["dent_0", "dent_1"]
    T0 = ex[0][1]
    assert np.allclose(X.apply_affine(T0[0], pos[:1]), pos[:1] - [0, 0.1, 0])
    assert np.allclose(T0[1], X.affine())
    # on the box: the dent of bone b moves the bone's anchor inward along -n
    b = 0
    a = int(np.argmin(np.linalg.norm(V_BOX - RIG.bone_pos[b], axis=1)))
    N, on = X.outward_vertex_normals(V_BOX, T_BOX)
    assert on[a]
    d = PM.example_displacement("dent_0")[a]
    assert d @ N[a] < 0.0                                    # inward
    assert np.linalg.norm(d) <= GEN.dent_depth * RIG.diag + 1e-12


def test_bend_preset_maps_centre_line_onto_arc():
    c = np.zeros(3)
    axes = np.eye(3)
    L, ang = 2.0, 60.0
    kappa = math.radians(ang) / L
    s = np.linspace(-1, 1, 9)
    P = np.stack([s, np.zeros_like(s), np.zeros_like(s)], axis=1)
    pos, rots = X.bend_map(P, c, axes, kappa)
    R = 1 / kappa
    centre_of_curvature = c + R * axes[2]
    assert np.allclose(np.linalg.norm(pos - centre_of_curvature, axis=1), R)
    arc = R * np.abs(np.arctan2(pos[:, 0], R - pos[:, 2]))
    assert np.allclose(arc, np.abs(s))                       # arc length preserved
    for k, Rm in enumerate(rots):
        assert np.allclose(Rm @ Rm.T, np.eye(3)) and np.isclose(np.linalg.det(Rm), 1.0)
        tangent = np.array([math.cos(kappa * s[k]), 0.0, math.sin(kappa * s[k])])
        assert np.allclose(Rm @ axes[0], tangent)
    ex = X.preset_bend(P, c, axes, L, ang)
    assert [n for n, _ in ex] == ["bend_pos", "bend_neg"]
    for b in range(len(P)):
        assert np.allclose(X.apply_affine(ex[0][1][b], P[b:b + 1])[0], pos[b])


def test_twist_and_squash_presets():
    rng = np.random.default_rng(6)
    P = rng.normal(size=(5, 3))
    c = np.zeros(3)
    axes = np.eye(3)
    ex = X.preset_twist(P, c, axes, 2.0, 40.0)
    for b in range(5):
        T = ex[0][1][b]
        q = X.apply_affine(T, P[b:b + 1])[0]
        assert np.isclose(np.linalg.norm(q[1:]), np.linalg.norm(P[b, 1:]))   # distance to axis
        assert np.isclose(q[0], P[b, 0])
        ang = math.radians(40.0) * P[b, 0] / 2.0
        assert np.isclose(math.atan2(np.cross(P[b, 1:], q[1:]), P[b, 1:] @ q[1:]), ang)
    sq = X.preset_squash(3, c, [0, 1, 0], 0.2)[0][1]
    S = sq[0][:, :3]
    assert np.isclose(np.linalg.det(S), 1.0) and np.isclose(S[1, 1], 0.8)
    assert np.allclose(sq[0], sq[2])


def test_authored_example_file():
    with open(os.path.join(ROOT, "examples", "wall_exemplar.json"), encoding="utf-8") as f:
        data = json.load(f)
    U = np.array([[-1.0, -0.5, -0.2], [1.0, 0.5, 0.2]])
    names, pos, ex_names, T = X.parse_example_file(data, U)
    assert len(names) == len(pos) == 9 and len(ex_names) == T.shape[0] == 12
    assert np.allclose(pos[names.index("centre")], [0, 0, 0])
    diag = np.linalg.norm(U[1] - U[0])
    k = ex_names.index("dent_front_centre")
    b = names.index("centre")
    assert np.allclose(T[k, b, :, 3], [0, 0, -0.05 * diag])
    others = [i for i in range(len(names)) if i != b]
    assert np.allclose(T[k, others], X.affine())
    bad = dict(data, examples=[{"name": "x", "transforms": {"nobody": {"translate": [0, 0, 1]}}}])
    try:
        X.parse_example_file(bad, U)
        raise AssertionError("unknown bone accepted")
    except ValueError:
        pass


# ----------------------------------------------------------------- bundle / responder

def test_modes_contract_and_rig():
    M = PM.modes.T @ (PM._lumped_mass_diagonal()[:, None] * PM.modes)
    assert np.allclose(M, np.eye(M.shape[0]), atol=1e-10)
    assert PM._lumped_mass_diagonal().shape == (3 * PM.num_vertices,)
    m3 = PM._lumped_mass_diagonal()
    for e in range(PM.num_examples):                         # modes span the examples
        d = PM.example_displacements[e].reshape(-1)
        proj = PM.modes @ (PM.modes.T @ (m3 * d))
        assert np.allclose(proj, d, atol=1e-10)
    assert np.allclose(RIG.W.sum(axis=1), 1.0) and RIG.W.min() >= 0.0 and RIG.W.max() <= 1.0
    assert len(PM.mode_energies) == PM.modes.shape[1] and np.all(PM.mode_energies >= 0)
    assert PM.mode_strain_magnitude(0).shape == (len(T_BOX),)


def test_determinism():
    _, rig2, pm2 = make_bundle(V_BOX, T_BOX, dict(num_bones=6, presets=["dent"]))
    assert np.array_equal(rig2.W, RIG.W) and np.array_equal(rig2.D, RIG.D)
    assert np.array_equal(rig2.bone_pos, RIG.bone_pos)
    _, rig3, _ = make_bundle(V_BOX, T_BOX, dict(num_bones=6, presets=["dent"], seed=7))
    assert not np.array_equal(rig3.bone_pos, RIG.bone_pos)
    outs = []
    for pm in (PM, pm2):
        st = fresh_state(pm)
        resp = responder()
        for _ in range(3):
            ctx, f = ctx_for(pm, st, TOP, DOWN)
            resp.respond(ctx, f, YIELD, Decision("plastic"))
        outs.append((st["cum"].copy(), st["exemplar_E"].copy()))
    assert np.array_equal(outs[0][0], outs[1][0]) and np.array_equal(outs[0][1], outs[1][1])


def _hammer(resp, st, n=200, magnitude=5e7):
    results = []
    for _ in range(n):
        ctx, f = ctx_for(PM, st, TOP, DOWN, magnitude=magnitude)
        results.append(resp.respond(ctx, f, YIELD, Decision("plastic", fallback="fracture")))
        if results[-1].get("exhausted"):
            break
    return results


def test_responder_dents_inward_and_saturates():
    st = fresh_state(PM)
    # absolute criterion only (saturation_ratio=0): run until nothing moves any more
    results = _hammer(responder(alpha=2.0, saturation_ratio=0.0), st)
    assert results[0]["fired"] and results[0]["reason"] == "dented"
    cum = st["cum"]
    near = np.linalg.norm(V_BOX - TOP, axis=1) < 0.15
    far = np.linalg.norm(V_BOX - TOP, axis=1) > 0.6
    assert (cum[near] @ DOWN).mean() > 0.0                   # pushed inward (down)
    assert np.linalg.norm(cum[near], axis=1).mean() > np.linalg.norm(cum[far], axis=1).mean()
    assert results[-1].get("exhausted") and results[-1]["reason"] == "saturated", results[-1]
    changes = [r["max_change"] for r in results if r["fired"]]
    assert changes[-1] < changes[0]                          # diminishing increments
    # saturation leaves the shape inside the examples' hull: |cum_i| <= max_e |D_e(i)|
    bound = np.linalg.norm(PM.example_displacements, axis=2).max(axis=0)
    assert np.all(np.linalg.norm(cum, axis=1) <= bound + 1e-12)
    # cum is exactly the skinned blend of the stored weights
    assert np.allclose(cum, X.blend_displacement(st["exemplar_E"], PM.example_displacements))
    # the relative criterion (default 0.05 of the same update on a fresh piece)
    # declares saturation earlier, exactly when the change falls below it
    st2 = fresh_state(PM)
    rel = _hammer(responder(alpha=2.0), st2)
    assert rel[-1]["reason"] == "saturated" and len(rel) <= len(results)
    assert rel[-1]["max_change"] < 0.05 * rel[-1]["fresh_change"]
    assert all(r["max_change"] >= 0.05 * r["fresh_change"] for r in rel[:-1])
    assert len(rel) > 1


def test_first_hit_is_never_saturation():
    """Weak hits, small alpha, poorly covered spots: a fresh piece always dents
    (or is below beta); saturation needs earlier deformation."""
    spots = [(TOP, DOWN), (np.array([-0.4, 0.4, 0.2]), np.array([0.0, -1.0, 0.0])),
             (np.array([0.6, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]))]
    for alpha in (0.03, 0.1, 1.0):
        for mag in (5e6, 5e7):
            for p, d in spots:
                st = fresh_state(PM)
                ctx, f = ctx_for(PM, st, p, d, magnitude=mag)
                out = responder(alpha=alpha).respond(ctx, f, YIELD, Decision("plastic"))
                assert out["reason"] in ("dented", "below_beta"), (alpha, mag, p, out)
                if out["fired"]:
                    assert np.isclose(out["max_change"], out["fresh_change"])


def test_smoothing_radius_changes_the_dent():
    """Eqs. 11-12 end to end: a larger gamma spreads the weight change further."""
    spread = []
    for g in (0.01, 0.5):
        st = fresh_state(PM)
        ctx, f = ctx_for(PM, st, TOP, DOWN, radius=0.12)
        responder(gamma=g).respond(ctx, f, YIELD, Decision("plastic"))
        changed = np.abs(st["exemplar_E"] - X.initial_weights(PM.num_vertices, PM.num_examples)).max(axis=1)
        spread.append(int(np.sum(changed > 1e-9)))
    assert spread[1] > spread[0], spread


def test_missing_weights_are_reconstructed_from_cum():
    st = fresh_state(PM)
    ctx, f = ctx_for(PM, st, TOP, DOWN)
    resp = responder()
    resp.respond(ctx, f, YIELD, Decision("plastic"))
    lost = dict(cum=st["cum"].copy(), fired=[])              # e.g. runtime profile swapped
    E, dE = resp.weights_for(PM, lost)
    assert np.allclose(X.blend_displacement(E, PM.example_displacements), lost["cum"], atol=1e-6)
    assert np.all(dE == 0.0)


def test_state_carry_through_fracture_subset():
    st = fresh_state(PM)
    resp = responder()
    for _ in range(3):
        ctx, f = ctx_for(PM, st, TOP, DOWN)
        resp.respond(ctx, f, YIELD, Decision("plastic"))
    keys = statecarry.responder_state_keys(resp)
    assert keys == ("exemplar_E", "exemplar_dE")
    pvi = np.flatnonzero(V_BOX[:, 0] < 0.0)                  # a fragment: the -x half
    base, extras = statecarry.subset_state(st, pvi, keys)
    assert base["fired"] == [] and np.array_equal(base["cum"], st["cum"][pvi])
    assert np.array_equal(extras["exemplar_E"], st["exemplar_E"][pvi])
    D_child = PM.example_displacements[:, pvi]               # child rows = parent rows
    assert np.allclose(X.blend_displacement(extras["exemplar_E"], D_child), base["cum"])
    # siblings copy; responders without declared keys get exactly the old dicts
    b2, e2 = statecarry.copy_state(st, keys)
    assert np.array_equal(e2["exemplar_dE"], st["exemplar_dE"]) and e2["exemplar_E"] is not st["exemplar_E"]
    legacy = REG.create("plastic.response", "realistic.single_mode")
    assert statecarry.responder_state_keys(legacy) == () and statecarry.responder_state_keys(None) == ()
    b3, e3 = statecarry.subset_state(st, pvi, ())
    assert e3 == {} and set(b3) == {"cum", "fired"}
    b4, e4 = statecarry.copy_state(dict(cum=st["cum"], fired=[1, 2], alpha=np.ones(3)), ())
    assert e4 == {} and b4["fired"] == [1, 2]
    assert statecarry.subset_state(st, np.array([len(V_BOX) + 5]), keys) is None
    inst_states = {7: dict(cum=base["cum"].copy(), fired=[])}
    statecarry.inject_extras(inst_states, {7: extras, 8: {"x": 1}})
    assert "exemplar_E" in inst_states[7] and 8 not in inst_states


def test_runtime_stays_numpy_only():
    """Fresh interpreter: build a bundle and run hits, then list heavy modules."""
    import subprocess
    code = (
        "import sys; sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
        "import test_exemplar as t\n"
        "st = t.fresh_state(t.PM); ctx, f = t.ctx_for(t.PM, st, t.TOP, t.DOWN)\n"
        "t.responder().respond(ctx, f, t.YIELD, t.Decision('plastic'))\n"
        "print([m for m in ('scipy', 'igl', 'fracture_utility', 'polyscope', 'mosek') if m in sys.modules])\n"
    ) % (ROOT, os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stdout


def test_generator_build_shares_root_rig_with_children():
    gen = REG.create("plastic.generator", {"plugin": "exemplar.skinned_examples",
                                           "params": dict(num_bones=6, presets=["dent"])})
    mats = types.SimpleNamespace(rho=np.full(len(T_BOX), 7800.0), sigma_y=np.full(len(T_BOX), 1e8))
    root = types.SimpleNamespace(vertices=V_BOX, elements=T_BOX, materials=mats, depth=0)
    pm_root = gen.build(root, None, dict(depth=0))
    # a child exactly as fracture_tree._build_child_from_piece makes it
    tet_mask = V_BOX[T_BOX].mean(axis=1)[:, 0] < 0.0
    used = np.unique(T_BOX[tet_mask])
    remap = -np.ones(len(V_BOX), dtype=np.int64)
    remap[used] = np.arange(len(used))
    child = types.SimpleNamespace(vertices=V_BOX[used], elements=remap[T_BOX[tet_mask]],
                                  materials=None, depth=1)
    pm_child = gen.build(child, None, dict(depth=1))
    assert np.array_equal(pm_child.root_vertex_indices, used)
    assert np.array_equal(pm_child.example_displacements, pm_root.example_displacements[:, used])
    assert np.array_equal(pm_child.skin_weights, pm_root.skin_weights[used])
    assert pm_child.example_names == pm_root.example_names
    assert np.isclose(pm_root.total_mass, 7800.0 * X.tet_volumes(V_BOX, T_BOX).sum())
    assert pm_child.reference_diag == pm_root.reference_diag
    # geodesics of the child stay inside the child (no path through the removed half)
    D = X.dense_rows(pm_child.nbr_indptr, pm_child.nbr_indices, pm_child.nbr_dist, len(used))
    assert np.all(np.diag(D) == 0.0)
    # a fresh generator without a root rig, or a node that is not part of the root, is refused
    gen2 = REG.create("plastic.generator", {"plugin": "exemplar.skinned_examples"})
    for g, node in ((gen2, child),
                    (gen, types.SimpleNamespace(vertices=V_BOX[used] + 0.01, elements=child.elements,
                                                materials=None, depth=1))):
        try:
            g.build(node, None, dict(depth=1))
            raise AssertionError("build should have refused")
        except ValueError:
            pass


def test_natural_step_is_scale_free_and_saturation_keeps_pending():
    outs = []
    for s in (1.0, 10.0):
        _, _, pm = make_bundle(s * V_BOX, T_BOX, dict(num_bones=6, presets=["dent"]))
        st = fresh_state(pm)
        ctx, f = ctx_for(pm, st, s * TOP, DOWN, radius=0.3 * s)
        # Eq. 4: dx = dt/m * j with m ~ s^3, so impulse_time ~ s^4 gives dx ~ s,
        # i.e. the same hit relative to the object's size.
        resp = responder(step="natural", max_step=1.0, beta_mode="absolute",
                         impulse_time=1e-3 * s ** 4)
        resp.respond(ctx, f, YIELD, Decision("plastic"))
        outs.append(st["exemplar_E"])
    assert np.allclose(outs[0], outs[1], atol=1e-9), np.abs(outs[0] - outs[1]).max()
    # saturation keeps the residual of earlier hits
    st = fresh_state(PM)
    resp = responder(iterations=1, lam=0.5)
    ctx, f = ctx_for(PM, st, TOP, DOWN)
    resp.respond(ctx, f, YIELD, Decision("plastic"))
    pending = st["exemplar_dE"].copy()
    assert np.abs(pending).max() > 0.0
    resp2 = responder(iterations=1, lam=0.5, min_increment=10.0)      # every hit counts as saturated
    out = resp2.respond(ctx, f, YIELD, Decision("plastic"))
    assert out["reason"] == "saturated" and np.array_equal(st["exemplar_dE"], pending)
    other = types.SimpleNamespace(piece=types.SimpleNamespace(plastic_modes=object()),
                                  plastic_state=fresh_state(PM))
    r = responder().respond(other, f, YIELD, Decision("plastic"))
    assert r["reason"] == "not_an_exemplar_bundle" and r["exhausted"]


def test_exemplar_profiles_validate_and_router_dents_first():
    from modegen import Profile
    for name in ("exemplar_dents", "exemplar_dent_only"):
        p = Profile.load(name)
        assert not p.validate(REG), (name, p.validate(REG))
        assert p.get("precompute.plastic.plugin") == "exemplar.skinned_examples"
        assert p.get("runtime.plastic_response.plugin") == "exemplar.impulse_blend"
    assert Profile.load("exemplar_dent_only").get("runtime.fracture_response") is None
    rt = Profile.load("exemplar_dents").build_runtime(REG)
    mk = lambda vm: dict(material_class="elastic-plastic", sigma_vm_max=vm, sigma_y_local=1.0,  # noqa: E731
                         sigma_c_local=3.0, excess_y=vm - 1.0, excess_c=vm - 3.0)
    assert rt.router.decide(None, mk(0.5)).branch == "none"
    d = rt.router.decide(None, mk(4.0))
    assert d.branch == "plastic" and d.fallback == "fracture"     # dents before it breaks
    assert rt.router.decide(None, mk(7.0)).branch == "fracture"   # > 2 x sigma_c breaks


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
