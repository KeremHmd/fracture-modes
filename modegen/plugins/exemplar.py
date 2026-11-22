"""Example-based plasticity after Jones, Thuerey, Shinar and Bargteil,
"Example-based Plastic Deformation of Rigid Bodies", ACM TOG 35(4), SIGGRAPH 2016.

* ``exemplar.skinned_examples`` (``plastic.generator``): precomputes a skinning
  rig (bones, skinning weights), a set of example poses (authored JSON file
  and/or procedural presets) and, per tree node, the example displacement
  fields and truncated geodesic neighbourhoods.
* ``exemplar.impulse_blend`` (``plastic.response``): turns each impact into
  per-vertex example-weight changes (Eqs. 4, 8, 11-13) and re-skins the piece.

The authoring format is described in the README. The numerics are in ``modegen/solvers/exemplar.py``.
"""

import json
import os

import numpy as np

from ..interfaces import Param, PlasticGenerator, PlasticResponder
from ..registry import PROJECT_ROOT, register

PRESETS = ("dent", "bend", "twist", "squash")


class _Rig:
    """Rig built once on the root node and shared by every node of the tree."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _resolve(path):
    if os.path.isabs(path) or os.path.isfile(path):
        return path
    cand = os.path.join(PROJECT_ROOT, path)
    if os.path.isfile(cand):
        return cand
    raise FileNotFoundError(f"example file {path!r} not found (also looked in {PROJECT_ROOT})")


@register("plastic.generator", "exemplar.skinned_examples")
class SkinnedExamples(PlasticGenerator):
    VERSION = "1.0.0"
    DESCRIPTION = ("Example-based plasticity rig (Jones et al. 2016): bones, skinning "
                   "weights and example poses (authored JSON and/or dent/bend/twist/squash "
                   "presets), built once on the root and evaluated on every node.")
    PARAMS = {
        "examples_file": Param(None, "authored example JSON (modegen.exemplar/1); None = presets only"),
        "presets": Param(["dent"], f"procedural example sets to add, any of {list(PRESETS)}"),
        "num_bones": Param(12, "automatic bones (farthest-point on surface vertices) when no file"),
        "seed": Param(0, "seed of the automatic bone placement"),
        "dent_depth": Param(0.05, "dent preset: bone translation, fraction of the root diagonal"),
        "bend_angle": Param(30.0, "bend preset: total bend angle (degrees) over the long axis"),
        "twist_angle": Param(30.0, "twist preset: total twist angle (degrees) over the long axis"),
        "squash": Param(0.15, "squash preset: compression fraction along squash_axis"),
        "squash_axis": Param("y", "squash preset axis", choices=("x", "y", "z", "long", "short")),
        "weights": Param("geodesic", "skinning weights: geodesic Gaussian, or clamped "
                                     "igl biharmonic", choices=("geodesic", "harmonic")),
        "weight_radius": Param(0.12, "geodesic weights: Gaussian sigma, fraction of the root diagonal"),
        "harmonic_k": Param(2, "harmonic weights: 1 harmonic, 2 biharmonic"),
        "propagation_radius": Param(0.25, "max geodesic propagation radius (Eq. 11) "
                                          "precomputed per node, fraction of the root diagonal"),
    }

    def setup(self):
        bad = [p for p in (self.presets or []) if p not in PRESETS]
        if bad:
            from ..interfaces import PluginError
            raise PluginError(f"{self.NAME}: unknown preset(s) {bad}; choose from {list(PRESETS)}")
        if not self.examples_file and not self.presets:
            from ..interfaces import PluginError
            raise PluginError(f"{self.NAME}: needs an examples_file or at least one preset")
        self._rig = None

    # ---------------------------------------------------------------- rig
    def build_rig(self, vertices, elements):
        from ..solvers import exemplar as X

        U = np.asarray(vertices, dtype=float)
        T = np.asarray(elements, dtype=np.int64)
        diag = X.bbox_diag(U)
        normals, on_surf = X.outward_vertex_normals(U, T)
        centre, axes, ext = X.principal_axes(U)
        surf = np.flatnonzero(on_surf)

        ex_names, ex_T = [], []
        if self.examples_file:
            with open(_resolve(self.examples_file), "r", encoding="utf-8") as f:
                data = json.load(f)
            bone_names, bone_pos, names, Ts = X.parse_example_file(data, U)
            ex_names += names
            ex_T += list(Ts)
        else:
            rng = np.random.default_rng(int(self.seed))
            start = int(rng.integers(len(surf)))
            pick = surf[X.farthest_point_sampling(U[surf], int(self.num_bones), start)]
            bone_pos = U[pick].copy()
            bone_names = [f"bone{i}" for i in range(len(pick))]

        B = len(bone_names)
        anchors = np.array([int(np.argmin(np.linalg.norm(U - p, axis=1))) for p in bone_pos])
        if len(set(anchors.tolist())) != B:
            raise ValueError("two bones snap to the same cage vertex; move them apart")
        surf_anchor = surf[[int(np.argmin(np.linalg.norm(U[surf] - p, axis=1))) for p in bone_pos]]

        for preset in self.presets or []:
            if preset == "dent":
                got = X.preset_dent(bone_pos, normals[surf_anchor], float(self.dent_depth) * diag)
            elif preset == "bend":
                got = X.preset_bend(bone_pos, centre, axes, ext[0], float(self.bend_angle))
            elif preset == "twist":
                got = X.preset_twist(bone_pos, centre, axes, ext[0], float(self.twist_angle))
            else:
                ax = {"x": np.eye(3)[0], "y": np.eye(3)[1], "z": np.eye(3)[2],
                      "long": axes[0], "short": axes[2]}[self.squash_axis]
                got = X.preset_squash(B, centre, ax, float(self.squash))
            for name, Tb in got:
                ex_names.append(name)
                ex_T.append(Tb)
        ex_T = np.stack(ex_T)

        if self.weights == "harmonic":
            W = X.harmonic_bone_weights(U, T, anchors, int(self.harmonic_k))
        else:
            ip, ix, dd = X.geodesic_table(U, T, sources=anchors)
            bone_dist = X.dense_rows(ip, ix, dd, U.shape[0])
            eucl = np.linalg.norm(U[None, :, :] - bone_pos[:, None, :], axis=2)
            W = X.geodesic_gaussian_weights(bone_dist, float(self.weight_radius) * diag, eucl)
        D = X.example_displacements(W, ex_T, U)
        scale = float(np.linalg.norm(D, axis=2).max()) if D.size else 0.0
        if scale <= 0.0:
            raise ValueError("the examples produce no displacement")
        return _Rig(U=U, diag=diag, W=W, D=D, scale=scale, bone_names=list(bone_names),
                    bone_pos=np.asarray(bone_pos, dtype=float), anchors=anchors,
                    ex_names=ex_names, ex_T=ex_T)

    # ---------------------------------------------------------------- build
    def build(self, node, elastic_modes, ctx):
        from ..solvers import exemplar as X

        depth = int(ctx.get("depth", getattr(node, "depth", 0)))
        if depth == 0:
            self._rig = None          # never reuse a rig from a previous tree build
            self._rig = self.build_rig(node.vertices, node.elements)
        if self._rig is None:
            # The root's plastic step was skipped or failed: building a rig on a
            # fragment would give its siblings rows of a different mesh. Refuse;
            # the tree builder records the error and the node gets no plastic data.
            raise ValueError("exemplar rig missing: the root node has no exemplar data "
                             "(root plastic step skipped or failed)")
        rig = self._rig
        U = np.asarray(node.vertices, dtype=float)
        T = np.asarray(node.elements, dtype=np.int64)
        rows = X.match_rows(rig.U, U, tol=1e-9 * max(rig.diag, 1e-12))
        if rows is None:
            raise ValueError("exemplar: node cage vertices are not a subset of the root "
                             "cage vertices; cannot share the root rig")
        mats = node.materials
        rho = getattr(mats, "rho", None) if mats is not None else None
        sigma_y = getattr(mats, "sigma_y", None) if mats is not None else None
        radius = float(self.propagation_radius) * rig.diag
        nb = X.geodesic_table(U, T, radius=radius)
        return X.ExemplarPlasticModes(
            nodes=U, elements=T, bone_names=rig.bone_names, bone_positions=rig.bone_pos,
            skin_weights=rig.W[rows], example_names=rig.ex_names,
            example_transforms=rig.ex_T, example_displacements=rig.D[:, rows],
            vertex_mass=X.lumped_vertex_mass(U, T, rho), root_vertex_indices=rows,
            reference_diag=rig.diag, example_scale=rig.scale, propagation_radius=radius,
            neighbourhoods=nb, sigma_y=sigma_y, generator_params=dict(self.params),
        )


@register("plastic.response", "exemplar.impulse_blend")
class ImpulseBlend(PlasticResponder):
    VERSION = "1.0.0"
    DESCRIPTION = ("Example-based plastic response (Jones et al. 2016, Eqs. 4, 8, 11-13): "
                   "impulses -> desired displacements -> example-weight changes, "
                   "geodesically smoothed and re-skinned. Saturates at the examples.")
    PARAMS = {
        "impulse_time": Param(1.0e-3, "seconds: per-vertex impulse j = impulse_time * force"),
        "dt": Param(1.0 / 60.0, "Eq. 4 timestep dt (seconds)"),
        "beta_mode": Param("yield", "'yield': beta = (sigma_y/sigma_vm) * max|j| (+ beta); "
                                    "'absolute': beta is an impulse (N s)",
                           choices=("yield", "absolute")),
        "beta": Param(0.0, "Eq. 4 yield threshold (N s); added in 'yield' mode"),
        "alpha": Param(1.0, "Eq. 8 step scale, in units of the largest example displacement"),
        "step": Param("normalized", "Eq. 8 (normalized) or Eq. 10 (natural) step",
                      choices=("normalized", "natural")),
        "gamma": Param(0.08, "Eq. 11 kernel radius, fraction of the root diagonal "
                             "(clamped to the generator's propagation_radius)"),
        "lam": Param(0.5, "Eq. 13 plasticity rate lambda in [0, 1)"),
        "iterations": Param(0, "Eq. 13 steps per hit; 0 = until lambda^n <= settle_tol"),
        "settle_tol": Param(1.0e-3, "pending fraction left after the automatic iterations"),
        "max_step": Param(0.05, "cap of |dx_i| per hit, fraction of the root diagonal"),
        "min_increment": Param(1.0e-4, "a hit that moves no vertex by more than this "
                                       "(fraction of the diagonal) means saturated"),
        "saturation_ratio": Param(0.05, "also saturated when the hit changes the shape by "
                                        "less than this fraction of what the same weight "
                                        "change does to an undeformed piece; 0 = off"),
    }
    PER_VERTEX_STATE_KEYS = ("exemplar_E", "exemplar_dE")

    def setup(self):
        if not 0.0 <= float(self.lam) < 1.0:
            from ..interfaces import PluginError
            raise PluginError(f"{self.NAME}: lam must be in [0, 1), got {self.lam}")

    def weights_for(self, pm, state):
        """Current example weights of a piece (initialised or reconstructed)."""
        from ..solvers import exemplar as X

        K = pm.num_examples
        V = pm.num_vertices
        E = state.get("exemplar_E")
        if E is not None and np.shape(E) == (V, K + 1):
            dE = state.get("exemplar_dE")
            if dE is None or np.shape(dE) != (V, K + 1):
                dE = np.zeros((V, K + 1))
            return np.asarray(E, dtype=float), np.asarray(dE, dtype=float)
        cum = np.asarray(state.get("cum", np.zeros((V, 3))), dtype=float)
        if np.abs(cum).max() > 0.0:
            E = X.reconstruct_weights(cum, pm.example_displacements)
        else:
            E = X.initial_weights(V, K)
        return E, np.zeros((V, K + 1))

    def respond(self, ctx, force, analysis, decision):
        from ..solvers import exemplar as X

        pm, st = ctx.piece.plastic_modes, ctx.plastic_state
        if not hasattr(pm, "example_displacements"):
            # This responder cannot deform a non-exemplar piece; report exhausted
            # so a router fallback (e.g. fracture) still applies.
            return dict(fired=False, reason="not_an_exemplar_bundle", exhausted=True)
        diag = pm.reference_diag
        E, dE_pending = self.weights_for(pm, st)
        D = pm.example_displacements

        # Eq. 4: impulses -> desired displacement, above the yield threshold beta
        j = float(self.impulse_time) * np.asarray(force, dtype=float)
        jmag = np.linalg.norm(j, axis=1)
        beta = float(self.beta)
        if self.beta_mode == "yield" and analysis is not None:
            ratio = float(analysis["sigma_y_local"]) / max(float(analysis["sigma_vm_max"]), 1e-300)
            beta += float(np.clip(ratio, 0.0, 1.0)) * float(jmag.max(initial=0.0))
        dx = X.desired_displacement(j, pm.total_mass, float(self.dt), beta,
                                    max_step=float(self.max_step) * diag)
        dxn = np.linalg.norm(dx, axis=1)
        if not np.any(dxn > 0.0):
            return dict(fired=False, reason="below_beta", beta=beta,
                        max_impulse=float(jmag.max(initial=0.0)))

        # Eq. 8 (or Eq. 10): example-weight step per impacted vertex
        # alpha is in example-amplitude units: Eq. 8 scales by 1/A, Eq. 10
        # (J^T dx has units length^2) by 1/A^2, so both are scale-free.
        A = pm.example_scale
        a_eff = float(self.alpha) / (A if self.step == "normalized" else A * A)
        de = X.weight_step(pm.jacobian(), dx, a_eff, self.step)
        if not np.any(de):
            return dict(fired=False, reason="no_matching_example", exhausted=True,
                        max_dx=float(dxn.max()))

        # Eqs. 11-12: geodesic smoothing of the weight change
        gamma = min(float(self.gamma) * diag, pm.propagation_radius)
        dE_hit = X.propagate(de, pm.nbr_indptr, pm.nbr_indices, pm.nbr_dist, gamma)
        dE = dE_pending + dE_hit

        # Eq. 13 (+ clamp / renormalise after every step)
        n = int(self.iterations) or X.iterations_for(float(self.lam), float(self.settle_tol))
        old = X.blend_displacement(E, D)
        E_new, dE_new = X.apply_schedule(E, dE, float(self.lam), n, clamp=True)
        new = X.blend_displacement(E_new, D)
        change = float(np.linalg.norm(new - old, axis=1).max())
        # Reference: what this hit's weight change does to an undeformed piece.
        # Saturated = (almost) nothing moves, or the shape moves far less than it
        # would on a fresh piece (the examples no longer follow the hit here).
        # Independent of alpha and of how well the examples cover the spot, and
        # never true on a fresh piece.
        fresh_change = 0.0
        if float(self.saturation_ratio) > 0.0:
            E0 = X.initial_weights(pm.num_vertices, pm.num_examples)
            E0_new, _ = X.apply_schedule(E0, dE_hit, float(self.lam), n, clamp=True)
            fresh_change = float(np.linalg.norm(X.blend_displacement(E0_new, D), axis=1).max())
        if (change < float(self.min_increment) * diag
                or change < float(self.saturation_ratio) * fresh_change):
            # Keep the weights and the residual of earlier hits
            # (this hit's own contribution is dropped, it cannot be applied).
            st["exemplar_E"] = E
            st["exemplar_dE"] = dE_pending
            return dict(fired=False, reason="saturated", exhausted=True,
                        max_dx=float(dxn.max()), max_change=change, fresh_change=fresh_change)

        st["exemplar_E"] = E_new
        st["exemplar_dE"] = dE_new
        st["cum"][:] = new
        gain = (E_new - E)[:, 1:].clip(0.0, None).sum(axis=0)
        used = [pm.example_names[k] for k in np.argsort(-gain)[:3] if gain[k] > 0.0]
        return dict(fired=True, reason="dented", max_dx=float(dxn.max()), max_change=change,
                    fresh_change=fresh_change,
                    max_cum=float(np.linalg.norm(new, axis=1).max()), beta=beta,
                    iterations=n, examples_used=used,
                    rest_weight_min=float(E_new[:, 0].min()),
                    pending=float(np.abs(dE_new).max()))
