"""Baseline plugins: the frozen fracture_utility behaviour, expressed as plugins.

The ``realistic`` profile wires these together and reproduces the baseline
pipeline (heterogeneous SOCP fracture modes, sparse plastic modes, stress
gate, Sellan contact projection, one plastic mode per impact).
"""

import numpy as np

from ..events import Decision
from ..interfaces import (
    ElasticGenerator, FaceWeighting, FractureGenerator, FractureResponder,
    ImpactModel, InitVectors, Param, PlasticGenerator, PlasticResponder,
    Router, TetWeighting, bbox_diagonal, distribute_force,
)
from ..registry import register


# ------------------------------------------------------------------ precompute

@register("fracture.generator", "realistic.sellan_socp")
class SellanSOCP(FractureGenerator):
    DESCRIPTION = ("Sellan et al. 2022 iterated SOCP fracture modes with "
                   "heterogeneous G_c, plus face-weight and init-vector hooks.")
    PARAMS = {
        "num_modes": Param(10, "number of fracture modes"),
        "d": Param(3, "mode dimension (3 = vector modes)", choices=(1, 3)),
        "max_iter": Param(10, "SOCP iterations per mode"),
        "tol": Param(1e-4, "per-mode convergence tolerance"),
        "omega": Param(0.01, "discontinuity-matrix scale"),
        "weights": Param({"plugin": "realistic.uniform_faces"},
                         "per-face toughness multiplier hook", plugin="fracture.weights"),
        "init": Param({"plugin": "realistic.laplacian"},
                      "initial-vector hook", plugin="fracture.init"),
    }

    def build(self, node, ctx):
        from fracture_utility.fracture_modes_parameters import fracture_modes_parameters
        from ..solvers.sellan_socp import HookedFractureModes

        k = int(min(self.num_modes, self.d * node.vertices.shape[0] - 2))
        params = fracture_modes_parameters(
            num_modes=k, d=int(self.d), max_iter=int(self.max_iter),
            tol=float(self.tol), omega=float(self.omega),
            verbose=bool(ctx.get("verbose", False)),
        )
        fm = HookedFractureModes(node.vertices, node.elements)
        return fm.compute_modes_hooked(params, node.materials, self.weights, self.init)


@register("fracture.weights", "realistic.uniform_faces")
class UniformFaces(FaceWeighting):
    DESCRIPTION = "No extra weighting: faces cost G_c(face) * area(face)."


@register("fracture.init", "realistic.laplacian")
class LaplacianInit(InitVectors):
    DESCRIPTION = "Default low-frequency Laplacian eigenmodes (baseline)."


@register("elastic.generator", "realistic.linear_elastic")
class LinearElastic(ElasticGenerator):
    DESCRIPTION = "Linear-elastic generalised eigenmodes K phi = w^2 M phi (rigid modes dropped)."
    PARAMS = {
        "num_modes": Param(30, "number of deformation modes"),
        "poisson_ratio": Param(0.3, "global Poisson ratio"),
        "default_E": Param(2.0e11, "Young's modulus when the material has no E"),
        "default_rho": Param(1.0e3, "density when the material has no rho"),
    }

    def build(self, node, ctx):
        from fracture_utility.elastic_modes import ElasticModes

        em = ElasticModes(nodes=node.vertices, elements=node.elements,
                          materials=node.materials,
                          poisson_ratio=float(self.poisson_ratio),
                          default_E=float(self.default_E),
                          default_rho=float(self.default_rho))
        em.assemble()
        k = int(min(self.num_modes, 3 * node.vertices.shape[0] - 7))
        em.compute_modes(num_modes=k, verbose=bool(ctx.get("verbose", False)))
        return em


@register("plastic.generator", "realistic.yield_modes")
class YieldModes(PlasticGenerator):
    DESCRIPTION = ("Sparse plastic modes: minimise sum sigma_y*vol*|eps| inside the "
                   "elastic subspace, with a per-tet weighting hook.")
    PARAMS = {
        "num_modes": Param(16, "number of plastic modes (<= elastic modes)"),
        "max_iter": Param(10, "SOCP iterations per mode"),
        "tol": Param(1e-4, "per-mode convergence tolerance"),
        "weights": Param({"plugin": "realistic.uniform_tets"},
                         "per-tet dissipation multiplier hook", plugin="plastic.weights"),
    }

    def build(self, node, elastic_modes, ctx):
        from ..solvers.weighted_plastic import WeightedPlasticModes

        w = None
        if self.weights is not None:
            w = self.weights.weights(node.vertices, node.elements, node.materials)
        pm = WeightedPlasticModes(elastic_modes, node.materials, tet_weights=w)
        pm.assemble()
        k = int(min(self.num_modes, elastic_modes.modes.shape[1]))
        pm.compute_modes(num_modes=k, max_iter=int(self.max_iter), tol=float(self.tol),
                         verbose=bool(ctx.get("verbose", False)))
        return pm


@register("plastic.weights", "realistic.uniform_tets")
class UniformTets(TetWeighting):
    DESCRIPTION = "No extra weighting: tets cost sigma_y * vol."


# ------------------------------------------------------------------ runtime

@register("impact", "realistic.contact_patch")
class ContactPatch(ImpactModel):
    DESCRIPTION = ("Point contact spread over a small Gaussian patch "
                   "(the baseline nodal-force model).")
    PARAMS = {"spread_radius": Param(0.05, "patch radius (0 = nearest vertex only)")}

    def force_field(self, ctx):
        d = np.linalg.norm(ctx.vertices - ctx.point[None, :], axis=1)
        r = float(self.spread_radius)
        if r <= 0.0:
            w = np.zeros_like(d)
            w[int(np.argmin(d))] = 1.0
        else:
            w = np.exp(-0.5 * (d / (0.5 * r)) ** 2)
            w[d > r] = 0.0
        return distribute_force(w, ctx.direction, ctx.magnitude)


@register("router", "realistic.stress_gate")
class StressGate(Router):
    DESCRIPTION = ("Baseline routing: rigid -> none; sigma_vm > sigma_c -> fracture; "
                   "elastic-plastic & sigma_vm > sigma_y -> plastic (fracture once exhausted).")
    PARAMS = {
        "yield_excess": Param(0.0, "extra Pa above sigma_y before plastic fires"),
        "fracture_excess": Param(0.0, "extra Pa above sigma_c before fracture fires"),
        "promote_on_exhaust": Param(True, "fracture when the plastic basis is exhausted"),
        "no_analysis_branch": Param("fracture", "branch when the piece has no elastic bundle",
                                    choices=("fracture", "none")),
    }

    def decide(self, ctx, a):
        if a is None:
            return Decision(self.no_analysis_branch, "no_elastic_analysis")
        mc = a["material_class"]
        if mc == "rigid":
            return Decision("none", "rigid_material")
        if a["excess_c"] > self.fracture_excess:
            return Decision("fracture", "sigma_c_exceeded")
        if mc != "elastic-plastic":
            return Decision("none", "brittle_below_sigma_c")
        if a["excess_y"] <= self.yield_excess:
            return Decision("none", "below_yield")
        return Decision("plastic", "yielded",
                        fallback="fracture" if self.promote_on_exhaust else None)


@register("fracture.response", "realistic.sellan_projection")
class SellanProjection(FractureResponder):
    DESCRIPTION = ("Sellan projection. mode='contact' is the exact baseline (one-hot "
                   "around the contact point); mode='field' projects the impact "
                   "model's force field instead, so tool shape matters.")
    PARAMS = {
        "threshold": Param(10.0, "piece-separation threshold"),
        "mode": Param("contact", "impact source", choices=("contact", "field")),
        "gain": Param(1.0, "scale of the unit-peak field profile (field mode)"),
        "use_wave": Param(True, "apply the precomputed wave-equation blur (field mode)"),
        "severity_exponent": Param(0.0, "0 = off (baseline). > 0: threshold is divided by "
                                        "(sigma_vm / sigma_c) ** exponent, so harder hits "
                                        "open more cracks"),
        "min_threshold_fraction": Param(0.05, "lower bound of the scaled threshold, as a "
                                              "fraction of 'threshold'"),
    }

    def effective_threshold(self, ctx, decision):
        thr = float(decision.params.get("threshold", self.threshold))
        a = ctx.scratch.get("analysis")
        if self.severity_exponent > 0.0 and a is not None and np.isfinite(a["sigma_c_local"]):
            severity = max(a["sigma_vm_max"] / max(a["sigma_c_local"], 1e-12), 1.0)
            thr = max(thr / severity ** float(self.severity_exponent),
                      thr * float(self.min_threshold_fraction))
        ctx.scratch["effective_threshold"] = thr
        return thr

    def project(self, ctx, force, decision):
        modes = ctx.piece.modes_instance
        thr = self.effective_threshold(ctx, decision)
        if self.mode == "contact":
            dim = modes.modes.shape[0] // modes.elements.shape[0]
            direction = ctx.direction if dim == 3 else np.array([1.0])
            modes.impact_projection(contact_point=ctx.point, threshold=thr,
                                    direction=direction)
            return True
        from ..projection import project_force_field
        project_force_field(modes, force, threshold=thr, gain=float(self.gain),
                            use_wave=bool(self.use_wave))
        return True


def plastic_projection(ctx, analysis):
    """M-projection of the elastic response onto the hit piece's plastic modes."""
    pm = ctx.piece.plastic_modes
    m = pm._lumped_mass_diagonal()
    return pm.modes.T @ (m * analysis["u_flat"])


@register("plastic.response", "realistic.single_mode")
class SingleMode(PlasticResponder):
    DESCRIPTION = ("Baseline: fire the best-matching unfired plastic mode with a fixed "
                   "amplitude; exhausted once every mode has fired.")
    PARAMS = {"amplitude_fraction": Param(0.05, "dent size as a fraction of the piece diagonal")}

    def respond(self, ctx, force, analysis, decision):
        pm, st = ctx.piece.plastic_modes, ctx.plastic_state
        k = pm.modes.shape[1]
        if len(st["fired"]) >= k:
            return dict(fired=False, reason="all_modes_exhausted", exhausted=True)
        alpha = plastic_projection(ctx, analysis)
        masked = alpha.copy()
        masked[list(st["fired"])] = 0.0
        if np.abs(masked).max() < 1e-12:
            return dict(fired=False, reason="no_matching_mode", alpha_scores=alpha)
        w = int(np.argmax(np.abs(masked)))
        sign = float(np.sign(masked[w]))
        mode = pm.modes[:, w].reshape(-1, 3)
        amp = self.amplitude_fraction * bbox_diagonal(ctx.rest_vertices) / max(np.abs(mode).max(), 1e-12)
        st["cum"] += sign * amp * mode
        st["fired"].append(w)
        return dict(fired=True, reason="yielded", winner=w, winner_sign=sign,
                    alpha_scores=alpha, mode_amplitude=amp,
                    fired_history=list(st["fired"]), modes_remaining=k - len(st["fired"]))
