"""Blunt-trauma plugins: wide impacts that dent locally before anything breaks.

* ``bludgeon.blunt_dome``   dome-shaped pressure patch (hammer / club face)
* ``bludgeon.blunt_gate``   router that needs sigma_vm >> sigma_c before breaking
* ``bludgeon.local_dent``   plastic responder that combines the top plastic
  modes and windows them around the contact, so every hit leaves a local,
  inward dent; dents saturate at ``max_depth`` and then the router's fallback
  (fracture) takes over. This intentionally trades strict modal purity for
  art-directable locality.
"""

import numpy as np

from ..events import Decision
from ..interfaces import (ImpactModel, Param, PlasticResponder, Router,
                          bbox_diagonal, distribute_force, smoothstep)
from ..registry import register
from .realistic import plastic_projection


@register("impact", "bludgeon.blunt_dome")
class BluntDome(ImpactModel):
    DESCRIPTION = "Dome-shaped pressure distribution over a wide contact patch."
    PARAMS = {
        "radius": Param(0.15, "patch radius (local units)"),
        "exponent": Param(1.5, "dome sharpness: w = (1 - (d/R)^2)^exponent"),
    }

    def force_field(self, ctx):
        d = np.linalg.norm(ctx.vertices - ctx.point[None, :], axis=1)
        r = float(ctx.extra.get("radius", self.radius))
        x = np.clip(1.0 - (d / r) ** 2, 0.0, None)
        return distribute_force(x ** float(self.exponent), ctx.direction, ctx.magnitude)


@register("router", "bludgeon.blunt_gate")
class BluntGate(Router):
    DESCRIPTION = ("Stress gate that only breaks when sigma_vm exceeds break_factor * "
                   "sigma_c; everything in between dents (then breaks once saturated).")
    PARAMS = {
        "break_factor": Param(1.8, "multiple of sigma_c needed to break outright"),
        "yield_excess": Param(0.0, "extra Pa above sigma_y before denting"),
        "dent_brittle": Param(False, "let brittle tets dent too (cartoon metal)"),
    }

    def decide(self, ctx, a):
        if a is None:
            return Decision("fracture", "no_elastic_analysis")
        if a["material_class"] == "rigid":
            return Decision("none", "rigid_material")
        if a["sigma_vm_max"] > self.break_factor * a["sigma_c_local"]:
            return Decision("fracture", "crushed")
        plastic_ok = a["material_class"] == "elastic-plastic" or self.dent_brittle
        if plastic_ok and a["excess_y"] > self.yield_excess:
            return Decision("plastic", "dented", fallback="fracture")
        if a["excess_c"] > 0.0:
            return Decision("fracture", "sigma_c_exceeded")
        return Decision("none", "below_yield")


@register("plastic.response", "bludgeon.local_dent")
class LocalDent(PlasticResponder):
    DESCRIPTION = "Windowed top-K plastic-mode dent, saturating at max_depth."
    PARAMS = {
        "top_k": Param(6, "number of best-matching plastic modes combined per hit"),
        "radius": Param(0.18, "dent window radius (local units)"),
        "depth": Param(0.03, "dent depth per hit, fraction of the piece diagonal"),
        "max_depth": Param(0.12, "saturation depth, fraction of the piece diagonal"),
        "consume_modes": Param(False, "mark used modes as fired (finite budget)"),
        "scale_with_excess": Param(True, "deeper dents for harder hits"),
    }

    def respond(self, ctx, force, analysis, decision):
        pm, st = ctx.piece.plastic_modes, ctx.plastic_state
        k = pm.modes.shape[1]
        diag = bbox_diagonal(ctx.rest_vertices)
        d = np.linalg.norm(ctx.vertices - ctx.point[None, :], axis=1)
        window = smoothstep(1.0 - d / float(self.radius))
        region = window > 0.5
        cap = float(self.max_depth) * diag
        cum_mag = np.linalg.norm(st["cum"], axis=1)
        if region.any() and cum_mag[region].max() >= 0.98 * cap:
            return dict(fired=False, reason="max_deformation_reached", exhausted=True)

        alpha = plastic_projection(ctx, analysis)
        avail = np.ones(k, dtype=bool)
        if self.consume_modes:
            avail[list(st["fired"])] = False
            if not avail.any():
                return dict(fired=False, reason="all_modes_exhausted", exhausted=True)
        scores = np.where(avail, np.abs(alpha), -1.0)
        top = [int(i) for i in np.argsort(-scores)[: int(self.top_k)] if scores[i] >= 0.0]
        combo = sum(alpha[i] * pm.modes[:, i].reshape(-1, 3) for i in top)
        dent = combo * window[:, None]
        if np.abs(dent).max() < 1e-12:
            dent = window[:, None] * ctx.direction[None, :]
        if float(np.sum(window * (dent @ ctx.direction))) < 0.0:
            dent = -dent
        s = 1.0
        if self.scale_with_excess and analysis is not None:
            s = float(np.clip(analysis["excess_y"] / max(analysis["sigma_y_local"], 1e-12), 0.25, 1.0))
        target = float(self.depth) * diag * s
        dent *= target / max(np.linalg.norm(dent, axis=1).max(), 1e-12)

        new = st["cum"] + dent
        mag = np.linalg.norm(new, axis=1)
        over = mag > cap
        new[over] *= (cap / mag[over])[:, None]
        st["cum"][:] = new
        if self.consume_modes:
            st["fired"].extend(top)
        return dict(fired=True, reason="dented", modes_used=top, dent_depth=target,
                    max_cum=float(np.linalg.norm(new, axis=1).max()),
                    fired_history=list(st["fired"]),
                    modes_remaining=k - len(st["fired"]))
