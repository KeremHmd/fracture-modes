"""Crush plugins: wide blunt impacts that only ever dent, with a hard,
algorithmic deformation limit. Everything stays inside the precomputed
plastic-mode basis, so the runtime never remeshes or re-solves anything.

* ``crush.ram``         wide, flat-faced ram (large contact radius, soft rim)
* ``crush.dent_only``   router that never chooses fracture
* ``crush.modal_dent``  purely modal plastic response with a ductility limit

Modal state
-----------
A piece's permanent deformation is stored as one coefficient per plastic
mode, ``alpha`` (K numbers). The displayed shape is ``Phi @ alpha`` where
``Phi`` is the precomputed (3V x K, M-orthonormal) plastic-mode matrix. A hit
computes a desired local dent, takes its M-projection onto the basis
(``dalpha = Phi^T M target``, the best band-limited approximation) and
updates ``alpha``. Topology never changes; each hit costs two K x 3V
matrix-vector products.

Deformation limit (all three act together)
------------------------------------------
1. Per-mode capacity: ``|alpha_k| <= c_k`` with
   ``c_k = ultimate_strain / max_tet_strain(mode k)``, i.e. no mode may be
   stretched past the material's ultimate strain.
2. Plastic-work budget: ``W(alpha) = sum_k e_k |alpha_k| <= work_budget *
   sum_k e_k c_k`` where ``e_k`` is the precomputed dissipation of mode k
   (the plastic-modes objective value). This is the piece's ductility.
3. Hardening: every increment is scaled by ``(1 - W / W_max) ** hardening``,
   so successive hits have diminishing effect and the shape converges.

When the budget is spent (or an increment would be negligible) the response
reports ``exhausted`` and the piece has reached its final state.
"""

import numpy as np

from ..events import Decision
from ..interfaces import (ImpactModel, Param, PlasticResponder, Router,
                          bbox_diagonal, distribute_force, smoothstep)
from ..registry import register


@register("impact", "crush.ram")
class Ram(ImpactModel):
    DESCRIPTION = "Wide flat-faced ram: large circular contact with a soft rim."
    PARAMS = {
        "radius": Param(0.3, "ram face radius (local units)"),
        "rim": Param(0.08, "width of the soft rim (local units)"),
        "reach": Param(0.12, "how far into the body the face presses (local units)"),
    }

    def force_field(self, ctx):
        rel = ctx.vertices - ctx.point[None, :]
        s = rel @ ctx.direction
        planar = np.linalg.norm(rel - s[:, None] * ctx.direction[None, :], axis=1)
        r = float(ctx.extra.get("radius", self.radius))
        face = smoothstep((r - planar) / max(float(self.rim), 1e-9))
        depth = np.exp(-np.clip(s, 0.0, None) / float(self.reach))
        depth[s < -float(self.reach)] = 0.0
        return distribute_force(face * depth, ctx.direction, ctx.magnitude)


@register("router", "crush.dent_only")
class DentOnly(Router):
    DESCRIPTION = "Never fractures: plastic when sigma_y is exceeded, otherwise nothing."
    PARAMS = {
        "yield_excess": Param(0.0, "extra Pa above sigma_y before denting"),
        "dent_brittle": Param(True, "brittle-class tets dent too (no fracture exists)"),
    }

    def decide(self, ctx, a):
        if a is None:
            return Decision("none", "no_elastic_analysis")
        mc = a["material_class"]
        if mc == "rigid":
            return Decision("none", "rigid_material")
        if mc == "brittle" and not self.dent_brittle:
            return Decision("none", "brittle_no_dent")
        if a["excess_y"] <= self.yield_excess:
            return Decision("none", "below_yield")
        return Decision("plastic", "crushed")


@register("plastic.response", "crush.modal_dent")
class ModalDent(PlasticResponder):
    DESCRIPTION = ("Purely modal dent (projection of a local target onto the plastic "
                   "basis) with per-mode strain caps, a plastic-work budget and "
                   "hardening; converges to a final state.")
    PARAMS = {
        "window_radius": Param(0.3, "radius of the desired dent (local units)"),
        "depth": Param(0.04, "desired dent depth per full-strength hit, fraction of diagonal"),
        "scale_with_excess": Param(True, "weaker hits dent less"),
        "ultimate_strain": Param(0.05, "per-mode cap: max strain any mode may reach"),
        "work_budget": Param(0.35, "usable fraction of the total plastic capacity (ductility)"),
        "hardening": Param(2.0, "increment scale = (1 - W/W_max) ** hardening"),
        "min_increment": Param(1e-4, "increments below this (fraction of diagonal) count as final"),
    }

    def setup(self):
        self._cache = {}

    def _limits(self, pm):
        key = id(pm)
        if key not in self._cache:
            K = pm.modes.shape[1]
            peak = np.array([float(pm.mode_strain_magnitude(k).max()) for k in range(K)])
            caps = float(self.ultimate_strain) / np.maximum(peak, 1e-12)
            energies = np.abs(np.asarray(pm.mode_energies, dtype=float))
            self._cache[key] = (caps, energies, float(energies @ caps))
        return self._cache[key]

    def respond(self, ctx, force, analysis, decision):
        pm, st = ctx.piece.plastic_modes, ctx.plastic_state
        Phi = pm.modes
        V = ctx.rest_vertices.shape[0]
        m = pm._lumped_mass_diagonal()
        caps, energies, capacity = self._limits(pm)
        w_max = float(self.work_budget) * capacity

        alpha = st.get("alpha")
        if alpha is None or len(alpha) != Phi.shape[1]:
            alpha = Phi.T @ (m * st["cum"].reshape(-1))
        work = float(energies @ np.abs(alpha))
        if work >= w_max * (1.0 - 1e-6):
            return dict(fired=False, reason="final_state", exhausted=True,
                        work_used=work / w_max)

        harden = (1.0 - work / w_max) ** float(self.hardening)
        diag = bbox_diagonal(ctx.rest_vertices)
        d = np.linalg.norm(ctx.vertices - ctx.point[None, :], axis=1)
        window = smoothstep(1.0 - d / float(self.window_radius))
        shape = analysis["u_flat"].reshape(V, 3) * window[:, None]
        if float(np.sum(window * (shape @ ctx.direction))) < 0.0:
            shape = -shape
        if np.abs(shape).max() < 1e-15:
            shape = window[:, None] * ctx.direction[None, :]
        shape /= max(np.linalg.norm(shape, axis=1).max(), 1e-15)
        s = 1.0
        if self.scale_with_excess:
            s = float(np.clip(analysis["excess_y"] / max(analysis["sigma_y_local"], 1e-12), 0.25, 1.0))
        target = float(self.depth) * diag * s * harden * shape

        dalpha = Phi.T @ (m * target.reshape(-1))
        new = np.clip(alpha + dalpha, -caps, caps)
        if float(energies @ np.abs(new)) > w_max:
            lo, hi = 0.0, 1.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                trial = np.clip(alpha + mid * dalpha, -caps, caps)
                if float(energies @ np.abs(trial)) <= w_max:
                    lo = mid
                else:
                    hi = mid
            new = np.clip(alpha + lo * dalpha, -caps, caps)

        change = (Phi @ (new - alpha)).reshape(V, 3)
        moved = float(np.linalg.norm(change, axis=1).max())
        if moved < float(self.min_increment) * diag:
            st["alpha"] = alpha
            return dict(fired=False, reason="final_state", exhausted=True,
                        work_used=float(energies @ np.abs(alpha)) / w_max)

        st["alpha"] = new
        st["cum"][:] = (Phi @ new).reshape(V, 3)
        new_work = float(energies @ np.abs(new))
        return dict(fired=True, reason="dented", work_used=new_work / w_max,
                    hardening_factor=harden, dent_depth=moved,
                    capped_modes=int(np.sum(np.isclose(np.abs(new), caps))))
