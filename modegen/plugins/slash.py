"""Slash plugins: elongated blade contacts and grain-directed cut planes.

* ``slash.blade_line``  force distributed along a short blade segment
* ``slash.grain``       face weighting that makes cuts prefer planes
                        perpendicular to a grain normal (wood, layered armour)
"""

import numpy as np

from ..interfaces import FaceWeighting, ImpactModel, Param, distribute_force
from ..registry import register


def _unit(v, fallback):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.asarray(fallback, dtype=float)


@register("impact", "slash.blade_line")
class BladeLine(ImpactModel):
    DESCRIPTION = ("Blade contact: force spread along a segment through the contact "
                   "point (axis from event params, the plugin param, or "
                   "direction x up).")
    PARAMS = {
        "length": Param(0.5, "blade contact length (local units)"),
        "width": Param(0.03, "blade edge width (local units)"),
        "blade_axis": Param(None, "fixed blade axis [x, y, z] (local frame)"),
    }

    def force_field(self, ctx):
        axis = ctx.extra.get("blade_axis", self.blade_axis)
        if axis is None:
            axis = np.cross(ctx.direction, [0.0, 1.0, 0.0])
            if np.linalg.norm(axis) < 1e-6:
                axis = np.cross(ctx.direction, [1.0, 0.0, 0.0])
        axis = _unit(axis, [1.0, 0.0, 0.0])
        half = 0.5 * float(self.length)
        rel = ctx.vertices - ctx.point[None, :]
        t = np.clip(rel @ axis, -half, half)
        d = np.linalg.norm(rel - t[:, None] * axis[None, :], axis=1)
        w = np.exp(-0.5 * (d / float(self.width)) ** 2)
        return distribute_force(w, ctx.direction, ctx.magnitude)


@register("fracture.weights", "slash.grain")
class Grain(FaceWeighting):
    DESCRIPTION = ("Lower toughness for faces whose normal is parallel to the grain "
                   "normal, so cut surfaces prefer planes perpendicular to it.")
    PARAMS = {
        "grain_normal": Param([1.0, 0.0, 0.0], "normal of the preferred cut planes"),
        "strength": Param(0.85, "0 = isotropic, 1 = fully grain-aligned"),
        "floor": Param(0.1, "minimum multiplier"),
    }

    def prepare(self, vertices, elements, materials):
        self._v = np.asarray(vertices, dtype=float)
        self._e = np.asarray(elements)
        self._g = _unit(self.grain_normal, [1.0, 0.0, 0.0])

    def multiplier(self, t1, t2):
        shared = list(set(self._e[t1].tolist()) & set(self._e[t2].tolist()))
        if len(shared) != 3:
            return 1.0
        a, b, c = self._v[shared]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-15:
            return 1.0
        align = abs(float(n @ self._g)) / nn
        return max(float(self.floor), 1.0 - float(self.strength) * align)
