"""Example user plugin: a ring-shaped impact (e.g. the rim of a pipe or a
hammer whose face is hollow). Use it in a profile as

    "tools": {"ring": {"plugin": "example.ring", "params": {"radius": 0.08}}}

or try it straight away on any modegen tree with ``--add-tool``::

    python tools\\modegen_gui.py --tree out\\trees\\x.modegen.pkl --add-tool ring=example.ring
"""

import numpy as np

from modegen import ImpactModel, Param, register
from modegen.interfaces import distribute_force


@register("impact", "example.ring")
class RingImpact(ImpactModel):
    DESCRIPTION = "Force concentrated on a ring around the contact point."
    PARAMS = {
        "radius": Param(0.08, "ring radius (local units)"),
        "width": Param(0.02, "ring thickness (local units)"),
    }

    def force_field(self, ctx):
        rel = ctx.vertices - ctx.point[None, :]
        s = rel @ ctx.direction
        planar = np.linalg.norm(rel - s[:, None] * ctx.direction[None, :], axis=1)
        w = np.exp(-0.5 * ((planar - self.radius) / self.width) ** 2)
        w *= np.exp(-np.abs(s) / (2.0 * self.radius))
        return distribute_force(w, ctx.direction, ctx.magnitude)
