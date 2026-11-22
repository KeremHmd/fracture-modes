"""Artistic / exaggerated plugins for game-style destruction.

* ``artistic.voronoi_shards``  non-SOCP fracture generator: Voronoi cells as
                               precomputed pieces (fast, many shards)
* ``artistic.trigger_happy``   router that breaks at a fraction of sigma_c
* ``artistic.explosive_kick``  post-processor that throws fragments outward
                               and spins them
"""

import numpy as np

from ..events import Decision
from ..interfaces import FractureGenerator, Param, PostProcessor, Router
from ..registry import register


@register("fracture.generator", "artistic.voronoi_shards")
class VoronoiShards(FractureGenerator):
    DESCRIPTION = ("Voronoi-cell pieces (farthest-point seeds, optional bias toward "
                   "low-G_c regions). No SOCP, no MOSEK.")
    PARAMS = {
        "num_cells": Param(24, "number of Voronoi cells per node"),
        "d": Param(3, "mode dimension", choices=(1, 3)),
        "seed": Param(0, "random seed (offset by node depth)"),
        "toughness_bias": Param(0.0, "> 0 puts more seeds where G_c is low"),
    }

    def build(self, node, ctx):
        from ..solvers.voronoi import VoronoiFractureModes

        T = node.elements.shape[0]
        fm = VoronoiFractureModes(node.vertices, node.elements)
        return fm.compute_modes_voronoi(
            num_cells=int(min(self.num_cells, T)), d=int(self.d),
            seed=int(self.seed) + int(ctx.get("depth", 0)),
            toughness_bias=float(self.toughness_bias),
            materials=node.materials, verbose=bool(ctx.get("verbose", False)),
        )


@register("router", "artistic.trigger_happy")
class TriggerHappy(Router):
    DESCRIPTION = "Breaks at fracture_fraction * sigma_c (or * sigma_y if no sigma_c)."
    PARAMS = {
        "fracture_fraction": Param(0.3, "fraction of sigma_c that already breaks"),
        "plastic_enabled": Param(True, "allow plastic dents below the break level"),
    }

    def decide(self, ctx, a):
        if a is None:
            return Decision("fracture", "no_elastic_analysis")
        if a["material_class"] == "rigid":
            return Decision("none", "rigid_material")
        ref = a["sigma_c_local"] if np.isfinite(a["sigma_c_local"]) else a["sigma_y_local"]
        if a["sigma_vm_max"] > self.fracture_fraction * ref:
            return Decision("fracture", "trigger_happy")
        if self.plastic_enabled and a["material_class"] == "elastic-plastic" and a["excess_y"] > 0.0:
            return Decision("plastic", "yielded", fallback="fracture")
        return Decision("none", "below_break_level")


@register("post", "artistic.explosive_kick")
class ExplosiveKick(PostProcessor):
    DESCRIPTION = "Throws new fragments away from the impact point and spins them."
    PARAMS = {
        "speed": Param(2.5, "radial launch speed"),
        "upward": Param(0.5, "extra upward component (relative to speed)"),
        "along_impact": Param(0.3, "extra push along the impact direction"),
        "spin": Param(8.0, "random angular velocity scale (rad/s)"),
        "seed": Param(0, "random seed"),
    }

    def setup(self):
        self._rng = np.random.default_rng(int(self.seed))

    def on_fracture(self, ctx, result):
        p = np.asarray(ctx.event.point, dtype=float)
        dirw = np.asarray(ctx.event.direction, dtype=float)
        dirw = dirw / max(np.linalg.norm(dirw), 1e-12)
        for inst in result.instances:
            lo, hi = inst.world_aabb()
            radial = 0.5 * (lo + hi) - p
            n = np.linalg.norm(radial)
            radial = radial / n if n > 1e-9 else self._rng.normal(size=3)
            inst.velocity = inst.velocity + self.speed * (
                radial + self.upward * np.array([0.0, 1.0, 0.0]) + self.along_impact * dirw)
            inst.angular_velocity = inst.angular_velocity + self._rng.normal(size=3) * self.spin
