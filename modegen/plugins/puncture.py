"""Puncture plugins: narrow, deep impacts that punch out local fragments.

* ``puncture.spike``        force concentrated along a thin channel into the body
* ``puncture.local_seeds``  SOCP init vectors = localised bumps, so the
                            precomputed fracture modes (and pieces) are small
                            and local instead of global low-frequency cuts
* ``puncture.local_hole``   projects the spike, then only lets components that
                            intersect the penetration channel detach
"""

import numpy as np

from ..interfaces import (FractureResponder, ImpactModel, InitVectors, Param,
                          bbox_diagonal, distribute_force)
from ..registry import register


def _farthest_points(points, k, seed=0):
    rng = np.random.default_rng(seed)
    n = points.shape[0]
    k = max(1, min(int(k), n))
    idx = [int(rng.integers(n))]
    d = np.linalg.norm(points - points[idx[0]], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(d))
        idx.append(nxt)
        d = np.minimum(d, np.linalg.norm(points - points[nxt], axis=1))
    return np.array(idx)


@register("impact", "puncture.spike")
class Spike(ImpactModel):
    DESCRIPTION = "Thin, deep penetration channel along the impact direction."
    PARAMS = {
        "radius": Param(0.02, "channel radius (local units)"),
        "depth": Param(0.35, "e-folding penetration depth (local units)"),
    }

    def force_field(self, ctx):
        rel = ctx.vertices - ctx.point[None, :]
        s = rel @ ctx.direction
        perp = np.linalg.norm(rel - s[:, None] * ctx.direction[None, :], axis=1)
        r = float(ctx.extra.get("radius", self.radius))
        w = np.exp(-0.5 * (perp / r) ** 2) * np.exp(-np.clip(s, 0.0, None) / float(self.depth))
        w[s < -2.0 * r] = 0.0
        return distribute_force(w, ctx.direction, ctx.magnitude)


@register("fracture.init", "puncture.local_seeds")
class LocalSeeds(InitVectors):
    DESCRIPTION = ("Localised Gaussian bumps at farthest-point seeds as SOCP init "
                   "vectors -> smaller, local fracture pieces.")
    PARAMS = {
        "sigma": Param(0.12, "bump width, fraction of the node diagonal"),
        "seed": Param(0, "random seed for the first seed point"),
    }

    def vectors(self, vertices, elements, num_modes, d):
        V = vertices.shape[0]
        sig = float(self.sigma) * bbox_diagonal(vertices)
        seeds = _farthest_points(vertices, num_modes, int(self.seed))
        out = np.zeros((d * V, num_modes))
        for i in range(num_modes):
            s = vertices[seeds[i % len(seeds)]]
            bump = np.exp(-0.5 * (np.linalg.norm(vertices - s, axis=1) / sig) ** 2)
            comp = i % d
            out[comp * V:(comp + 1) * V, i] = bump
        return out


@register("fracture.response", "puncture.local_hole")
class LocalHole(FractureResponder):
    DESCRIPTION = ("Field projection, then only components touching the penetration "
                   "channel may detach; the rest stay attached to the body.")
    PARAMS = {
        "threshold": Param(6.0, "piece-separation threshold"),
        "gain": Param(1.0, "scale of the unit-peak spike profile"),
        "channel_radius": Param(0.1, "radius of the region allowed to detach"),
        "channel_depth": Param(0.5, "depth of the region allowed to detach"),
        "use_wave": Param(True, "apply the precomputed wave-equation blur"),
    }

    def project(self, ctx, force, decision):
        from ..projection import piece_centroids, project_force_field, restrict_separation

        modes = ctx.piece.modes_instance
        project_force_field(modes, force, threshold=float(self.threshold),
                            gain=float(self.gain), use_wave=bool(self.use_wave))
        c = piece_centroids(modes)
        rel = c - ctx.point[None, :]
        s = rel @ ctx.direction
        perp = np.linalg.norm(rel - s[:, None] * ctx.direction[None, :], axis=1)
        rr = float(self.channel_radius)
        inside = (s > -rr) & (s < float(self.channel_depth)) & (perp < rr)
        restrict_separation(modes, inside)
        return True
