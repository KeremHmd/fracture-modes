"""Plugin-driven runtime world.

``PluginWorld`` reuses the frozen ``WorldState`` / ``ActiveInstance``
(rigid-body stepping, compound instances, plastic-state-aware rendering) and
replaces only ``apply_impact`` with a pipeline whose every stage is a plugin:

    impact model  ->  static elastic analysis  ->  router
        -> plastic responder   (permanent deformation, may fall back to fracture)
        -> fracture responder  (projection) -> component spawning
    -> post processors

Component spawning (sibling redistribution, plastic-state inheritance through
``parent_vertex_indices``, kick velocities) mirrors the baseline exactly.
Since 0.3.0 a plastic responder may declare extra per-vertex state keys
(``PER_VERTEX_STATE_KEYS``) that are carried the same way as ``cum``
(``modegen/statecarry.py``); responders that declare none are unaffected.
"""

import numpy as np

from fracture_utility.world_state import ActiveInstance, WorldState

from .analysis import static_elastic_analysis
from .events import Decision, ImpactContext, ImpactEvent, ImpactResult
from .profile import Profile
from .registry import get_registry
from .statecarry import copy_state, inject_extras, responder_state_keys, subset_state


class PluginWorld(WorldState):
    def __init__(self, runtime, profile=None):
        w = runtime.world
        super().__init__(gravity=tuple(w["gravity"]), ground_y=w.get("ground_y"),
                         restitution=w.get("restitution", 0.3),
                         impact_kick=w.get("impact_kick", 0.5))
        self.runtime = runtime
        self.profile = profile
        self.poisson_ratio = float(w.get("poisson_ratio", 0.3))
        self.last_result = None
        self.history = []
        self.lift, self.smooth_iterations, self.smooth_weight = getattr(
            runtime, "render", ("nearest", 0, 0.5))

    # ------------------------------------------------------------ spawning (lift)
    def spawn(self, pieces, position=None, orientation=None, velocity=None,
              angular_velocity=None, plastic_states=None):
        """Like ``WorldState.spawn``. With the default nearest lift (and no
        smoothing) it is exactly the frozen code; otherwise the instance lifts
        cage displacement to the fine mesh with ``LiftedInstance``."""
        if self.lift == "nearest" and self.smooth_iterations <= 0:
            return super().spawn(pieces, position=position, orientation=orientation,
                                 velocity=velocity, angular_velocity=angular_velocity,
                                 plastic_states=plastic_states)
        inst = LiftedInstance(pieces, position=position, orientation=orientation,
                              velocity=velocity, angular_velocity=angular_velocity,
                              plastic_states=plastic_states)
        inst.lift = self.lift
        inst.smooth_iterations = self.smooth_iterations
        inst.smooth_weight = self.smooth_weight
        self.active_instances.append(inst)
        return inst

    @classmethod
    def from_profile(cls, profile, registry=None):
        reg = registry or get_registry()
        return cls(profile.build_runtime(reg), profile=profile)

    @classmethod
    def from_tree(cls, tree, runtime_profile=None, registry=None):
        """Runtime from the profile stored in the tree, optionally overridden."""
        from .compat import check_tree
        check_tree(tree, registry=registry or get_registry())
        if runtime_profile is not None:
            prof = runtime_profile
        elif getattr(tree, "modegen_profile", None) is not None:
            prof = Profile.from_dict(tree.modegen_profile)
        else:
            prof = Profile()
        return cls.from_profile(prof, registry)

    @property
    def tool_names(self):
        return sorted(self.runtime.tools)

    # ------------------------------------------------------------ context
    def _make_context(self, event, hit, piece):
        local_pt = np.asarray(hit.world_to_local(event.point), dtype=float)
        local_dir = np.asarray(hit.world_vector_to_local(event.direction), dtype=float)
        n = np.linalg.norm(local_dir)
        local_dir = local_dir / n if n > 1e-12 else np.array([0.0, -1.0, 0.0])
        state = hit.plastic_states.get(id(piece))
        if state is None and getattr(piece, "plastic_modes", None) is not None:
            state = dict(cum=np.zeros((piece.vertices.shape[0], 3)), fired=[])
            hit.plastic_states[id(piece)] = state
        rest = np.asarray(piece.vertices, dtype=float)
        verts = rest + state["cum"] if state is not None else rest
        return ImpactContext(
            world=self, event=event, instance=hit, piece=piece,
            point=local_pt, direction=local_dir, magnitude=float(event.magnitude),
            rest_vertices=rest, vertices=verts, elements=np.asarray(piece.elements),
            materials=piece.materials, plastic_state=state, extra=dict(event.params),
        )

    # ------------------------------------------------------------ impact
    def apply_impact(self, event, verbose=False):
        if not isinstance(event, ImpactEvent):
            raise TypeError("PluginWorld.apply_impact takes an ImpactEvent")
        event.time = self.time
        tool_name, impact_model = self.runtime.tool(event.tool)
        event.tool = tool_name

        hit = self.find_hit_instance(event.point)
        if hit is None:
            return self._finish(ImpactResult("none", "missed", missed=True, tool=tool_name), None, verbose)
        piece = hit.find_hit_piece(event.point)
        if piece is None:
            return self._finish(ImpactResult("none", "no_piece", instances=[hit],
                                             hit_instance=hit, tool=tool_name), None, verbose)

        ctx = self._make_context(event, hit, piece)
        force = np.asarray(impact_model.force_field(ctx), dtype=float)
        if force.shape != ctx.rest_vertices.shape:
            raise ValueError(f"impact model {impact_model.NAME} returned {force.shape}, "
                             f"expected {ctx.rest_vertices.shape}")
        ctx.scratch["force"] = force
        analysis = static_elastic_analysis(piece, force, self.poisson_ratio)
        ctx.scratch["analysis"] = analysis
        decision = self.runtime.router.decide(ctx, analysis)
        result = ImpactResult(decision.branch, decision.reason, decision=decision,
                              analysis=analysis, instances=[hit], hit_instance=hit,
                              hit_piece=piece, tool=tool_name)

        if decision.branch == "plastic":
            result = self._do_plastic(ctx, force, analysis, decision, result)
        elif decision.branch == "fracture":
            result = self._do_fracture(ctx, force, decision, result)
        return self._finish(result, ctx, verbose)

    def _do_plastic(self, ctx, force, analysis, decision, result):
        responder = self.runtime.plastic_response
        if responder is None or ctx.plastic_state is None or analysis is None:
            result.branch, result.reason = "none", "no_plastic_bundle"
            if decision.fallback == "fracture":
                return self._do_fracture(ctx, force, decision, result, promoted=True)
            return result
        pr = responder.respond(ctx, force, analysis, decision)
        result.plastic_result = pr
        if pr.get("fired"):
            result.branch, result.reason = "plastic", pr.get("reason", "yielded")
            return result
        if decision.fallback == "fracture" and pr.get("exhausted"):
            return self._do_fracture(ctx, force, decision, result, promoted=True)
        result.branch, result.reason = "none", f"plastic_no_fire:{pr.get('reason')}"
        return result

    def _do_fracture(self, ctx, force, decision, result, promoted=False):
        piece, hit = ctx.piece, ctx.instance
        label = "promoted_fracture" if promoted else "fracture"
        if self.runtime.fracture_response is None:
            result.branch = "none"
            result.reason = ("final_state" if promoted else "fracture_disabled")
            return result
        if piece.is_atomic or not piece.children or piece.modes_instance is None:
            result.branch, result.reason = "none", "atomic_piece"
            return result
        ok = self.runtime.fracture_response.project(ctx, force, decision)
        if not ok:
            result.branch, result.reason = "none", "fracture_cancelled"
            return result
        new = self._spawn_after_projection(hit, piece)
        if len(new) == 1 and new[0] is hit:
            result.branch, result.reason = "none", "no_separation"
            return result
        result.branch = label
        result.reason = decision.reason if not promoted else "plastic_exhausted"
        result.instances = new
        result.fracture_info = dict(
            n_new_instances=len(new),
            n_pieces=int(getattr(piece.modes_instance, "n_pieces_after_impact", 0)),
        )
        return result

    def _finish(self, result, ctx, verbose):
        if ctx is not None:
            for post in self.runtime.post:
                if result.branch in ("fracture", "promoted_fracture"):
                    post.on_fracture(ctx, result)
                elif result.branch == "plastic":
                    post.on_plastic(ctx, result)
                else:
                    post.on_none(ctx, result)
        info = dict(branch=result.branch, reason=result.reason, analysis=result.analysis,
                    plastic_result=result.plastic_result, tool=result.tool)
        for inst in result.instances or []:
            inst.last_impact_info = info
        self.last_result = result
        self.history.append((self.time, result.tool, result.branch, result.reason))
        if verbose:
            a = result.analysis
            extra = ""
            if a is not None:
                extra = (f" class={a['material_class']} sigma_vm={a['sigma_vm_max']:.3g}"
                         f" sigma_y={a['sigma_y_local']:.3g} sigma_c={a['sigma_c_local']:.3g}")
            print(f"[{result.tool}] {result.branch.upper()} ({result.reason}){extra}")
        return result

    # ------------------------------------------------------------ spawning
    def _spawn_after_projection(self, hit, hit_piece):
        """Group children by post-impact component and spawn one instance each
        (mirrors the frozen WorldState._fire_fracture_branch tail)."""
        modes = hit_piece.modes_instance
        if int(getattr(modes, "n_pieces_after_impact", 1)) <= 1:
            return [hit]
        plai = np.asarray(modes.piece_labels_after_impact, dtype=int)
        comp_to_children = {}
        for idx, child in enumerate(hit_piece.children):
            if idx < len(plai):
                comp_to_children.setdefault(int(plai[idx]), []).append(child)
        if len(comp_to_children) <= 1:
            return [hit]

        def centroid(p):
            v = p.vertices if p.vertices is not None and p.vertices.size else p.fine_vertices
            return None if v is None or v.size == 0 else v.mean(axis=0)

        comp_pieces = {c: list(ch) for c, ch in comp_to_children.items()}
        for sib in [p for p in hit.pieces if p is not hit_piece]:
            s_c = centroid(sib)
            best, best_d = next(iter(comp_pieces)), float("inf")
            if s_c is not None:
                for c, plist in comp_to_children.items():
                    for ch in plist:
                        c_c = centroid(ch)
                        if c_c is not None:
                            d = float(np.linalg.norm(c_c - s_c))
                            if d < best_d:
                                best, best_d = c, d
            comp_pieces[best].append(sib)

        parent_states = getattr(hit, "plastic_states", {}) or {}
        hp_state = parent_states.get(id(hit_piece))
        # Extra per-vertex state keys declared by the plastic responder (0.3.0);
        # () for every pre-0.3 responder, which keeps the old cum/fired-only carry.
        keys = responder_state_keys(self.runtime.plastic_response)
        hp_c = centroid(hit_piece)
        hp_c = np.zeros(3) if hp_c is None else hp_c
        self.remove(hit)
        new_instances = []
        for c, plist in comp_pieces.items():
            cents = [x for x in (centroid(p) for p in plist) if x is not None]
            comp_c = np.mean(cents, axis=0) if cents else hp_c
            kick = comp_c - hp_c
            kn = float(np.linalg.norm(kick))
            kick = kick / kn if kn > 1e-9 else np.zeros(3)
            kick_world = hit.local_vector_to_world(kick) * self.impact_kick

            carry, extras = {}, {}
            for p in plist:
                st = parent_states.get(id(p))
                if st is not None:
                    carry[id(p)], extras[id(p)] = copy_state(st, keys)
                    continue
                if hp_state is None:
                    continue
                pvi = getattr(p, "parent_vertex_indices", None)
                if pvi is None:
                    from scipy.spatial import cKDTree
                    _, pvi = cKDTree(hit_piece.vertices).query(p.vertices)
                    p.parent_vertex_indices = pvi
                sub = subset_state(hp_state, pvi, keys)
                if sub is not None:
                    carry[id(p)], extras[id(p)] = sub

            inst = self.spawn(
                plist, position=hit.position.copy(), orientation=hit.orientation.copy(),
                velocity=hit.velocity + kick_world,
                angular_velocity=hit.angular_velocity.copy(),
                plastic_states=carry or None,
            )
            inject_extras(inst.plastic_states, extras)
            new_instances.append(inst)
        return new_instances


class LiftedInstance(ActiveInstance):
    """``ActiveInstance`` whose fine mesh follows the cage by a configurable
    lift (``runtime.render``): ``barycentric`` (cage tet containing each fine
    vertex) and/or ``smooth_iterations`` of umbrella smoothing on the fine
    mesh. Rendering, hit tests and AABBs all use the lifted vertices."""

    lift = "barycentric"
    smooth_iterations = 0
    smooth_weight = 0.5

    def piece_deformed_fine_vertices(self, piece):
        from . import lift as _lift
        v = getattr(piece, "fine_vertices", None)
        if v is None:
            return None
        state = self.plastic_states.get(id(piece))
        if state is None:
            return v
        cum = state["cum"]
        if self.lift == "barycentric":
            data = _lift.node_barycentric(piece)
            d = _lift.lift_barycentric(cum, piece.elements, *data)
        elif self.lift == "surface":
            d = _lift.lift_corners(cum, *_lift.node_surface(piece))
        else:
            c2f = getattr(piece, "cage_to_fine_map", None)
            if c2f is None or c2f.shape[0] != v.shape[0]:
                c2f = _lift.nearest_map(piece.vertices, v)
                piece.cage_to_fine_map = c2f
            d = cum[c2f]
        if self.smooth_iterations > 0:
            nb = getattr(piece, "_modegen_fine_neighbours", None)
            if nb is None or len(nb[2]) != len(v):
                nb = _lift.uniform_neighbours(piece.fine_triangles, len(v))
                piece._modegen_fine_neighbours = nb
            d = _lift.smooth_displacement(d, piece.fine_triangles, self.smooth_iterations,
                                          self.smooth_weight, neighbours=nb)
        return v + d


# ---------------------------------------------------------------- helpers

def sample_surface_point(world, rng, instance=None):
    """Area-weighted random point on the current (moved, deformed) world
    surface. Returns ``(point, instance)`` or ``(None, None)``."""
    insts = [instance] if instance is not None else list(world.active_instances)
    cands = []
    for inst in insts:
        mesh = inst.world_combined_mesh()
        if mesh is None:
            continue
        V, F = mesh
        a = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
        if a.sum() > 0:
            cands.append((inst, V, F, a))
    if not cands:
        return None, None
    totals = np.array([c[3].sum() for c in cands])
    inst, V, F, a = cands[int(rng.choice(len(cands), p=totals / totals.sum()))]
    f = F[int(rng.choice(len(a), p=a / a.sum()))]
    r1, r2 = rng.random(2)
    s = np.sqrt(r1)
    p = (1 - s) * V[f[0]] + s * (1 - r2) * V[f[1]] + s * r2 * V[f[2]]
    return p, inst


def inward_direction(instance, point):
    """Unit vector from ``point`` toward the instance's surface centroid."""
    mesh = instance.world_combined_mesh()
    c = mesh[0].mean(axis=0) if mesh is not None else instance.position
    d = c - np.asarray(point, dtype=float)
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else np.array([0.0, -1.0, 0.0])
