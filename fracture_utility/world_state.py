"""Runtime world state for multi-impact fracture demos"""

import numpy as np


class ActiveInstance:
    """A single rigid body in the world, made of one or more pieces"""

    def __init__(self, pieces, position=None, orientation=None,
                 velocity=None, angular_velocity=None,
                 plastic_states=None):
        # Accept either a single FractureNode or a list.
        if isinstance(pieces, (list, tuple)):
            self.pieces = list(pieces)
        else:
            self.pieces = [pieces]
        if not self.pieces:
            raise ValueError("ActiveInstance requires at least one piece")
        # First piece used for legacy ``.node`` access.
        self.node = self.pieces[0]
        self.position = (
            np.zeros(3) if position is None
            else np.asarray(position, dtype=float).copy()
        )
        self.orientation = (
            np.eye(3) if orientation is None
            else np.asarray(orientation, dtype=float).copy()
        )
        self.velocity = (
            np.zeros(3) if velocity is None
            else np.asarray(velocity, dtype=float).copy()
        )
        self.angular_velocity = (
            np.zeros(3) if angular_velocity is None
            else np.asarray(angular_velocity, dtype=float).copy()
        )

        # Per-piece plastic runtime state. Keyed by id(piece) so multiple
        # instances sharing the same FractureNode still get their own state.
        # A piece without plastic_modes gets no entry (rendered undeformed).
        self.plastic_states = {}
        if plastic_states is not None:
            for pid, state in plastic_states.items():
                self.plastic_states[pid] = dict(
                    cum=np.asarray(state["cum"], dtype=float).copy(),
                    fired=list(state.get("fired", [])),
                )
        for p in self.pieces:
            if getattr(p, "plastic_modes", None) is None:
                continue
            if id(p) in self.plastic_states:
                continue
            self.plastic_states[id(p)] = dict(
                cum=np.zeros((p.vertices.shape[0], 3), dtype=float),
                fired=[],
            )

    def piece_plastic_state(self, piece):
        return self.plastic_states.get(id(piece))

    def piece_deformed_fine_vertices(self, piece):
        """Rest-space fine vertices + lifted cumulative plastic displacement."""
        v = getattr(piece, "fine_vertices", None)
        if v is None:
            return None
        state = self.plastic_states.get(id(piece))
        if state is None:
            return v
        c2f = getattr(piece, "cage_to_fine_map", None)
        # Rebuild the cage-to-fine map on the fly if it's absent or was built
        # against a mismatched vertex set (older tree pickles from before the
        # nearest-cage-map fix).
        if c2f is None or c2f.shape[0] != v.shape[0]:
            from scipy.spatial import cKDTree as _cKDTree
            _, c2f = _cKDTree(piece.vertices).query(v)
            piece.cage_to_fine_map = c2f
        return v + state["cum"][c2f]


    def world_to_local(self, point):
        return self.orientation.T @ (np.asarray(point, dtype=float) - self.position)

    def local_to_world(self, point):
        return self.orientation @ np.asarray(point, dtype=float) + self.position

    def world_vector_to_local(self, vector):
        return self.orientation.T @ np.asarray(vector, dtype=float)

    def local_vector_to_world(self, vector):
        return self.orientation @ np.asarray(vector, dtype=float)

    def world_fine_vertices(self):
        """Return concatenated world-space fine vertices from every piece (deformed)."""
        parts = []
        for p in self.pieces:
            v = self.piece_deformed_fine_vertices(p)
            if v is None or v.size == 0:
                continue
            parts.append((self.orientation @ v.T).T + self.position)
        if not parts:
            return None
        return np.vstack(parts)

    def world_fine_triangles_offsets(self):
        """Yield ``(piece, vertex_offset, triangle_offset)`` per piece"""
        v_off = 0
        f_off = 0
        for p in self.pieces:
            v = self.piece_deformed_fine_vertices(p)
            f = getattr(p, "fine_triangles", None)
            if v is None or v.size == 0 or f is None or f.size == 0:
                continue
            yield p, v_off, f_off
            v_off += int(v.shape[0])
            f_off += int(f.shape[0])

    def world_combined_mesh(self):
        """Concatenated world-space fine mesh of every piece in this instance"""
        V_parts = []
        F_parts = []
        v_off = 0
        for p in self.pieces:
            v = self.piece_deformed_fine_vertices(p)
            f = getattr(p, "fine_triangles", None)
            if v is None or v.size == 0 or f is None or f.size == 0:
                continue
            V_parts.append((self.orientation @ v.T).T + self.position)
            F_parts.append(np.asarray(f) + v_off)
            v_off += int(v.shape[0])
        if not V_parts:
            return None
        return np.vstack(V_parts), np.vstack(F_parts)

    def _piece_world_aabb(self, p):
        v = self.piece_deformed_fine_vertices(p)
        if v is None or v.size == 0:
            return None
        wv = (self.orientation @ v.T).T + self.position
        return wv.min(axis=0), wv.max(axis=0)

    def world_aabb(self):
        """Union of all pieces' AABBs in world space."""
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        any_piece = False
        for p in self.pieces:
            box = self._piece_world_aabb(p)
            if box is None:
                continue
            any_piece = True
            lo = np.minimum(lo, box[0])
            hi = np.maximum(hi, box[1])
        if not any_piece:
            return self.position - 0.5, self.position + 0.5
        return lo, hi

    def contains_world_point(self, world_point):
        """True if any piece's AABB contains the point."""
        p = np.asarray(world_point, dtype=float)
        for piece in self.pieces:
            box = self._piece_world_aabb(piece)
            if box is None:
                continue
            lo, hi = box
            if np.all(p >= lo) and np.all(p <= hi):
                return True
        return False

    def find_hit_piece(self, world_point):
        """Return the piece in ``self.pieces`` whose AABB contains the point"""
        p = np.asarray(world_point, dtype=float)
        best_piece = None
        best_dist = float("inf")
        for piece in self.pieces:
            box = self._piece_world_aabb(piece)
            if box is None:
                continue
            lo, hi = box
            if not (np.all(p >= lo) and np.all(p <= hi)):
                continue
            v = self.piece_deformed_fine_vertices(piece)
            if v is None:
                continue
            wv = (self.orientation @ v.T).T + self.position
            centroid = wv.mean(axis=0)
            d = float(np.linalg.norm(p - centroid))
            if d < best_dist:
                best_dist = d
                best_piece = piece
        return best_piece


class WorldState:
    """A collection of active fracture instances with simple dynamics."""

    def __init__(self, gravity=(0.0, -9.81, 0.0), ground_y=None,
                 restitution=0.3, impact_kick=0.5):
        self.active_instances = []
        self.gravity = np.asarray(gravity, dtype=float)
        self.ground_y = (None if ground_y is None else float(ground_y))
        self.restitution = float(restitution)
        self.impact_kick = float(impact_kick)
        self.time = 0.0


    def spawn(self, pieces, position=None, orientation=None,
              velocity=None, angular_velocity=None,
              plastic_states=None):
        """Spawn a new active instance"""
        instance = ActiveInstance(
            pieces, position=position, orientation=orientation,
            velocity=velocity, angular_velocity=angular_velocity,
            plastic_states=plastic_states,
        )
        self.active_instances.append(instance)
        return instance

    def remove(self, instance):
        if instance in self.active_instances:
            self.active_instances.remove(instance)


    def step(self, dt):
        """Advance all active instances by dt under gravity + ground."""
        for instance in self.active_instances:
            instance.velocity = instance.velocity + self.gravity * dt
            instance.position = instance.position + instance.velocity * dt
            if self.ground_y is not None:
                lo, _ = instance.world_aabb()
                if lo[1] < self.ground_y:
                    instance.position[1] += (self.ground_y - lo[1])
                    if instance.velocity[1] < 0.0:
                        instance.velocity[1] = -instance.velocity[1] * self.restitution
                    instance.velocity[0] *= (1.0 - 0.1)
                    instance.velocity[2] *= (1.0 - 0.1)
            omega = instance.angular_velocity
            wnorm = float(np.linalg.norm(omega))
            if wnorm > 1e-9:
                axis = omega / wnorm
                theta = wnorm * dt
                K = np.array([
                    [0.0, -axis[2], axis[1]],
                    [axis[2], 0.0, -axis[0]],
                    [-axis[1], axis[0], 0.0],
                ])
                R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
                instance.orientation = R @ instance.orientation
        self.time += dt

    def find_hit_instance(self, world_point):
        """Return the first active instance containing the world point, or None."""
        for instance in self.active_instances:
            if instance.contains_world_point(world_point):
                return instance
        return None

    def apply_impact(self, world_point, world_direction, threshold=10.0,
                     magnitude=1.0e6, spread_radius=0.05,
                     poisson_ratio=0.3,
                     yield_excess_threshold=0.0,
                     fracture_excess_threshold=0.0,
                     mode_amplitude=None,
                     verbose=False):
        """Route an impact via plastic / fracture per material class + thresholds.

        When the hit piece has a plastic bundle (``piece.plastic_modes``) and a
        material with ``sigma_y``, the impact is analysed via a static elastic
        response and routed as:

          - ``rigid`` peak-tet or nothing yielded  -> no response
          - ``sigma_vm > sigma_c`` at peak         -> FRACTURE branch (existing)
          - ``elastic-plastic`` + ``sigma_vm > sigma_y`` + modes remain
                                                    -> PLASTIC branch (fires one mode)
          - plastic modes exhausted, still yielding -> promoted to FRACTURE

        The last return-value entry (extra dict) reports what happened; the
        first entries are the list of active instances after the routing (for
        backwards compatibility with the fracture-only pipeline).
        """
        hit = self.find_hit_instance(world_point)
        if hit is None:
            if verbose:
                print(f"Impact at {world_point} missed all instances")
            return []

        hit_piece = hit.find_hit_piece(world_point)
        if hit_piece is None:
            return [hit]

        local_pt = hit.world_to_local(world_point)
        local_dir = hit.world_vector_to_local(
            np.asarray(world_direction, dtype=float)
        )

        # --- Routing decision (only if the piece has a plastic bundle) ---
        analysis = None
        if hit_piece.plastic_modes is not None:
            analysis = hit_piece.analyze_impact(
                contact_local=local_pt, direction_local=local_dir,
                magnitude=magnitude, spread_radius=spread_radius,
                poisson_ratio=poisson_ratio,
            )

        if analysis is not None:
            mc = analysis["material_class"]
            if mc == "rigid":
                if verbose:
                    print(f"Impact on rigid piece (depth {hit_piece.depth}) — no response")
                setattr(hit, "last_impact_info", dict(branch="none",
                        reason="rigid_material", analysis=analysis))
                return [hit]

            if analysis["excess_c"] > fracture_excess_threshold:
                if verbose:
                    print(f"Impact triggers FRACTURE branch "
                          f"(sigma_vm={analysis['sigma_vm_max']:.3g} > "
                          f"sigma_c={analysis['sigma_c_local']:.3g})")
                return self._fire_fracture_branch(
                    hit, hit_piece, local_pt, local_dir, threshold,
                    verbose=verbose,
                    extra_info=dict(branch="fracture", reason="sigma_c_exceeded",
                                    analysis=analysis),
                )

            if mc != "elastic-plastic":
                if verbose:
                    print(f"Brittle piece below sigma_c — no response")
                setattr(hit, "last_impact_info", dict(branch="none",
                        reason="brittle_below_sigma_c", analysis=analysis))
                return [hit]

            if analysis["excess_y"] <= yield_excess_threshold:
                if verbose:
                    print(f"Impact below yield — no response")
                setattr(hit, "last_impact_info", dict(branch="none",
                        reason="below_yield", analysis=analysis))
                return [hit]

            # Plastic branch
            fired, pm_result = _fire_plastic_on_piece(
                hit, hit_piece,
                local_pt, local_dir, magnitude,
                spread_radius=spread_radius,
                poisson_ratio=poisson_ratio,
                yield_excess_threshold=yield_excess_threshold,
                mode_amplitude=mode_amplitude,
            )
            if fired:
                if verbose:
                    print(f"Impact triggers PLASTIC branch on depth-{hit_piece.depth} "
                          f"piece: fired mode {pm_result['winner']} "
                          f"(history={pm_result['fired_history']}, "
                          f"remaining={pm_result['modes_remaining']})")
                setattr(hit, "last_impact_info", dict(branch="plastic",
                        reason="yielded", analysis=analysis,
                        plastic_result=pm_result))
                return [hit]

            if pm_result["reason"] == "all_modes_exhausted":
                if verbose:
                    print("Plastic modes exhausted — promoting to FRACTURE")
                return self._fire_fracture_branch(
                    hit, hit_piece, local_pt, local_dir, threshold,
                    verbose=verbose,
                    extra_info=dict(branch="promoted_fracture",
                                    reason="plastic_exhausted",
                                    analysis=analysis,
                                    plastic_result=pm_result),
                )

            if verbose:
                print(f"Plastic no-fire ({pm_result['reason']}) — no response")
            setattr(hit, "last_impact_info", dict(branch="none",
                    reason=f"plastic_no_fire:{pm_result['reason']}",
                    analysis=analysis, plastic_result=pm_result))
            return [hit]

        # --- Fracture-only fallback (no plastic bundle on this piece) ---
        return self._fire_fracture_branch(
            hit, hit_piece, local_pt, local_dir, threshold,
            verbose=verbose,
            extra_info=dict(branch="fracture", reason="no_plastic_bundle"),
        )

    def _fire_fracture_branch(self, hit, hit_piece, local_pt, local_dir,
                              threshold, verbose=False, extra_info=None):
        """Run the fracture pipeline on the hit piece and spawn new instances.

        Returns the list of instances that now represent the impact site.
        On no-fracture (atomic piece, single component after projection, etc.)
        the hit instance is returned unchanged.
        """
        if (hit_piece.is_atomic
                or not hit_piece.children
                or hit_piece.modes_instance is None):
            if verbose:
                print(
                    f"Impact on atomic piece (depth {hit_piece.depth}) — "
                    f"no further fracture"
                )
            info = dict(extra_info or {})
            info["branch"] = "none"
            info["reason"] = info.get("reason", "atomic_piece")
            setattr(hit, "last_impact_info", info)
            return [hit]

        modes = hit_piece.modes_instance
        try:
            modes.impact_projection(
                contact_point=np.asarray(local_pt, dtype=float),
                direction=np.asarray(local_dir, dtype=float),
                threshold=threshold,
            )
        except Exception as e:
            print(f"Impact projection failed: {e}")
            return [hit]

        n_pieces_after = int(getattr(modes, "n_pieces_after_impact", 1))
        if n_pieces_after <= 1:
            if verbose:
                print(
                    f"Impact projected on piece at depth {hit_piece.depth}; "
                    f"result is one connected piece — no separation"
                )
            info = dict(extra_info or {})
            info["branch"] = "none"
            info["reason"] = info.get("reason", "no_separation")
            setattr(hit, "last_impact_info", info)
            return [hit]

        plai = np.asarray(modes.piece_labels_after_impact, dtype=int)

        comp_to_children = {}
        for child_idx, child_node in enumerate(hit_piece.children):
            if child_idx >= len(plai):
                continue
            comp = int(plai[child_idx])
            comp_to_children.setdefault(comp, []).append(child_node)

        if len(comp_to_children) <= 1:
            if verbose:
                print("Single component — no actual fracture")
            info = dict(extra_info or {})
            info["branch"] = "none"
            info["reason"] = info.get("reason", "single_component")
            setattr(hit, "last_impact_info", info)
            return [hit]

        siblings = [p for p in hit.pieces if p is not hit_piece]
        comp_pieces = {
            comp: list(children) for comp, children in comp_to_children.items()
        }

        def _centroid(piece):
            v = piece.vertices
            if v is None or v.size == 0:
                v = piece.fine_vertices
            if v is None or v.size == 0:
                return None
            return v.mean(axis=0)

        for sib in siblings:
            s_c = _centroid(sib)
            if s_c is None:
                first_comp = next(iter(comp_pieces))
                comp_pieces[first_comp].append(sib)
                continue
            best_comp = None
            best_dist = float("inf")
            for comp, plist in comp_to_children.items():
                for ch in plist:
                    c_c = _centroid(ch)
                    if c_c is None:
                        continue
                    d = float(np.linalg.norm(c_c - s_c))
                    if d < best_dist:
                        best_dist = d
                        best_comp = comp
            if best_comp is None:
                best_comp = next(iter(comp_pieces))
            comp_pieces[best_comp].append(sib)

        self.remove(hit)
        hit_piece_centroid = _centroid(hit_piece)
        if hit_piece_centroid is None:
            hit_piece_centroid = np.zeros(3)
        new_instances = []

        # Preserve the parent instance's plastic state. Two sources:
        #   (1) siblings that were already in ``hit`` keep their state as-is
        #       (piece identity unchanged, dict key already right).
        #   (2) children of ``hit_piece`` inherit a SUBSET of hit_piece's
        #       cumulative plastic displacement, using each child's
        #       ``parent_vertex_indices`` to map the parent-cage-node array
        #       down to the child-cage-node array. ``fired`` resets because
        #       the child has its own precomputed plastic-modes basis with
        #       different mode indices.
        parent_states = getattr(hit, "plastic_states", {}) or {}
        hit_piece_state = parent_states.get(id(hit_piece))

        for comp, plist in comp_pieces.items():
            centroids = [c for c in (_centroid(p) for p in plist) if c is not None]
            if centroids:
                comp_centroid = np.mean(centroids, axis=0)
            else:
                comp_centroid = hit_piece_centroid
            kick_local = comp_centroid - hit_piece_centroid
            knorm = float(np.linalg.norm(kick_local))
            if knorm > 1e-9:
                kick_local /= knorm
            else:
                kick_local = np.zeros(3)
            kick_world = hit.local_vector_to_world(kick_local) * self.impact_kick

            carry_states = {}
            for p in plist:
                # (1) Sibling from the parent instance — reuse state directly.
                st = parent_states.get(id(p))
                if st is not None:
                    carry_states[id(p)] = dict(
                        cum=st["cum"].copy(),
                        fired=list(st["fired"]),
                    )
                    continue
                # (2) Child of hit_piece — subset hit_piece's plastic cum onto
                # this child via parent_vertex_indices. Only makes sense if
                # both indices exist (older pickles skip it silently).
                if hit_piece_state is None:
                    continue
                pvi = getattr(p, "parent_vertex_indices", None)
                if pvi is None:
                    # Older pickles didn't cache this — reconstruct via nearest
                    # match against the parent's cage vertices. Child cage
                    # vertices are a strict subset of the parent's, so nearest
                    # is exact (distance ~ 0).
                    from scipy.spatial import cKDTree as _cKDTree
                    _, pvi = _cKDTree(hit_piece.vertices).query(p.vertices)
                    p.parent_vertex_indices = pvi
                cum_parent = hit_piece_state["cum"]
                if pvi.max() >= cum_parent.shape[0]:
                    continue
                child_cum = cum_parent[pvi].copy()
                if child_cum.shape[0] != p.vertices.shape[0]:
                    continue
                carry_states[id(p)] = dict(cum=child_cum, fired=[])

            new_inst = self.spawn(
                plist,
                position=hit.position.copy(),
                orientation=hit.orientation.copy(),
                velocity=hit.velocity + kick_world,
                angular_velocity=hit.angular_velocity.copy(),
                plastic_states=carry_states if carry_states else None,
            )
            new_instances.append(new_inst)
            info = dict(extra_info or {})
            info.setdefault("branch", "fracture")
            info.setdefault("reason", "fractured")
            info["n_new_instances"] = len(new_instances)
            setattr(new_inst, "last_impact_info", info)

        if verbose:
            print(
                f"Fracture at piece depth {hit_piece.depth}: "
                f"replaced by {len(new_instances)} component instance(s) "
                f"({sum(len(i.pieces) for i in new_instances)} pieces total)"
            )
        return new_instances


def _fire_plastic_on_piece(hit, hit_piece, contact_local, direction_local,
                           magnitude, spread_radius=0.0, poisson_ratio=0.3,
                           yield_excess_threshold=0.0, mode_amplitude=None):
    """Fire one plastic mode on ``hit_piece`` using the state stored on ``hit``.

    Since ``PlasticModes`` is shared across all instances that reference the
    same FractureNode, we swap in the per-instance state before delegating to
    ``pm.apply_impact`` and swap it back out after — the pm's internal
    attributes end up unchanged; only the per-instance state stored on the
    ActiveInstance is mutated.
    """
    pm = hit_piece.plastic_modes
    if pm is None:
        return False, dict(fired=False, reason="no_plastic_bundle")

    # Older pickles predate the runtime-state attributes on PlasticModes.
    pm._ensure_runtime()

    state = hit.plastic_states.get(id(hit_piece))
    if state is None:
        state = dict(
            cum=np.zeros((hit_piece.vertices.shape[0], 3), dtype=float),
            fired=[],
        )
        hit.plastic_states[id(hit_piece)] = state

    saved_cum = pm.cumulative_displacement
    saved_fired = pm.fired_modes
    pm.cumulative_displacement = state["cum"].copy()
    pm.fired_modes = list(state["fired"])
    try:
        result = pm.apply_impact(
            contact_point=contact_local, direction=direction_local,
            magnitude=magnitude, spread_radius=spread_radius,
            poisson_ratio=poisson_ratio,
            yield_excess_threshold=yield_excess_threshold,
            mode_amplitude=mode_amplitude,
        )
        new_cum = pm.cumulative_displacement.copy()
        new_fired = list(pm.fired_modes)
    finally:
        pm.cumulative_displacement = saved_cum
        pm.fired_modes = saved_fired

    if result["fired"]:
        state["cum"] = new_cum
        state["fired"] = new_fired
    return bool(result["fired"]), result
