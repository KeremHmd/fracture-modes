"""United precomputed asset: elastic + plastic + fracture bundles + material.

Runtime routing model (bend / break / wobble):
    - Compute the static elastic response and peak per-tet sigma_vm.
    - Look up the material class at the peak-excess tet.
    - `rigid`               -> no response.
    - sigma_vm > sigma_c    -> FRACTURE branch (existing pipeline).
    - `elastic-plastic`  and  sigma_vm > sigma_y  and  modes remaining
                            -> PLASTIC branch (fires one plastic mode).
    - `elastic-plastic` but plastic exhausted while still yielding
                            -> promote to FRACTURE.
    - otherwise             -> "elastic bounce" (no-op; no elastic dynamics here).

First-order pose alignment:
    When fracture fires after plastic has already deformed the mesh, we render
    the fractured fine mesh at the plastically-displaced vertex positions. The
    displacement is lifted from cage nodes to fine-mesh vertices by a
    precomputed nearest-cage-node map (built once at asset-construction time).
    This is accurate for small plastic strains; the proper F_p pushforward is
    future work.
"""

import pickle

import numpy as np
from scipy.spatial import cKDTree

from .material_field import MaterialField, MATERIAL_CLASSES


class PlasticFractureAsset:
    """Precomputed united asset (elastic + plastic + fracture) with a routed runtime."""

    def __init__(self, elastic_modes, plastic_modes, fracture_modes_obj,
                 material, cage_to_fine_map=None):
        self.em = elastic_modes
        self.pm = plastic_modes
        self.fm = fracture_modes_obj
        self.material = material

        if cage_to_fine_map is None:
            fv = getattr(self.fm, "fine_vertices", None)
            if fv is not None:
                tree = cKDTree(self.em.nodes)
                _, self.cage_to_fine_map = tree.query(fv)
            else:
                self.cage_to_fine_map = None
        else:
            self.cage_to_fine_map = cage_to_fine_map

        self.is_fractured = False
        self.last_impact_result = None
        self.pm.reset_runtime()

    def reset(self):
        """Un-fire all plastic modes and clear the fracture flag."""
        self.pm.reset_runtime()
        self.is_fractured = False
        self.last_impact_result = None

    def _material_class_at(self, tet_idx):
        if self.material.material_class is None:
            return "elastic-plastic"
        code = int(self.material.material_class[tet_idx])
        return MATERIAL_CLASSES[code]

    def _sigma_c_array(self):
        if self.material.sigma_c is not None:
            return self.material.sigma_c
        return np.full(self.material.num_tets, np.inf, dtype=np.float64)

    def apply_impact(self, contact, direction, magnitude,
                     spread_radius=0.0, poisson_ratio=0.3,
                     yield_excess_threshold=0.0,
                     fracture_excess_threshold=0.0,
                     fracture_projection_threshold=10.0,
                     mode_amplitude=None):
        """Route an impact to plastic or fracture based on material + thresholds.

        Returns a dict describing what happened; see class docstring for the routing
        table. The dict always contains ``branch`` ∈ {'none', 'plastic', 'fracture',
        'promoted_fracture'} and ``reason``.
        """
        f = self.pm._nodal_force(contact, direction, magnitude, spread_radius)
        u_e_flat = self.pm._static_elastic_response(f)
        eps_e = self.pm._per_tet_strain(u_e_flat)
        sigma_vm = self.pm._per_tet_sigma_vm(eps_e, poisson_ratio)

        sigma_y_arr = self.material.sigma_y
        sigma_c_arr = self._sigma_c_array()
        excess_y = sigma_vm - sigma_y_arr
        excess_c = sigma_vm - sigma_c_arr
        peak_tet = int(np.argmax(np.maximum(excess_y, excess_c)))
        sigma_vm_peak = float(sigma_vm[peak_tet])
        mc_name = self._material_class_at(peak_tet)

        result = dict(
            branch="none", reason=None,
            plastic_result=None, fracture_result=None,
            sigma_vm_max=sigma_vm_peak,
            sigma_y_local=float(sigma_y_arr[peak_tet]),
            sigma_c_local=float(sigma_c_arr[peak_tet]),
            excess_y=float(excess_y[peak_tet]),
            excess_c=float(excess_c[peak_tet]),
            peak_material_class=mc_name,
            peak_tet=peak_tet,
        )

        if self.is_fractured:
            result["reason"] = "already_fractured"
            self.last_impact_result = result
            return result

        if mc_name == "rigid":
            result["reason"] = "rigid_material"
            self.last_impact_result = result
            return result

        if excess_c[peak_tet] > fracture_excess_threshold:
            self._fire_fracture(contact, direction,
                                fracture_projection_threshold)
            result["branch"] = "fracture"
            result["reason"] = "sigma_c_exceeded"
            result["fracture_result"] = self._fracture_summary()
            self.last_impact_result = result
            return result

        if mc_name != "elastic-plastic":
            result["reason"] = "brittle_below_sigma_c"
            self.last_impact_result = result
            return result

        if excess_y[peak_tet] <= yield_excess_threshold:
            result["reason"] = "below_yield"
            self.last_impact_result = result
            return result

        pm_result = self.pm.apply_impact(
            contact_point=contact, direction=direction, magnitude=magnitude,
            spread_radius=spread_radius, poisson_ratio=poisson_ratio,
            yield_excess_threshold=yield_excess_threshold,
            mode_amplitude=mode_amplitude,
        )
        if pm_result["fired"]:
            result["branch"] = "plastic"
            result["reason"] = "yielded"
            result["plastic_result"] = pm_result
            self.last_impact_result = result
            return result

        if pm_result["reason"] == "all_modes_exhausted":
            self._fire_fracture(contact, direction,
                                fracture_projection_threshold)
            result["branch"] = "promoted_fracture"
            result["reason"] = "plastic_exhausted"
            result["fracture_result"] = self._fracture_summary()
            self.last_impact_result = result
            return result

        result["reason"] = f"plastic_no_fire:{pm_result['reason']}"
        result["plastic_result"] = pm_result
        self.last_impact_result = result
        return result

    def _fire_fracture(self, contact, direction, threshold):
        contact_arr = np.asarray(contact, dtype=np.float64).reshape(3)
        dir_arr = np.asarray(direction, dtype=np.float64).reshape(3)
        d_norm = np.linalg.norm(dir_arr)
        dir_unit = dir_arr / d_norm if d_norm > 1e-12 else np.array([1.0, 0.0, 0.0])
        self.fm.impact_projection(
            contact_point=contact_arr, threshold=threshold, direction=dir_unit,
        )
        self.is_fractured = True

    def _fracture_summary(self):
        labels = getattr(self.fm, "fine_vertex_labels_after_impact", None)
        n = int(getattr(self.fm, "n_pieces_after_impact", 0))
        return dict(
            n_pieces=n,
            labels=None if labels is None else np.asarray(labels).copy(),
        )

    def current_fine_vertices(self, display_scale=1.0):
        """Fine-mesh vertices with cumulative plastic deformation lifted from cage."""
        if not hasattr(self.fm, "fine_vertices") or self.fm.fine_vertices is None:
            raise RuntimeError(
                "fracture_modes.impact_precomputation must have been called before "
                "PlasticFractureAsset can render."
            )
        rest = self.fm.fine_vertices
        if self.cage_to_fine_map is None:
            return rest
        lift = self.pm.cumulative_displacement[self.cage_to_fine_map]
        return rest + display_scale * lift

    def current_fine_triangles(self):
        return self.fm.fine_triangles

    def current_fine_labels(self):
        """Per-fine-vertex piece label if fractured, else None."""
        if self.is_fractured and hasattr(self.fm, "fine_vertex_labels_after_impact"):
            return self.fm.fine_vertex_labels_after_impact
        return None

    def save(self, path):
        """Pickle the whole bundle (fracture_modes carries sparse matrices)."""
        with open(path, "wb") as f:
            pickle.dump(dict(
                em=self.em, pm=self.pm, fm=self.fm,
                material=self.material,
                cage_to_fine_map=self.cage_to_fine_map,
            ), f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(
            elastic_modes=data["em"],
            plastic_modes=data["pm"],
            fracture_modes_obj=data["fm"],
            material=data["material"],
            cage_to_fine_map=data.get("cage_to_fine_map"),
        )
        if obj.pm.elastic_modes is None:
            obj.pm.elastic_modes = obj.em
        return obj

    def __repr__(self):
        return (
            f"PlasticFractureAsset("
            f"cage_verts={self.em.num_vertices}, "
            f"tets={self.em.num_tets}, "
            f"fine_verts={self.fm.fine_vertices.shape[0]}, "
            f"elastic_modes={self.em.modes.shape[1]}, "
            f"plastic_modes={self.pm.modes.shape[1]}, "
            f"fracture_modes={self.fm.modes.shape[1]}, "
            f"fired_plastic={len(self.pm.fired_modes)}, "
            f"fractured={self.is_fractured})"
        )
