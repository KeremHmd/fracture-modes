"""Heterogeneous-aware subclass of fracture_modes"""

from .fracture_modes import fracture_modes
from .fracture_modes_parameters import fracture_modes_parameters
from .compute_fracture_modes_heterogeneous import (
    compute_fracture_modes_heterogeneous,
)
from .material_field import MaterialField


class fracture_modes_heterogeneous(fracture_modes):
    """``fracture_modes`` with an optional per-tet material field """

    def compute_modes(self, parameters=None, materials=None):
        """Compute modes, optionally with heterogeneous toughness"""
        if parameters is None:
            parameters = fracture_modes_parameters()
        if materials is None:
            return super().compute_modes(parameters=parameters)

        if not isinstance(materials, MaterialField):
            raise TypeError(
                "materials must be a MaterialField instance "
                f"(got {type(materials).__name__})"
            )

        (
            self.exploded_vertices,
            self.exploded_elements,
            self.modes,
            self.labels,
            self.tet_to_vertex_matrix,
            self.tet_neighbors,
            self.massmatrix,
            self.unexploded_to_exploded_matrix,
        ) = compute_fracture_modes_heterogeneous(
            self.vertices, self.elements, parameters, materials
        )
        self.verbose = parameters.verbose

    def impact_precomputation(self, v_fine=None, f_fine=None, wave_h=1 / 30,
                              upper_envelope=False, smoothing_lambda=100,
                              delaunay_remesh=False, delaunay_kwargs=None):
        """Same as the parent's impact_precomputation plus an optional Delaunay remeshing pass on cap-face edges"""
        super().impact_precomputation(
            v_fine=v_fine, f_fine=f_fine, wave_h=wave_h,
            upper_envelope=upper_envelope, smoothing_lambda=smoothing_lambda,
        )
        if delaunay_remesh:
            from .delaunay_remesh import apply_delaunay_to_fracture_modes
            apply_delaunay_to_fracture_modes(
                self, v_fine, f_fine, **(delaunay_kwargs or {})
            )
