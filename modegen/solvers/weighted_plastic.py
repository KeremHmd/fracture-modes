"""PlasticModes whose dissipation weights are scaled by a TetWeighting plugin.

Runtime yield checks keep using the true ``sigma_y``; only the sparsity
objective of the mode computation sees the extra per-tet weights.
"""

import numpy as np

from fracture_utility.plastic_modes import (
    PlasticModes, precompute_tet_operators, build_plastic_group_operator,
)


class WeightedPlasticModes(PlasticModes):
    def __init__(self, elastic_modes, materials, tet_weights=None):
        super().__init__(elastic_modes=elastic_modes, materials=materials)
        self.tet_weights = (None if tet_weights is None
                            else np.asarray(tet_weights, dtype=float).reshape(-1))

    def assemble(self):
        (self.B_per_tet, self.vol_per_tet,
         self.gdof_per_tet) = precompute_tet_operators(self.nodes, self.elements)
        w = self.sigma_y_per_tet.copy()
        if self.tet_weights is not None:
            w = w * np.maximum(self.tet_weights, 1e-9)
        self.B_group = build_plastic_group_operator(
            self.B_per_tet, self.vol_per_tet, self.gdof_per_tet, w, self.num_vertices,
        )
        self.B_reduced = self.B_group @ self.elastic_modes.modes
        return self
