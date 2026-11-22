"""Sellan et al. SOCP fracture modes with two plugin hooks.

Identical to ``fracture_utility.compute_fracture_modes_heterogeneous`` except:

* face weights: ``G_c(face) * area(face)`` is multiplied by a
  ``FaceWeighting`` plugin's ``multiplier(t1, t2)``;
* initial vectors: an ``InitVectors`` plugin may replace the Laplacian
  eigenmodes that seed each mode's iteration.

With the default ``realistic.uniform_faces`` + ``realistic.laplacian`` hooks
the result is the same computation as the frozen baseline.
"""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, kron, eye
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import connected_components
import igl

from fracture_utility.explode_mesh_heterogeneous import explode_mesh_heterogeneous
from fracture_utility.conic_solve import conic_solve
from fracture_utility.massmatrix_tets import massmatrix_tets
from fracture_utility.sparse_sqrt import sparse_sqrt
from fracture_utility.tictoc import tic, toc
from fracture_utility.material_field import MaterialField
from fracture_utility.fracture_modes_heterogeneous import fracture_modes_heterogeneous


class WeightedMaterialField(MaterialField):
    """MaterialField whose per-face toughness is scaled by a FaceWeighting plugin."""

    def __init__(self, base, weighting, floor=1.0e-6):
        super().__init__(
            num_tets=base.num_tets, G_c=base.G_c, rho=base.rho, E=base.E,
            sigma_y=base.sigma_y, sigma_c=base.sigma_c,
            material_class=base.material_class,
            interface_rule=base.interface_rule,
        )
        self._weighting = weighting
        self._floor = float(floor)

    def face_G_c(self, t1, t2):
        m = float(self._weighting.multiplier(int(t1), int(t2)))
        return super().face_G_c(t1, t2) * max(m, self._floor)

    def face_G_c_vectorized(self, t1_arr, t2_arr):
        base = super().face_G_c_vectorized(t1_arr, t2_arr)
        m = np.array([self._weighting.multiplier(int(a), int(b))
                      for a, b in zip(t1_arr, t2_arr)], dtype=float)
        return base * np.maximum(m, self._floor)


def compute_fracture_modes_hooked(vertices, elements, parameters,
                                  materials=None, weighting=None, init=None):
    num_tets = elements.shape[0]
    if materials is None:
        materials = MaterialField.homogeneous(num_tets)
    if weighting is not None:
        weighting.prepare(vertices, elements, materials)
        materials = WeightedMaterialField(materials, weighting)

    if parameters.verbose:
        print("Starting hooked fracture mode computation:",
              parameters.num_modes, "modes,", vertices.shape[0], "vertices,",
              num_tets, "tets")
        tic()

    blockdiag_kron = eye(parameters.d)

    # Step 1: initial vectors (plugin hook, default = Laplacian eigenmodes)
    init_vectors = None
    if init is not None:
        init_vectors = init.vectors(vertices, elements, parameters.num_modes, parameters.d)
    if init_vectors is None:
        laplacian_unexploded = igl.cotmatrix(vertices, elements)
        massmatrix_unexploded = kron(
            blockdiag_kron, massmatrix_tets(vertices, elements), format="csc"
        )
        Q_unexploded = kron(blockdiag_kron, laplacian_unexploded, format="csc")
        _, init_vectors = eigsh(
            -Q_unexploded, parameters.num_modes, massmatrix_unexploded,
            which="LM", sigma=0,
        )
        init_vectors = np.real(init_vectors)
    else:
        init_vectors = np.asarray(init_vectors, dtype=float)
        expected = (parameters.d * vertices.shape[0], parameters.num_modes)
        if init_vectors.shape != expected:
            raise ValueError(
                f"init plugin returned shape {init_vectors.shape}, expected {expected} "
                f"(component-major d*V x num_modes)"
            )

    # Step 2: explode (face weights hook lives inside the material field)
    (
        exploded_vertices, exploded_elements, discontinuity_matrix,
        unexploded_to_exploded_matrix, tet_to_vertex_matrix, tet_neighbors,
    ) = explode_mesh_heterogeneous(vertices, elements, materials=materials, num_quad=1)

    discontinuity_matrix_full = kron(
        parameters.omega * blockdiag_kron, discontinuity_matrix, format="coo"
    )
    unexploded_to_exploded_matrix_full = kron(
        blockdiag_kron, unexploded_to_exploded_matrix, format="csc"
    )
    tet_to_vertex_matrix_full = kron(blockdiag_kron, tet_to_vertex_matrix, format="csc")
    laplacian_exploded = igl.cotmatrix(exploded_vertices, exploded_elements)
    massmatrix_exploded = massmatrix_tets(exploded_vertices, exploded_elements)
    Q = kron(blockdiag_kron, laplacian_exploded, format="csc")
    M = kron(blockdiag_kron, massmatrix_exploded, format="csc")

    # Step 3: iterate
    UU = unexploded_to_exploded_matrix_full @ init_vectors
    UU = tet_to_vertex_matrix_full.T @ UU
    Q = coo_matrix(tet_to_vertex_matrix_full.T @ Q @ tet_to_vertex_matrix_full)
    M = coo_matrix(tet_to_vertex_matrix_full.T @ M @ tet_to_vertex_matrix_full)
    discontinuity_matrix_full = coo_matrix(
        discontinuity_matrix_full @ tet_to_vertex_matrix_full
    )

    if not np.all(np.isfinite(UU)):
        raise ValueError("initial vectors contain NaN/inf (degenerate cage?)")
    Us = []
    failed_solves = 0
    labels_full = np.zeros((num_tets, parameters.num_modes))
    for k in range(parameters.num_modes):
        iter_num = 0
        diff = 1.0
        c = UU[:, k]
        while diff > parameters.tol and iter_num < parameters.max_iter:
            cprev = c
            Ui = conic_solve(discontinuity_matrix_full, M, Us, c, parameters.d)
            nrm2 = float(np.dot(Ui, M @ Ui))
            if not np.isfinite(nrm2) or nrm2 <= 1e-30:
                # MOSEK returned a degenerate (zero / non-finite) solution, which
                # happens on some coarse or sliver-heavy cages. Keep the last
                # valid iterate instead of feeding NaNs into the next solve.
                if parameters.verbose:
                    print(f"  mode {k + 1}: degenerate SOCP solution at iteration "
                          f"{iter_num}; keeping previous iterate")
                failed_solves += 1
                break
            c = Ui / np.sqrt(nrm2)
            diff = np.max(np.abs(c - cprev))
            iter_num += 1
        cc = np.reshape(c, (-1, parameters.d), order="F")
        dist = np.linalg.norm(cc[tet_neighbors[:, 0], :] - cc[tet_neighbors[:, 1], :], axis=1)
        actual = tet_neighbors[dist < 0.1, :]
        adj = csr_matrix(
            (np.ones(actual.shape[0]), (actual[:, 0], actual[:, 1])),
            shape=(exploded_elements.shape[0], exploded_elements.shape[0]), dtype=int,
        )
        n_components, labels = connected_components(adj)
        labels_full[:, k] = labels
        Us.append(c)
        UU[:, k] = c
        if parameters.verbose:
            print(f"  mode {k + 1}: {iter_num} iterations, {n_components} pieces")

    if failed_solves:
        print(f"  warning: {failed_solves} degenerate SOCP solve(s); affected modes "
              f"fall back to their last valid iterate (try a finer cage)")
    if parameters.verbose:
        print("Hooked fracture modes done in", round(toc(silence=True), 2), "s")

    return (exploded_vertices, exploded_elements, UU, labels_full,
            tet_to_vertex_matrix, tet_neighbors, M, unexploded_to_exploded_matrix)


class HookedFractureModes(fracture_modes_heterogeneous):
    """``fracture_modes_heterogeneous`` driven by plugin hooks."""

    def compute_modes_hooked(self, parameters, materials=None, weighting=None, init=None):
        (
            self.exploded_vertices, self.exploded_elements, self.modes,
            self.labels, self.tet_to_vertex_matrix, self.tet_neighbors,
            self.massmatrix, self.unexploded_to_exploded_matrix,
        ) = compute_fracture_modes_hooked(
            self.vertices, self.elements, parameters, materials, weighting, init
        )
        self.verbose = parameters.verbose
        return self
