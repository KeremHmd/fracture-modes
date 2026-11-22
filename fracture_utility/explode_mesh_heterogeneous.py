"""Heterogeneous variant of explode_mesh"""

import numpy as np
from scipy.sparse import csr_matrix
import igl

from .material_field import MaterialField


def explode_mesh_heterogeneous(vertices, elements, materials, num_quad=1):
    """Explode the tet mesh and build a G_c-weighted discontinuity matrix"""
    if not isinstance(materials, MaterialField):
        raise TypeError(
            "materials must be a MaterialField instance "
            f"(got {type(materials).__name__})"
        )
    if materials.num_tets != elements.shape[0]:
        raise ValueError(
            f"MaterialField was built for {materials.num_tets} tets, "
            f"but elements has {elements.shape[0]} tets"
        )

    # --- Identical to the original up to the face-weight assignment. ---

    vert_indeces = np.squeeze(np.reshape(elements, (-1, 1)))
    exploded_vert_indeces = np.linspace(
        0, 4 * elements.shape[0] - 1, 4 * elements.shape[0], dtype=int
    )
    exploded_elements = np.reshape(exploded_vert_indeces, (-1, 4))
    exploded_vertices = vertices[vert_indeces, :]

    # tet -> 4 exploded vertices
    I = exploded_vert_indeces
    J = np.kron(
        np.linspace(0, elements.shape[0] - 1, elements.shape[0], dtype=int),
        np.array([1.0, 1.0, 1.0, 1.0], dtype=int),
    )
    vals = np.ones(I.shape)
    tet_to_vertex_matrix = csr_matrix(
        (vals, (I, J)), shape=(4 * elements.shape[0], elements.shape[0])
    )

    # unexploded -> exploded
    J = vert_indeces
    I = exploded_vert_indeces
    vals = np.ones((I.shape[0]))
    unexploded_to_exploded_matrix = csr_matrix(
        (vals, (I, J)), shape=(4 * elements.shape[0], vertices.shape[0])
    )

    # Discontinuity matrix
    if num_quad == 3:
        quad_weights = np.array(
            [[2 / 3, 1 / 3, 1 / 3],
             [1 / 3, 2 / 3, 1 / 3],
             [1 / 3, 1 / 3, 2 / 3]]
        )
    elif num_quad == 1:
        quad_weights = np.array([[1.0, 0.0, 0.0]])
    else:
        raise ValueError("num_quad must be 1 or 3")

    TT, TTi = igl.tet_tet_adjacency(elements)

    tet_neighbors_j = np.reshape(TT, (-1, 1))
    tet_neighbors_i = np.reshape(
        np.kron(
            np.linspace(0, elements.shape[0] - 1, elements.shape[0], dtype=int),
            np.array([1.0, 1.0, 1.0, 1.0], dtype=int),
        ),
        (-1, 1),
    )
    tet_neighbors = np.hstack((tet_neighbors_i, tet_neighbors_j))
    tet_neighbors = tet_neighbors[tet_neighbors_j[:, 0] > -1, :]

    I = np.zeros(6 * num_quad * 4 * exploded_elements.shape[0])
    J = np.zeros(6 * num_quad * 4 * exploded_elements.shape[0])
    vals = np.zeros(6 * num_quad * 4 * exploded_elements.shape[0])
    num_tets = elements.shape[0]

    tet_face_ordering = np.array(
        [[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 0, 3]]
    )
    all_faces = np.vstack(
        (
            elements[:, [0, 1, 2]],
            elements[:, [0, 1, 3]],
            elements[:, [1, 2, 3]],
            elements[:, [2, 0, 3]],
        )
    )
    areas = igl.doublearea(vertices, all_faces)

    for i in range(elements.shape[0]):
        tet_1 = i
        for nn in range(4):
            tet_2 = TT[i, nn]
            if tet_2 > -1 and tet_1 > tet_2:
                face_area = areas[nn * elements.shape[0] + i]

                # =============================================================
                # ONLY MATHEMATICAL DIFFERENCE FROM explode_mesh.py:
                # multiply face_area by the per-face G_c.
                # =============================================================
                weight = materials.face_G_c(tet_1, tet_2) * face_area

                unexploded_indeces_tet_1 = elements[
                    tet_1, tet_face_ordering[nn, :]
                ]
                unexploded_indeces_tet_2 = elements[
                    tet_2, tet_face_ordering[TTi[i, nn], :]
                ]
                exploded_indeces_tet_1 = (
                    4 * tet_1 + tet_face_ordering[nn, :]
                )
                exploded_indeces_tet_2 = (
                    4 * tet_2 + tet_face_ordering[TTi[i, nn], :]
                )

                argsort_1 = np.argsort(unexploded_indeces_tet_1)
                argsort_2 = np.argsort(unexploded_indeces_tet_2)

                for qi in range(num_quad):
                    base = 6 * (qi + num_quad * nn + num_quad * 4 * i)
                    row = 4 * num_tets * qi + 4 * i + nn

                    I[base + 0] = row
                    I[base + 1] = row
                    I[base + 2] = row
                    I[base + 3] = row
                    I[base + 4] = row
                    I[base + 5] = row

                    J[base + 0] = exploded_indeces_tet_1[argsort_1[0]]
                    J[base + 1] = exploded_indeces_tet_2[argsort_2[0]]
                    J[base + 2] = exploded_indeces_tet_1[argsort_1[1]]
                    J[base + 3] = exploded_indeces_tet_2[argsort_2[1]]
                    J[base + 4] = exploded_indeces_tet_1[argsort_1[2]]
                    J[base + 5] = exploded_indeces_tet_2[argsort_2[2]]

                    vals[base + 0] = quad_weights[qi, 0] * weight
                    vals[base + 1] = -quad_weights[qi, 0] * weight
                    vals[base + 2] = quad_weights[qi, 1] * weight
                    vals[base + 3] = -quad_weights[qi, 1] * weight
                    vals[base + 4] = quad_weights[qi, 2] * weight
                    vals[base + 5] = -quad_weights[qi, 2] * weight
            else:
                for qi in range(num_quad):
                    base = 6 * (qi + num_quad * nn + num_quad * 4 * i)
                    I[base + 0] = -1
                    I[base + 1] = -1
                    I[base + 2] = -1
                    I[base + 3] = -1
                    I[base + 4] = -1
                    I[base + 5] = -1

    J = J[I > -1]
    vals = vals[I > -1]
    I = I[I > -1]

    discontinuity_matrix = csr_matrix(
        (vals, (I, J)),
        shape=(num_quad * 4 * exploded_elements.shape[0],
               exploded_vertices.shape[0]),
    )

    num_nonzeros = np.diff(discontinuity_matrix.indptr)
    discontinuity_matrix = discontinuity_matrix[num_nonzeros != 0]

    return (
        exploded_vertices,
        exploded_elements,
        discontinuity_matrix,
        unexploded_to_exploded_matrix,
        tet_to_vertex_matrix,
        tet_neighbors,
    )
