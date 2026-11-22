"""Voronoi shard "fracture modes" - an alternative, non-SOCP generator.

Seeds are farthest-point sampled over tet centroids (optionally denser where
G_c is low), every tet joins its nearest seed, and each cell becomes one
piecewise-constant, M-normalised mode. The resulting object satisfies the
same contract as the SOCP modes, so ``impact_precomputation`` extracts the
cells as precomputed pieces and the whole runtime (projection thresholds,
multi-impact tree, plastic routing) works unchanged. Shards are cheap to
compute (no MOSEK) and read well for exaggerated / artistic destruction.
"""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, kron, eye
from scipy.sparse.csgraph import connected_components

from fracture_utility.explode_mesh_heterogeneous import explode_mesh_heterogeneous
from fracture_utility.massmatrix_tets import massmatrix_tets
from fracture_utility.material_field import MaterialField
from fracture_utility.fracture_modes_heterogeneous import fracture_modes_heterogeneous


def farthest_point_seeds(points, k, rng, weights=None):
    n = points.shape[0]
    k = max(1, min(int(k), n))
    first = int(rng.integers(n)) if weights is None else int(rng.choice(n, p=weights / weights.sum()))
    seeds = [first]
    d = np.linalg.norm(points - points[first], axis=1)
    for _ in range(k - 1):
        score = d if weights is None else d * weights
        nxt = int(np.argmax(score))
        seeds.append(nxt)
        d = np.minimum(d, np.linalg.norm(points - points[nxt], axis=1))
    return np.array(seeds, dtype=int)


class VoronoiFractureModes(fracture_modes_heterogeneous):
    def compute_modes_voronoi(self, num_cells=24, d=3, seed=0, toughness_bias=0.0,
                              materials=None, verbose=False):
        vertices = np.asarray(self.vertices, dtype=float)
        elements = np.asarray(self.elements)
        T = elements.shape[0]
        rng = np.random.default_rng(seed)

        homog = MaterialField.homogeneous(T)
        (ev, ee, _D, u2e, t2v, tet_neighbors) = explode_mesh_heterogeneous(
            vertices, elements, materials=homog, num_quad=1
        )
        blk = eye(d)
        M_ex = kron(blk, massmatrix_tets(ev, ee), format="csc")
        t2v_full = kron(blk, t2v, format="csc")
        M = coo_matrix(t2v_full.T @ M_ex @ t2v_full)
        Mcsr = M.tocsr()

        centroids = vertices[elements].mean(axis=1)
        weights = None
        if toughness_bias > 0.0 and materials is not None:
            g = np.asarray(materials.G_c, dtype=float)
            weights = (1.0 / np.maximum(g, 1e-12)) ** float(toughness_bias)
        seeds = farthest_point_seeds(centroids, num_cells, rng, weights)
        dist = np.linalg.norm(centroids[:, None, :] - centroids[seeds][None, :, :], axis=2)
        cell = np.argmin(dist, axis=1)

        modes = np.zeros((d * T, len(seeds)))
        labels = np.zeros((T, len(seeds)))
        for j in range(len(seeds)):
            inside = cell == j
            if not inside.any():
                continue
            comp = j % d
            vec = np.zeros(d * T)
            vec[comp * T + np.where(inside)[0]] = 1.0
            nrm = np.sqrt(vec @ (Mcsr @ vec))
            modes[:, j] = vec / max(nrm, 1e-12)
            same = inside[tet_neighbors[:, 0]] == inside[tet_neighbors[:, 1]]
            pairs = tet_neighbors[same]
            adj = csr_matrix((np.ones(pairs.shape[0]), (pairs[:, 0], pairs[:, 1])),
                             shape=(T, T), dtype=int)
            _, lab = connected_components(adj, directed=False)
            labels[:, j] = lab

        self.exploded_vertices = ev
        self.exploded_elements = ee
        self.modes = modes
        self.labels = labels
        self.tet_to_vertex_matrix = t2v
        self.tet_neighbors = tet_neighbors
        self.massmatrix = M
        self.unexploded_to_exploded_matrix = u2e
        self.verbose = verbose
        self.voronoi_cells = cell
        if verbose:
            print(f"Voronoi modes: {len(seeds)} cells on {T} tets")
        return self
