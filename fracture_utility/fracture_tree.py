"""Recursive precomputed fracture-mode hierarchy"""

import pickle

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from .fracture_modes_heterogeneous import fracture_modes_heterogeneous
from .fracture_modes_parameters import fracture_modes_parameters
from .material_field import MaterialField, MATERIAL_CLASSES
from .elastic_modes import ElasticModes
from .plastic_modes import PlasticModes

def tet_mesh_volume(vertices, elements):
    """Total (absolute) volume of a tetrahedral mesh."""
    if elements is None or len(elements) == 0:
        return 0.0
    elements = np.asarray(elements, dtype=np.int64)
    vertices = np.asarray(vertices, dtype=float)
    v0 = vertices[elements[:, 0]]
    v1 = vertices[elements[:, 1]]
    v2 = vertices[elements[:, 2]]
    v3 = vertices[elements[:, 3]]
    vols = np.einsum("ij,ij->i", np.cross(v1 - v0, v2 - v0), v3 - v0) / 6.0
    return float(np.abs(vols).sum())


class FractureNode:
    """A single node in a fracture tree"""

    def __init__(self, vertices, elements, materials=None,
                 fine_vertices=None, fine_triangles=None, depth=0,
                 face_status=None):
        self.vertices = np.asarray(vertices, dtype=float)
        self.elements = np.asarray(elements, dtype=np.int32)
        self.materials = materials
        self.fine_vertices = (
            None if fine_vertices is None
            else np.asarray(fine_vertices, dtype=float)
        )
        self.fine_triangles = (
            None if fine_triangles is None
            else np.asarray(fine_triangles, dtype=np.int32)
        )
        self.face_status = (
            None if face_status is None
            else np.asarray(face_status, dtype=np.int8)
        )
        self.depth = int(depth)
        self.volume = tet_mesh_volume(self.vertices, self.elements)
        self.modes_instance = None
        self.children = []
        self.is_atomic = False
        # Optional per-node plastic pipeline bundles (populated by precompute_node
        # when --num-elastic-modes / --num-plastic-modes are set). None-safe:
        # existing trees without these attributes still work end-to-end.
        self.elastic_modes = None
        self.plastic_modes = None
        self.cage_to_fine_map = None

    def analyze_impact(self, contact_local, direction_local, magnitude,
                       spread_radius=0.0, poisson_ratio=0.3):
        """Static-elastic-response analysis of an impact on this node.

        Returns ``None`` when the node has no plastic bundle (which also means
        the sigma_y / sigma_c / material_class fields aren't available). Callers
        should treat that as "no plastic pipeline on this node" and fall back
        to the fracture-only path.

        Otherwise returns a dict with per-tet sigma_vm, the peak-excess tet
        index, its material class, and the sigma_y / sigma_c thresholds there.
        """
        pm = self.plastic_modes
        if pm is None or self.materials is None or self.materials.sigma_y is None:
            return None
        f = pm._nodal_force(contact_local, direction_local, magnitude, spread_radius)
        u_e_flat = pm._static_elastic_response(f)
        eps_e = pm._per_tet_strain(u_e_flat)
        sigma_vm = pm._per_tet_sigma_vm(eps_e, poisson_ratio)
        sigma_y = self.materials.sigma_y
        sigma_c = (self.materials.sigma_c if self.materials.sigma_c is not None
                   else np.full_like(sigma_y, np.inf))
        excess_y = sigma_vm - sigma_y
        excess_c = sigma_vm - sigma_c
        peak_tet = int(np.argmax(np.maximum(excess_y, excess_c)))
        if self.materials.material_class is not None:
            mc_name = MATERIAL_CLASSES[int(self.materials.material_class[peak_tet])]
        else:
            mc_name = "elastic-plastic"
        return dict(
            sigma_vm=sigma_vm,
            peak_tet=peak_tet,
            sigma_vm_max=float(sigma_vm[peak_tet]),
            sigma_y_local=float(sigma_y[peak_tet]),
            sigma_c_local=float(sigma_c[peak_tet]),
            excess_y=float(excess_y[peak_tet]),
            excess_c=float(excess_c[peak_tet]),
            material_class=mc_name,
        )

    def num_tets(self):
        return int(self.elements.shape[0])

    def num_pieces(self):
        if self.modes_instance is None:
            return 0
        return int(getattr(self.modes_instance, "precomputed_num_pieces", 0))

    def __repr__(self):
        return (
            f"FractureNode(depth={self.depth}, tets={self.num_tets()}, "
            f"volume={self.volume:.4g}, children={len(self.children)}, "
            f"atomic={self.is_atomic})"
        )


class FractureTree:
    """A precomputed hierarchy of fracture nodes"""

    def __init__(self, root, depth_limit=1, volume_threshold=0.0,
                 min_tets_for_fracture=10):
        self.root = root
        self.depth_limit = int(depth_limit)
        self.volume_threshold = float(volume_threshold)
        self.min_tets_for_fracture = int(min_tets_for_fracture)

    @classmethod
    def from_mesh(cls, v_fine, f_fine, cage_tet_nodes, cage_tet_elements,
                  materials=None, params=None,
                  depth_limit=1, volume_threshold=0.0,
                  min_tets_for_fracture=10, verbose=True,
                  delaunay_remesh=False, delaunay_kwargs=None,
                  num_elastic_modes=0, num_plastic_modes=0,
                  poisson_ratio=0.3,
                  default_E=2.0e11, default_rho=1.0e3,
                  plastic_max_iter=10, plastic_tol=1.0e-4):
        """Build the tree from an input mesh + cage tet mesh + materials"""
        if params is None:
            params = fracture_modes_parameters()
        root = FractureNode(
            vertices=cage_tet_nodes,
            elements=cage_tet_elements,
            materials=materials,
            fine_vertices=v_fine,
            fine_triangles=f_fine,
            depth=0,
        )
        tree = cls(
            root,
            depth_limit=depth_limit,
            volume_threshold=volume_threshold,
            min_tets_for_fracture=min_tets_for_fracture,
        )
        precompute_node(
            root, params, depth_limit, volume_threshold,
            min_tets_for_fracture=min_tets_for_fracture, verbose=verbose,
            delaunay_remesh=delaunay_remesh,
            delaunay_kwargs=delaunay_kwargs,
            num_elastic_modes=num_elastic_modes,
            num_plastic_modes=num_plastic_modes,
            poisson_ratio=poisson_ratio,
            default_E=default_E, default_rho=default_rho,
            plastic_max_iter=plastic_max_iter, plastic_tol=plastic_tol,
        )
        if verbose:
            print(f"FractureTree built: {tree.num_nodes()} total nodes")
        return tree

    def num_nodes(self):
        def _count(node):
            return 1 + sum(_count(c) for c in node.children)
        return _count(self.root)

    def walk(self):
        """Yield every node in the tree (root first)."""
        def _walk(node):
            yield node
            for child in node.children:
                yield from _walk(child)
        yield from _walk(self.root)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)


def precompute_node(node, params, depth_limit, volume_threshold,
                    min_tets_for_fracture=10, verbose=True,
                    delaunay_remesh=False, delaunay_kwargs=None,
                    num_elastic_modes=0, num_plastic_modes=0,
                    poisson_ratio=0.3,
                    default_E=2.0e11, default_rho=1.0e3,
                    plastic_max_iter=10, plastic_tol=1.0e-4):
    """Recursively compute fracture modes for the node and its descendants.

    When ``num_elastic_modes > 0`` and the node's material carries an ``E``
    field (or ``default_E`` fallback), this also builds a per-node
    :class:`ElasticModes` bundle. When ``num_plastic_modes > 0`` and the node's
    material carries a ``sigma_y`` field, a per-node :class:`PlasticModes`
    bundle is built on top of that elastic subspace, and a nearest-cage-node
    map from the fine mesh to the cage tets is stored for Stage-7 first-order
    pose alignment. All bundles are attached to the node and inherited
    naturally by children (each child gets its own bundle rebuilt from its
    subsetted tet mesh + materials).
    """
    indent = "  " * node.depth
    if node.depth >= depth_limit:
        node.is_atomic = True
        if verbose:
            print(f"{indent}[depth {node.depth}] atomic (depth limit)")
        return
    if node.volume < volume_threshold:
        node.is_atomic = True
        if verbose:
            print(f"{indent}[depth {node.depth}] atomic (volume {node.volume:.4g} < {volume_threshold})")
        return
    if node.num_tets() < min_tets_for_fracture:
        node.is_atomic = True
        if verbose:
            print(f"{indent}[depth {node.depth}] atomic ({node.num_tets()} tets < {min_tets_for_fracture})")
        return
    if node.fine_vertices is None or node.fine_triangles is None:
        node.is_atomic = True
        if verbose:
            print(f"{indent}[depth {node.depth}] atomic (missing fine mesh)")
        return

    if verbose:
        print(f"{indent}[depth {node.depth}] compute_modes "
              f"({node.num_tets()} tets, vol={node.volume:.4g})...")

    modes = fracture_modes_heterogeneous(node.vertices, node.elements)
    modes.compute_modes(parameters=params, materials=node.materials)
    modes.impact_precomputation(
        v_fine=node.fine_vertices,
        f_fine=node.fine_triangles,
        delaunay_remesh=delaunay_remesh,
        delaunay_kwargs=delaunay_kwargs,
    )
    node.modes_instance = modes

    if num_elastic_modes > 0:
        try:
            em = ElasticModes(
                nodes=node.vertices, elements=node.elements,
                materials=node.materials, poisson_ratio=poisson_ratio,
                default_E=default_E, default_rho=default_rho,
            )
            em.assemble()
            em_k = min(num_elastic_modes, 3 * node.vertices.shape[0] - 7)
            em.compute_modes(num_modes=em_k, verbose=False)
            node.elastic_modes = em
            if verbose:
                print(f"{indent}[depth {node.depth}] elastic modes: {em_k}")
        except Exception as e:
            if verbose:
                print(f"{indent}[depth {node.depth}] elastic-modes build failed: {e}")
            node.elastic_modes = None

    if (num_plastic_modes > 0 and node.elastic_modes is not None
            and node.materials is not None and node.materials.sigma_y is not None):
        try:
            pm = PlasticModes(elastic_modes=node.elastic_modes, materials=node.materials)
            pm.assemble()
            pm_k = min(num_plastic_modes, node.elastic_modes.modes.shape[1])
            pm.compute_modes(
                num_modes=pm_k,
                max_iter=plastic_max_iter, tol=plastic_tol,
                verbose=False,
            )
            node.plastic_modes = pm
            # Build the cage-to-fine map against the vertices actually used at
            # render time (node.fine_vertices), not the further-carved output
            # of impact_precomputation which may have a different vertex count.
            tree_kd = cKDTree(node.vertices)
            _, node.cage_to_fine_map = tree_kd.query(node.fine_vertices)
            if verbose:
                print(f"{indent}[depth {node.depth}] plastic modes: {pm_k}")
        except Exception as e:
            if verbose:
                print(f"{indent}[depth {node.depth}] plastic-modes build failed: {e}")
            node.plastic_modes = None

    n_pieces = int(getattr(modes, "precomputed_num_pieces", 0))
    if n_pieces <= 1:
        node.is_atomic = True
        if verbose:
            print(f"{indent}[depth {node.depth}] atomic (1 piece)")
        return

    if verbose:
        print(f"{indent}[depth {node.depth}] precomputed {n_pieces} pieces")

    for piece_idx in range(n_pieces):
        child = _build_child_from_piece(node, piece_idx)
        if child is None:
            continue
        precompute_node(
            child, params, depth_limit, volume_threshold,
            min_tets_for_fracture=min_tets_for_fracture, verbose=verbose,
            delaunay_remesh=delaunay_remesh,
            delaunay_kwargs=delaunay_kwargs,
            num_elastic_modes=num_elastic_modes,
            num_plastic_modes=num_plastic_modes,
            poisson_ratio=poisson_ratio,
            default_E=default_E, default_rho=default_rho,
            plastic_max_iter=plastic_max_iter, plastic_tol=plastic_tol,
        )
        node.children.append(child)


def _build_child_from_piece(parent, piece_idx):
    """Construct a child FractureNode from one piece of the parent's segmentation"""
    modes = parent.modes_instance
    tet_mask = np.asarray(modes.all_modes_labels) == piece_idx
    if not tet_mask.any():
        return None

    # Subset the tet mesh
    child_elements_raw = parent.elements[tet_mask]
    used_vertex_indices = np.unique(child_elements_raw.flatten())
    vremap = -np.ones(parent.vertices.shape[0], dtype=np.int32)
    vremap[used_vertex_indices] = np.arange(used_vertex_indices.shape[0], dtype=np.int32)
    child_elements = vremap[child_elements_raw]
    child_vertices = parent.vertices[used_vertex_indices, :]

    # Subset materials — carry ALL fields the parent had (G_c + optional
    # sigma_y / sigma_c / E / rho / material_class) so the child can drive
    # its own plastic + fracture pipelines.
    child_materials = None
    if parent.materials is not None:
        child_G_c = parent.materials.G_c[tet_mask]
        if child_G_c.size > 0:
            mkw = dict(
                num_tets=int(child_elements.shape[0]),
                G_c=child_G_c,
                interface_rule=parent.materials.interface_rule,
            )
            if parent.materials.sigma_y is not None:
                mkw["sigma_y"] = parent.materials.sigma_y[tet_mask]
            if parent.materials.sigma_c is not None:
                mkw["sigma_c"] = parent.materials.sigma_c[tet_mask]
            if parent.materials.E is not None:
                mkw["E"] = parent.materials.E[tet_mask]
            if parent.materials.rho is not None:
                mkw["rho"] = parent.materials.rho[tet_mask]
            if parent.materials.material_class is not None:
                mkw["material_class"] = parent.materials.material_class[tet_mask]
            child_materials = MaterialField(**mkw)

    # Subset the fine (rendering) mesh
    P2V = modes.piece_to_fine_vertices_matrix
    col = P2V[:, piece_idx]
    if hasattr(col, "toarray"):
        fine_vertex_mask = np.asarray(col.toarray()).reshape(-1) > 0
    else:
        fine_vertex_mask = np.asarray(col).reshape(-1) > 0
    if not fine_vertex_mask.any():
        return None
    fine_v_indices = np.where(fine_vertex_mask)[0]
    fine_v_set = set(int(v) for v in fine_v_indices)
    tri_in_piece = np.array(
        [all(int(v) in fine_v_set for v in tri) for tri in modes.fine_triangles],
        dtype=bool,
    )
    if not tri_in_piece.any():
        return None
    sub_tris_raw = modes.fine_triangles[tri_in_piece]
    used_fine_v = np.unique(sub_tris_raw.flatten())
    fremap = -np.ones(modes.fine_vertices.shape[0], dtype=np.int32)
    fremap[used_fine_v] = np.arange(used_fine_v.shape[0], dtype=np.int32)
    sub_tris = fremap[sub_tris_raw]
    sub_verts = modes.fine_vertices[used_fine_v, :]

    # If the parent's modes ran Delaunay, the per-face status array is 1:1 with the parent's fine_triangles. Subset it the same way.
    parent_status = getattr(modes, "delaunay_face_status", None)
    child_status = None
    if parent_status is not None:
        ps = np.asarray(parent_status)
        if ps.shape[0] == modes.fine_triangles.shape[0]:
            child_status = ps[tri_in_piece].astype(np.int8)

    child = FractureNode(
        vertices=child_vertices,
        elements=child_elements,
        materials=child_materials,
        fine_vertices=sub_verts,
        fine_triangles=sub_tris,
        depth=parent.depth + 1,
        face_status=child_status,
    )
    # Record how this child's cage vertices map back into the parent's cage
    # vertex indexing. The runtime uses this to subset the parent's cumulative
    # plastic displacement (per-cage-node) onto each child when a fracture
    # splits the parent — the resulting fragment stays bent along with its
    # inherited shape rather than snapping back to rest.
    child.parent_vertex_indices = used_vertex_indices.astype(np.int64).copy()
    return child
