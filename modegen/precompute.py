"""Profile-driven recursive precomputation of a fracture tree.

Same recursion, gating and child construction as the frozen
``fracture_utility.fracture_tree.precompute_node``, but every mode family is
produced by the generator plugins named in the profile:

    precompute.fracture -> fracture.generator   (required; drives the tree topology)
    precompute.elastic  -> elastic.generator    (optional; impact analysis subspace)
    precompute.plastic  -> plastic.generator    (optional; needs elastic + sigma_y)

The resulting object is a normal ``FractureTree`` (so the baseline GUIs can
still open it) with extra attributes: ``modegen_profile`` (the full
merged profile dict), ``modegen_api_version`` and ``modegen_stats``, and since
0.3.0 the version metadata ``modegen_tree_format``, ``modegen_version``,
``modegen_profile_schema_version`` and ``modegen_plugins`` (names and
versions of the precompute plugins).
"""

import time

import numpy as np

from ._version import TREE_FORMAT_VERSION, __version__
from .interfaces import PLUGIN_API_VERSION, Plugin
from .profile import Profile
from .registry import get_registry


def precompute_plugin_versions(gens):
    """``{slot: [{category, name, version}, ...]}`` for every precompute plugin
    of a ``PrecomputeConfig``, nested hooks (weights, init) included."""
    out = {}
    for slot in ("fracture", "elastic", "plastic"):
        top = getattr(gens, slot, None)
        found, stack = [], [top]
        while stack:
            p = stack.pop(0)
            if not isinstance(p, Plugin):
                continue
            found.append(dict(category=p.CATEGORY, name=p.NAME, version=str(p.VERSION)))
            for v in p.params.values():
                stack.extend(v if isinstance(v, list) else [v])
        out[slot] = found
    return out


def build_tree(v_fine, f_fine, cage_nodes, cage_elements, materials=None,
               profile=None, registry=None, depth_limit=1, volume_threshold=0.0,
               min_tets_for_fracture=10, delaunay_remesh=False,
               delaunay_kwargs=None, verbose=True, solver_verbose=False):
    from fracture_utility.fracture_tree import FractureNode, FractureTree

    profile = profile or Profile()
    reg = registry or get_registry()
    gens = profile.build_precompute(reg)

    root = FractureNode(
        vertices=cage_nodes, elements=cage_elements, materials=materials,
        fine_vertices=v_fine, fine_triangles=f_fine, depth=0,
    )
    tree = FractureTree(root, depth_limit=depth_limit,
                        volume_threshold=volume_threshold,
                        min_tets_for_fracture=min_tets_for_fracture)
    opts = dict(depth_limit=int(depth_limit), volume_threshold=float(volume_threshold),
                min_tets=int(min_tets_for_fracture), delaunay_remesh=bool(delaunay_remesh),
                delaunay_kwargs=delaunay_kwargs, verbose=verbose,
                solver_verbose=solver_verbose)
    stats = dict(nodes=0, atomic=0, fracture_nodes=0, elastic_nodes=0,
                 plastic_nodes=0, errors=[], seconds=0.0)
    t0 = time.time()
    _build_node(root, gens, opts, stats)
    stats["seconds"] = round(time.time() - t0, 2)

    tree.modegen_profile = profile.to_dict()
    tree.modegen_api_version = PLUGIN_API_VERSION
    tree.modegen_stats = stats
    # Version metadata (tree format 1, added in 0.3.0; older trees simply lack it).
    tree.modegen_tree_format = TREE_FORMAT_VERSION
    tree.modegen_version = __version__
    tree.modegen_profile_schema_version = profile.schema_version
    tree.modegen_plugins = precompute_plugin_versions(gens)
    if verbose:
        print(f"modegen tree built in {stats['seconds']} s: {tree.num_nodes()} nodes "
              f"({stats['fracture_nodes']} fracturable, {stats['plastic_nodes']} with "
              f"plastic modes, {len(stats['errors'])} generator errors)")
    return tree


def _atomic(node, why, opts, stats):
    node.is_atomic = True
    stats["atomic"] += 1
    if opts["verbose"]:
        print(f"{'  ' * node.depth}[depth {node.depth}] atomic ({why})")


def _build_node(node, gens, opts, stats):
    from fracture_utility.fracture_tree import _build_child_from_piece

    stats["nodes"] += 1
    ind = "  " * node.depth
    if node.depth >= opts["depth_limit"]:
        return _atomic(node, "depth limit", opts, stats)
    if node.volume < opts["volume_threshold"]:
        return _atomic(node, f"volume {node.volume:.4g}", opts, stats)
    if node.num_tets() < opts["min_tets"]:
        return _atomic(node, f"{node.num_tets()} tets", opts, stats)
    if node.fine_vertices is None or node.fine_triangles is None:
        return _atomic(node, "missing fine mesh", opts, stats)

    ctx = dict(verbose=opts["solver_verbose"], depth=node.depth)
    if opts["verbose"]:
        print(f"{ind}[depth {node.depth}] {gens.fracture.NAME} on {node.num_tets()} tets ...")
    try:
        fm = gens.fracture.build(node, ctx)
        fm.impact_precomputation(
            v_fine=node.fine_vertices, f_fine=node.fine_triangles,
            delaunay_remesh=opts["delaunay_remesh"],
            delaunay_kwargs=opts["delaunay_kwargs"],
        )
    except Exception as exc:
        stats["errors"].append((node.depth, "fracture", repr(exc)))
        return _atomic(node, f"fracture generator failed: {exc!r}", opts, stats)
    node.modes_instance = fm
    node.modegen_generators = dict(fracture=gens.fracture.NAME)
    stats["fracture_nodes"] += 1

    em = None
    if gens.elastic is not None:
        try:
            em = gens.elastic.build(node, ctx)
            node.elastic_modes = em
            node.modegen_generators["elastic"] = gens.elastic.NAME
            stats["elastic_nodes"] += 1
        except Exception as exc:
            stats["errors"].append((node.depth, "elastic", repr(exc)))
            if opts["verbose"]:
                print(f"{ind}  elastic generator failed: {exc!r}")

    if (gens.plastic is not None and em is not None and node.materials is not None
            and node.materials.sigma_y is not None):
        try:
            from scipy.spatial import cKDTree
            pm = gens.plastic.build(node, em, ctx)
            node.plastic_modes = pm
            _, node.cage_to_fine_map = cKDTree(node.vertices).query(node.fine_vertices)
            # Barycentric lift data (0.4.0, optional runtime.render.lift); old
            # trees without it compute it lazily (modegen/lift.py).
            from .lift import node_barycentric, node_surface
            node_barycentric(node)
            node_surface(node)
            node.modegen_generators["plastic"] = gens.plastic.NAME
            stats["plastic_nodes"] += 1
        except Exception as exc:
            stats["errors"].append((node.depth, "plastic", repr(exc)))
            if opts["verbose"]:
                print(f"{ind}  plastic generator failed: {exc!r}")

    n_pieces = int(getattr(fm, "precomputed_num_pieces", 0))
    if opts["verbose"]:
        extras = ", ".join(f"{k}={v}" for k, v in node.modegen_generators.items())
        print(f"{ind}  -> {n_pieces} pieces ({extras})")
    if n_pieces <= 1:
        return _atomic(node, "1 piece", opts, stats)

    for piece_idx in range(n_pieces):
        child = _build_child_from_piece(node, piece_idx)
        if child is None:
            continue
        _build_node(child, gens, opts, stats)
        node.children.append(child)
