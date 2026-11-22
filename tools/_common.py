"""Shared helpers for the modegen tools (path setup, mesh + material loading)."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402


def version_string():
    from modegen._version import PROFILE_SCHEMA_VERSION, TREE_FORMAT_VERSION, __version__
    from modegen.interfaces import PLUGIN_API_VERSION
    return (f"modegen {__version__} (plugin API {PLUGIN_API_VERSION}, profile schema "
            f"{PROFILE_SCHEMA_VERSION}, tree format {TREE_FORMAT_VERSION})")


def add_version_arg(parser):
    parser.add_argument("--version", action="version", version=version_string())


def check_loaded_tree(tree, path, reg):
    """Warn (never fail) when a loaded tree was written by a different version."""
    from modegen.compat import check_tree
    return check_tree(tree, source=path, registry=reg)


def add_plugin_args(parser):
    parser.add_argument("--plugin-dir", action="append", default=[],
                        help="Extra folder of plugin .py files (repeatable).")
    parser.add_argument("--no-user-plugins", action="store_true",
                        help="Do not load ./user_plugins.")


def registry_from_args(args):
    from modegen import get_registry
    reg = get_registry(extra_dirs=args.plugin_dir, include_user=not args.no_user_plugins)
    for path, err, _tb in reg.load_errors:
        print(f"[plugins] failed to load {path}: {err}")
    return reg


def load_cage(mesh_path, num_faces):
    import igl
    import tetgen
    import gpytoolbox
    from gpytoolbox.copyleft import lazy_cage

    v, f = igl.read_triangle_mesh(mesh_path)
    v = gpytoolbox.normalize_points(v)
    vc, fc = lazy_cage(v, f, num_faces=num_faces)
    nodes, elements = tetgen.TetGen(vc, fc).tetrahedralize()
    return v, f, nodes, elements


def banded_material(nodes, elements, num_bands=3, axis=1):
    """Synthetic 'helmet' material with every field set (used by the self-test)."""
    from fracture_utility.material_field import MaterialField

    c = nodes[elements].mean(axis=1)[:, axis]
    t = (c - c.min()) / max(c.max() - c.min(), 1e-12)
    inside = (np.clip((t * num_bands).astype(int), 0, num_bands - 1) % 2) == 1
    pick = lambda a, b: np.where(inside, a, b).astype(float)  # noqa: E731
    return MaterialField(
        num_tets=elements.shape[0],
        G_c=pick(5.0, 1.0), sigma_y=pick(8e7, 3e8), sigma_c=pick(2e8, 1e9),
        E=pick(5e10, 2e11), rho=pick(1000.0, 7800.0),
        material_class=np.where(inside, "elastic-plastic", "brittle"),
    )
