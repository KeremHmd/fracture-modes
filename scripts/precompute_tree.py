# CLI to precompute a multi-level FractureTree and pickle it to disk.
# Each node in the tree can optionally carry its own elastic + plastic mode
# bundles, so the united runtime (multi_impact_gui.py) can route impacts to
# plastic bending on the hit piece before falling back to fracture.
#
# Usage:
#   python scripts\precompute_tree.py [MESH] [--materials FILE.npz] --depth N --output OUT.pkl
#                                     [--num-faces N] [--num-modes K]
#                                     [--volume-threshold V] [--min-tets-for-fracture T]
#                                     [--delaunay-remesh] [--no-surface-cleanup]
#                                     [--num-elastic-modes N] [--num-plastic-modes N]
#                                     [--poisson-ratio V]
#                                     [--default-E V] [--default-rho V]
#                                     [--plastic-max-iter N] [--plastic-tol V]

import argparse
import os
from datetime import datetime

import numpy as np
import igl
import tetgen

from context import fracture_utility as fracture
from context import gpytoolbox
from gpytoolbox.copyleft import lazy_cage


parser = argparse.ArgumentParser(
    description="Precompute a multi-level FractureTree and pickle to disk."
)
parser.add_argument("mesh", nargs="?", default="data/bunny_oded.obj")
parser.add_argument("--materials", default=None,
                    help="Optional MaterialField .npz. If absent, homogeneous.")
parser.add_argument("--num-faces", type=int, default=2000)
parser.add_argument("--num-modes", type=int, default=10)
parser.add_argument("--depth", type=int, default=1,
                    help="Tree depth. 1 = root only (equivalent to single-level "
                         "pipeline). 2 = root + one level of children. Etc.")
parser.add_argument("--volume-threshold", type=float, default=0.0,
                    help="Pieces with cage-tet volume below this are not "
                         "further fractured (atomic).")
parser.add_argument("--min-tets-for-fracture", type=int, default=10,
                    help="Pieces with fewer cage tets than this are atomic.")
parser.add_argument("--delaunay-remesh", action="store_true",
                    help="Run Dyer 2007 §3.1 Delaunay remesh on each node's "
                         "carved cap faces (and planar surface edges). The "
                         "per-face modification status is stored on each "
                         "FractureNode so the multi-impact GUI can paint "
                         "flips/splits.")
parser.add_argument("--no-surface-cleanup", action="store_true",
                    help="With --delaunay-remesh, restrict the algorithm to "
                         "cap-cap edges only (no surface-flip cleanup).")
parser.add_argument("--num-elastic-modes", type=int, default=0,
                    help="If > 0, build a per-node ElasticModes bundle "
                         "(needed by the plastic pipeline). Typical: 20-30.")
parser.add_argument("--num-plastic-modes", type=int, default=0,
                    help="If > 0 and materials has sigma_y, build a per-node "
                         "PlasticModes bundle. Requires --num-elastic-modes > 0. "
                         "Typical: 8-16.")
parser.add_argument("--poisson-ratio", type=float, default=0.3,
                    help="Global Poisson ratio for the elastic pipeline.")
parser.add_argument("--default-E", type=float, default=2.0e11,
                    help="Fallback Young's modulus (Pa) when materials has no E field.")
parser.add_argument("--default-rho", type=float, default=1.0e3,
                    help="Fallback density (kg/m^3) when materials has no rho field.")
parser.add_argument("--plastic-max-iter", type=int, default=10,
                    help="Max iterations per plastic-mode SOCP.")
parser.add_argument("--plastic-tol", type=float, default=1.0e-4,
                    help="Convergence tolerance per plastic-mode SOCP.")
parser.add_argument("--output", required=True,
                    help="Path to write the pickled tree.")
args = parser.parse_args()


print(f"Loading {args.mesh}...")
v_fine, f_fine = igl.read_triangle_mesh(args.mesh)
v_fine = gpytoolbox.normalize_points(v_fine)
v_cage, f_cage = lazy_cage(v_fine, f_fine, num_faces=args.num_faces)
tgen = tetgen.TetGen(v_cage, f_cage)
nodes, elements = tgen.tetrahedralize()
N_tets = elements.shape[0]
print(f"  {N_tets} tets")


materials = None
if args.materials is not None:
    materials = fracture.MaterialField.from_file(args.materials, num_tets=N_tets)
    print(f"  {repr(materials)}")
else:
    print("  homogeneous (no --materials)")


params = fracture.fracture_modes_parameters(num_modes=args.num_modes, verbose=False, d=3)

t_start = datetime.now()
tree = fracture.fracture_tree.FractureTree.from_mesh(
    v_fine=v_fine, f_fine=f_fine,
    cage_tet_nodes=nodes, cage_tet_elements=elements,
    materials=materials, params=params,
    depth_limit=args.depth,
    volume_threshold=args.volume_threshold,
    min_tets_for_fracture=args.min_tets_for_fracture,
    verbose=True,
    delaunay_remesh=args.delaunay_remesh,
    delaunay_kwargs=(
        {"clean_surface_edges": not args.no_surface_cleanup}
        if args.delaunay_remesh else None
    ),
    num_elastic_modes=args.num_elastic_modes,
    num_plastic_modes=args.num_plastic_modes,
    poisson_ratio=args.poisson_ratio,
    default_E=args.default_E,
    default_rho=args.default_rho,
    plastic_max_iter=args.plastic_max_iter,
    plastic_tol=args.plastic_tol,
)
t_end = datetime.now()


os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
tree.save(args.output)
print(f"\nSaved: {args.output}")
print(f"Total nodes: {tree.num_nodes()}")
print(f"Precomputation time: {(t_end - t_start).total_seconds():.1f} s")
