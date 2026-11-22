# One-shot precomputation of a united PlasticFractureAsset (elastic + plastic
# + fracture bundles + material). Bundles all three physics into a single .pkl
# so the united GUI has one thing to load.
#
# Usage:
#   python scripts\precompute_asset.py [MESH] --materials FILE.npz --output OUT.asset.pkl
#                                      [--num-faces N]
#                                      [--num-elastic-modes N]
#                                      [--num-plastic-modes N]
#                                      [--num-fracture-modes N]
#                                      [--poisson-ratio V]
#                                      [--default-E V] [--default-rho V]
#                                      [--plastic-max-iter N] [--plastic-tol V]

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
    description="Precompute a united PlasticFractureAsset (elastic + plastic + fracture)."
)
parser.add_argument("mesh", nargs="?", default="data/bunny_oded.obj")
parser.add_argument("--materials", required=True,
                    help="MaterialField .npz. Should have G_c, sigma_y, sigma_c, "
                         "and material_class set to exercise all routing branches.")
parser.add_argument("--num-faces", type=int, default=2000)
parser.add_argument("--num-elastic-modes", type=int, default=30)
parser.add_argument("--num-plastic-modes", type=int, default=16)
parser.add_argument("--num-fracture-modes", type=int, default=10)
parser.add_argument("--poisson-ratio", type=float, default=0.3)
parser.add_argument("--default-E", type=float, default=2.0e11)
parser.add_argument("--default-rho", type=float, default=1.0e3)
parser.add_argument("--plastic-max-iter", type=int, default=10)
parser.add_argument("--plastic-tol", type=float, default=1.0e-4)
parser.add_argument("--fracture-omega", type=float, default=0.01)
parser.add_argument("--output", required=True, help="Output .asset.pkl path.")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()


def _t():
    return datetime.now()


t_start = _t()
print(f"[{(_t()-t_start).total_seconds():.1f}s] Loading {args.mesh} ...")
v_fine, f_fine = igl.read_triangle_mesh(args.mesh)
v_fine = gpytoolbox.normalize_points(v_fine)
v_cage, f_cage = lazy_cage(v_fine, f_fine, num_faces=args.num_faces)
tgen = tetgen.TetGen(v_cage, f_cage)
nodes, elements = tgen.tetrahedralize()
N_tets = elements.shape[0]
N_verts = nodes.shape[0]
print(f"  cage: {N_verts} verts, {N_tets} tets")


print(f"[{(_t()-t_start).total_seconds():.1f}s] Loading materials: {args.materials}")
materials = fracture.MaterialField.from_file(args.materials, num_tets=N_tets)
print(f"  {repr(materials)}")
if materials.sigma_y is None:
    raise SystemExit(
        "materials.sigma_y required. Re-author with --sigma_y-inside in set_materials.py."
    )
if materials.sigma_c is None:
    print("  WARNING: materials.sigma_c is None — fracture branch cannot fire on "
          "sigma_c threshold. Add --sigma_c-inside to set_materials.py.")
if materials.material_class is None:
    print("  WARNING: materials.material_class is None — treating all tets as elastic-plastic.")


print(f"\n[{(_t()-t_start).total_seconds():.1f}s] Building ElasticModes ({args.num_elastic_modes} modes) ...")
em = fracture.ElasticModes(
    nodes=nodes, elements=elements, materials=materials,
    poisson_ratio=args.poisson_ratio,
    default_E=args.default_E, default_rho=args.default_rho,
)
em.assemble()
em.compute_modes(num_modes=args.num_elastic_modes, verbose=args.verbose)
print(f"  done. lowest omega^2 = {em.eigenvalues[0]:.4g}, freq_min = {em.frequencies_hz[0]:.4g} Hz")


print(f"\n[{(_t()-t_start).total_seconds():.1f}s] Building PlasticModes ({args.num_plastic_modes} modes) ...")
pm = fracture.PlasticModes(elastic_modes=em, materials=materials)
pm.assemble()
pm.compute_modes(
    num_modes=args.num_plastic_modes,
    max_iter=args.plastic_max_iter, tol=args.plastic_tol,
    verbose=args.verbose,
)
print(f"  done. energies: min={pm.mode_energies.min():.4g}  max={pm.mode_energies.max():.4g}")


print(f"\n[{(_t()-t_start).total_seconds():.1f}s] Building fracture modes ({args.num_fracture_modes} modes) ...")
fm = fracture.fracture_modes_heterogeneous(nodes, elements)
params = fracture.fracture_modes_parameters(
    num_modes=args.num_fracture_modes, verbose=args.verbose, d=3, omega=args.fracture_omega,
)
fm.compute_modes(parameters=params, materials=materials)
print(f"  computed {fm.modes.shape[1]} fracture modes.")

print(f"\n[{(_t()-t_start).total_seconds():.1f}s] Running fracture impact_precomputation ...")
fm.impact_precomputation(v_fine=v_fine, f_fine=f_fine)
print(f"  fine mesh (carved): {fm.fine_vertices.shape[0]} verts, {fm.fine_triangles.shape[0]} faces")


print(f"\n[{(_t()-t_start).total_seconds():.1f}s] Bundling asset ...")
asset = fracture.PlasticFractureAsset(
    elastic_modes=em, plastic_modes=pm, fracture_modes_obj=fm,
    material=materials,
)
print(f"  {repr(asset)}")

os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
asset.save(args.output)
print(f"\n[{(_t()-t_start).total_seconds():.1f}s] Saved: {args.output}")
