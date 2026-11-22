# CLI to precompute sparse plastic modes on a tet mesh, searching within a
# precomputed elastic-modes subspace to guarantee smooth (non-spike) modes.
#
# Solves the reduced SOCP
#   min sum_tet sigma_y * vol * || eps_phys(Phi_e @ alpha)|_tet ||_2
#   s.t. alpha_j^T alpha_k = 0 for previous modes j, alpha^T c_alpha = 1
# via the same MOSEK conic-solve machinery as the fracture pipeline, but with a
# per-element strain group-operator (d = 6) instead of the per-face jump operator.
# The elastic subspace eliminates rigid-body zero-modes and single-vertex spikes;
# every plastic mode is a smooth linear combination of low-frequency elastic modes.
#
# Requires materials.sigma_y to be set, and a precomputed .elastic.npz bundle
# from scripts/compute_elastic_modes.py that was built on the same mesh + cage.
#
# Usage:
#   python scripts\compute_plastic_modes.py [MESH] --materials FILE.npz
#                                           --elastic-modes ELASTIC.npz
#                                           --output OUT.plastic.npz
#                                           [--num-faces N] [--num-modes K]
#                                           [--max-iter N] [--tol V]
#                                           [--strain-threshold V]
#                                           [--preview] [--preview-amplitude A]

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
    description="Precompute sparse plastic modes on a tet mesh."
)
parser.add_argument("mesh", nargs="?", default="data/bunny_oded.obj")
parser.add_argument("--materials", required=True,
                    help="MaterialField .npz (must have sigma_y set).")
parser.add_argument("--elastic-modes", required=True,
                    help=".elastic.npz bundle from scripts/compute_elastic_modes.py. "
                         "Must have been computed on the same mesh + cage as this run "
                         "so num_tets matches.")
parser.add_argument("--num-faces", type=int, default=2000,
                    help="Cage face count. Must match the material file's tet count.")
parser.add_argument("--num-modes", type=int, default=8,
                    help="Number of plastic modes to compute (must be <= number of "
                         "elastic modes in --elastic-modes).")
parser.add_argument("--max-iter", type=int, default=10,
                    help="Max iterations per mode in the SOCP loop.")
parser.add_argument("--tol", type=float, default=1.0e-4,
                    help="Convergence tolerance for per-mode iteration.")
parser.add_argument("--strain-threshold", type=float, default=1.0e-6,
                    help="Diagnostic: tets with per-tet |eps_phys| * sigma_y * vol "
                         "above this fraction of the sum are counted as 'yielded'.")
parser.add_argument("--output", required=True, help="Output .plastic.npz path.")
parser.add_argument("--preview", action="store_true",
                    help="Open Polyscope after solving; cycle modes with a slider.")
parser.add_argument("--preview-amplitude", type=float, default=0.1,
                    help="Displacement scale factor for the preview window.")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()


print(f"Loading {args.mesh} ...")
v_fine, f_fine = igl.read_triangle_mesh(args.mesh)
v_fine = gpytoolbox.normalize_points(v_fine)
v_cage, f_cage = lazy_cage(v_fine, f_fine, num_faces=args.num_faces)
tgen = tetgen.TetGen(v_cage, f_cage)
nodes, elements = tgen.tetrahedralize()
N_tets = elements.shape[0]
N_verts = nodes.shape[0]
print(f"  cage: {N_verts} verts, {N_tets} tets")

materials = fracture.MaterialField.from_file(args.materials, num_tets=N_tets)
print(f"  {repr(materials)}")
if materials.sigma_y is None:
    raise SystemExit(
        f"{args.materials} has no sigma_y field. Re-author it with "
        "--sigma_y-inside / --sigma_y-outside in set_materials.py."
    )

print(f"\nLoading elastic subspace: {args.elastic_modes}")
em = fracture.ElasticModes.from_file(args.elastic_modes)
print(f"  {repr(em)}")
if em.num_tets != N_tets:
    raise SystemExit(
        f"num_tets mismatch: elastic bundle has {em.num_tets} tets but this "
        f"run's mesh + --num-faces={args.num_faces} tetrahedralises to {N_tets}."
    )


pm = fracture.PlasticModes(elastic_modes=em, materials=materials)

t_start = datetime.now()
print("\nAssembling B_group and reduced group operator ...")
pm.assemble()
t_asm = datetime.now()
print(f"  assembly: {(t_asm - t_start).total_seconds():.2f} s")

print(f"\nSolving SOCP for {args.num_modes} modes "
      f"(max_iter={args.max_iter}, tol={args.tol}) ...")
pm.compute_modes(
    num_modes=args.num_modes,
    max_iter=args.max_iter,
    tol=args.tol,
    strain_threshold=args.strain_threshold,
    verbose=True,
)
t_end = datetime.now()
print(f"  solve:    {(t_end - t_asm).total_seconds():.2f} s")
print(f"  total:    {(t_end - t_start).total_seconds():.2f} s")

os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
pm.to_file(args.output, elastic_modes_path=args.elastic_modes)
print(f"\nSaved: {args.output}")

print("\nPer-mode summary:")
for i in range(pm.modes.shape[1]):
    mag = pm.mode_strain_magnitude(i)
    print(f"  mode {i:3d}: energy={pm.mode_energies[i]:.4g}   "
          f"yielded_tets={pm.n_yielded_per_mode[i]:5d} / {N_tets}   "
          f"strain_max={mag.max():.4g}   strain_p95={np.percentile(mag, 95):.4g}")


if args.preview:
    import polyscope as ps
    import polyscope.imgui as psim

    print("\nOpening Polyscope preview ...")
    ps.init()
    ps.set_program_name("compute_plastic_modes preview")
    ps.set_ground_plane_mode("none")

    state = {"idx": 0, "amp": float(args.preview_amplitude)}

    ps_rest = ps.register_surface_mesh("rest cage", v_cage, f_cage)
    ps_rest.set_transparency(0.20)

    ps_def = ps.register_volume_mesh(
        f"mode {state['idx']}",
        nodes + pm.mode_displacement(state["idx"], state["amp"]),
        tets=elements,
    )

    def refresh():
        global ps_def
        ps_def.remove()
        disp = pm.mode_displacement(state["idx"], state["amp"])
        strain_mag = pm.mode_strain_magnitude(state["idx"])
        ps_def = ps.register_volume_mesh(
            f"mode {state['idx']}", nodes + disp, tets=elements,
        )
        ps_def.add_scalar_quantity(
            "|eps_phys| (Frobenius)", strain_mag,
            defined_on="cells", enabled=True, cmap="viridis",
        )
        ps_def.add_scalar_quantity(
            "|displacement|", np.linalg.norm(disp, axis=1),
            defined_on="vertices", enabled=False, cmap="viridis",
        )

    refresh()

    def cb():
        ch_i, state["idx"] = psim.SliderInt(
            "Mode index", state["idx"], v_min=0, v_max=pm.modes.shape[1] - 1,
        )
        ch_a, state["amp"] = psim.SliderFloat(
            "Amplitude", state["amp"], v_min=-2.0, v_max=2.0,
        )
        if ch_i or ch_a:
            refresh()
        psim.Text(f"mode {state['idx']}  energy={pm.mode_energies[state['idx']]:.4g}")
        psim.Text(f"yielded tets: {pm.n_yielded_per_mode[state['idx']]} / {N_tets}")
        mag = pm.mode_strain_magnitude(state["idx"])
        psim.Text(f"|eps_phys| max={mag.max():.4g}  p95={np.percentile(mag, 95):.4g}")

    ps.set_user_callback(cb)
    ps.show()
