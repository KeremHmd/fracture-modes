# CLI to precompute a linear-elastic modal subspace on a tet mesh.
#
# Assembles K and M (linear elasticity with per-tet Young's modulus E and density rho
# from an optional MaterialField, plus a global Poisson ratio) and solves
# K phi = omega^2 M phi via shift-invert ARPACK. Drops the 6 rigid-body modes and
# saves the remaining low-frequency deformation subspace to an .npz.
#
# Usage:
#   python scripts\compute_elastic_modes.py [MESH] --output OUT.elastic.npz
#                                           [--materials FILE.npz]
#                                           [--num-faces N] [--num-modes K]
#                                           [--poisson-ratio V]
#                                           [--default-E V] [--default-rho V]
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
    description="Precompute linear-elastic modes on a tet mesh."
)
parser.add_argument("mesh", nargs="?", default="data/bunny_oded.obj")
parser.add_argument("--materials", default=None,
                    help="Optional MaterialField .npz for per-tet E and rho. "
                         "If absent or the field is unset, default_E / default_rho apply.")
parser.add_argument("--num-faces", type=int, default=2000,
                    help="Cage face count. Must match downstream scripts.")
parser.add_argument("--num-modes", type=int, default=30,
                    help="Number of deformation modes to keep (after dropping rigid-body).")
parser.add_argument("--poisson-ratio", type=float, default=0.3,
                    help="Global Poisson ratio (per-material-class in future).")
parser.add_argument("--default-E", type=float, default=1.0e9,
                    help="Young's modulus (Pa) fallback when materials has no E field.")
parser.add_argument("--default-rho", type=float, default=1.0e3,
                    help="Density (kg/m^3) fallback when materials has no rho field.")
parser.add_argument("--output", required=True, help="Output .elastic.npz path.")
parser.add_argument("--keep-rigid-body", action="store_true",
                    help="Keep the 6 rigid-body modes in the output (default: dropped). "
                         "Useful as a sanity check: with this flag the first "
                         "6 modes should be near-zero-frequency translations and rotations, "
                         "and modes 7+ should be bending patterns.")
parser.add_argument("--preview", action="store_true",
                    help="Open Polyscope after solving; cycle through modes with the slider.")
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


materials = None
if args.materials is not None:
    materials = fracture.MaterialField.from_file(args.materials, num_tets=N_tets)
    print(f"  {repr(materials)}")
else:
    print("  no --materials; using default_E / default_rho for all tets")


em = fracture.ElasticModes(
    nodes=nodes, elements=elements,
    materials=materials,
    poisson_ratio=args.poisson_ratio,
    default_E=args.default_E,
    default_rho=args.default_rho,
)

t_start = datetime.now()
print(f"\nAssembling K, M (3V = {3 * N_verts} DOFs) ...")
em.assemble()
t_asm = datetime.now()
print(f"  assembly: {(t_asm - t_start).total_seconds():.2f} s")

print(f"\nSolving generalised eigenproblem for {args.num_modes} modes "
      f"(keep_rigid_body={args.keep_rigid_body}) ...")
em.compute_modes(num_modes=args.num_modes,
                 drop_rigid_body=not args.keep_rigid_body,
                 verbose=True)
t_end = datetime.now()
print(f"  solve:    {(t_end - t_asm).total_seconds():.2f} s")
print(f"  total:    {(t_end - t_start).total_seconds():.2f} s")

os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
em.to_file(args.output)
print(f"\nSaved: {args.output}")

print("\nFirst 10 frequencies (Hz):")
for i, f in enumerate(em.frequencies_hz[:10]):
    print(f"  mode {i:3d}: omega^2 = {em.eigenvalues[i]:.4g}   freq = {f:.4g} Hz")


if args.preview:
    import polyscope as ps
    import polyscope.imgui as psim

    print("\nOpening Polyscope preview ... use the 'Mode index' slider to cycle. Close window to exit.")
    ps.init()
    ps.set_program_name("compute_elastic_modes preview")
    ps.set_ground_plane_mode("none")

    state = {"idx": 0, "amp": float(args.preview_amplitude)}

    ps_rest = ps.register_surface_mesh("rest cage", v_cage, f_cage)
    ps_rest.set_transparency(0.25)

    ps_def = ps.register_volume_mesh(
        f"mode {state['idx']}", nodes + em.mode_displacement(state["idx"], state["amp"]),
        tets=elements,
    )

    def refresh():
        global ps_def
        ps_def.remove()
        disp = em.mode_displacement(state["idx"], state["amp"])
        ps_def = ps.register_volume_mesh(
            f"mode {state['idx']}", nodes + disp, tets=elements,
        )
        magnitude = np.linalg.norm(disp, axis=1)
        ps_def.add_scalar_quantity(
            "|displacement|", magnitude,
            defined_on="vertices", enabled=True, cmap="viridis",
        )

    refresh()

    def cb():
        ch_i, state["idx"] = psim.SliderInt(
            "Mode index", state["idx"], v_min=0, v_max=em.modes.shape[1] - 1,
        )
        ch_a, state["amp"] = psim.SliderFloat(
            "Amplitude", state["amp"], v_min=-1.0, v_max=1.0,
        )
        if ch_i or ch_a:
            refresh()
        psim.Text(f"omega^2 = {em.eigenvalues[state['idx']]:.4g}")
        psim.Text(f"freq    = {em.frequencies_hz[state['idx']]:.4g} Hz")

    ps.set_user_callback(cb)
    ps.show()
