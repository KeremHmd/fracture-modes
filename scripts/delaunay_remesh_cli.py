# Standalone Delaunay remesh (Dyer 2007 §3.1) of cap-face regions in a carved mesh.
#
# Usage:
#   python scripts\delaunay_remesh_cli.py --original ORIG.obj --carved IN.obj --output OUT.obj
#                                         [--max-iterations N] [--dihedral-tol-deg D]
#                                         [--split-delta DELTA] [--no-surface-cleanup]

import argparse
import os

import igl

from context import fracture_utility as fracture


parser = argparse.ArgumentParser(
    description="Delaunay remesh (cap-face restricted) per Dyer et al. 2007 §3.1."
)
parser.add_argument("--original", required=True,
                    help="Path to the original (un-fractured) input mesh.")
parser.add_argument("--carved", required=True,
                    help="Path to the carved (fractured) mesh.")
parser.add_argument("--output", required=True,
                    help="Where to write the remeshed mesh.")
parser.add_argument("--max-iterations", type=int, default=2000)
parser.add_argument("--dihedral-tol-deg", type=float, default=5.0,
                    help="Planar-vs-physical threshold for cap-edge "
                         "classification (degrees).")
parser.add_argument("--split-delta", type=float, default=None,
                    help="δ in Dyer Eq. (1). If unset, auto-picks from "
                         "shortest cap edge.")
parser.add_argument("--no-surface-cleanup", action="store_true",
                    help="Restrict to cap-cap edges only. By default, "
                         "planar surface-surface edges (where mesh_boolean "
                         "split an input triangle into subdivisions) are "
                         "also rewired by flips.")
args = parser.parse_args()


v_orig, f_orig = igl.read_triangle_mesh(args.original)
v_carved, f_carved = igl.read_triangle_mesh(args.carved)
print(f"Original: {v_orig.shape[0]} v, {f_orig.shape[0]} f")
print(f"Carved:   {v_carved.shape[0]} v, {f_carved.shape[0]} f")


cap_idx = fracture.cap_detail.cap_face_indices(
    v_carved, f_carved, v_orig, f_orig
)
print(f"Cap faces: {len(cap_idx)} of {f_carved.shape[0]}")

V_new, F_new, cap_idx_new, _, face_status = fracture.delaunay_remesh.delaunay_remesh_cap_faces(
    v_carved, f_carved, cap_idx,
    max_iterations=args.max_iterations,
    dihedral_tol_deg=args.dihedral_tol_deg,
    split_delta=args.split_delta,
    clean_surface_edges=not args.no_surface_cleanup,
)
n_flipped = int(((face_status & 1) != 0).sum())
n_split = int(((face_status & 2) != 0).sum())
n_untouched = int((face_status == 0).sum())
print(f"After remesh: {V_new.shape[0]} v, {F_new.shape[0]} f "
      f"({len(cap_idx_new)} cap faces, "
      f"+{V_new.shape[0] - v_carved.shape[0]} verts "
      f"+{F_new.shape[0] - f_carved.shape[0]} faces)")
print(f"  flipped: {n_flipped}   split: {n_split}   untouched: {n_untouched}")


os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
igl.write_obj(args.output, V_new, F_new)
print(f"Wrote: {args.output}")
