# Standalone cap-face detail enrichment (fractal value noise) for a carved mesh.
#
# Usage:
#   python scripts\enrich_detail.py --original ORIG.obj --carved IN.obj --output OUT.obj
#                                   [--amplitude A] [--frequency F]
#                                   [--octaves N] [--seed S]

import argparse
import os
import igl

from context import fracture_utility as fracture


parser = argparse.ArgumentParser(
    description="Procedural cap-face detail enrichment for fractured meshes."
)
parser.add_argument("--original", required=True,
                    help="Path to the original (un-fractured) input mesh.")
parser.add_argument("--carved", required=True,
                    help="Path to the carved (fractured) mesh to enrich.")
parser.add_argument("--output", required=True,
                    help="Where to write the enriched mesh.")
parser.add_argument("--amplitude", type=float, default=0.01,
                    help="Displacement amplitude in mesh units. 0.01 = ~1%% of bunny.")
parser.add_argument("--frequency", type=float, default=20.0,
                    help="Spatial frequency of the noise (cells per unit length).")
parser.add_argument("--octaves", type=int, default=4,
                    help="Number of fractal noise octaves.")
parser.add_argument("--seed", type=int, default=0,
                    help="RNG seed for reproducibility.")
args = parser.parse_args()


v_orig, f_orig = igl.read_triangle_mesh(args.original)
v_carved, f_carved = igl.read_triangle_mesh(args.carved)
print(f"Original: {v_orig.shape[0]} v, {f_orig.shape[0]} f")
print(f"Carved:   {v_carved.shape[0]} v, {f_carved.shape[0]} f")


cap_idx = fracture.cap_detail.cap_face_indices(
    v_carved, f_carved, v_orig, f_orig
)
print(f"Cap faces: {len(cap_idx)} of {f_carved.shape[0]}")

v_enriched = fracture.cap_detail.displace_cap_faces(
    v_carved, f_carved, v_orig, f_orig,
    amplitude=args.amplitude,
    frequency=args.frequency,
    octaves=args.octaves,
    seed=args.seed,
    cap_idx=cap_idx,
)


os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
igl.write_obj(args.output, v_enriched, f_carved)
print(f"Wrote: {args.output}")
print(
    f"Settings used: amplitude={args.amplitude}, frequency={args.frequency}, "
    f"octaves={args.octaves}, seed={args.seed}"
)
