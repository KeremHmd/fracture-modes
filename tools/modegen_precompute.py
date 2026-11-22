# Precompute a plugin-driven fracture tree (fracture + elastic + plastic mode families).
#
# Usage:
#   python tools\modegen_precompute.py [MESH] --profile NAME|PATH --output OUT.modegen.pkl
#                                       [--materials FILE.npz] [--depth N] [--num-faces N]
#                                       [--set key.path=value ...] [--plugin-dir DIR ...]
#                                       [--volume-threshold V] [--min-tets-for-fracture T]
#                                       [--delaunay-remesh] [--no-surface-cleanup]
#                                       [--dump-profile FILE.json] [--solver-verbose] [--quiet]
#
# Example overrides:
#   --set precompute.fracture.params.num_modes=20
#   --set precompute.plastic=null
#   --set "precompute.fracture.params.weights={\"plugin\": \"slash.grain\"}"

import argparse
import os
from datetime import datetime

import _common  # noqa: F401  (path setup)

parser = argparse.ArgumentParser(description="Plugin-driven tree precomputation.")
parser.add_argument("mesh", nargs="?", default=os.path.join(_common.ROOT, "data", "bunny_oded.obj"))
parser.add_argument("--profile", default="realistic", help="Profile name (profiles/) or path.")
parser.add_argument("--set", action="append", default=[], help="Profile override key.path=value.")
parser.add_argument("--materials", default=None, help="MaterialField .npz (sigma_y needed for plastic).")
parser.add_argument("--depth", type=int, default=2)
parser.add_argument("--num-faces", type=int, default=2000)
parser.add_argument("--volume-threshold", type=float, default=0.0)
parser.add_argument("--min-tets-for-fracture", type=int, default=10)
parser.add_argument("--delaunay-remesh", action="store_true")
parser.add_argument("--no-surface-cleanup", action="store_true")
parser.add_argument("--output", required=True)
parser.add_argument("--dump-profile", default=None, help="Write the merged profile JSON here.")
parser.add_argument("--solver-verbose", action="store_true")
parser.add_argument("--quiet", action="store_true")
_common.add_plugin_args(parser)
_common.add_version_arg(parser)
args = parser.parse_args()

from modegen import Profile  # noqa: E402
from modegen.precompute import build_tree  # noqa: E402

reg = _common.registry_from_args(args)
profile = Profile.load(args.profile).apply_overrides(args.set)
errors = profile.validate(reg)
if errors:
    raise SystemExit("Profile is invalid:\n  " + "\n  ".join(errors))
print(f"Profile: {profile.name}  ({profile.source})")
if args.dump_profile:
    profile.save(args.dump_profile)

print(f"Loading {args.mesh} (cage faces {args.num_faces}) ...")
v_fine, f_fine, nodes, elements = _common.load_cage(args.mesh, args.num_faces)
print(f"  {nodes.shape[0]} cage verts, {elements.shape[0]} tets")

materials = None
if args.materials:
    from fracture_utility.material_field import MaterialField
    materials = MaterialField.from_file(args.materials, num_tets=elements.shape[0])
    print(f"  {materials!r}")
elif profile.get("precompute.plastic"):
    print("  no --materials: plastic generator needs sigma_y, plastic modes will be skipped")

t0 = datetime.now()
tree = build_tree(
    v_fine, f_fine, nodes, elements, materials=materials, profile=profile, registry=reg,
    depth_limit=args.depth, volume_threshold=args.volume_threshold,
    min_tets_for_fracture=args.min_tets_for_fracture,
    delaunay_remesh=args.delaunay_remesh,
    delaunay_kwargs=({"clean_surface_edges": not args.no_surface_cleanup}
                     if args.delaunay_remesh else None),
    verbose=not args.quiet, solver_verbose=args.solver_verbose,
)
os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
tree.save(args.output)
print(f"\nSaved {args.output}  ({tree.num_nodes()} nodes, "
      f"{(datetime.now() - t0).total_seconds():.1f} s)")
for err in tree.modegen_stats["errors"]:
    print("  generator error:", err)
