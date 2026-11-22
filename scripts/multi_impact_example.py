# Headless multi-impact demo on a precomputed FractureTree (fires two scripted impacts, writes OBJs).
#
# Usage:
#   python scripts\multi_impact_example.py --tree TREE.pkl [--threshold T] [--output DIR]

import argparse
import os
from datetime import datetime

import numpy as np
import igl

from context import fracture_utility as fracture


parser = argparse.ArgumentParser(
    description="Headless multi-impact demo using a precomputed FractureTree."
)
parser.add_argument("--tree", required=True,
                    help="Path to a .pkl tree from precompute_tree.py.")
parser.add_argument("--threshold", type=float, default=10.0,
                    help="Impact-projection threshold parameter.")
parser.add_argument("--output", default=None,
                    help="Output folder (auto-generated under out/ if omitted).")
args = parser.parse_args()


if args.output is None:
    out_dir = os.path.join(
        "out", "multi_impact", datetime.now().strftime("%Y%m%d_%H%M%S")
    )
else:
    out_dir = args.output
os.makedirs(out_dir, exist_ok=True)
print(f"Outputs will go to: {out_dir}")


print(f"Loading tree from {args.tree}...")
tree = fracture.fracture_tree.FractureTree.load(args.tree)
print(f"  depth_limit={tree.depth_limit}, total nodes={tree.num_nodes()}")


world = fracture.world_state.WorldState(gravity=(0.0, 0.0, 0.0), ground_y=None)
root_inst = world.spawn(tree.root)
print(f"Spawned root with {root_inst.node.num_tets()} tets, "
      f"{0 if root_inst.node.fine_vertices is None else root_inst.node.fine_vertices.shape[0]} fine verts")


def save_world_snapshot(world, tag):
    for k, instance in enumerate(world.active_instances):
        combined = instance.world_combined_mesh()
        if combined is None:
            continue
        V, F = combined
        path = os.path.join(out_dir, f"{tag}_instance_{k:03d}.obj")
        igl.write_obj(path, V, F)
    print(f"  wrote {len(world.active_instances)} instance OBJ(s) for tag '{tag}'")


v = tree.root.fine_vertices
f = tree.root.fine_triangles
B, FI = igl.random_points_on_mesh(64, v, f)
B_full = np.vstack(
    (B[:, 0], B[:, 0], B[:, 0], B[:, 1], B[:, 1], B[:, 1], B[:, 2], B[:, 2], B[:, 2])
).T
P = (
    B_full[:, 0:3] * v[f[FI, 0], :]
    + B_full[:, 3:6] * v[f[FI, 1], :]
    + B_full[:, 6:9] * v[f[FI, 2], :]
)
direction = np.array([1.0, 0.0, 0.0])

print("\n=== Initial state ===")
save_world_snapshot(world, "00_initial")

print("\n=== Impact #1 ===")
hit_point_1 = P[0]
print(f"Impact at world point {hit_point_1}")
new_after_1 = world.apply_impact(hit_point_1, direction, threshold=args.threshold, verbose=True)
print(f"Now {len(world.active_instances)} active instance(s)")
save_world_snapshot(world, "01_after_impact1")

print("\n=== Impact #2 ===")
hit_point_2 = P[1]
print(f"Impact at world point {hit_point_2}")
new_after_2 = world.apply_impact(hit_point_2, direction, threshold=args.threshold, verbose=True)
print(f"Now {len(world.active_instances)} active instance(s)")
save_world_snapshot(world, "02_after_impact2")

print(f"\nDone. Outputs in {out_dir}")
