# Headless impact runner: fire random or scripted impacts on a modegen tree,
# step physics in between, dump OBJs + a JSON log. Useful for batch tests and
# offline content generation.
#
# Usage:
#   python tools\modegen_headless.py --tree TREE.modegen.pkl
#                                    [--impacts N | --script impacts.json]
#                                    [--tool NAME] [--magnitude V] [--direction inward|X,Y,Z]
#                                    [--steps-after N] [--dt V] [--no-gravity] [--ground-y V]
#                                    [--profile NAME|PATH] [--set key=value ...]
#                                    [--seed N] [--output-dir DIR] [--no-obj]
#
# Script format: a JSON list of objects with optional keys
#   point [x,y,z] (default: random surface point), direction "inward"|[x,y,z],
#   magnitude, tool, params {...}, steps_after

import argparse
import json
import os
from datetime import datetime

import numpy as np

import _common

parser = argparse.ArgumentParser(description="modegen headless impact runner.")
parser.add_argument("--tree", required=True)
parser.add_argument("--impacts", type=int, default=5)
parser.add_argument("--script", default=None)
parser.add_argument("--tool", default=None)
parser.add_argument("--magnitude", type=float, default=5.0e6)
parser.add_argument("--direction", default="inward")
parser.add_argument("--steps-after", type=int, default=10)
parser.add_argument("--dt", type=float, default=1.0 / 60.0)
parser.add_argument("--no-gravity", action="store_true")
parser.add_argument("--ground-y", type=float, default=None)
parser.add_argument("--profile", default=None)
parser.add_argument("--set", action="append", default=[])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output-dir", default=None)
parser.add_argument("--no-obj", action="store_true")
_common.add_plugin_args(parser)
_common.add_version_arg(parser)
args = parser.parse_args()

from fracture_utility.fracture_tree import FractureTree  # noqa: E402
from modegen import ImpactEvent, Profile  # noqa: E402
from modegen.runtime import PluginWorld, inward_direction, sample_surface_point  # noqa: E402

reg = _common.registry_from_args(args)
tree = FractureTree.load(args.tree)
_common.check_loaded_tree(tree, args.tree, reg)
profile = (Profile.load(args.profile) if args.profile else
           Profile.from_dict(tree.modegen_profile) if getattr(tree, "modegen_profile", None)
           else Profile())
profile.apply_overrides(args.set)
world = PluginWorld.from_profile(profile, reg)
if args.no_gravity:
    world.gravity = np.zeros(3)
world.ground_y = args.ground_y
world.spawn(tree.root)
rng = np.random.default_rng(args.seed)

out_dir = args.output_dir or os.path.join(
    _common.ROOT, "out", "modegen_headless", datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(out_dir, exist_ok=True)

if args.script:
    with open(args.script, "r", encoding="utf-8") as f:
        plan = json.load(f)
else:
    plan = [{} for _ in range(args.impacts)]


def save(tag):
    if args.no_obj:
        return
    import igl
    for k, inst in enumerate(world.active_instances):
        mesh = inst.world_combined_mesh()
        if mesh is not None:
            igl.write_obj(os.path.join(out_dir, f"{tag}_inst{k:03d}.obj"), *mesh)


log = []
save("00_initial")
for i, step in enumerate(plan, start=1):
    if step.get("point") is not None:
        point = np.asarray(step["point"], dtype=float)
        inst = world.find_hit_instance(point) or world.active_instances[0]
    else:
        point, inst = sample_surface_point(world, rng)
    if point is None:
        print("Nothing left to hit.")
        break
    d = step.get("direction", args.direction)
    if isinstance(d, str) and d == "inward":
        direction = inward_direction(inst, point)
    else:
        direction = np.asarray(d if not isinstance(d, str) else [float(x) for x in d.split(",")], dtype=float)
    ev = ImpactEvent(point=point, direction=direction,
                     magnitude=float(step.get("magnitude", args.magnitude)),
                     tool=step.get("tool", args.tool), params=step.get("params", {}))
    r = world.apply_impact(ev, verbose=True)
    for _ in range(int(step.get("steps_after", args.steps_after))):
        world.step(args.dt)
    log.append(dict(index=i, tool=r.tool, branch=r.branch, reason=r.reason,
                    point=point.tolist(), magnitude=ev.magnitude,
                    instances=len(world.active_instances), time=world.time))
    save(f"{i:02d}_{r.branch}")

with open(os.path.join(out_dir, "log.json"), "w", encoding="utf-8") as f:
    json.dump(log, f, indent=2)
counts = {}
for e in log:
    counts[e["branch"]] = counts.get(e["branch"], 0) + 1
print(f"\n{len(log)} impacts: {counts}; {len(world.active_instances)} instances; output in {out_dir}")
