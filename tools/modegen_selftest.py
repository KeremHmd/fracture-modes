# End-to-end self-test: for every profile, build a small tree on a coarse cage
# with a synthetic banded material (all fields set), then fire escalating
# impacts with every tool and report which branches were exercised.
#
# Usage:
#   python tools\modegen_selftest.py [--profiles NAME ...] [--num-faces N] [--depth N]
#                                    [--impacts N] [--mesh PATH] [--keep] [--full]
#
# --full keeps the profiles' own mode counts (slow); default shrinks them.

import argparse
import os
import time
import traceback

import numpy as np

import _common

parser = argparse.ArgumentParser(description="modegen end-to-end self-test.")
parser.add_argument("--profiles", nargs="*", default=None)
parser.add_argument("--num-faces", type=int, default=2000)
parser.add_argument("--depth", type=int, default=2)
parser.add_argument("--impacts", type=int, default=8)
parser.add_argument("--mesh", default=os.path.join(_common.ROOT, "data", "bunny_oded.obj"))
parser.add_argument("--keep", action="store_true", help="Save trees to out/selftest/.")
parser.add_argument("--full", action="store_true", help="Do not shrink mode counts.")
_common.add_plugin_args(parser)
_common.add_version_arg(parser)
args = parser.parse_args()

from modegen import ImpactEvent, Profile  # noqa: E402
from modegen.precompute import build_tree  # noqa: E402
from modegen.runtime import PluginWorld, inward_direction, sample_surface_point  # noqa: E402

FAST = {
    "precompute.fracture.params.num_modes": 4,
    "precompute.fracture.params.num_cells": 8,
    "precompute.elastic.params.num_modes": 12,
    "precompute.plastic.params.num_modes": 6,
}

reg = _common.registry_from_args(args)
names = args.profiles or Profile.available()
print(f"Coarse cage ({args.num_faces} faces) for {args.mesh} ...")
v, f, nodes, elements = _common.load_cage(args.mesh, args.num_faces)
mats = _common.banded_material(nodes, elements)
print(f"  {nodes.shape[0]} verts, {elements.shape[0]} tets; {mats!r}\n")

rows = []
for name in names:
    t0 = time.time()
    try:
        prof = Profile.load(name)
        if not args.full:
            for key, val in FAST.items():
                parent = key.rsplit(".", 2)[0]
                spec = prof.get(parent)
                if isinstance(spec, dict) and "plugin" in spec:
                    cls = reg.get({"fracture": "fracture.generator", "elastic": "elastic.generator",
                                   "plastic": "plastic.generator"}[parent.split(".")[1]], spec["plugin"])
                    if key.rsplit(".", 1)[1] in cls.PARAMS:
                        prof.set(key, val)
        tree = build_tree(v, f, nodes, elements, materials=mats, profile=prof, registry=reg,
                          depth_limit=args.depth, verbose=False)
        if args.keep:
            os.makedirs(os.path.join(_common.ROOT, "out", "selftest"), exist_ok=True)
            tree.save(os.path.join(_common.ROOT, "out", "selftest", f"{name}.modegen.pkl"))
        world = PluginWorld.from_tree(tree, registry=reg)
        world.spawn(tree.root)
        rng = np.random.default_rng(0)
        tools = world.tool_names
        branches = {}
        for i in range(args.impacts):
            p, inst = sample_surface_point(world, rng)
            if p is None:
                break
            mag = 10.0 ** (5.0 + 3.0 * i / max(args.impacts - 1, 1))
            r = world.apply_impact(ImpactEvent(point=p, direction=inward_direction(inst, p),
                                               magnitude=mag, tool=tools[i % len(tools)]))
            branches[r.branch] = branches.get(r.branch, 0) + 1
            for _ in range(5):
                world.step(1.0 / 60.0)
        st = tree.modegen_stats
        rows.append((name, "PASS", time.time() - t0, tree.num_nodes(), st["plastic_nodes"],
                     len(st["errors"]), branches, len(world.active_instances)))
    except Exception:
        rows.append((name, "FAIL", time.time() - t0, 0, 0, 0, traceback.format_exc(), 0))

print(f"\n{'profile':16s} {'status':6s} {'sec':>6s} {'nodes':>5s} {'plast':>5s} {'gErr':>4s} {'inst':>4s}  branches")
for name, status, sec, nn, pn, ge, br, ni in rows:
    if status == "PASS":
        print(f"{name:16s} {status:6s} {sec:6.1f} {nn:5d} {pn:5d} {ge:4d} {ni:4d}  {br}")
    else:
        print(f"{name:16s} {status:6s} {sec:6.1f}\n{br}")
raise SystemExit(0 if all(r[1] == "PASS" for r in rows) else 1)
