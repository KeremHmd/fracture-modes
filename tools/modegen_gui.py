# Interactive multi-impact GUI for modegen trees, with selectable tools (impact plugins).
#
# Usage:
#   python tools\modegen_gui.py --tree TREE.modegen.pkl
#                               [--profile NAME|PATH] [--set key.path=value ...]
#                               [--add-tool NAME=PLUGIN ...] [--plugin-dir DIR ...]
#                               [--magnitude V] [--seed N]
#                               [--point X Y Z] [--sample-index N]
#
# Root samples are seeded (same --seed -> same points on every tree built from
# the same mesh), so "Root sample #N" hits the identical spot across trees.
# --point adds a "Fire at --point" button for an exact world coordinate.
#
# The runtime half of the profile stored in the tree is used unless --profile
# is given (then its runtime half replaces it; precomputed modes are unchanged).

import argparse
import os
from datetime import datetime

import numpy as np

import _common
from _scene import depth_color, plastic_scalar

parser = argparse.ArgumentParser(description="modegen interactive impact GUI.")
parser.add_argument("--tree", required=True)
parser.add_argument("--profile", default=None, help="Runtime profile override.")
parser.add_argument("--set", action="append", default=[])
parser.add_argument("--add-tool", action="append", default=[], help="NAME=PLUGIN (repeatable).")
parser.add_argument("--magnitude", type=float, default=5.0e6)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--point", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="Exact world point for the 'Fire at --point' button.")
parser.add_argument("--sample-index", type=int, default=0, help="Initial root sample index.")
_common.add_plugin_args(parser)
_common.add_version_arg(parser)
args = parser.parse_args()

import igl  # noqa: E402
import polyscope as ps  # noqa: E402
import polyscope.imgui as psim  # noqa: E402

from fracture_utility.fracture_tree import FractureTree  # noqa: E402
from modegen import ImpactEvent, Profile  # noqa: E402
from modegen.runtime import PluginWorld, inward_direction, sample_surface_point  # noqa: E402

reg = _common.registry_from_args(args)
tree = FractureTree.load(args.tree)
_common.check_loaded_tree(tree, args.tree, reg)
if args.profile:
    profile = Profile.load(args.profile)
elif getattr(tree, "modegen_profile", None) is not None:
    profile = Profile.from_dict(tree.modegen_profile)
else:
    print("Tree has no modegen profile (baseline tree?) - using the realistic runtime.")
    profile = Profile()
profile.apply_overrides(args.set)
for spec in args.add_tool:
    name, plugin = spec.split("=", 1)
    profile.set(f"runtime.tools.{name}", {"plugin": plugin})
print(f"Tree: {tree.num_nodes()} nodes, depth_limit {tree.depth_limit}; runtime profile {profile.name}")

rng = np.random.default_rng(args.seed)
out_dir = os.path.join(_common.ROOT, "out", "modegen_gui", datetime.now().strftime("%Y%m%d_%H%M%S"))

S = dict(world=None, tool=None, log_mag=float(np.log10(max(args.magnitude, 1.0))),
         dir_mode=0, last_point=None, last_inst=None, playing=False, gravity=True,
         ground=True, ground_y=-0.6, dt=1.0 / 60.0, heatmap=True, dirty=True,
         handles={}, counter=0, marker=None, sample_idx=int(args.sample_index))
DIR_MODES = ["inward", "-Y (down)", "+X", "-X", "-Z", "+Z"]
FIXED = {1: [0, -1, 0], 2: [1, 0, 0], 3: [-1, 0, 0], 4: [0, 0, -1], 5: [0, 0, 1]}

v_root, f_root = tree.root.fine_vertices, tree.root.fine_triangles
_srng = np.random.default_rng(args.seed)
_area = 0.5 * np.linalg.norm(np.cross(v_root[f_root[:, 1]] - v_root[f_root[:, 0]],
                                      v_root[f_root[:, 2]] - v_root[f_root[:, 0]]), axis=1)
_tri = f_root[_srng.choice(len(_area), size=256, p=_area / _area.sum())]
_r = _srng.random((256, 2))
_s = np.sqrt(_r[:, :1])
ROOT_SAMPLES = ((1 - _s) * v_root[_tri[:, 0]] + _s * (1 - _r[:, 1:]) * v_root[_tri[:, 1]]
                + _s * _r[:, 1:] * v_root[_tri[:, 2]])


def new_world():
    w = PluginWorld.from_profile(profile, reg)
    w.spawn(tree.root)
    S["tool"] = S["tool"] if S["tool"] in w.runtime.tools else w.runtime.default_tool
    return w


def sync(force=False):
    w = S["world"]
    current = {id(i): i for i in w.active_instances}
    for k in list(S["handles"]):
        if force or k not in current:
            S["handles"][k][1].remove()
            del S["handles"][k]
    for k, inst in current.items():
        mesh = inst.world_combined_mesh()
        if mesh is None:
            continue
        V, F = mesh
        h = S["handles"].get(k)
        if h is not None and h[2] == V.shape[0]:
            h[1].update_vertex_positions(V)
        else:
            if h is not None:
                h[1].remove()
            S["counter"] += 1
            handle = ps.register_surface_mesh(
                f"piece_{S['counter']:04d}_d{inst.node.depth}", V, F)
            handle.set_color(depth_color(inst.node.depth))
            S["handles"][k] = (inst, handle, V.shape[0])
            S["dirty"] = True
    if S["dirty"]:
        for inst, handle, _n in S["handles"].values():
            sc = plastic_scalar(inst) if S["heatmap"] else None
            if sc is not None and sc.max() > 0:
                handle.add_scalar_quantity("plastic |u|", sc, defined_on="vertices",
                                           enabled=True, cmap="viridis")
        S["dirty"] = False


def fire(point, inst):
    if point is None:
        print("No surface to hit.")
        return
    d = (inward_direction(inst, point) if S["dir_mode"] == 0
         else np.asarray(FIXED[S["dir_mode"]], dtype=float))
    ev = ImpactEvent(point=np.asarray(point, dtype=float), direction=d,
                     magnitude=10.0 ** S["log_mag"], tool=S["tool"])
    S["world"].apply_impact(ev, verbose=True)
    S["last_point"], S["last_inst"] = point, inst
    S["dirty"] = True
    sync()
    if S["marker"] is not None:
        S["marker"].remove()
    S["marker"] = ps.register_point_cloud("impact", np.asarray(point).reshape(1, 3))
    S["marker"].set_color((1.0, 0.0, 0.0))
    S["marker"].set_radius(0.012)


def callback():
    w = S["world"]
    psim.Text(f"Runtime profile: {profile.name}   instances: {len(w.active_instances)}   t={w.time:.2f}s")
    r = w.last_result
    if r is not None:
        psim.Text(f"LAST: [{r.tool}] {r.branch.upper()} ({r.reason})")
        if r.analysis is not None:
            a = r.analysis
            psim.Text(f"  class={a['material_class']} sigma_vm={a['sigma_vm_max']:.3g} "
                      f"sigma_y={a['sigma_y_local']:.3g} sigma_c={a['sigma_c_local']:.3g}")
        if r.plastic_result:
            pr = r.plastic_result
            psim.Text("  plastic: " + ", ".join(f"{k}={pr[k]}" for k in
                      ("winner", "modes_used", "modes_remaining", "dent_depth", "work_used",
                       "hardening_factor", "capped_modes") if k in pr))

    psim.Text("--- Tool ---")
    for i, name in enumerate(sorted(w.runtime.tools)):
        if i:
            psim.SameLine()
        label = f"[{name}]" if name == S["tool"] else name
        if psim.Button(label):
            S["tool"] = name
    psim.Text(f"  {w.runtime.tools[S['tool']].NAME}")
    _, S["log_mag"] = psim.SliderFloat("log10 magnitude", S["log_mag"], v_min=3.0, v_max=10.0)
    psim.Text(f"  magnitude = {10.0 ** S['log_mag']:.3g}")
    fr = w.runtime.fracture_response
    if fr is not None and "threshold" in fr.params:
        _, fr.params["threshold"] = psim.SliderFloat(
            "Fracture threshold", float(fr.params["threshold"]), v_min=0.5, v_max=500.0)
        if "severity_exponent" in fr.params:
            _, fr.params["severity_exponent"] = psim.SliderFloat(
                "Severity exponent (0=off)", float(fr.params["severity_exponent"]),
                v_min=0.0, v_max=2.0)
        if r is not None and getattr(r, "fracture_info", None):
            psim.Text(f"  last pieces: {r.fracture_info.get('n_pieces')}")
    for i, label in enumerate(DIR_MODES):
        if i:
            psim.SameLine()
        if psim.Button(f"[{label}]" if i == S["dir_mode"] else label):
            S["dir_mode"] = i

    psim.Text("--- Fire ---")
    if psim.Button("Random point on current pieces"):
        fire(*sample_surface_point(w, rng))
    psim.SameLine()
    if psim.Button("Again at last point") and S["last_point"] is not None:
        hit = w.find_hit_instance(S["last_point"])
        fire(S["last_point"], hit or S["last_inst"])
    ch, S["sample_idx"] = psim.InputInt("Root sample #", S["sample_idx"], step=1, step_fast=10)
    S["sample_idx"] = int(np.clip(S["sample_idx"], 0, len(ROOT_SAMPLES) - 1))
    if psim.Button("Fire at root sample"):
        p = ROOT_SAMPLES[S["sample_idx"]]
        fire(p, w.find_hit_instance(p) or w.active_instances[0])
    psim.Text(f"  sample point = {np.round(ROOT_SAMPLES[S['sample_idx']], 4)}")
    if args.point is not None and psim.Button(f"Fire at --point {args.point}"):
        p = np.asarray(args.point, dtype=float)
        fire(p, w.find_hit_instance(p) or w.active_instances[0])

    psim.Text("--- Simulation ---")
    _, S["playing"] = psim.Checkbox("Play (real time)", S["playing"])
    psim.SameLine()
    if psim.Button("Step"):
        w.step(S["dt"])
        sync()
    ch_g, S["gravity"] = psim.Checkbox("Gravity", S["gravity"])
    psim.SameLine()
    ch_f, S["ground"] = psim.Checkbox("Ground", S["ground"])
    _, S["ground_y"] = psim.SliderFloat("Ground y", S["ground_y"], v_min=-2.0, v_max=0.0)
    _, S["dt"] = psim.SliderFloat("dt", S["dt"], v_min=1.0 / 240.0, v_max=1.0 / 15.0)
    w.gravity = np.array(profile.get("runtime.world.gravity") if S["gravity"] else [0.0, 0.0, 0.0], dtype=float)
    w.ground_y = S["ground_y"] if S["ground"] else None
    if S["playing"]:
        w.step(S["dt"])
        sync()

    psim.Text("--- View ---")
    ch_h, S["heatmap"] = psim.Checkbox("Plastic heatmap", S["heatmap"])
    if ch_h:
        sync(force=True)
    if psim.Button("Reset camera"):
        ps.reset_camera_to_home_view()
    psim.SameLine()
    if psim.Button("Reset world"):
        S["world"] = new_world()
        sync(force=True)
    psim.SameLine()
    if psim.Button("Save OBJs"):
        os.makedirs(out_dir, exist_ok=True)
        for k, inst in enumerate(w.active_instances):
            mesh = inst.world_combined_mesh()
            if mesh is not None:
                igl.write_obj(os.path.join(out_dir, f"instance_{k:03d}.obj"), *mesh)
        print("Saved to", out_dir)

    psim.Text("--- History ---")
    for t, tool, branch, reason in w.history[-8:]:
        psim.Text(f"  t={t:6.2f} [{tool}] {branch} ({reason})")


ps.init()
ps.set_program_name("modegen")
ps.set_ground_plane_mode("none")
S["world"] = new_world()
sync(force=True)
ps.set_user_callback(callback)
ps.show()
