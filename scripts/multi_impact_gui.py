# Interactive multi-impact demo GUI using a precomputed FractureTree.
# Routes impacts to plastic bending OR fracture on the hit piece, based on
# the piece's material class + sigma_y / sigma_c thresholds. Pieces retain
# their plastic state across fractures — a fragment that was already dented
# stays dented after it breaks off, and can be dented further.
#
# Usage:
#   python scripts\multi_impact_gui.py --tree TREE.pkl
#                                      [--threshold T] [--magnitude V]
#                                      [--spread-radius V] [--poisson-ratio V]
#                                      [--yield-threshold V] [--fracture-threshold V]

import argparse
import os
from datetime import datetime

import numpy as np
import igl
import polyscope as ps
import polyscope.imgui as psim

from context import fracture_utility as fracture


parser = argparse.ArgumentParser(
    description="Interactive multi-impact demo on a precomputed FractureTree."
)
parser.add_argument("--tree", required=True,
                    help="Path to a .pkl tree from precompute_tree.py.")
parser.add_argument("--threshold", type=float, default=10.0,
                    help="Sellan-style fracture projection threshold.")
parser.add_argument("--magnitude", type=float, default=1.0e6,
                    help="Physical impact magnitude (used by the plastic router).")
parser.add_argument("--spread-radius", type=float, default=0.05)
parser.add_argument("--poisson-ratio", type=float, default=0.3)
parser.add_argument("--yield-threshold", type=float, default=0.0,
                    help="Extra Pa above sigma_y required to fire a plastic mode.")
parser.add_argument("--fracture-threshold", type=float, default=0.0,
                    help="Extra Pa above sigma_c required to fire fracture.")
args = parser.parse_args()


print(f"Loading tree from {args.tree}...")
tree = fracture.fracture_tree.FractureTree.load(args.tree)
print(f"  depth_limit={tree.depth_limit}, total nodes={tree.num_nodes()}")

v_root = tree.root.fine_vertices
f_root = tree.root.fine_triangles
B, FI = igl.random_points_on_mesh(256, v_root, f_root)
B_full = np.vstack(
    (B[:, 0], B[:, 0], B[:, 0], B[:, 1], B[:, 1], B[:, 1], B[:, 2], B[:, 2], B[:, 2])
).T
P = (
    B_full[:, 0:3] * v_root[f_root[FI, 0], :]
    + B_full[:, 3:6] * v_root[f_root[FI, 1], :]
    + B_full[:, 6:9] * v_root[f_root[FI, 2], :]
)


out_dir = os.path.join(
    "out", "multi_impact_gui", datetime.now().strftime("%Y%m%d_%H%M%S")
)
os.makedirs(out_dir, exist_ok=True)


world = None
ind = 0
threshold = float(args.threshold)
magnitude = float(args.magnitude)
spread_radius = float(args.spread_radius)
poisson_ratio = float(args.poisson_ratio)
yield_threshold = float(args.yield_threshold)
fracture_threshold = float(args.fracture_threshold)
direction = np.array([0.0, -1.0, 0.0])
gravity_on = False
dt = 1.0 / 60.0
instance_meshes = []
ps_impact_pts = None
last_spawned = []
show_delaunay_ops = False
show_plastic_heatmap = True
last_impact_info = None


def reset_world():
    global world, instance_meshes
    for handle in instance_meshes:
        handle.remove()
    instance_meshes = []
    g = (0.0, -9.81, 0.0) if gravity_on else (0.0, 0.0, 0.0)
    world = fracture.world_state.WorldState(gravity=g, ground_y=None)
    world.spawn(tree.root)
    refresh_instance_meshes()


def refresh_instance_meshes():
    global instance_meshes
    for handle in instance_meshes:
        handle.remove()
    instance_meshes = []
    for k, inst in enumerate(world.active_instances):
        combined = inst.world_combined_mesh()
        if combined is None:
            continue
        V, F = combined
        h = ps.register_surface_mesh(
            f"instance_{k:03d}_depth{inst.node.depth}_pieces{len(inst.pieces)}",
            V, F,
        )
        c = [(0.7, 0.7, 0.75),
             (0.95, 0.85, 0.3),
             (0.95, 0.5, 0.2),
             (0.9, 0.3, 0.4)]
        h.set_color(c[min(inst.node.depth, len(c) - 1)])

        if show_delaunay_ops:
            fs_parts = []
            any_status = False
            for piece in inst.pieces:
                f = getattr(piece, "fine_triangles", None)
                if f is None or f.size == 0:
                    continue
                ps_status = getattr(piece, "face_status", None)
                if ps_status is None:
                    fs_parts.append(np.zeros(f.shape[0], dtype=np.int8))
                else:
                    ps_status = np.asarray(ps_status, dtype=np.int8)
                    if ps_status.shape[0] != f.shape[0]:
                        fs_parts.append(np.zeros(f.shape[0], dtype=np.int8))
                    else:
                        fs_parts.append(ps_status)
                        any_status = True
            if any_status and fs_parts:
                combined_fs = np.concatenate(fs_parts).astype(float)
                h.add_scalar_quantity(
                    "Delaunay ops (0=untouched, 1=flip, 2=split, 3=both)",
                    combined_fs,
                    defined_on="faces",
                    enabled=True,
                    cmap="phase",
                    vminmax=(0.0, 3.0),
                )

        if show_plastic_heatmap:
            plastic_parts = []
            any_plastic = False
            for piece in inst.pieces:
                v = getattr(piece, "fine_vertices", None)
                if v is None or v.size == 0:
                    continue
                st = inst.plastic_states.get(id(piece))
                c2f = getattr(piece, "cage_to_fine_map", None)
                if st is not None and c2f is not None:
                    per_v = np.linalg.norm(st["cum"][c2f], axis=1)
                    if per_v.max() > 0:
                        any_plastic = True
                    plastic_parts.append(per_v)
                else:
                    plastic_parts.append(np.zeros(v.shape[0]))
            if plastic_parts:
                combined_pl = np.concatenate(plastic_parts).astype(float)
                h.add_scalar_quantity(
                    "cumulative plastic |u|", combined_pl,
                    defined_on="vertices",
                    enabled=any_plastic,
                    cmap="viridis",
                )

        instance_meshes.append(h)


def fire_impact(point):
    global ps_impact_pts, last_spawned, last_impact_info
    new_insts = world.apply_impact(
        point, direction,
        threshold=threshold, magnitude=magnitude,
        spread_radius=spread_radius, poisson_ratio=poisson_ratio,
        yield_excess_threshold=yield_threshold,
        fracture_excess_threshold=fracture_threshold,
        verbose=True,
    )
    last_spawned = new_insts or []
    # Grab the routing info from whichever instance now holds it.
    last_impact_info = None
    for inst in (last_spawned or world.active_instances):
        info = getattr(inst, "last_impact_info", None)
        if info is not None:
            last_impact_info = info
            break
    refresh_instance_meshes()
    if ps_impact_pts is not None:
        ps_impact_pts.remove()
    ps_impact_pts = ps.register_point_cloud(
        "impact point", point.reshape(1, 3)
    )
    ps_impact_pts.set_color((1.0, 0.0, 0.0))
    ps_impact_pts.set_radius(0.012)


def _focus_camera_on(instances):
    if not instances:
        return
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for inst in instances:
        a, b = inst.world_aabb()
        lo = np.minimum(lo, a)
        hi = np.maximum(hi, b)
    center = (lo + hi) / 2.0
    extent = float(np.linalg.norm(hi - lo))
    if extent < 1e-6:
        extent = 1.0
    cam_pos = center + np.array([0.0, 0.0, max(extent * 2.0, 0.5)])
    ps.look_at(cam_pos, center)


ps.init()
ps.set_program_name("Multi-impact fracture demo")
ps.set_ground_plane_mode("none")
reset_world()


def callback():
    global ind, threshold, magnitude, spread_radius, poisson_ratio
    global yield_threshold, fracture_threshold
    global gravity_on, dt, show_delaunay_ops, show_plastic_heatmap
    psim.Text(f"Tree depth_limit: {tree.depth_limit}")
    psim.Text(f"Total tree nodes: {tree.num_nodes()}")
    psim.Text(f"Active instances: {len(world.active_instances)}")
    psim.Text(f"Sim time: {world.time:.3f} s")

    if last_impact_info is not None:
        li = last_impact_info
        branch = li.get("branch", "?")
        reason = li.get("reason", "?")
        psim.Text(f"LAST IMPACT: {branch.upper()}  ({reason})")
        a = li.get("analysis")
        if a is not None:
            psim.Text(f"  peak class={a['material_class']}  "
                      f"sigma_vm={a['sigma_vm_max']:.3g}  "
                      f"sigma_y={a['sigma_y_local']:.3g}  "
                      f"sigma_c={a['sigma_c_local']:.3g}")
            psim.Text(f"  excess_y={a['excess_y']:+.3g}   "
                      f"excess_c={a['excess_c']:+.3g}")
        pr = li.get("plastic_result")
        if pr is not None and pr.get("fired"):
            psim.Text(f"  plastic winner={pr['winner']}  "
                      f"history={pr['fired_history']}  "
                      f"remaining={pr['modes_remaining']}")

    psim.Text("--- Impact controls ---")
    ch_t, threshold = psim.SliderFloat(
        "Fracture projection threshold (Sellan)", threshold,
        v_min=0.0, v_max=1000.0,
    )
    _, magnitude = psim.SliderFloat(
        "Magnitude (Pa driver)", magnitude, v_min=0.0, v_max=1.0e8,
    )
    _, spread_radius = psim.SliderFloat(
        "Spread radius", spread_radius, v_min=0.0, v_max=0.5,
    )
    _, yield_threshold = psim.SliderFloat(
        "Yield excess threshold (Pa)", yield_threshold,
        v_min=0.0, v_max=5.0e8,
    )
    _, fracture_threshold = psim.SliderFloat(
        "Fracture excess threshold (Pa)", fracture_threshold,
        v_min=0.0, v_max=5.0e9,
    )
    ch_i, ind = psim.InputInt(
        "Impact index", ind, step=1, step_fast=10
    )
    if ch_i:
        ind = max(0, min(P.shape[0] - 1, ind))

    if psim.Button("Apply impact at index"):
        fire_impact(P[ind])
    if psim.Button("Random impact (next index)"):
        ind = (ind + 1) % P.shape[0]
        fire_impact(P[ind])

    psim.Text("--- Dynamics ---")
    ch_g, gravity_on = psim.Checkbox("Enable gravity", gravity_on)
    if ch_g:
        pass
    ch_dt, dt = psim.SliderFloat(
        "dt (seconds)", dt, v_min=1.0 / 240.0, v_max=1.0 / 10.0
    )
    if psim.Button("Step physics (one frame)"):
        old_gravity = world.gravity.copy()
        world.gravity = np.array(
            (0.0, -9.81, 0.0) if gravity_on else (0.0, 0.0, 0.0)
        )
        world.step(dt)
        refresh_instance_meshes()

    psim.Text("--- Visualization ---")
    ch_d, show_delaunay_ops = psim.Checkbox(
        "Paint Delaunay ops (flip/split per face)", show_delaunay_ops
    )
    ch_p, show_plastic_heatmap = psim.Checkbox(
        "Paint cumulative plastic |u|", show_plastic_heatmap
    )
    if ch_d or ch_p:
        refresh_instance_meshes()

    psim.Text("--- Camera ---")
    if psim.Button("Reset camera (fit all)"):
        ps.reset_camera_to_home_view()
    if psim.Button(f"Focus on last spawned piece(s) [{len(last_spawned)}]"):
        if last_spawned:
            _focus_camera_on(last_spawned)
        else:
            print("No last-spawned pieces to focus on yet.")
    if psim.Button("Focus on all active pieces"):
        _focus_camera_on(world.active_instances)

    psim.Text("--- Scene ---")
    if psim.Button("Reset to root"):
        reset_world()
    if psim.Button("Save active instances to disk"):
        for k, inst in enumerate(world.active_instances):
            combined = inst.world_combined_mesh()
            if combined is None:
                continue
            V, F = combined
            path = os.path.join(out_dir, f"instance_{k:03d}.obj")
            igl.write_obj(path, V, F)
        print(f"Saved {len(world.active_instances)} OBJ(s) to {out_dir}")


ps.set_user_callback(callback)
ps.show()
