# United impact-driven demo for a PlasticFractureAsset. Each impact is routed
# to plastic bending or brittle fracture based on material class + thresholds:
#
#   sigma_vm > sigma_c              -> FRACTURE branch (any non-rigid class)
#   elastic-plastic + sigma_vm > sigma_y + plastic modes remaining
#                                   -> PLASTIC branch (fires one mode)
#   elastic-plastic + plastic exhausted + still yielding
#                                   -> promote to FRACTURE
#   rigid                           -> no response
#   otherwise                       -> elastic bounce (no visible change here)
#
# Usage:
#   python scripts\plastic_fracture_gui.py --asset OUT.asset.pkl
#                                          [--num-sample-points N]
#                                          [--magnitude V] [--spread-radius V]
#                                          [--poisson-ratio V]
#                                          [--yield-threshold V] [--fracture-threshold V]
#                                          [--fracture-projection-threshold V]
#                                          [--display-scale V] [--explode-scale V]

import argparse

import numpy as np

from context import fracture_utility as fracture


parser = argparse.ArgumentParser(
    description="United PlasticFractureAsset demo (plastic + fracture routing)."
)
parser.add_argument("--asset", required=True, help=".asset.pkl from scripts/precompute_asset.py.")
parser.add_argument("--num-sample-points", type=int, default=256)
parser.add_argument("--magnitude", type=float, default=1.0e6)
parser.add_argument("--spread-radius", type=float, default=0.05)
parser.add_argument("--poisson-ratio", type=float, default=0.3)
parser.add_argument("--yield-threshold", type=float, default=0.0,
                    help="Extra Pa above sigma_y required to trigger plastic firing.")
parser.add_argument("--fracture-threshold", type=float, default=0.0,
                    help="Extra Pa above sigma_c required to trigger fracture.")
parser.add_argument("--fracture-projection-threshold", type=float, default=10.0,
                    help="Sellan-style projection threshold for the fracture branch.")
parser.add_argument("--display-scale", type=float, default=1.0)
parser.add_argument("--explode-scale", type=float, default=0.0,
                    help="If >0 after fracture, pieces are moved outward from the "
                         "global centroid to make fracture surfaces visible.")
args = parser.parse_args()


print(f"Loading asset: {args.asset}")
asset = fracture.PlasticFractureAsset.load(args.asset)
print(f"  {repr(asset)}")


import polyscope as ps
import polyscope.imgui as psim


print("\nOpening united demo. Impact index / Apply / Random cycle contact points.")
print("HUD reports the routing branch (plastic / fracture / promoted / none).")

ps.init()
ps.set_program_name("plastic_fracture_gui")
ps.set_ground_plane_mode("none")


V_fine_rest = asset.fm.fine_vertices
F_fine = asset.fm.fine_triangles

rng = np.random.default_rng(0)
n_samples = min(args.num_sample_points, V_fine_rest.shape[0])
P_samples = V_fine_rest[rng.choice(V_fine_rest.shape[0], size=n_samples, replace=False)]

state = {
    "impact_idx": 0,
    "contact": P_samples[0].copy(),
    "magnitude": float(args.magnitude),
    "spread": float(args.spread_radius),
    "poisson": float(args.poisson_ratio),
    "yield_threshold": float(args.yield_threshold),
    "fracture_threshold": float(args.fracture_threshold),
    "fracture_projection_threshold": float(args.fracture_projection_threshold),
    "display_scale": float(args.display_scale),
    "explode_scale": float(args.explode_scale),
}


def piece_exploded_positions(verts, labels, scale):
    if scale <= 0.0 or labels is None or labels.size == 0:
        return verts.copy()
    labels = labels.astype(int)
    n_p = int(labels.max()) + 1
    if n_p <= 1:
        return verts.copy()
    centers = np.zeros((n_p, 3))
    counts = np.zeros(n_p)
    for p in range(n_p):
        mask = labels == p
        c = int(mask.sum())
        if c > 0:
            centers[p] = verts[mask].mean(axis=0)
            counts[p] = c
    valid = counts > 0
    if not valid.any():
        return verts.copy()
    global_center = centers[valid].mean(axis=0)
    disp = verts.copy()
    for p in range(n_p):
        mask = labels == p
        if counts[p] > 0:
            disp[mask] += scale * (centers[p] - global_center)
    return disp


ps_mesh = ps.register_surface_mesh(
    "fine mesh (deformed rest pose)",
    asset.current_fine_vertices(display_scale=state["display_scale"]),
    F_fine,
)
ps_contact = ps.register_point_cloud("contact", state["contact"].reshape(1, 3))
ps_contact.set_color((1.0, 0.0, 0.0))
ps_contact.set_radius(0.012)


def refresh_mesh():
    verts_deformed = asset.current_fine_vertices(display_scale=state["display_scale"])
    labels = asset.current_fine_labels()
    if labels is not None:
        verts_show = piece_exploded_positions(verts_deformed, labels, state["explode_scale"])
    else:
        verts_show = verts_deformed
    ps_mesh.update_vertex_positions(verts_show)
    ps_contact.update_point_positions(state["contact"].reshape(1, 3))

    disp_mag = np.linalg.norm(asset.pm.cumulative_displacement, axis=1)
    lift = disp_mag[asset.cage_to_fine_map] if asset.cage_to_fine_map is not None else np.zeros(V_fine_rest.shape[0])
    ps_mesh.add_scalar_quantity(
        "cumulative plastic |u|", lift,
        defined_on="vertices", enabled=(labels is None), cmap="viridis",
    )
    if labels is not None:
        ps_mesh.add_scalar_quantity(
            "fracture pieces", labels.astype(float),
            defined_on="vertices", enabled=True, cmap="phase",
            vminmax=(0.0, float(labels.max() + 1)),
        )


def fire_impact():
    r = asset.apply_impact(
        contact=state["contact"],
        direction=[0.0, -1.0, 0.0],
        magnitude=state["magnitude"],
        spread_radius=state["spread"],
        poisson_ratio=state["poisson"],
        yield_excess_threshold=state["yield_threshold"],
        fracture_excess_threshold=state["fracture_threshold"],
        fracture_projection_threshold=state["fracture_projection_threshold"],
    )
    branch = r["branch"]
    if branch == "plastic":
        pr = r["plastic_result"]
        print(f"  [PLASTIC]   fired mode {pr['winner']}  "
              f"history={pr['fired_history']}  remaining={pr['modes_remaining']}")
    elif branch in ("fracture", "promoted_fracture"):
        fr = r["fracture_result"]
        print(f"  [{branch.upper()}]  {fr['n_pieces']} pieces  reason={r['reason']}")
    else:
        print(f"  [none]      reason={r['reason']}  "
              f"peak_class={r['peak_material_class']}  "
              f"sigma_vm={r['sigma_vm_max']:.3g}  "
              f"sigma_y={r['sigma_y_local']:.3g}  sigma_c={r['sigma_c_local']:.3g}")
    refresh_mesh()


def cb():
    r = asset.last_impact_result
    psim.Text(f"Contact: {state['contact']}")
    psim.Text(f"Fractured: {asset.is_fractured}   "
              f"Plastic fires: {len(asset.pm.fired_modes)} / {asset.pm.modes.shape[1]}")
    if r is not None:
        psim.Text(f"peak class = {r['peak_material_class']}   "
                  f"sigma_vm = {r['sigma_vm_max']:.3g}   "
                  f"sigma_y = {r['sigma_y_local']:.3g}   "
                  f"sigma_c = {r['sigma_c_local']:.3g}")
        psim.Text(f"excess_y = {r['excess_y']:+.3g}   excess_c = {r['excess_c']:+.3g}")
        psim.Text(f"LAST BRANCH: {r['branch'].upper()}   reason: {r['reason']}")
        if r["plastic_result"] is not None and r["plastic_result"].get("fired"):
            pr = r["plastic_result"]
            psim.Text(f"  plastic winner={pr['winner']}   history={pr['fired_history']}")
        if r["fracture_result"] is not None:
            fr = r["fracture_result"]
            psim.Text(f"  fracture pieces = {fr['n_pieces']}")

    psim.Text("--- Impact selection ---")
    ch_ii, state["impact_idx"] = psim.InputInt(
        "Impact index", state["impact_idx"], step=1, step_fast=10,
    )
    if ch_ii:
        state["impact_idx"] = max(0, min(P_samples.shape[0] - 1, state["impact_idx"]))
    if psim.Button("Apply impact at index"):
        state["contact"] = P_samples[state["impact_idx"]].copy()
        fire_impact()
    if psim.Button("Random impact (next index)"):
        state["impact_idx"] = (state["impact_idx"] + 1) % P_samples.shape[0]
        state["contact"] = P_samples[state["impact_idx"]].copy()
        fire_impact()

    psim.Text("--- Impact params ---")
    _, state["magnitude"] = psim.SliderFloat(
        "Magnitude", state["magnitude"], v_min=0.0, v_max=1.0e8,
    )
    _, state["spread"] = psim.SliderFloat(
        "Spread radius", state["spread"], v_min=0.0, v_max=0.5,
    )

    psim.Text("--- Thresholds ---")
    _, state["yield_threshold"] = psim.SliderFloat(
        "Plastic yield excess (Pa)", state["yield_threshold"],
        v_min=0.0, v_max=5.0e8,
    )
    _, state["fracture_threshold"] = psim.SliderFloat(
        "Fracture excess (Pa)", state["fracture_threshold"],
        v_min=0.0, v_max=5.0e9,
    )
    _, state["fracture_projection_threshold"] = psim.SliderFloat(
        "Fracture projection threshold (Sellan)", state["fracture_projection_threshold"],
        v_min=0.0, v_max=1000.0,
    )

    psim.Text("--- Display ---")
    ch_ds, state["display_scale"] = psim.SliderFloat(
        "Display scale", state["display_scale"], v_min=0.1, v_max=10.0,
    )
    ch_es, state["explode_scale"] = psim.SliderFloat(
        "Explode pieces (after fracture)", state["explode_scale"],
        v_min=0.0, v_max=1.0,
    )
    if ch_ds or ch_es:
        refresh_mesh()

    psim.Text("--- Reset ---")
    if psim.Button("Reset asset (unfire everything)"):
        asset.reset()
        refresh_mesh()


refresh_mesh()
ps.set_user_callback(cb)
ps.show()
