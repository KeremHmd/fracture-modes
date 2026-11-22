# Interactive Polyscope GUI for authoring a per-tet MaterialField and saving to .npz.
#
# Paints G_c and, optionally, sigma_y / sigma_c / E / rho / material_class using an
# axis-split region selector. Any field the user leaves in "disabled" mode is not
# written to the .npz, so the fracture-only pipeline is unaffected.
#
# Usage:
#   python scripts\paint_materials.py [--mesh MESH] [--num-faces N] [--output FILE.npz]

import argparse
import os
from datetime import datetime
import numpy as np
import igl
import tetgen
import polyscope as ps
import polyscope.imgui as psim
from context import fracture_utility as fracture
from context import gpytoolbox
from gpytoolbox.copyleft import lazy_cage


parser = argparse.ArgumentParser(
    description="Interactive MaterialField authoring (axis-split regions)."
)
parser.add_argument("--mesh", default="data/bunny_oded.obj")
parser.add_argument("--num-faces", type=int, default=2000)
parser.add_argument(
    "--output",
    default=None,
    help="Output .npz path. Default: materials/<meshname>_<timestamp>.materials.npz",
)
args = parser.parse_args()

if args.output is None:
    mesh_name = os.path.splitext(os.path.basename(args.mesh))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output = os.path.join(
        "materials", f"{mesh_name}_{ts}.materials.npz"
    )
os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)


v_fine, f_fine = igl.read_triangle_mesh(args.mesh)
v_fine = gpytoolbox.normalize_points(v_fine)
v_cage, f_cage = lazy_cage(v_fine, f_fine, num_faces=args.num_faces)
tgen = tetgen.TetGen(v_cage, f_cage)
nodes, elements = tgen.tetrahedralize()
N_tets = elements.shape[0]
centroids = nodes[elements].mean(axis=1)
bbox_lo = centroids.min(axis=0)
bbox_hi = centroids.max(axis=0)
print(f"Loaded {args.mesh}: {N_tets} tets")
print(f"Centroid bbox: lo={bbox_lo}, hi={bbox_hi}")
print(f"Default save path: {args.output}")


MATERIAL_CLASS_CHOICES = ("rigid", "brittle", "elastic-plastic")

FIELDS = (
    {"key": "G_c",     "label": "G_c",     "kind": "scalar",   "default_in": 0.05,  "default_out": 1.0,   "enabled": True,  "vmin": 0.001, "vmax": 20.0},
    {"key": "sigma_y", "label": "sigma_y", "kind": "scalar",   "default_in": 2.0e8, "default_out": 5.0e7, "enabled": False, "vmin": 1e5,   "vmax": 1e10},
    {"key": "sigma_c", "label": "sigma_c", "kind": "scalar",   "default_in": 4.0e8, "default_out": 1.0e8, "enabled": False, "vmin": 1e5,   "vmax": 1e10},
    {"key": "E",       "label": "E",       "kind": "scalar",   "default_in": 2.0e11, "default_out": 7.0e10, "enabled": False, "vmin": 1e6,   "vmax": 5e11},
    {"key": "rho",     "label": "rho",     "kind": "scalar",   "default_in": 7800.0, "default_out": 100.0,  "enabled": False, "vmin": 1.0,   "vmax": 20000.0},
    {"key": "material_class", "label": "material_class", "kind": "class", "default_in": "elastic-plastic", "default_out": "brittle", "enabled": False},
)

state = {
    "axis": 1,
    "threshold": float((bbox_lo[1] + bbox_hi[1]) / 2),
    "inside_above": 1,
    "fields": {f["key"]: {"in": f["default_in"], "out": f["default_out"], "enabled": f["enabled"]} for f in FIELDS},
    "overlay_field": "G_c",
    "save_message": "",
}


def compute_mask():
    coord = centroids[:, state["axis"]]
    if state["inside_above"]:
        return coord > state["threshold"]
    return coord <= state["threshold"]


def compute_scalar_field(key):
    fs = state["fields"][key]
    return np.where(compute_mask(), fs["in"], fs["out"]).astype(float)


def compute_material_class_labels():
    fs = state["fields"]["material_class"]
    return np.where(compute_mask(), fs["in"], fs["out"])


def refresh_overlay():
    key = state["overlay_field"]
    if key == "material_class":
        labels = compute_material_class_labels()
        codes = np.array([MATERIAL_CLASS_CHOICES.index(l) for l in labels], dtype=float)
        ps_vol.add_scalar_quantity(
            "material_class (0=rigid,1=brittle,2=elastic-plastic)",
            codes, defined_on="cells", enabled=True, cmap="phase",
            vminmax=(0.0, len(MATERIAL_CLASS_CHOICES) - 1.0),
        )
    else:
        ps_vol.add_scalar_quantity(
            key, compute_scalar_field(key),
            defined_on="cells", enabled=True, cmap="viridis",
        )


ps.init()
ps.set_ground_plane_mode("none")
ps.set_program_name("paint_materials")
ps_vol = ps.register_volume_mesh("cage tets", nodes, tets=elements)
refresh_overlay()
ps.reset_camera_to_home_view()


def callback():
    psim.Text(f"Mesh: {args.mesh}  ({N_tets} tets)")
    psim.Text(f"Save path: {args.output}")

    psim.Text("--- Region selector ---")
    changed_axis, new_axis = psim.InputInt(
        "Axis (0=x, 1=y, 2=z)", state["axis"], step=1, step_fast=1
    )
    if changed_axis:
        new_axis = max(0, min(2, new_axis))
        state["axis"] = new_axis
        state["threshold"] = float(
            (bbox_lo[new_axis] + bbox_hi[new_axis]) / 2
        )
        refresh_overlay()

    lo = float(bbox_lo[state["axis"]])
    hi = float(bbox_hi[state["axis"]])
    ch, new_t = psim.SliderFloat(
        "Threshold", state["threshold"], v_min=lo, v_max=hi
    )
    if ch:
        state["threshold"] = new_t
        refresh_overlay()

    ch, new_dir = psim.InputInt(
        "Inside above? (1=above, 0=below)", state["inside_above"],
        step=1, step_fast=1,
    )
    if ch:
        state["inside_above"] = 1 if new_dir else 0
        refresh_overlay()

    psim.Text("--- Overlay field ---")
    for f in FIELDS:
        if psim.Button(f"Show: {f['label']}"):
            state["overlay_field"] = f["key"]
            refresh_overlay()
        psim.SameLine()
    psim.Text("")

    psim.Text("--- Field values ---")
    for f in FIELDS:
        key = f["key"]
        fs = state["fields"][key]
        ch, fs["enabled"] = psim.Checkbox(f"Save {f['label']}", fs["enabled"])
        if fs["enabled"]:
            if f["kind"] == "scalar":
                ch_in, fs["in"] = psim.SliderFloat(
                    f"{f['label']} inside", fs["in"],
                    v_min=f["vmin"], v_max=f["vmax"],
                )
                ch_out, fs["out"] = psim.SliderFloat(
                    f"{f['label']} outside", fs["out"],
                    v_min=f["vmin"], v_max=f["vmax"],
                )
                if (ch_in or ch_out) and state["overlay_field"] == key:
                    refresh_overlay()
            elif f["kind"] == "class":
                for idx, choice in enumerate(MATERIAL_CLASS_CHOICES):
                    if psim.Button(f"inside={choice}##in_{key}_{idx}"):
                        fs["in"] = choice
                        if state["overlay_field"] == key:
                            refresh_overlay()
                    psim.SameLine()
                psim.Text(f"(inside={fs['in']})")
                for idx, choice in enumerate(MATERIAL_CLASS_CHOICES):
                    if psim.Button(f"outside={choice}##out_{key}_{idx}"):
                        fs["out"] = choice
                        if state["overlay_field"] == key:
                            refresh_overlay()
                    psim.SameLine()
                psim.Text(f"(outside={fs['out']})")

    if psim.Button("Save MaterialField"):
        kwargs = {"num_tets": N_tets, "G_c": compute_scalar_field("G_c"),
                  "interface_rule": "min"}
        if state["fields"]["sigma_y"]["enabled"]:
            kwargs["sigma_y"] = compute_scalar_field("sigma_y")
        if state["fields"]["sigma_c"]["enabled"]:
            kwargs["sigma_c"] = compute_scalar_field("sigma_c")
        if state["fields"]["E"]["enabled"]:
            kwargs["E"] = compute_scalar_field("E")
        if state["fields"]["rho"]["enabled"]:
            kwargs["rho"] = compute_scalar_field("rho")
        if state["fields"]["material_class"]["enabled"]:
            kwargs["material_class"] = compute_material_class_labels()
        mf = fracture.MaterialField(**kwargs)
        mf.to_file(args.output)
        state["save_message"] = f"Saved: {args.output}"
        print(state["save_message"])
        print(repr(mf))

    psim.Text(state["save_message"])


ps.set_user_callback(callback)
ps.show()
