# Single-pipeline heterogeneous fracture-modes GUI (counterpart to scripts/fracture_gui.py).
#
# Usage:
#   python scripts\fracture_gui_heterogeneous.py [MESH] [--materials FILE.npz] [--num-faces N]

import argparse
import os
from datetime import datetime
import numpy as np
import igl
import polyscope as ps
import polyscope.imgui as psim
import tetgen
from context import fracture_utility as fracture
from context import gpytoolbox
from gpytoolbox.copyleft import lazy_cage


parser = argparse.ArgumentParser(description="Heterogeneous fracture-modes GUI.")
parser.add_argument("mesh", nargs="?", default="data/bunny_oded.obj")
parser.add_argument("--materials", default=None, help="Path to a .npz MaterialField file.")
parser.add_argument("--num-faces", type=int, default=2000)
args = parser.parse_args()

print("Loading GUI...")


v, f = igl.read_triangle_mesh(args.mesh)
v = gpytoolbox.normalize_points(v)

B, FI = igl.random_points_on_mesh(1000, v, f)
B = np.vstack(
    (B[:, 0], B[:, 0], B[:, 0], B[:, 1], B[:, 1], B[:, 1], B[:, 2], B[:, 2], B[:, 2])
).T
P = (
    B[:, 0:3] * v[f[FI, 0], :]
    + B[:, 3:6] * v[f[FI, 1], :]
    + B[:, 6:9] * v[f[FI, 2], :]
)
n = igl.per_face_normals(v, f, np.array([[0.0, 0.0, 0.0]]))
N = n[FI, :]

params = fracture.fracture_modes_parameters(num_modes=3, verbose=True, d=1)

v_fine = v
f_fine = f
v, f = lazy_cage(v_fine, f_fine, num_faces=args.num_faces)
tgen = tetgen.TetGen(v, f)
nodes, elements = tgen.tetrahedralize()
N_tets = elements.shape[0]
print(f"Tetrahedralized: {N_tets} tets.")


materials = None
if args.materials is not None:
    materials = fracture.MaterialField.from_file(args.materials, num_tets=N_tets)
    print(f"Loaded materials from {args.materials}")
    print(f"  {repr(materials)}")
else:
    print("No --materials given; this run is homogeneous (same as fracture_gui.py).")


out_dir = os.path.join("out", "gui", datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(out_dir, exist_ok=True)
print(f"Outputs will go to: {out_dir}")


show_full_impact = False
impact_text = ""
modes_text = ""
computed_modes = False
showing_modes = False
showing_input = True
showing_impact = False
ps_vol = []
ind = 0
impact_threshold = 10
modes = None
contact_point = nodes[1, :]
direction = np.array([1])
explode_scale = 0.0
input_v_count = v_fine.shape[0]
input_f_count = f_fine.shape[0]


ps.init()
ps.set_ground_plane_mode("none")
ps.set_transparency_mode("pretty")
ps.set_transparency_render_passes(2)
ps.set_program_name("Fracture Modes GUI (heterogeneous)")

ps_input_mesh = ps.register_surface_mesh("input mesh", v_fine, f_fine)
ps_cage = ps.register_surface_mesh("cage mesh", v, f)
ps_cage.set_transparency(0.4)

if materials is not None:
    ps_tets = ps.register_volume_mesh("cage tets (G_c)", nodes, tets=elements)
    ps_tets.add_scalar_quantity(
        "G_c", materials.G_c, defined_on="cells", enabled=True, cmap="viridis"
    )
    ps_tets.set_enabled(False)


def piece_exploded_positions(verts, labels, scale):
    if scale <= 0.0 or labels.size == 0:
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
    if valid.sum() == 0:
        return verts.copy()
    global_center = centers[valid].mean(axis=0)
    displaced = verts.copy()
    for p in range(n_p):
        mask = labels == p
        if counts[p] > 0:
            displaced[mask] += scale * (centers[p] - global_center)
    return displaced


def callback():
    global modes_text, computed_modes, showing_modes, showing_input, showing_impact
    global ps_input_mesh, ps_cage, ps_vol, modes
    global UU, ps_impact_mesh, ps_fracture_mesh, ps_impact_projected_mesh
    global ind, impact_threshold, impact_text
    global contact_point, direction
    global labels_fine_1d, fine_vertices_1d, fine_triangles_1d
    global modes_1d, labels_1d
    global off_tets_x, off_tets_x_exploded
    global off_x, off_y, off
    global show_full_impact, explode_scale

    psim.Text(f"Mesh: {args.mesh}  ({N_tets} tets)")
    if materials is not None:
        psim.Text(
            f"Materials: G_c [{materials.G_c.min():.3g}, {materials.G_c.max():.3g}], "
            f"rule={materials.interface_rule}"
        )
    else:
        psim.Text("Materials: <none> (homogeneous)")
    psim.Text(f"Output dir: {out_dir}")

    _, params.num_modes = psim.InputInt(
        "Number of modes", params.num_modes, step=1, step_fast=10
    )

    if psim.Button("Compute modes"):
        modes_text = "Computing modes..."
        if materials is not None:
            modes = fracture.fracture_modes_heterogeneous(nodes, elements)
            modes.compute_modes(parameters=params, materials=materials)
        else:
            modes = fracture.fracture_modes(nodes, elements)
            modes.compute_modes(parameters=params)
        modes.impact_precomputation(v_fine=v_fine, f_fine=f_fine)
        labels_fine_1d = modes.fine_labels.copy()
        fine_vertices_1d = modes.fine_vertices.copy()
        fine_triangles_1d = modes.fine_triangles.copy()
        modes_1d = modes.modes.copy()
        labels_1d = modes.labels.copy()
        UU = modes_1d.copy()
        UU = np.vstack((np.zeros((2 * UU.shape[0], UU.shape[1])), UU))
        off = 0.0 * modes.fine_vertices
        off_x = off.copy()
        off_y = off.copy()
        off_x[:, 0] = 1.2
        off_y[:, 2] = 1.2
        computed_modes = True
        modes_text = "Computed modes"

    if computed_modes:
        if psim.Button("Write segmented modes"):
            modes.write_segmented_modes(os.path.join(out_dir, "modes"))
            modes_text = f"Wrote modes to {out_dir}/modes_mode_*.obj"

    psim.Text(modes_text)

    if psim.Button("Show modes"):
        if showing_input:
            ps_input_mesh.remove()
            ps_cage.remove()
            showing_input = False
        ps_vol = []
        ps.set_transparency_mode("none")
        off_local = 0.0 * fine_vertices_1d
        off_x_local = off_local.copy()
        off_y_local = off_local.copy()
        off_x_local[:, 0] = 1.2
        off_y_local[:, 2] = 1.2
        for i in range(UU.shape[1]):
            mesh_name = f"Mode number {i}"
            scalar_map_name = f"Labels for mode number {i}"
            ps_vol.append(
                ps.register_surface_mesh(
                    mesh_name,
                    vertices=fine_vertices_1d
                    + (i // 4) * off_y_local
                    + (i % 4) * off_x_local,
                    faces=fine_triangles_1d,
                )
            )
            ps_vol[i].add_scalar_quantity(
                scalar_map_name,
                labels_fine_1d[:, i],
                enabled=True,
                cmap="phase",
                vminmax=(0, np.max(labels_fine_1d[:, i]) + 1.0),
            )
        ps.reset_camera_to_home_view()
        showing_modes = True

    if psim.Button("Show input"):
        if showing_modes:
            for i in range(len(ps_vol)):
                ps_vol[i].remove()
            showing_modes = False
        if showing_impact:
            ps_impact_mesh.remove()
            ps_fracture_mesh.remove()
            showing_impact = False
        if not showing_input:
            ps_input_mesh = ps.register_surface_mesh("input mesh", v_fine, f_fine)
            ps_cage = ps.register_surface_mesh("cage mesh", v, f)
            ps_cage.set_transparency(0.4)
            showing_input = True

    if psim.Button("Impact mode"):
        ps.set_transparency_mode("none")
        showing_impact = True
        if showing_modes:
            for i in range(len(ps_vol)):
                ps_vol[i].remove()
            showing_modes = False
        if showing_input:
            ps_input_mesh.remove()
            ps_cage.remove()
            showing_input = False
        off_local = 0.0 * modes.fine_vertices
        off_x = off_local.copy()
        off_y = off_local.copy()
        off_x[:, 0] = 1.2
        off_y[:, 2] = 1.2
        off_tets = 0.0 * modes.vertices
        off_tets_x = off_tets.copy()
        off_tets_x[:, 0] = 1.2
        off_tets_exploded = 0.0 * modes.exploded_vertices
        off_tets_x_exploded = off_tets_exploded.copy()
        off_tets_x_exploded[:, 0] = 1.2
        ps_impact_mesh = ps.register_volume_mesh(
            "impact mesh", modes.vertices, tets=modes.elements
        )
        if show_full_impact:
            ps_impact_projected_mesh = ps.register_volume_mesh(
                "projected impact mesh",
                modes.exploded_vertices - off_tets_x_exploded,
                tets=modes.exploded_elements,
            )
        else:
            ps_impact_projected_mesh = ps.register_volume_mesh(
                "projected impact mesh",
                modes.vertices - off_tets_x,
                tets=modes.elements,
            )
        ps_fracture_mesh = ps.register_surface_mesh(
            "fracture mesh", modes.fine_vertices - 2 * off_x, modes.fine_triangles
        )
        ps.reset_camera_to_home_view()

    if showing_impact:
        if modes is not None and hasattr(modes, "fine_vertices") and modes.fine_vertices is not None:
            psim.Text(
                f"Input mesh: {input_v_count} verts, {input_f_count} faces"
            )
            psim.Text(
                f"Carved mesh: {modes.fine_vertices.shape[0]} verts, "
                f"{modes.fine_triangles.shape[0]} faces"
            )
            psim.Text(
                f"Added by carving: "
                f"+{modes.fine_vertices.shape[0] - input_v_count} verts, "
                f"+{modes.fine_triangles.shape[0] - input_f_count} faces"
            )

        ch, impact_threshold = psim.SliderFloat(
            "Threshold", impact_threshold, v_min=0.0, v_max=1000.0
        )
        if ch and modes.impact_projected:
            modes.impact_projection(
                contact_point=contact_point,
                threshold=impact_threshold,
                direction=direction,
            )
            ps_fracture_mesh.add_scalar_quantity(
                "fracture", modes.fine_vertex_labels_after_impact, enabled=True
            )
            displaced = piece_exploded_positions(
                modes.fine_vertices,
                modes.fine_vertex_labels_after_impact,
                explode_scale,
            )
            ps_fracture_mesh.update_vertex_positions(displaced - 2 * off_x)

        ch_e, explode_scale = psim.SliderFloat(
            "Explode pieces", explode_scale, v_min=0.0, v_max=1.0
        )
        if ch_e and modes is not None and modes.impact_projected:
            displaced = piece_exploded_positions(
                modes.fine_vertices,
                modes.fine_vertex_labels_after_impact,
                explode_scale,
            )
            ps_fracture_mesh.update_vertex_positions(displaced - 2 * off_x)
        if psim.Button("Generate random impact"):
            contact_point = P[ind, :]
            direction = np.array([1])
            ind = ind + 1
            modes.impact_projection(
                contact_point=contact_point,
                threshold=impact_threshold,
                direction=direction,
            )
            if direction.shape[0] == 1:
                ps_impact_mesh.add_scalar_quantity(
                    "impact",
                    np.squeeze(modes.impact_vis),
                    enabled=True,
                    defined_on="vertices",
                )
                ps_impact_projected_mesh.add_scalar_quantity(
                    "projected impact",
                    modes.piece_to_tet_matrix @ modes.projected_impact,
                    enabled=True,
                    defined_on="cells",
                )
            else:
                ps_impact_mesh.add_scalar_quantity(
                    "impact",
                    np.linalg.norm(
                        np.reshape(modes.impact_vis, (-1, 3), order="F"), axis=1
                    ),
                    enabled=True,
                    defined_on="vertices",
                )
                ps_impact_projected_mesh.add_scalar_quantity(
                    "projected impact",
                    np.linalg.norm(
                        modes.piece_to_tet_matrix
                        @ np.reshape(modes.projected_impact, (-1, 3), order="F"),
                        axis=1,
                    ),
                    enabled=True,
                    defined_on="cells",
                )
            ps_fracture_mesh.add_scalar_quantity(
                "fracture", modes.fine_vertex_labels_after_impact, enabled=True
            )
            displaced = piece_exploded_positions(
                modes.fine_vertices,
                modes.fine_vertex_labels_after_impact,
                explode_scale,
            )
            ps_fracture_mesh.update_vertex_positions(displaced - 2 * off_x)
            impact_text = f"Impact time: {modes.t_impact * 1000} ms"
        if psim.Button("Save segmented output"):
            save_path = os.path.join(out_dir, "output.obj")
            modes.write_segmented_output(save_path)
            impact_text = f"Saved: {save_path}"

    psim.Text(impact_text)


ps.set_user_callback(callback)
ps.show()
