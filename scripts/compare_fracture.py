# Four-bunny side-by-side fracture comparison GUI (baseline / homo / hetero / hetero+detail).
#
# Usage:
#   python scripts\compare_fracture.py [MESH] --materials FILE.npz
#                                      [--num-faces N] [--num-modes K]

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


parser = argparse.ArgumentParser(
    description="Four-way fracture comparison: baseline / homo / hetero / hetero+enriched."
)
parser.add_argument("mesh", nargs="?", default="data/bunny_oded.obj")
parser.add_argument("--materials", required=True,
                    help="Path to a .npz MaterialField file.")
parser.add_argument("--num-faces", type=int, default=2000)
parser.add_argument("--num-modes", type=int, default=10)
args = parser.parse_args()

print("Loading...")


v_fine, f_fine = igl.read_triangle_mesh(args.mesh)
v_fine = gpytoolbox.normalize_points(v_fine)
v_cage, f_cage = lazy_cage(v_fine, f_fine, num_faces=args.num_faces)
tgen = tetgen.TetGen(v_cage, f_cage)
nodes, elements = tgen.tetrahedralize()
N_tets = elements.shape[0]
print(f"  {N_tets} tets")


materials = fracture.MaterialField.from_file(args.materials, num_tets=N_tets)
print(f"  {repr(materials)}")


B, FI = igl.random_points_on_mesh(1000, v_fine, f_fine)
B_full = np.vstack(
    (B[:, 0], B[:, 0], B[:, 0], B[:, 1], B[:, 1], B[:, 1], B[:, 2], B[:, 2], B[:, 2])
).T
P = (
    B_full[:, 0:3] * v_fine[f_fine[FI, 0], :]
    + B_full[:, 3:6] * v_fine[f_fine[FI, 1], :]
    + B_full[:, 6:9] * v_fine[f_fine[FI, 2], :]
)


bbox_lo = v_fine.min(axis=0)
bbox_hi = v_fine.max(axis=0)
spacing = (bbox_hi[0] - bbox_lo[0]) * 1.4
off_baseline = np.array([-1.5 * spacing, 0.0, 0.0])
off_homo     = np.array([-0.5 * spacing, 0.0, 0.0])
off_hetero   = np.array([ 0.5 * spacing, 0.0, 0.0])
off_enriched = np.array([ 1.5 * spacing, 0.0, 0.0])


out_dir = os.path.join("out", "compare", datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(out_dir, exist_ok=True)
print(f"Outputs will go to: {out_dir}")


params = fracture.fracture_modes_parameters(
    num_modes=args.num_modes, verbose=True, d=3
)
direction = np.array([1.0, 0.0, 0.0])

modes_homo = None
modes_hetero = None
computed = False
ind = 0
explode_scale = 0.0
impact_threshold = 10.0
last_contact_point = None

homo_cap_face_idx = None
hetero_cap_face_idx = None

enrich_amplitude = 0.01
enrich_frequency = 20.0
enrich_seed = 0
enrich_octaves = 4

fourth_V = None
fourth_F = None
fourth_cap_idx = None
fourth_vpiece = None
fourth_face_status = None
enriched_vertices = None
delaunay_applied = False
delaunay_dihedral_tol = 5.0
delaunay_max_iter = 2000
delaunay_clean_surface = True
show_delaunay_ops = True


ps.init()
ps.set_program_name("Fracture comparison")
ps.set_ground_plane_mode("none")

ps_baseline = ps.register_surface_mesh(
    "baseline (input)", v_fine + off_baseline, f_fine
)
ps_homo = None
ps_hetero = None
ps_enriched = None
ps_homo_caps = None
ps_hetero_caps = None
ps_enriched_caps = None
ps_impact_pts = None


def piece_exploded(verts, labels, scale):
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
    if not valid.any():
        return verts.copy()
    global_center = centers[valid].mean(axis=0)
    disp = verts.copy()
    for p in range(n_p):
        mask = labels == p
        if counts[p] > 0:
            disp[mask] += scale * (centers[p] - global_center)
    return disp


def _broken_piece_indices():
    if (modes_hetero is None
            or not modes_hetero.impact_projected
            or int(modes_hetero.n_pieces_after_impact) <= 1):
        return set()
    plai = np.asarray(modes_hetero.piece_labels_after_impact, dtype=int)
    pn = np.asarray(modes_hetero.piece_neighbors, dtype=int)
    broken = set()
    for k in range(pn.shape[0]):
        a, b = int(pn[k, 0]), int(pn[k, 1])
        if int(plai[a]) != int(plai[b]):
            broken.add(a)
            broken.add(b)
    return broken


def _attach_delaunay_ops_scalars(enabled=True):
    if fourth_face_status is None or fourth_F is None:
        return False
    broken = _broken_piece_indices()
    if not broken:
        return False
    full_status = fourth_face_status.astype(float).copy()
    face_piece = fourth_vpiece[fourth_F[:, 0]]
    in_broken = np.isin(face_piece, np.asarray(list(broken), dtype=face_piece.dtype))
    full_status[~in_broken] = 0.0
    if ps_enriched is not None:
        ps_enriched.add_scalar_quantity(
            "Delaunay ops (0=untouched, 1=flip, 2=split, 3=both)",
            full_status,
            defined_on="faces",
            enabled=enabled,
            cmap="phase",
            vminmax=(0.0, 3.0),
        )
    if ps_enriched_caps is not None and fourth_cap_idx is not None:
        cap_status = full_status[fourth_cap_idx]
        ps_enriched_caps.add_scalar_quantity(
            "Delaunay ops (0=untouched, 1=flip, 2=split, 3=both)",
            cap_status,
            defined_on="faces",
            enabled=enabled,
            cmap="phase",
            vminmax=(0.0, 3.0),
        )
    return True


def fourth_vertex_labels_after_impact():
    if modes_hetero is None or fourth_vpiece is None:
        return None
    if not modes_hetero.impact_projected:
        return None
    piece_lbls = np.asarray(modes_hetero.piece_labels_after_impact, dtype=int)
    return piece_lbls[fourth_vpiece.astype(int)]


def recompute_enriched():
    global enriched_vertices
    if fourth_V is None or fourth_F is None:
        return
    enriched_vertices = fracture.cap_detail.displace_cap_faces(
        fourth_V, fourth_F,
        v_fine, f_fine,
        amplitude=enrich_amplitude,
        frequency=enrich_frequency,
        octaves=enrich_octaves,
        seed=enrich_seed,
        cap_idx=fourth_cap_idx,
    )


def reset_fourth_to_hetero():
    global fourth_V, fourth_F, fourth_cap_idx, fourth_vpiece, delaunay_applied
    global fourth_face_status
    fourth_V = modes_hetero.fine_vertices.copy()
    fourth_F = modes_hetero.fine_triangles.copy()
    fourth_cap_idx = hetero_cap_face_idx.copy()
    P2V = modes_hetero.piece_to_fine_vertices_matrix
    fourth_vpiece = np.asarray(P2V.argmax(axis=1)).reshape(-1).astype(np.int64)
    fourth_face_status = np.zeros(fourth_F.shape[0], dtype=np.int8)
    delaunay_applied = False


def apply_delaunay_to_fourth():
    global fourth_V, fourth_F, fourth_cap_idx, fourth_vpiece, delaunay_applied
    global fourth_face_status
    if modes_hetero is None or fourth_V is None:
        return
    V_old = int(fourth_V.shape[0])
    F_old = int(fourth_F.shape[0])
    V_new, F_new, cap_new, vpl_new, face_status = (
        fracture.delaunay_remesh.delaunay_remesh_cap_faces(
            fourth_V, fourth_F, fourth_cap_idx,
            vertex_piece_labels=fourth_vpiece,
            max_iterations=delaunay_max_iter,
            dihedral_tol_deg=delaunay_dihedral_tol,
            clean_surface_edges=delaunay_clean_surface,
        )
    )
    fourth_V = V_new
    fourth_F = F_new
    fourth_cap_idx = cap_new
    fourth_vpiece = vpl_new if vpl_new is not None else fourth_vpiece
    fourth_face_status = face_status
    delaunay_applied = True
    n_flipped = int(((face_status & 1) != 0).sum())
    n_split = int(((face_status & 2) != 0).sum())
    n_untouched_cap = int((face_status[cap_new] == 0).sum())
    print(
        f"Delaunay remesh applied to 4th bunny: "
        f"verts {V_old} -> {V_new.shape[0]} (+{V_new.shape[0]-V_old}), "
        f"faces {F_old} -> {F_new.shape[0]} (+{F_new.shape[0]-F_old}), "
        f"cap faces: {len(cap_new)}"
    )
    print(
        f"  Per-face modification status: "
        f"flipped={n_flipped}, split={n_split}, "
        f"untouched caps={n_untouched_cap}"
    )


def register_fourth_bunny():
    global ps_enriched, ps_enriched_caps
    if ps_enriched is not None:
        ps_enriched.remove()
        ps_enriched = None
    if ps_enriched_caps is not None:
        ps_enriched_caps.remove()
        ps_enriched_caps = None
    ps_enriched = ps.register_surface_mesh(
        "heterogeneous + enriched",
        enriched_vertices + off_enriched,
        fourth_F,
    )
    ps_enriched_caps = ps.register_surface_mesh(
        "enriched cap faces (new)",
        enriched_vertices + off_enriched,
        fourth_F[fourth_cap_idx],
    )
    ps_enriched_caps.set_color((1.0, 0.9, 0.2))


def update_clones():
    if modes_homo is None or not modes_homo.impact_projected:
        if ps_homo is not None:
            ps_homo.update_vertex_positions(modes_homo.fine_vertices + off_homo)
            if ps_homo_caps is not None:
                ps_homo_caps.update_vertex_positions(modes_homo.fine_vertices + off_homo)
        if ps_hetero is not None:
            ps_hetero.update_vertex_positions(modes_hetero.fine_vertices + off_hetero)
            if ps_hetero_caps is not None:
                ps_hetero_caps.update_vertex_positions(modes_hetero.fine_vertices + off_hetero)
        if ps_enriched is not None and enriched_vertices is not None:
            ps_enriched.update_vertex_positions(enriched_vertices + off_enriched)
            if ps_enriched_caps is not None:
                ps_enriched_caps.update_vertex_positions(enriched_vertices + off_enriched)
        return

    h_labels = modes_homo.fine_vertex_labels_after_impact
    h_disp = piece_exploded(modes_homo.fine_vertices, h_labels, explode_scale)
    h_vminmax = (0, max(1.0, float(h_labels.max() + 1)))
    ps_homo.update_vertex_positions(h_disp + off_homo)
    ps_homo.add_scalar_quantity(
        "pieces", h_labels, enabled=True, cmap="phase", vminmax=h_vminmax,
    )
    if ps_homo_caps is not None:
        ps_homo_caps.update_vertex_positions(h_disp + off_homo)
        ps_homo_caps.add_scalar_quantity(
            "pieces", h_labels, enabled=True, cmap="phase", vminmax=h_vminmax,
        )

    het_labels = modes_hetero.fine_vertex_labels_after_impact
    het_disp = piece_exploded(modes_hetero.fine_vertices, het_labels, explode_scale)
    het_vminmax = (0, max(1.0, float(het_labels.max() + 1)))
    ps_hetero.update_vertex_positions(het_disp + off_hetero)
    ps_hetero.add_scalar_quantity(
        "pieces", het_labels, enabled=True, cmap="phase", vminmax=het_vminmax,
    )
    if ps_hetero_caps is not None:
        ps_hetero_caps.update_vertex_positions(het_disp + off_hetero)
        ps_hetero_caps.add_scalar_quantity(
            "pieces", het_labels, enabled=True, cmap="phase", vminmax=het_vminmax,
        )

    if ps_enriched is not None and enriched_vertices is not None:
        labels_4 = fourth_vertex_labels_after_impact()
        if labels_4 is None:
            ps_enriched.update_vertex_positions(enriched_vertices + off_enriched)
            if ps_enriched_caps is not None:
                ps_enriched_caps.update_vertex_positions(enriched_vertices + off_enriched)
        else:
            want_delaunay_overlay = (
                delaunay_applied
                and show_delaunay_ops
                and fourth_face_status is not None
                and int(modes_hetero.n_pieces_after_impact) > 1
            )
            vm4 = (0, max(1.0, float(labels_4.max() + 1)))
            enr_disp = piece_exploded(enriched_vertices, labels_4, explode_scale)
            ps_enriched.update_vertex_positions(enr_disp + off_enriched)
            ps_enriched.add_scalar_quantity(
                "pieces", labels_4,
                enabled=(not want_delaunay_overlay),
                cmap="phase", vminmax=vm4,
            )
            if ps_enriched_caps is not None:
                ps_enriched_caps.update_vertex_positions(enr_disp + off_enriched)
                ps_enriched_caps.add_scalar_quantity(
                    "pieces", labels_4,
                    enabled=(not want_delaunay_overlay),
                    cmap="phase", vminmax=vm4,
                )
            if want_delaunay_overlay:
                attached = _attach_delaunay_ops_scalars(enabled=True)
                if not attached and ps_enriched is not None:
                    ps_enriched.add_scalar_quantity(
                        "pieces", labels_4,
                        enabled=True, cmap="phase", vminmax=vm4,
                    )
                    if ps_enriched_caps is not None:
                        ps_enriched_caps.add_scalar_quantity(
                            "pieces", labels_4,
                            enabled=True, cmap="phase", vminmax=vm4,
                        )


def apply_impact(point):
    global ps_impact_pts, last_contact_point
    last_contact_point = point
    modes_homo.impact_projection(
        contact_point=point, threshold=impact_threshold, direction=direction
    )
    modes_hetero.impact_projection(
        contact_point=point, threshold=impact_threshold, direction=direction
    )
    update_clones()
    impact_pts = np.vstack(
        [point + off_baseline, point + off_homo, point + off_hetero, point + off_enriched]
    )
    if ps_impact_pts is not None:
        ps_impact_pts.remove()
    ps_impact_pts = ps.register_point_cloud("impact point", impact_pts)
    ps_impact_pts.set_color((1.0, 0.0, 0.0))
    ps_impact_pts.set_radius(0.015)


def compute_modes_button():
    global modes_homo, modes_hetero, computed
    global ps_homo, ps_hetero
    global ps_homo_caps, ps_hetero_caps
    global homo_cap_face_idx, hetero_cap_face_idx

    print("Computing homogeneous modes...")
    modes_homo = fracture.fracture_modes(nodes, elements)
    modes_homo.compute_modes(parameters=params)
    modes_homo.impact_precomputation(v_fine=v_fine, f_fine=f_fine)
    print("Computing heterogeneous modes...")
    modes_hetero = fracture.fracture_modes_heterogeneous(nodes, elements)
    modes_hetero.compute_modes(parameters=params, materials=materials)
    modes_hetero.impact_precomputation(v_fine=v_fine, f_fine=f_fine)

    homo_cap_face_idx = fracture.cap_detail.cap_face_indices(
        modes_homo.fine_vertices, modes_homo.fine_triangles, v_fine, f_fine
    )
    hetero_cap_face_idx = fracture.cap_detail.cap_face_indices(
        modes_hetero.fine_vertices, modes_hetero.fine_triangles, v_fine, f_fine
    )
    print(
        f"Cap faces (new from mesh_boolean): "
        f"homo {len(homo_cap_face_idx)}, hetero {len(hetero_cap_face_idx)}"
    )

    ps_homo = ps.register_surface_mesh(
        "homogeneous", modes_homo.fine_vertices + off_homo, modes_homo.fine_triangles
    )
    ps_hetero = ps.register_surface_mesh(
        "heterogeneous", modes_hetero.fine_vertices + off_hetero, modes_hetero.fine_triangles
    )

    reset_fourth_to_hetero()
    recompute_enriched()
    register_fourth_bunny()

    ps_homo_caps_local = ps.register_surface_mesh(
        "homo cap faces (new)",
        modes_homo.fine_vertices + off_homo,
        modes_homo.fine_triangles[homo_cap_face_idx],
    )
    ps_homo_caps_local.set_color((1.0, 0.9, 0.2))
    ps_hetero_caps_local = ps.register_surface_mesh(
        "hetero cap faces (new)",
        modes_hetero.fine_vertices + off_hetero,
        modes_hetero.fine_triangles[hetero_cap_face_idx],
    )
    ps_hetero_caps_local.set_color((1.0, 0.9, 0.2))

    global ps_homo_caps, ps_hetero_caps
    ps_homo_caps = ps_homo_caps_local
    ps_hetero_caps = ps_hetero_caps_local

    computed = True
    print("Done. Use the GUI controls to fire impacts.")


def callback():
    global ind, explode_scale, impact_threshold, computed
    global enrich_amplitude, enrich_frequency, enrich_seed
    global delaunay_dihedral_tol, show_delaunay_ops, delaunay_clean_surface

    psim.Text(f"Output dir: {out_dir}")
    psim.Text(f"Tets: {N_tets}  modes: {args.num_modes}  d=3")
    psim.Text(
        f"G_c: [{materials.G_c.min():.3g}, {materials.G_c.max():.3g}]  "
        f"rule={materials.interface_rule}"
    )

    if not computed:
        if psim.Button("Compute modes (homo + hetero)"):
            compute_modes_button()
        psim.Text("Click above to compute. Will take ~25 seconds.")
        return

    psim.Text(f"Input mesh:   {v_fine.shape[0]} v, {f_fine.shape[0]} f")
    psim.Text(
        f"Homo carved:   {modes_homo.fine_vertices.shape[0]} v, "
        f"{modes_homo.fine_triangles.shape[0]} f  "
        f"({len(homo_cap_face_idx)} new cap faces)"
    )
    psim.Text(
        f"Hetero carved: {modes_hetero.fine_vertices.shape[0]} v, "
        f"{modes_hetero.fine_triangles.shape[0]} f  "
        f"({len(hetero_cap_face_idx)} new cap faces)"
    )
    psim.Text(
        f"4th bunny:     {fourth_V.shape[0]} v, {fourth_F.shape[0]} f  "
        f"(Delaunay applied: {delaunay_applied})"
    )
    if delaunay_applied and fourth_face_status is not None:
        cap_status = fourth_face_status[fourth_cap_idx]
        n_cap = int(len(fourth_cap_idx))
        n_flipped = int(((cap_status & 1) != 0).sum())
        n_split = int(((cap_status & 2) != 0).sum())
        n_both = int(((cap_status & 3) == 3).sum())
        n_untouched = int((cap_status == 0).sum())
        psim.Text(
            f"  Delaunay ops on cap faces: "
            f"flipped {n_flipped}, split {n_split} "
            f"(both {n_both}), untouched {n_untouched} of {n_cap}"
        )
    if modes_homo.impact_projected:
        n_h = int(modes_homo.n_pieces_after_impact)
        n_het = int(modes_hetero.n_pieces_after_impact)
        marker = "  <-- DIFFER" if n_h != n_het else ""
        psim.Text(f"Last impact: homo {n_h} pieces, hetero {n_het} pieces{marker}")

    ch_t, impact_threshold = psim.SliderFloat(
        "Threshold", impact_threshold, v_min=0.0, v_max=1000.0
    )
    if ch_t and last_contact_point is not None:
        apply_impact(last_contact_point)

    ch_e, explode_scale = psim.SliderFloat(
        "Explode pieces", explode_scale, v_min=0.0, v_max=0.5
    )
    if ch_e:
        update_clones()

    psim.Text("--- Cap-face Delaunay remesh (4th bunny only) ---")
    ch_dh, delaunay_dihedral_tol = psim.SliderFloat(
        "Delaunay coplanarity tol (deg)", delaunay_dihedral_tol,
        v_min=0.1, v_max=30.0,
    )
    _, delaunay_clean_surface = psim.Checkbox(
        "Also flip planar surface edges (cleans split-surface slivers)",
        delaunay_clean_surface,
    )
    if not delaunay_applied:
        if psim.Button("Apply Delaunay remesh (Dyer 2007 Sec. 3.1)"):
            apply_delaunay_to_fourth()
            recompute_enriched()
            register_fourth_bunny()
            update_clones()
    else:
        if psim.Button("Revert 4th bunny to plain heterogeneous"):
            reset_fourth_to_hetero()
            recompute_enriched()
            register_fourth_bunny()
            update_clones()
        ch_show, show_delaunay_ops = psim.Checkbox(
            "Show flip/split overlay (overrides piece colors on 4th bunny)",
            show_delaunay_ops,
        )
        if ch_show:
            register_fourth_bunny()
            update_clones()

    psim.Text("--- Cap-detail enrichment (4th bunny only) ---")
    ch_a, enrich_amplitude = psim.SliderFloat(
        "Detail amplitude", enrich_amplitude, v_min=0.0, v_max=0.05
    )
    ch_f, enrich_frequency = psim.SliderFloat(
        "Detail frequency", enrich_frequency, v_min=1.0, v_max=80.0
    )
    ch_s, enrich_seed = psim.InputInt(
        "Detail seed", enrich_seed, step=1, step_fast=10
    )
    if ch_a or ch_f or ch_s:
        recompute_enriched()
        update_clones()

    psim.Text("--- Impact ---")
    ch_i, ind = psim.InputInt("Impact index", ind, step=1, step_fast=10)
    if ch_i:
        ind = max(0, min(P.shape[0] - 1, ind))
        apply_impact(P[ind])

    if psim.Button("Random impact (next index)"):
        ind = (ind + 1) % P.shape[0]
        apply_impact(P[ind])

    if psim.Button("Save fractured meshes"):
        if last_contact_point is None:
            print("No impact yet; nothing to save.")
        else:
            homo_path = os.path.join(out_dir, "output_homo.obj")
            hetero_path = os.path.join(out_dir, "output_hetero.obj")
            enriched_path = os.path.join(out_dir, "output_hetero_enriched.obj")
            modes_homo.write_segmented_output(homo_path)
            modes_hetero.write_segmented_output(hetero_path)
            igl.write_obj(enriched_path, enriched_vertices, fourth_F)
            print(f"Saved: {homo_path}")
            print(f"Saved: {hetero_path}")
            print(f"Saved: {enriched_path}  (Delaunay applied: {delaunay_applied})")


ps.set_user_callback(callback)
ps.show()
