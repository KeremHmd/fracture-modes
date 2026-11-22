# CLI to fire a scripted impact through a precomputed elastic-modes bundle and
# visualize (or dump) the resulting free-vibration animation with damping.
#
# Usage:
#   python scripts\elastic_example.py --elastic-modes OUT.elastic.npz
#                                     [--contact-point X Y Z]
#                                     [--direction X Y Z] [--magnitude V]
#                                     [--num-frames N] [--dt V]
#                                     [--rayleigh-alpha V] [--rayleigh-beta V]
#                                     [--spread-radius V]
#                                     [--dump-frames] [--output-dir DIR]
#                                     [--preview] [--display-scale V]
#
# --preview opens a Polyscope window with a "Frame" slider. --dump-frames writes
# frame_XXX.obj for each time step into --output-dir. Neither flag is required;
# with neither, the script prints summary statistics and exits.

import argparse
import os
from datetime import datetime

import numpy as np
import igl

from context import fracture_utility as fracture


parser = argparse.ArgumentParser(
    description="Elastic-response projection of a scripted impact."
)
parser.add_argument("--elastic-modes", required=True,
                    help=".elastic.npz bundle from scripts/compute_elastic_modes.py.")
parser.add_argument("--contact-point", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="World-space contact point for the initial impact. Default: "
                         "highest-y vertex (a nice 'tap on top' location for the bunny). "
                         "In --preview you can also cycle through pre-sampled surface "
                         "points with the 'Impact index' input.")
parser.add_argument("--num-sample-points", type=int, default=256,
                    help="How many random surface points to pre-sample so the preview "
                         "GUI can cycle through impact locations without a re-run.")
parser.add_argument("--direction", type=float, nargs=3, default=[0.0, -1.0, 0.0],
                    metavar=("X", "Y", "Z"),
                    help="Impact direction (will be normalised).")
parser.add_argument("--magnitude", type=float, default=1.0,
                    help="Impulse magnitude (arbitrary units in this stage).")
parser.add_argument("--num-frames", type=int, default=120,
                    help="Number of animation frames.")
parser.add_argument("--dt", type=float, default=1.0 / 240.0,
                    help="Time step per frame (seconds).")
parser.add_argument("--rayleigh-alpha", type=float, default=0.5,
                    help="Mass-proportional damping coefficient (1/s).")
parser.add_argument("--rayleigh-beta", type=float, default=1.0e-5,
                    help="Stiffness-proportional damping coefficient (s).")
parser.add_argument("--spread-radius", type=float, default=0.0,
                    help="Spread the impact force over vertices within this "
                         "radius via Gaussian weights. 0 = nearest vertex only.")
parser.add_argument("--dump-frames", action="store_true",
                    help="Write frame_XXX.obj for each frame into --output-dir.")
parser.add_argument("--output-dir", default=None,
                    help="Output folder; default: out/elastic_example/<timestamp>/.")
parser.add_argument("--preview", action="store_true",
                    help="Open Polyscope with a Frame slider.")
parser.add_argument("--display-scale", type=float, default=1.0,
                    help="Multiplier applied to displacements for visualisation. "
                         "Modal amplitudes here are arbitrary until we calibrate.")
args = parser.parse_args()


print(f"Loading elastic bundle: {args.elastic_modes}")
em = fracture.ElasticModes.from_file(args.elastic_modes)
print(f"  {repr(em)}")

if args.contact_point is None:
    top_v = int(np.argmax(em.nodes[:, 1]))
    contact = em.nodes[top_v].copy()
    print(f"  --contact-point defaulted to vertex {top_v} (highest y): {contact}")
else:
    contact = np.asarray(args.contact_point, dtype=np.float64)

print(f"\nFiring impact: contact={contact}, direction={args.direction}, "
      f"magnitude={args.magnitude}")
print(f"  frames={args.num_frames}, dt={args.dt}, "
      f"rayleigh=(alpha={args.rayleigh_alpha}, beta={args.rayleigh_beta}), "
      f"spread_radius={args.spread_radius}")

t_start = datetime.now()
disps, strains, times = em.project_impact(
    contact_point=contact,
    direction=args.direction,
    magnitude=args.magnitude,
    num_frames=args.num_frames,
    dt=args.dt,
    rayleigh_alpha=args.rayleigh_alpha,
    rayleigh_beta=args.rayleigh_beta,
    spread_radius=args.spread_radius,
)
t_end = datetime.now()
print(f"  projected {args.num_frames} frames in "
      f"{(t_end - t_start).total_seconds():.3f} s")

disp_mag = np.linalg.norm(disps, axis=2)
vm_strain = np.stack([em.per_tet_von_mises_strain(strains[i]) for i in range(args.num_frames)])
print(f"\n  peak |displacement|: {disp_mag.max():.4g}  at frame "
      f"{int(np.unravel_index(disp_mag.argmax(), disp_mag.shape)[0])}")
print(f"  peak von-Mises strain: {vm_strain.max():.4g}  at frame "
      f"{int(np.unravel_index(vm_strain.argmax(), vm_strain.shape)[0])}")
print(f"  final |displacement|: {disp_mag[-1].max():.4g}   (should decay if damping is set)")


if args.dump_frames or args.output_dir:
    out_dir = args.output_dir or os.path.join(
        "out", "elastic_example", datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(out_dir, exist_ok=True)
    F_cage = igl.boundary_facets(em.elements.astype(np.int32))
    F_cage = F_cage[:, [0, 2, 1]]
    for i in range(args.num_frames):
        path = os.path.join(out_dir, f"frame_{i:04d}.obj")
        igl.write_obj(path, em.nodes + args.display_scale * disps[i], F_cage)
    print(f"\nDumped {args.num_frames} frame OBJs to {out_dir}")


if args.preview:
    import polyscope as ps
    import polyscope.imgui as psim

    print("\nOpening Polyscope preview.")
    print("  - 'Frame' slider     : cycle through the current impact's animation.")
    print("  - 'Impact index'     : select a pre-sampled surface point.")
    print("  - 'Apply / Random'   : re-project the impact using current params.")
    print("  - Magnitude / damping sliders re-project on change.")
    ps.init()
    ps.set_program_name("elastic_example preview")
    ps.set_ground_plane_mode("none")

    F_cage = igl.boundary_facets(em.elements.astype(np.int32))
    F_cage = F_cage[:, [0, 2, 1]]

    P = em.nodes[np.random.default_rng(0).choice(
        em.num_vertices, size=min(args.num_sample_points, em.num_vertices),
        replace=False,
    )]

    state = {
        "idx": 0,
        "impact_idx": 0,
        "display_scale": float(args.display_scale),
        "contact": contact.copy(),
        "direction": np.asarray(args.direction, dtype=np.float64).copy(),
        "magnitude": float(args.magnitude),
        "alpha": float(args.rayleigh_alpha),
        "beta": float(args.rayleigh_beta),
        "spread": float(args.spread_radius),
        "disps": disps,
        "strains": strains,
        "times": times,
        "disp_mag": disp_mag,
        "vm_strain": vm_strain,
    }

    ps_mesh = ps.register_surface_mesh(
        "elastic response",
        em.nodes + state["display_scale"] * state["disps"][state["idx"]],
        F_cage,
    )
    ps_mesh.add_scalar_quantity(
        "|displacement|", state["disp_mag"][state["idx"]],
        defined_on="vertices", enabled=True, cmap="viridis",
    )
    ps_contact = ps.register_point_cloud("contact", state["contact"].reshape(1, 3))
    ps_contact.set_color((1.0, 0.0, 0.0))
    ps_contact.set_radius(0.012)

    def re_project():
        d, s, t = em.project_impact(
            contact_point=state["contact"],
            direction=state["direction"],
            magnitude=state["magnitude"],
            num_frames=args.num_frames,
            dt=args.dt,
            rayleigh_alpha=state["alpha"],
            rayleigh_beta=state["beta"],
            spread_radius=state["spread"],
        )
        state["disps"] = d
        state["strains"] = s
        state["times"] = t
        state["disp_mag"] = np.linalg.norm(d, axis=2)
        state["vm_strain"] = np.stack(
            [em.per_tet_von_mises_strain(s[i]) for i in range(args.num_frames)]
        )
        state["idx"] = 0

    def refresh_mesh():
        ps_mesh.update_vertex_positions(
            em.nodes + state["display_scale"] * state["disps"][state["idx"]]
        )
        ps_mesh.add_scalar_quantity(
            "|displacement|", state["disp_mag"][state["idx"]],
            defined_on="vertices", enabled=True, cmap="viridis",
        )
        ps_contact.update_point_positions(state["contact"].reshape(1, 3))

    def cb():
        psim.Text(f"Contact: {state['contact']}")
        psim.Text(f"Magnitude={state['magnitude']:.3g}   spread={state['spread']:.3g}")
        psim.Text(f"Rayleigh alpha={state['alpha']:.3g}   beta={state['beta']:.3g}")
        psim.Text(f"t = {state['times'][state['idx']]:.4f} s")
        psim.Text(f"peak |u| this frame     = {state['disp_mag'][state['idx']].max():.4g}")
        psim.Text(f"peak vonMises this frame= {state['vm_strain'][state['idx']].max():.4g}")

        psim.Text("--- Playback ---")
        ch_i, state["idx"] = psim.SliderInt(
            "Frame", state["idx"], v_min=0, v_max=args.num_frames - 1,
        )
        ch_s, state["display_scale"] = psim.SliderFloat(
            "Display scale", state["display_scale"], v_min=0.01, v_max=100.0,
        )
        if ch_i or ch_s:
            refresh_mesh()

        psim.Text("--- Impact selection ---")
        ch_ii, state["impact_idx"] = psim.InputInt(
            "Impact index", state["impact_idx"], step=1, step_fast=10,
        )
        if ch_ii:
            state["impact_idx"] = max(0, min(P.shape[0] - 1, state["impact_idx"]))
        if psim.Button("Apply impact at index"):
            state["contact"] = P[state["impact_idx"]].copy()
            re_project()
            refresh_mesh()
        if psim.Button("Random impact (next index)"):
            state["impact_idx"] = (state["impact_idx"] + 1) % P.shape[0]
            state["contact"] = P[state["impact_idx"]].copy()
            re_project()
            refresh_mesh()

        psim.Text("--- Impact params (re-project on change) ---")
        ch_m, state["magnitude"] = psim.SliderFloat(
            "Magnitude", state["magnitude"], v_min=0.0, v_max=100.0,
        )
        ch_sp, state["spread"] = psim.SliderFloat(
            "Spread radius", state["spread"], v_min=0.0, v_max=0.5,
        )
        ch_a, state["alpha"] = psim.SliderFloat(
            "Rayleigh alpha", state["alpha"], v_min=0.0, v_max=50.0,
        )
        ch_b, state["beta"] = psim.SliderFloat(
            "Rayleigh beta", state["beta"], v_min=0.0, v_max=1e-2,
        )
        if ch_m or ch_sp or ch_a or ch_b:
            re_project()
            refresh_mesh()

    ps.set_user_callback(cb)
    ps.show()
