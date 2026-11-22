# Impact-driven demo for a precomputed plastic-modes bundle.
#
# Semantics (mirrors the fracture pipeline's discrete-event model): each impact
# that passes the yield threshold fires ONE plastic mode. The fired mode's
# shape is added to the cumulative rest pose with a fixed per-mode amplitude,
# giving a predictable dent. Successive impacts fire successive modes (picked
# by best projection score among unfired modes). Once all modes have fired,
# further impacts return "all_modes_exhausted" — max plastic deformation is
# reached (fracture would take over from here in the full pipeline).
#
# Usage:
#   python scripts\plastic_example.py --plastic-modes OUT.plastic.npz
#                                     --elastic-modes OUT.elastic.npz
#                                     --materials FILE.npz
#                                     [--num-sample-points N]
#                                     [--magnitude V] [--spread-radius V]
#                                     [--poisson-ratio V]
#                                     [--yield-threshold V]
#                                     [--mode-amplitude V]
#                                     [--display-scale V]

import argparse

import numpy as np
import igl

from context import fracture_utility as fracture


parser = argparse.ArgumentParser(
    description="Impact-driven plastic-modes demo (one impact = one mode fires)."
)
parser.add_argument("--plastic-modes", required=True,
                    help=".plastic.npz bundle from scripts/compute_plastic_modes.py.")
parser.add_argument("--elastic-modes", required=True,
                    help=".elastic.npz bundle it was computed on (same mesh + cage).")
parser.add_argument("--materials", required=True,
                    help="MaterialField .npz (must have sigma_y set).")
parser.add_argument("--num-sample-points", type=int, default=256,
                    help="Pre-sampled surface points cycled by the GUI Impact index.")
parser.add_argument("--magnitude", type=float, default=1.0e6,
                    help="Impulse magnitude. Bump higher if nothing yields.")
parser.add_argument("--spread-radius", type=float, default=0.05,
                    help="Gaussian spread of the impact force. 0 = single vertex.")
parser.add_argument("--poisson-ratio", type=float, default=0.3)
parser.add_argument("--yield-threshold", type=float, default=0.0,
                    help="Extra Pa above sigma_y required to trigger yielding "
                         "(0 = any excess fires; higher = stricter).")
parser.add_argument("--mode-amplitude", type=float, default=None,
                    help="Per-firing amplitude (0.05 * mesh_diag / max|mode| if unset).")
parser.add_argument("--display-scale", type=float, default=1.0)
args = parser.parse_args()


print(f"Loading elastic modes: {args.elastic_modes}")
em = fracture.ElasticModes.from_file(args.elastic_modes)
print(f"  {repr(em)}")

print(f"Loading plastic modes: {args.plastic_modes}")
pm = fracture.PlasticModes.from_file(args.plastic_modes, elastic_modes=em)
print(f"  {repr(pm)}")

print(f"Loading materials: {args.materials}")
materials = fracture.MaterialField.from_file(args.materials, num_tets=em.num_tets)
print(f"  {repr(materials)}")
if materials.sigma_y is None:
    raise SystemExit("materials.sigma_y is required.")

pm.reset_runtime()


import polyscope as ps
import polyscope.imgui as psim


print("\nOpening Polyscope demo.")
print("  - 'Impact index' + Apply / Random cycle contact points.")
print("  - Each impact past the yield threshold fires ONE plastic mode.")
print("  - After all modes fire, the mesh is at max plastic deformation.")

ps.init()
ps.set_program_name("plastic_example demo")
ps.set_ground_plane_mode("none")

F_cage = igl.boundary_facets(em.elements.astype(np.int32))
F_cage = F_cage[:, [0, 2, 1]]

rng = np.random.default_rng(0)
sample_size = min(args.num_sample_points, em.num_vertices)
P = em.nodes[rng.choice(em.num_vertices, size=sample_size, replace=False)]

state = {
    "impact_idx": 0,
    "contact": P[0].copy(),
    "magnitude": float(args.magnitude),
    "spread": float(args.spread_radius),
    "poisson": float(args.poisson_ratio),
    "yield_threshold": float(args.yield_threshold),
    "mode_amplitude": args.mode_amplitude,
    "display_scale": float(args.display_scale),
    "last_result": None,
}

ps_mesh = ps.register_surface_mesh(
    "cumulative rest pose",
    em.nodes + state["display_scale"] * pm.cumulative_displacement,
    F_cage,
)
ps_contact = ps.register_point_cloud("contact", state["contact"].reshape(1, 3))
ps_contact.set_color((1.0, 0.0, 0.0))
ps_contact.set_radius(0.012)

ps_tets = ps.register_volume_mesh(
    "cage tets (sigma_vm elastic)",
    em.nodes + state["display_scale"] * pm.cumulative_displacement,
    tets=em.elements,
)
ps_tets.set_enabled(False)

ps_tets.add_scalar_quantity(
    "sigma_y (material)", pm.sigma_y_per_tet,
    defined_on="cells", enabled=False, cmap="viridis",
)


def refresh_mesh():
    verts_now = em.nodes + state["display_scale"] * pm.cumulative_displacement
    ps_mesh.update_vertex_positions(verts_now)
    ps_tets.update_vertex_positions(verts_now)
    ps_contact.update_point_positions(state["contact"].reshape(1, 3))

    disp_mag = np.linalg.norm(pm.cumulative_displacement, axis=1)
    ps_mesh.add_scalar_quantity(
        "cumulative |u| per vertex", disp_mag,
        defined_on="vertices", enabled=True, cmap="viridis",
    )

    r = state.get("last_result")
    if r is not None and r.get("elastic_strain") is not None:
        eps_e = r["elastic_strain"]
        E = em.E_per_tet
        nu = state["poisson"]
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        tr = eps_e[:, 0] + eps_e[:, 1] + eps_e[:, 2]
        sig = np.empty_like(eps_e)
        sig[:, 0] = lam * tr + 2 * mu * eps_e[:, 0]
        sig[:, 1] = lam * tr + 2 * mu * eps_e[:, 1]
        sig[:, 2] = lam * tr + 2 * mu * eps_e[:, 2]
        sig[:, 3] = mu * eps_e[:, 3]; sig[:, 4] = mu * eps_e[:, 4]; sig[:, 5] = mu * eps_e[:, 5]
        p = (sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3
        s = sig.copy(); s[:, 0] -= p; s[:, 1] -= p; s[:, 2] -= p
        s_dd = (s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2
                + 2 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))
        sigma_vm = np.sqrt(1.5 * s_dd)
        ps_tets.add_scalar_quantity(
            "sigma_vm (elastic response)", sigma_vm,
            defined_on="cells", enabled=True, cmap="viridis",
        )
        ps_tets.add_scalar_quantity(
            "yield_ratio = sigma_vm / sigma_y",
            sigma_vm / np.maximum(pm.sigma_y_per_tet, 1e-16),
            defined_on="cells", enabled=False, cmap="viridis",
        )


def fire_impact():
    r = pm.apply_impact(
        contact_point=state["contact"],
        direction=[0.0, -1.0, 0.0],
        magnitude=state["magnitude"],
        spread_radius=state["spread"],
        poisson_ratio=state["poisson"],
        yield_excess_threshold=state["yield_threshold"],
        mode_amplitude=state["mode_amplitude"],
    )
    state["last_result"] = r
    if r["fired"]:
        print(f"  fired mode {r['winner']} (sign={r['winner_sign']:+.0f}, "
              f"amp={r['mode_amplitude']:.4g})   history={r['fired_history']}   "
              f"remaining={r['modes_remaining']}")
    else:
        print(f"  no fire — reason={r['reason']}   "
              f"peak sigma_vm={r['sigma_vm_max']:.3g}  vs sigma_y={r['sigma_y_local']:.3g}   "
              f"excess={r['excess_max']:+.3g}")
    refresh_mesh()


def cb():
    r = state["last_result"]
    psim.Text(f"Contact: {state['contact']}")
    psim.Text(f"Magnitude={state['magnitude']:.3g}  spread={state['spread']:.3g}")
    if r is not None:
        psim.Text(f"peak sigma_vm = {r['sigma_vm_max']:.3g}   sigma_y = {r['sigma_y_local']:.3g}")
        psim.Text(f"excess        = {r['excess_max']:+.3g}   (threshold = {state['yield_threshold']:.3g})")
        if r["fired"]:
            psim.Text(f"LAST IMPACT: fired mode {r['winner']} (sign {r['winner_sign']:+.0f})")
        else:
            psim.Text(f"LAST IMPACT: no fire ({r['reason']})")
        psim.Text(f"fired history: {r['fired_history']}")
        psim.Text(f"modes remaining: {r['modes_remaining']} / {pm.modes.shape[1]}")
    psim.Text(f"cumulative |u| max = {float(np.abs(pm.cumulative_displacement).max()):.4g}")

    psim.Text("--- Impact selection ---")
    ch_ii, state["impact_idx"] = psim.InputInt(
        "Impact index", state["impact_idx"], step=1, step_fast=10,
    )
    if ch_ii:
        state["impact_idx"] = max(0, min(P.shape[0] - 1, state["impact_idx"]))
    if psim.Button("Apply impact at index"):
        state["contact"] = P[state["impact_idx"]].copy()
        fire_impact()
    if psim.Button("Random impact (next index)"):
        state["impact_idx"] = (state["impact_idx"] + 1) % P.shape[0]
        state["contact"] = P[state["impact_idx"]].copy()
        fire_impact()

    psim.Text("--- Impact params ---")
    _, state["magnitude"] = psim.SliderFloat(
        "Magnitude", state["magnitude"], v_min=0.0, v_max=1.0e8,
    )
    _, state["spread"] = psim.SliderFloat(
        "Spread radius", state["spread"], v_min=0.0, v_max=0.5,
    )
    _, state["yield_threshold"] = psim.SliderFloat(
        "Yield excess threshold (Pa)", state["yield_threshold"],
        v_min=0.0, v_max=5.0e8,
    )
    ch_ds, state["display_scale"] = psim.SliderFloat(
        "Display scale", state["display_scale"], v_min=0.1, v_max=10.0,
    )
    if ch_ds:
        refresh_mesh()

    psim.Text("--- Reset ---")
    if psim.Button("Reset rest pose (unfire all modes)"):
        pm.reset_runtime()
        state["last_result"] = None
        refresh_mesh()


ps.set_user_callback(cb)
ps.show()
