# CLI to generate a MaterialField .npz from a named pattern.
#
# Usage:
#   python scripts\set_materials.py [MESH] --output FILE.npz --pattern PATTERN
#                                   [--G_c-inside V] [--G_c-outside V] [--invert]
#                                   [--sigma_y-inside V] [--sigma_y-outside V]
#                                   [--sigma_c-inside V] [--sigma_c-outside V]
#                                   [--material-class-inside CLS] [--material-class-outside CLS]
#                                   [--E-inside V] [--E-outside V]
#                                   [--rho-inside V] [--rho-outside V]
#                                   [--axis 0|1|2] [--threshold T] [--above]
#                                   [--center X Y Z] [--radius R]
#                                   [--bbox-min X Y Z] [--bbox-max X Y Z]
#                                   [--G_c-at-lo V] [--G_c-at-hi V] [--num-bands N]
#                                   [--interface-rule min|harmonic|mean] [--preview]
#
# Patterns: uniform, axis-split, sphere, box, gradient, bands.
# Material classes: rigid, brittle, elastic-plastic.
# The sigma_y/sigma_c/E/rho/material_class fields are optional; when unset the .npz
# does not carry them and the fracture-only pipeline behaves exactly as before.

import argparse
import os

import numpy as np
import igl
import tetgen

from context import fracture_utility as fracture
from context import gpytoolbox
from gpytoolbox.copyleft import lazy_cage


parser = argparse.ArgumentParser(
    description="Generate a MaterialField file from a named distribution pattern."
)

parser.add_argument("mesh", nargs="?", default="data/bunny_oded.obj")
parser.add_argument("--num-faces", type=int, default=2000,
                    help="Cage face count. Must match the value used by downstream "
                         "scripts (paint_materials, example_heterogeneous, etc.).")
parser.add_argument("--output", required=True, help="Output .npz path.")
parser.add_argument("--interface-rule", default="min",
                    choices=["min", "harmonic", "mean"],
                    help="Combining rule for per-face G_c at material interfaces.")
parser.add_argument("--preview", action="store_true",
                    help="Open Polyscope to inspect the G_c field on the cage tets.")

parser.add_argument("--pattern", required=True,
                    choices=["uniform", "axis-split", "sphere", "box", "gradient", "bands"],
                    help="Pattern type.")
parser.add_argument("--G_c-inside", dest="G_c_inside", type=float, default=10.0,
                    help="G_c inside the pattern region.")
parser.add_argument("--G_c-outside", dest="G_c_outside", type=float, default=1.0,
                    help="G_c outside the pattern region.")
parser.add_argument("--invert", action="store_true",
                    help="Swap inside and outside G_c values.")

parser.add_argument("--axis", type=int, default=1, choices=[0, 1, 2],
                    help="Axis (0=x, 1=y, 2=z).")
parser.add_argument("--threshold", type=float, default=None,
                    help="Axis threshold for --pattern axis-split (defaults to midpoint).")
parser.add_argument("--above", action="store_true",
                    help="For --pattern axis-split: inside = strictly above threshold "
                         "(default: below).")

parser.add_argument("--center", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="Sphere center (defaults to centroid mean).")
parser.add_argument("--radius", type=float, default=None,
                    help="Sphere radius (defaults to ~20%% of bbox diagonal).")

parser.add_argument("--bbox-min", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"))
parser.add_argument("--bbox-max", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"))

parser.add_argument("--G_c-at-lo", dest="G_c_at_lo", type=float, default=None,
                    help="G_c at the low end of --axis (defaults to --G_c-outside).")
parser.add_argument("--G_c-at-hi", dest="G_c_at_hi", type=float, default=None,
                    help="G_c at the high end of --axis (defaults to --G_c-inside).")

parser.add_argument("--num-bands", type=int, default=4,
                    help="Number of bands for --pattern bands (alternating in/out).")

parser.add_argument("--sigma_y-inside", dest="sigma_y_inside", type=float, default=None,
                    help="Yield stress (Pa) inside the pattern region. Omit to skip sigma_y.")
parser.add_argument("--sigma_y-outside", dest="sigma_y_outside", type=float, default=None,
                    help="Yield stress (Pa) outside the pattern region.")
parser.add_argument("--sigma_c-inside", dest="sigma_c_inside", type=float, default=None,
                    help="Fracture stress (Pa) inside the pattern region. Omit to skip sigma_c.")
parser.add_argument("--sigma_c-outside", dest="sigma_c_outside", type=float, default=None,
                    help="Fracture stress (Pa) outside the pattern region.")
parser.add_argument("--E-inside", dest="E_inside", type=float, default=None,
                    help="Young's modulus (Pa) inside the pattern region.")
parser.add_argument("--E-outside", dest="E_outside", type=float, default=None,
                    help="Young's modulus (Pa) outside the pattern region.")
parser.add_argument("--rho-inside", dest="rho_inside", type=float, default=None,
                    help="Density (kg/m^3) inside the pattern region.")
parser.add_argument("--rho-outside", dest="rho_outside", type=float, default=None,
                    help="Density (kg/m^3) outside the pattern region.")
parser.add_argument("--material-class-inside", dest="material_class_inside",
                    default=None,
                    choices=["rigid", "brittle", "elastic-plastic"],
                    help="Material class label for the inside region.")
parser.add_argument("--material-class-outside", dest="material_class_outside",
                    default=None,
                    choices=["rigid", "brittle", "elastic-plastic"],
                    help="Material class label for the outside region.")

args = parser.parse_args()


print(f"Loading {args.mesh}...")
v_fine, f_fine = igl.read_triangle_mesh(args.mesh)
v_fine = gpytoolbox.normalize_points(v_fine)
v_cage, f_cage = lazy_cage(v_fine, f_fine, num_faces=args.num_faces)
tgen = tetgen.TetGen(v_cage, f_cage)
nodes, elements = tgen.tetrahedralize()
N_tets = elements.shape[0]
centroids = nodes[elements].mean(axis=1)
print(f"  {N_tets} cage tets")
print(f"  Centroid bbox: lo={centroids.min(axis=0)}  hi={centroids.max(axis=0)}")


inside_value = args.G_c_outside if args.invert else args.G_c_inside
outside_value = args.G_c_inside if args.invert else args.G_c_outside

pattern_mask = None

if args.pattern == "uniform":
    G_c = np.full(N_tets, inside_value, dtype=float)
    pattern_mask = np.ones(N_tets, dtype=bool)
    region_summary = f"uniform G_c = {inside_value}"

elif args.pattern == "axis-split":
    axis_name = "xyz"[args.axis]
    threshold = args.threshold
    if threshold is None:
        threshold = float(
            (centroids[:, args.axis].min() + centroids[:, args.axis].max()) / 2
        )
    mask = centroids[:, args.axis] > threshold
    if not args.above:
        mask = ~mask
    G_c = np.where(mask, inside_value, outside_value).astype(float)
    pattern_mask = mask
    region_summary = (
        f"axis-split on {axis_name} at {threshold:.4f}, inside = "
        f"{'above' if args.above else 'below'} threshold"
    )

elif args.pattern == "sphere":
    center = (
        np.asarray(args.center, dtype=float)
        if args.center is not None
        else centroids.mean(axis=0)
    )
    if args.radius is not None:
        radius = float(args.radius)
    else:
        bbox_diag = float(np.linalg.norm(centroids.max(axis=0) - centroids.min(axis=0)))
        radius = 0.2 * bbox_diag
    d = np.linalg.norm(centroids - center, axis=1)
    mask = d <= radius
    G_c = np.where(mask, inside_value, outside_value).astype(float)
    pattern_mask = mask
    region_summary = f"sphere center={tuple(center)} radius={radius:.4f}"

elif args.pattern == "box":
    bbox_lo = (
        np.asarray(args.bbox_min, dtype=float)
        if args.bbox_min is not None
        else centroids.min(axis=0)
    )
    bbox_hi = (
        np.asarray(args.bbox_max, dtype=float)
        if args.bbox_max is not None
        else centroids.max(axis=0)
    )
    mask = np.all((centroids >= bbox_lo) & (centroids <= bbox_hi), axis=1)
    G_c = np.where(mask, inside_value, outside_value).astype(float)
    pattern_mask = mask
    region_summary = f"box lo={tuple(bbox_lo)} hi={tuple(bbox_hi)}"

elif args.pattern == "gradient":
    axis_name = "xyz"[args.axis]
    G_lo = args.G_c_at_lo if args.G_c_at_lo is not None else outside_value
    G_hi = args.G_c_at_hi if args.G_c_at_hi is not None else inside_value
    c = centroids[:, args.axis]
    span = float(c.max() - c.min())
    if span < 1e-12:
        G_c = np.full(N_tets, (G_lo + G_hi) * 0.5, dtype=float)
    else:
        t = (c - c.min()) / span
        G_c = (G_lo + t * (G_hi - G_lo)).astype(float)
    pattern_mask = np.ones(N_tets, dtype=bool)
    region_summary = (
        f"gradient on {axis_name}: lo={G_lo:.4f} -> hi={G_hi:.4f}"
    )

elif args.pattern == "bands":
    axis_name = "xyz"[args.axis]
    c = centroids[:, args.axis]
    span = float(c.max() - c.min())
    if span < 1e-12:
        G_c = np.full(N_tets, inside_value, dtype=float)
        pattern_mask = np.ones(N_tets, dtype=bool)
    else:
        t = (c - c.min()) / span
        n = max(1, int(args.num_bands))
        idx = np.clip((t * n).astype(int), 0, n - 1)
        mask = (idx % 2 == 0)
        G_c = np.where(mask, inside_value, outside_value).astype(float)
        pattern_mask = mask
    region_summary = (
        f"bands on {axis_name}: {args.num_bands} alternating bands"
    )

else:
    raise SystemExit(f"Unknown pattern: {args.pattern}")


if not np.all(G_c > 0.0):
    raise SystemExit(
        "G_c contains non-positive values — both --G_c-inside and "
        "--G_c-outside must be strictly positive."
    )


def _build_optional_field(name, inside_v, outside_v):
    if inside_v is None and outside_v is None:
        return None
    if outside_v is None:
        outside_v = inside_v
    elif inside_v is None:
        inside_v = outside_v
    if args.invert:
        inside_v, outside_v = outside_v, inside_v
    arr = np.where(pattern_mask, inside_v, outside_v).astype(float)
    if np.any(arr <= 0.0):
        raise SystemExit(f"{name} contains non-positive values; must be > 0.")
    return arr


def _build_material_class_field(inside_label, outside_label):
    if inside_label is None and outside_label is None:
        return None
    if outside_label is None:
        outside_label = inside_label
    elif inside_label is None:
        inside_label = outside_label
    if args.invert:
        inside_label, outside_label = outside_label, inside_label
    labels = np.where(pattern_mask, inside_label, outside_label)
    return labels


sigma_y_field = _build_optional_field(
    "sigma_y", args.sigma_y_inside, args.sigma_y_outside
)
sigma_c_field = _build_optional_field(
    "sigma_c", args.sigma_c_inside, args.sigma_c_outside
)
E_field = _build_optional_field("E", args.E_inside, args.E_outside)
rho_field = _build_optional_field("rho", args.rho_inside, args.rho_outside)
material_class_field = _build_material_class_field(
    args.material_class_inside, args.material_class_outside
)


materials = fracture.MaterialField(
    num_tets=N_tets, G_c=G_c,
    rho=rho_field, E=E_field,
    sigma_y=sigma_y_field, sigma_c=sigma_c_field,
    material_class=material_class_field,
    interface_rule=args.interface_rule,
)
print(f"\nPattern: {region_summary}")
print(repr(materials))
print(f"Soft tets (G_c == min):  {int((G_c == G_c.min()).sum())} of {N_tets}")
print(f"Tough tets (G_c == max): {int((G_c == G_c.max()).sum())} of {N_tets}")

os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
materials.to_file(args.output)
print(f"\nSaved: {args.output}")


if args.preview:
    import polyscope as ps
    print("\nOpening Polyscope preview — close the window to exit.")
    ps.init()
    ps.set_program_name("set_materials preview")
    ps.set_ground_plane_mode("none")
    ps_vol = ps.register_volume_mesh("cage tets", nodes, tets=elements)
    ps_vol.add_scalar_quantity(
        "G_c", G_c, defined_on="cells", enabled=True, cmap="viridis"
    )
    ps.reset_camera_to_home_view()
    ps.show()
