# Heterogeneous fracture-modes example (counterpart to scripts/example.py).
#
# Usage:
#   python scripts\example_heterogeneous.py [--materials FILE.npz] [--save-materials FILE.npz]

import argparse
import os
from datetime import datetime
import numpy as np
import igl
import tetgen
from context import fracture_utility as fracture
from context import gpytoolbox
from gpytoolbox.copyleft import lazy_cage


parser = argparse.ArgumentParser(
    description="Heterogeneous fracture-mode example."
)
parser.add_argument(
    "--materials",
    metavar="FILE",
    default=None,
    help="Load per-tet MaterialField from FILE.npz (see MaterialField.from_file).",
)
parser.add_argument(
    "--save-materials",
    metavar="FILE",
    default=None,
    help="Save the MaterialField in use to FILE.npz before computing modes.",
)
args = parser.parse_args()


v_fine, f_fine = igl.read_triangle_mesh("data/bunny_oded.obj")
v_fine = gpytoolbox.normalize_points(v_fine)
v, f = lazy_cage(v_fine, f_fine, num_faces=2000)

tgen = tetgen.TetGen(v, f)
nodes, elements = tgen.tetrahedralize()
N_tets = elements.shape[0]


if args.materials is not None:
    materials = fracture.MaterialField.from_file(args.materials, num_tets=N_tets)
    print(f"Loaded materials from {args.materials}")
else:
    centroids = nodes[elements].mean(axis=1)
    y = centroids[:, 1]
    y_threshold = y.min() + 2.0 / 3.0 * (y.max() - y.min())
    G_c = np.where(y > y_threshold, 0.05, 1.0).astype(float)
    materials = fracture.MaterialField(
        num_tets=N_tets, G_c=G_c, interface_rule="min"
    )
    print("Using default hardcoded MaterialField (soft top third).")

print("Material field:", repr(materials))

if args.save_materials is not None:
    materials.to_file(args.save_materials)
    print(f"Saved materials to {args.save_materials}")


out_dir = os.path.join(
    "out", "heterogeneous", datetime.now().strftime("%Y%m%d_%H%M%S")
)
os.makedirs(out_dir, exist_ok=True)
print(f"Outputs will be written to: {out_dir}")

modes = fracture.fracture_modes_heterogeneous(nodes, elements)
params = fracture.fracture_modes_parameters(num_modes=10, verbose=True, d=3)
modes.compute_modes(parameters=params, materials=materials)
modes.impact_precomputation(v_fine=v_fine, f_fine=f_fine)
modes.write_segmented_modes(os.path.join(out_dir, "modes"))

contact_point = nodes[1, :]
direction = np.array([1.0, 0.0, 0.0])
modes.impact_projection(contact_point=contact_point, direction=direction)
new_contact_point = nodes[5, :]
modes.impact_projection(contact_point=new_contact_point, direction=direction)
modes.write_segmented_output(os.path.join(out_dir, "output.obj"))
print("Example (heterogeneous) ran successfully!")
