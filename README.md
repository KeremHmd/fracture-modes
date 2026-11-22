# Breaking Good: Fracture Modes for Realtime Destruction

This is the code accompanying the *Transactions on Graphics* paper ["Breaking Good: Fracture Modes for Realtime Destruction"](https://www.silviasellan.com/pdf/papers/fracture-harmonics.pdf), by Silvia Sellán, Jack Luong, Leticia Mattos Da Silva,
Aravind Ramakrishnan, Yuchuan Yang and Alec Jacobson and the *NeurIPS Datasets & Benchmarks 2022* paper ["Breaking Bad: A Dataset for Geometric Fracture and Reassembly"](https://breaking-bad-dataset.github.io) by Silvia Sellán, Yun-Chun Chen, Ziyi Wu, Animesh Garg and Alec Jacobson.

> **This fork: A Modular Approach for Fracture Modeling.** On top of the original method, this fork adds
> heterogeneous fracture toughness, a multi-impact fracture tree, cut-surface clean-up, elastic and
> sparse plastic modes (permanent dents before breaking), a modular framework (`modegen`) in which every
> stage of mode generation and impact response can be swapped, and an example-based plasticity module.
> The original API and scripts are unchanged. See [Extensions in this fork](#extensions) below.

## Installation

To install this library, start by cloning this repository 
```bash
git clone https://github.com/sgsellan/fracture-modes.git
# or this fork: git clone https://github.com/KeremHmd/fracture-modes.git
```
Create a [`conda` environment](https://docs.conda.io/projects/conda/en/latest/index.html)\*
```bash
conda create -n fracture-modes python=3.9
conda activate fracture-modes
```
> \*_for Apple Silicon (M1/M2) users, the igl, scikit-sparse, and tetgen conda packages are currently not available for arm64. To ensure that you create a x86_64 environment use instead_
> ```
> CONDA_SUBDIR=osx-64 conda create -n fracture-modes python=3.9
> conda activate fracture-modes
> conda config --env --set subdir osx-64  # make sure that conda commands in this environment use intel packages
> ```

Then, install all dependencies:
```bash
python -m pip install -r requirements.txt
conda install -c conda-forge igl scikit-sparse tetgen
```
Finally, obtain a (free if you're academic) mosek licence [here](https://www.mosek.com/products/academic-licenses/), and install it as a python package:
```bash
conda install -c mosek mosek
```

You can validate your installation by running 
```bash
python scripts/example.py
``` 
which should complete in around 15 seconds and return a verbose wall of text ending with "Example ran successfully!". Outputs are written to `out/example/<timestamp>/`.

The extensions in this fork were developed and tested with the following versions: `numpy==1.23.5`, `scipy==1.9.3`, `igl==2.2.1`, `scikit-sparse==0.4.8`, `scikit-image==0.19.3`, `polyscope==1.3.1`, `tetgen==0.6.4`, `gpytoolbox==0.0.1` and `mosek`. On Windows, tet index arrays passed to libigl must be `int32`; the new code takes care of this.

## Use for fracture simulation

You can use this library by adding this repository to your python path and importing `fracture_utility`. The following sample code will load a mesh and compute its fracture modes (see `scripts/example.py`):

```python
import numpy as np
import igl
import fracture_utility as fracture
from gpytoolbox.copyleft import lazy_cage

# This is the "fine mesh", i.e. the mesh we use for rendering
v_fine, f_fine = igl.read_triangle_mesh("data/bunny_oded.obj")
v_fine = gpytoolbox.normalize_points(v_fine)
# This is the "cage mesh", i.e. the coarser mesh that we will tetrahedralize and use for the physical simulation
v, f = lazy_cage(v_fine,f_fine,num_faces=100)
# Tetrahedralize
tgen = tetgen.TetGen(v,f)
nodes, elements =  tgen.tetrahedralize()
# Initialize fracture mode class
modes = fracture.fracture_modes(nodes,elements) 
# Set parameters for call to fracture modes
params = fracture.fracture_modes_parameters(num_modes=10,verbose=True,d=3)
# Compute fracture modes
modes.compute_modes(parameters=params)
```

After running this code, the class attribute `modes.modes` will be populated with a `3num_tets` by `10` matrix containing the displacements of each tetrahedron in the mesh (row) for each mode (column).

Any runtime impact can be projected into our computed modes to obtain a realtime fracture with `modes.impact_projection`; for example, 
```python
# We need to precompute some stuff that we will only need to do once
modes.impact_precomputation(v_fine=v_fine,f_fine=f_fine)
# Optionally, you can use this to save each mode's segmentation in the current directory
modes.write_segmented_modes("output_modes")
contact_point = nodes[1,:]
direction = np.array([1.0,0.0,0.0])
# First projection, this should be fast
modes.impact_projection(contact_point=contact_point,direction=direction)
# Second projection, this should be fast
new_contact_point = nodes[5,:]
modes.impact_projection(contact_point=new_contact_point,direction=direction)
# Write segmented output to obj
modes.write_segmented_output("output.obj")
```

We also provide a graphical interface that you can use to get a quick idea of how our algorithm works. Open it by running 
```bash
python scripts/fracture_gui.py PATH/TO/FINE/MESH.obj
```
Your mesh will appear on screen surrounded by a coarse cage. You can tweak the size of the cage if you want using the slider. You can also change the number of modes you want to compute:

![](assets/screencap1.png)

You can then press "Compute Modes". The console will show the progress, it shouldn't take more than one minute unless your cage is very fine. Once it is done, a message saying "Modes Computed" will appear. You can then press "Show Modes" to visualize all the possible fractures our modes identified:

![](assets/screencap2.png)

Then, you can go to "Impact Mode", and click "Generate Random Impact". Three meshes will appear: one will show the norm of the impact as a scalar field on the cage mesh, the middle one will show the impact's projection onto our modes on the cage mesh, and the final mesh will be the fine mesh, segmented to show the pieces it would break on according to our algorithm with that given impact. Timing details will also appear on screen

![](assets/screencap3.png)

Optionally, you can now press on "Save segmented output" to write the fine output to an `.obj` file. See `assets/sample_use.mp4` for a full recorded GUI example.

<!----><a name="dataset"></a>
## Use for fracture dataset generation

As shown in our paper ["Breaking Bad: A Dataset for Geometric Fracture and Reassembly"](https://breaking-bad-dataset.github.io), our code can also be used to simulate many fractures for a specific object. The documented funcionality necessary to replicate our dataset generation can be found inside the `fracture_utility/generate_fractures.py` function. An example use can be found in the `scripts/example_dataset_generation.py` script:

```python
# Load dependencies
import fracture_utility as fracture
import sys
import os
# Read input mesh
filename = "data/bunny_oded.obj"
if len(sys.argv)>1:
    filename = sys.argv[1]

# Choose output directory
output_dir = os.path.splitext(os.path.basename(filename))[0]

# Call dataset generation without volume constraint and with a 5000 cage size.
fracture.generate_fractures(filename,output_dir=output_dir,verbose=True,compressed=False,cage_size=5000,volume_constraint=0.00)
```


<!----><a name="extensions"></a>
## Extensions in this fork

All additions are new modules and scripts; the original `fracture_utility` functions and scripts keep their behaviour. Every new script lists its full command line in its header comment. Generated files go to `out/` (git-ignored), and material files to `materials/`.

| Feature | Main files |
|---|---|
| Heterogeneous fracture toughness | `fracture_utility/material_field.py`, `*_heterogeneous.py`, `scripts/set_materials.py`, `scripts/paint_materials.py` |
| Cut-surface detail and Delaunay clean-up | `fracture_utility/cap_detail.py`, `fracture_utility/delaunay_remesh.py` |
| Multi-impact fracture tree | `fracture_utility/fracture_tree.py`, `fracture_utility/world_state.py` |
| Elastic and sparse plastic modes | `fracture_utility/elastic_modes.py`, `plastic_modes.py`, `plastic_state.py` |
| Modular mode generation | `modegen/`, `profiles/`, `tools/`, `user_plugins/` |
| Example-based plasticity | `modegen/plugins/exemplar.py`, `modegen/solvers/exemplar.py`, `examples/` |

### Heterogeneous fracture toughness

The original modes minimise the total cut area, `sum_f area(f) * ||[u]_f||`, which implicitly assumes the same fracture toughness everywhere. Here every face is weighted by a per-tetrahedron toughness `G_c`, i.e. `sum_f G_c(f) * area(f) * ||[u]_f||`, so cracks prefer weak material. At a face between two materials the weaker one is used by default (`--interface-rule min|harmonic|mean`). A uniform `G_c` reproduces the original modes.

A material field is stored as a `.npz` file. Create one from a named pattern (`uniform`, `axis-split`, `sphere`, `box`, `gradient`, `bands`), or paint it interactively:
```bash
python scripts/set_materials.py --output materials/bunny_stiff_head.materials.npz --pattern axis-split --axis 1 --threshold 0.15 --above --G_c-inside 10.0 --G_c-outside 1.0 --preview
python scripts/paint_materials.py --output materials/custom.materials.npz
```
Run the heterogeneous pipeline headless, in the original-style GUI, or side by side with the original (baseline / homogeneous / heterogeneous / heterogeneous with cut-surface detail):
```bash
python scripts/example_heterogeneous.py --materials materials/bunny_stiff_head.materials.npz
python scripts/fracture_gui_heterogeneous.py --materials materials/bunny_stiff_head.materials.npz
python scripts/compare_fracture.py --materials materials/bunny_stiff_head.materials.npz
```

### Cut-surface detail and Delaunay clean-up

Carving pieces leaves flat cut faces with thin sliver triangles. `cap_detail` adds seeded fractal noise to the cut faces (the outer surface is kept), and `delaunay_remesh` restores the Delaunay condition on them with edge flips and splits (Dyer, Zhang and Möller, SGP 2007). Both sides of a crack get identical operations, so pieces still fit together. Either step can also be applied to any carved mesh:
```bash
python scripts/delaunay_remesh_cli.py --original data/bunny_oded.obj --carved CARVED.obj --output CLEANED.obj
python scripts/enrich_detail.py --original data/bunny_oded.obj --carved CARVED.obj --output ROUGH.obj
```

### Elastic and sparse plastic modes

Elastic (vibration) modes are the lowest solutions of `K phi = omega^2 M phi`, built from per-tetrahedron stiffness `E` and density `rho`. At a hit they give the deformation and the stress in every tetrahedron, which is compared with the material's yield stress `sigma_y` and fracture stress `sigma_c`.

Plastic modes are the cheapest permanent dents, found with the same recipe as fracture modes but a different cost:
`min sum_t sigma_y(t) * vol(t) * ||eps_t(u)||`, where `u` is a mix of the 30 smoothest elastic modes (this keeps dents smooth instead of producing single-vertex spikes). The yield check uses a J2 return mapping.

Material files can carry `sigma_y`, `sigma_c`, `E`, `rho` and a material class (`rigid`, `brittle`, `elastic-plastic`) through the extra `set_materials.py` flags (`--sigma_y-inside`, `--sigma_c-inside`, `--E-inside`, `--rho-inside`, `--material-class-inside` and their `-outside` versions). Step-by-step tools:
```bash
python scripts/compute_elastic_modes.py --materials FILE.npz --output out/bunny.elastic.npz --preview
python scripts/compute_plastic_modes.py --materials FILE.npz --elastic-modes out/bunny.elastic.npz --output out/bunny.plastic.npz --preview
python scripts/plastic_example.py --plastic-modes out/bunny.plastic.npz --elastic-modes out/bunny.elastic.npz --materials FILE.npz
python scripts/plastic_unit_test.py
```
`precompute_asset.py` bundles fracture, elastic and plastic modes into one file, and `plastic_fracture_gui.py` shows a hit ending in no change, a dent, or a break.

### Multi-impact fracture tree

The original method handles one impact. `precompute_tree.py` gives every precomputed piece its own modes, recursively up to a depth and size limit, so pieces can be hit and broken again. Only the opened cracks separate; pieces that stay connected move as one body. Optionally every node also gets elastic and plastic modes, and dents are kept when a piece breaks.
```bash
python scripts/precompute_tree.py --materials materials/bunny_stiff_head.materials.npz --depth 2 --delaunay-remesh --output out/trees/bunny_d2.pkl
python scripts/multi_impact_gui.py --tree out/trees/bunny_d2.pkl
python scripts/multi_impact_example.py --tree out/trees/bunny_d2.pkl
```
Add `--num-elastic-modes 30 --num-plastic-modes 16` to `precompute_tree.py` (with a material file that has `sigma_y` and `sigma_c`) to enable denting.

### Modular mode generation (`modegen`)

`modegen` turns the pipeline into eleven interchangeable stages. A JSON *profile* picks one plugin per stage; the `realistic` profile reproduces the pipeline above.

| Before any hit (precompute) | At a hit (runtime) |
|---|---|
| `fracture.generator`: how fracture modes are found | `impact`: shape of the hit (force on each cage point) |
| `fracture.weights`: per-face cost multipliers | `router`: no change, dent, or break |
| `fracture.init`: starting vectors for the solver | `fracture.response`: which cracks open |
| `elastic.generator`: elastic modes | `plastic.response`: how a dent is applied |
| `plastic.generator`: how dents are found | `post`: extra effects (velocities, logging) |
| `plastic.weights`: per-tetrahedron dent cost multipliers | |

Shipped profiles: `realistic`, `blunt_trauma`, `blunt_dent_only`, `puncture`, `slash`, `crush`, `exploding_wall` (Voronoi cells instead of the optimiser), `exemplar_dents` and `exemplar_dent_only`. Any value can be overridden on the command line, e.g. `--set precompute.fracture.params.num_modes=20` or `--set precompute.plastic=null` (`null` disables a stage). Profiles are merged onto the `realistic` defaults.
```bash
python tools/modegen_plugins.py                 # list plugins
python tools/modegen_plugins.py --profiles      # validate every profile
python tools/modegen_selftest.py                # end-to-end check on a coarse mesh
python tools/modegen_precompute.py --profile blunt_trauma --materials FILE.npz --depth 2 --output out/trees/bunny_blunt.modegen.pkl
python tools/modegen_gui.py --tree out/trees/bunny_blunt.modegen.pkl
python tools/modegen_headless.py --tree out/trees/bunny_blunt.modegen.pkl --impacts 8
```
`modegen_gui.py --profile NAME` swaps the runtime behaviour on an existing tree without recomputing modes. `modegen_catalog.py` prints a Markdown catalog of all plugins and their parameters.

**Writing a plugin.** Put any `*.py` file in `user_plugins/` (or a folder listed in `MODEGEN_PLUGIN_PATH`, or `--plugin-dir`). Subclass a category base from `modegen.interfaces`, declare parameters with `Param`, and register it:
```python
import numpy as np
from modegen import ImpactModel, Param, register
from modegen.interfaces import distribute_force

@register("impact", "mystudio.knuckles")
class Knuckles(ImpactModel):
    PARAMS = {"radius": Param(0.015, "patch radius")}
    def force_field(self, ctx):
        d = np.linalg.norm(ctx.vertices - ctx.point[None, :], axis=1)
        return distribute_force(np.exp(-0.5 * (d / self.radius) ** 2), ctx.direction, ctx.magnitude)
```
Use it with `--add-tool knuckles=mystudio.knuckles` or from a profile. Elastic and plastic vectors are interleaved (`3*v + axis`); fracture vectors are component-major (`axis*V + v`). See `user_plugins/example_ring_impact.py` for a complete example.

### Example-based plasticity

After Jones, Thuerey, Shinar and Bargteil, "Example-based Plastic Deformation of Rigid Bodies" (ACM TOG 2016). A few dented example shapes are prepared beforehand, either from built-in presets (dent, bend, twist, squash) or from a JSON file of handle positions and handle moves (`examples/wall_exemplar.json`). Each hit above a limit moves the nearby points toward the examples, fading with distance; once the dents are used up, the piece breaks (`exemplar_dents`) or stays as it is (`exemplar_dent_only`). Dents are drawn with a smooth cage-to-mesh mapping. Example on a steel wall:
```bash
python tools/make_test_meshes.py
python scripts/set_materials.py data/wall.obj --output materials/wall_ductile.materials.npz --pattern uniform --G_c-inside 1 --G_c-outside 1 --sigma_y-inside 1e8 --sigma_y-outside 1e8 --sigma_c-inside 4e8 --sigma_c-outside 4e8 --E-inside 2e11 --E-outside 2e11 --rho-inside 7800 --rho-outside 7800 --material-class-inside elastic-plastic --material-class-outside elastic-plastic
python tools/modegen_precompute.py data/wall.obj --profile exemplar_dents --materials materials/wall_ductile.materials.npz --depth 2 --output out/trees/wall_exemplar.modegen.pkl
python tools/modegen_gui.py --tree out/trees/wall_exemplar.modegen.pkl --magnitude 2e7 --point 0 0 0.07
python tools/modegen_headless.py --tree out/trees/wall_exemplar.modegen.pkl --script examples/wall_hits_saturate.json --no-gravity
```
For hand-made examples add `--set precompute.plastic.params.examples_file=examples/wall_exemplar.json --set "precompute.plastic.params.presets=[]"` to the precompute command.

### Tests

```bash
python tests/run_all.py
```
The unit tests use numpy only (plus scipy for two runtime tests); `tools/modegen_selftest.py` checks the full stack.


## Known Issues

Please do not hesitate to [contact me](sgsellan@cs.toronto.edu) if you find any issues or bugs in this code, or you struggle to run it in any way.

## Licensing

This code is released free for academic, non-commercial, non-military use. Contact [the authors](mailto:sgsellan@cs.toronto.edu) for commercial usage licensing.
