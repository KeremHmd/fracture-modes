# Authored examples for `exemplar.skinned_examples`

* `wall_exemplar.json`: 9 bones and 12 examples (centre/left/right dents from
  either side, top-corner folds, top lean) for `data/wall.obj`.

Use one with:

    --set precompute.plastic.params.examples_file=examples/wall_exemplar.json --set "precompute.plastic.params.presets=[]"

The format is described in the main README.
* `wall_hits.json`: 13 scripted hits on the wall: weak and medium dents, then one 2e8 hit
  that breaks it outright. Coordinates are world = asset frame at spawn.
* `wall_hits_saturate.json`: 41 repeated 2e7 hits on the wall's front centre, enough to
  saturate the dent so that `exemplar_dents` breaks it through the router's fallback.

    conda run -n fracture-modes python tools/modegen_headless.py --tree out/trees/wall_exemplar.modegen.pkl --script examples/wall_hits_saturate.json --no-gravity

Points must lie inside the fine mesh's bounding box (the hit test uses it). For the
wall, front-face points at z = 0.07 hit; points at the cage face z = 0.1 missed.
