"""Scene helpers shared by modegen_gui.py and modegen_snapshot.py (no polyscope import here)."""

import numpy as np

# Piece colour by tree depth (root, depth 1, depth 2, deeper).
COLORS = [(0.7, 0.7, 0.75), (0.95, 0.85, 0.3), (0.95, 0.5, 0.2), (0.9, 0.3, 0.4)]


def depth_color(depth):
    return COLORS[min(int(depth), len(COLORS) - 1)]


def fine_displacement(inst, piece):
    """Lifted plastic displacement of one piece's fine vertices (local frame), or None.

    It is whatever the instance renders (``piece_deformed_fine_vertices``)
    minus the rest fine vertices, so it follows the instance's lift.
    """
    v = getattr(piece, "fine_vertices", None)
    if v is None or getattr(piece, "fine_triangles", None) is None:
        return None
    dv = inst.piece_deformed_fine_vertices(piece)
    if dv is None:
        return None
    if inst.plastic_states.get(id(piece)) is None:
        return np.zeros(v.shape)
    return np.asarray(dv, dtype=float) - np.asarray(v, dtype=float)


def plastic_scalar(inst):
    """Per-fine-vertex |plastic displacement| for the whole instance (render order)."""
    parts = []
    for p in inst.pieces:
        d = fine_displacement(inst, p)
        if d is None:
            continue
        parts.append(np.linalg.norm(d, axis=1))
    return np.concatenate(parts) if parts else None
