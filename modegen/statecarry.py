"""Per-piece plastic state carried through fracture (numpy only).

The runtime keeps one state dict per (instance, piece). It always has ``cum``
(V x 3 cage displacement) and ``fired``. When a piece splits, the state must be
copied to sibling pieces and subset onto child pieces through the child's
``parent_vertex_indices``.

A plastic responder may declare extra per-vertex keys in
``PER_VERTEX_STATE_KEYS`` (e.g. ``exemplar.impulse_blend`` keeps its example
weights there). Declared keys are copied or subset exactly like ``cum``.
Undeclared keys are dropped, as before 0.3.0, so responders that declare
nothing behave exactly as they did.
"""

import numpy as np


def responder_state_keys(responder):
    """The per-vertex state keys a responder declares (``()`` if none)."""
    keys = getattr(responder, "PER_VERTEX_STATE_KEYS", ()) or ()
    return tuple(k for k in keys if k not in ("cum", "fired"))


def copy_state(state, keys=()):
    """State of a piece that keeps its own vertices (a sibling piece).

    Returns ``(base, extras)``: ``base`` is the ``{cum, fired}`` dict the
    runtime always carried; ``extras`` holds copies of the declared keys.
    """
    base = dict(cum=np.asarray(state["cum"]).copy(), fired=list(state["fired"]))
    extras = {k: np.array(state[k], copy=True) for k in keys if state.get(k) is not None}
    return base, extras


def subset_state(parent_state, parent_vertex_indices, keys=()):
    """State of a child piece cut out of the parent piece.

    Child vertex ``i`` is parent vertex ``parent_vertex_indices[i]``. Returns
    ``(base, extras)`` or ``None`` if the indices do not fit the parent state.
    ``fired`` starts empty for the child (baseline behaviour).
    """
    pvi = np.asarray(parent_vertex_indices, dtype=np.int64)
    cum = np.asarray(parent_state["cum"])
    if pvi.size == 0 or pvi.max() >= cum.shape[0]:
        return None
    base = dict(cum=cum[pvi].copy(), fired=[])
    extras = {}
    for k in keys:
        v = parent_state.get(k)
        if v is None:
            continue
        v = np.asarray(v)
        if v.ndim >= 1 and v.shape[0] == cum.shape[0]:
            extras[k] = v[pvi].copy()
    return base, extras


def inject_extras(instance_states, extras_by_piece):
    """Put carried extra keys back into a freshly spawned instance's
    ``plastic_states`` (the frozen ``ActiveInstance`` only copies cum/fired)."""
    for pid, extras in extras_by_piece.items():
        st = instance_states.get(pid)
        if st is not None and extras:
            st.update(extras)
