"""Plugin base classes. Every extension point of modegen is one of these.

A plugin is a small class that declares its tunable parameters in ``PARAMS``
and implements the one or two methods of its category. Plugins are created
from profile specs (``{"plugin": "name", "params": {...}}``) by the registry,
which validates parameter names, applies defaults and instantiates nested
plugin specs (e.g. the face-weighting plugin used inside a fracture generator).

Precompute-time categories
    fracture.generator  build a fracture-modes object for one tree node
    fracture.weights    per-face toughness multiplier (hook of SOCP generators)
    fracture.init       initial vectors for the SOCP iteration (hook)
    elastic.generator   build the elastic subspace used for impact analysis
    plastic.generator   build the plastic-modes basis for one node
    plastic.weights     per-tet multiplier of the plastic dissipation weights

Runtime categories
    impact              turn an impact event into a per-vertex force field
    router              decide plastic / fracture / none for one impact
    fracture.response   project the impact onto the fracture modes
    plastic.response    turn the impact into a permanent deformation
    post                observe / modify the world after each impact
"""

import copy
from dataclasses import dataclass

import numpy as np

PLUGIN_API_VERSION = "1.0"


class PluginError(Exception):
    """Raised for unknown plugins, bad parameters or incompatible API versions."""


@dataclass(frozen=True)
class Param:
    """Declaration of one plugin parameter.

    ``plugin`` names a category when the parameter holds a nested plugin spec
    (a string, a ``{"plugin": ..., "params": ...}`` dict, ``None``, or for
    ``many=True`` a list of those). ``choices`` restricts plain values.
    """

    default: object = None
    doc: str = ""
    plugin: str = None
    many: bool = False
    choices: tuple = None


def _resolve_nested(category, value):
    if value is None or isinstance(value, Plugin):
        return value
    from .registry import registry as _registry
    _registry.load_builtins()
    return _registry.create(category, value)


class Plugin:
    """Base class of every plugin."""

    CATEGORY = None
    NAME = None
    API_VERSION = PLUGIN_API_VERSION
    #: The plugin's own SemVer. Bump it when its behaviour or parameters change;
    #: trees record the versions of the generators that built them.
    VERSION = "1.0.0"
    DESCRIPTION = ""
    PARAMS = {}

    def __init__(self, **params):
        allowed = self.PARAMS
        unknown = sorted(set(params) - set(allowed))
        if unknown:
            raise PluginError(
                f"{self.NAME or type(self).__name__}: unknown parameter(s) "
                f"{unknown}; allowed: {sorted(allowed)}"
            )
        resolved = {k: copy.deepcopy(p.default) for k, p in allowed.items()}
        resolved.update(params)
        for k, p in allowed.items():
            if p.choices is not None and resolved[k] not in p.choices:
                raise PluginError(
                    f"{self.NAME}: parameter {k}={resolved[k]!r} not in {p.choices}"
                )
        for k, p in allowed.items():
            if p.plugin is None:
                continue
            v = resolved[k]
            if p.many:
                resolved[k] = [_resolve_nested(p.plugin, x) for x in (v or [])]
            else:
                resolved[k] = _resolve_nested(p.plugin, v)
        self.__dict__["params"] = resolved
        self.setup()

    def setup(self):
        """Optional hook called after parameters are resolved."""

    def __getattr__(self, name):
        if name == "params":
            raise AttributeError(name)
        params = self.__dict__.get("params", {})
        if name in params:
            return params[name]
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    @classmethod
    def describe(cls):
        return dict(
            category=cls.CATEGORY, name=cls.NAME, description=cls.DESCRIPTION,
            api_version=cls.API_VERSION, version=str(cls.VERSION),
            params={k: dict(default=p.default, doc=p.doc, plugin=p.plugin,
                            many=p.many, choices=p.choices)
                    for k, p in cls.PARAMS.items()},
        )

    def __repr__(self):
        return f"<{self.CATEGORY}:{self.NAME} {self.params}>"


# --------------------------------------------------------------------------
# Precompute-time categories
# --------------------------------------------------------------------------

class FractureGenerator(Plugin):
    """Builds a ``fracture_modes``-compatible object for a tree node.

    The returned object must have run its mode computation (``modes``,
    ``labels``, ``tet_neighbors``, ``massmatrix``, explode matrices set) but
    NOT ``impact_precomputation`` - the tree builder runs that so the piece
    extraction, Delaunay cleanup and child construction stay identical for
    every generator.
    """

    CATEGORY = "fracture.generator"

    def build(self, node, ctx):
        raise NotImplementedError


class FaceWeighting(Plugin):
    """Per-face toughness multiplier, applied on top of G_c(face) * area(face)."""

    CATEGORY = "fracture.weights"

    def prepare(self, vertices, elements, materials):
        """Called once per node before any ``multiplier`` call."""

    def multiplier(self, t1, t2):
        return 1.0


class InitVectors(Plugin):
    """Initial vectors of the iterative SOCP (one per mode).

    Return ``None`` to use the default low-frequency Laplacian eigenmodes, or
    an array of shape ``(d * V, num_modes)`` in component-major order
    (all x entries, then all y entries, then all z entries).
    """

    CATEGORY = "fracture.init"

    def vectors(self, vertices, elements, num_modes, d):
        return None


class ElasticGenerator(Plugin):
    """Builds the elastic subspace (``ElasticModes``) for a node."""

    CATEGORY = "elastic.generator"

    def build(self, node, ctx):
        raise NotImplementedError


class PlasticGenerator(Plugin):
    """Builds the plastic-modes basis (``PlasticModes``) for a node."""

    CATEGORY = "plastic.generator"

    def build(self, node, elastic_modes, ctx):
        raise NotImplementedError


class TetWeighting(Plugin):
    """Per-tet multiplier of the plastic dissipation weight sigma_y * vol."""

    CATEGORY = "plastic.weights"

    def weights(self, vertices, elements, materials):
        return np.ones(elements.shape[0])


# --------------------------------------------------------------------------
# Runtime categories
# --------------------------------------------------------------------------

class ImpactModel(Plugin):
    """Turns an impact into a per-cage-vertex force field (V, 3), local frame."""

    CATEGORY = "impact"

    def force_field(self, ctx):
        raise NotImplementedError


class Router(Plugin):
    """Decides which branch (plastic / fracture / none) handles an impact."""

    CATEGORY = "router"

    def decide(self, ctx, analysis):
        raise NotImplementedError


class FractureResponder(Plugin):
    """Projects an impact onto the hit piece's fracture modes.

    Must leave ``n_pieces_after_impact`` and ``piece_labels_after_impact`` set
    on ``ctx.piece.modes_instance`` and return True, or return False to cancel.
    """

    CATEGORY = "fracture.response"

    def project(self, ctx, force, decision):
        raise NotImplementedError


class PlasticResponder(Plugin):
    """Turns an impact into a permanent deformation of the hit piece.

    Mutates ``ctx.plastic_state`` (``cum`` per-cage-vertex displacement and the
    ``fired`` mode list) and returns a dict with at least ``fired`` (bool) and
    ``reason`` (str).

    ``PER_VERTEX_STATE_KEYS`` (optional, since 0.3.0) names extra entries of
    ``ctx.plastic_state`` whose first axis is the cage vertex. The runtime
    copies them to sibling pieces and subsets them onto fragments through
    ``parent_vertex_indices`` when the piece fractures, exactly like ``cum``.
    Undeclared extra keys are not carried.
    """

    CATEGORY = "plastic.response"
    PER_VERTEX_STATE_KEYS = ()

    def respond(self, ctx, force, analysis, decision):
        raise NotImplementedError


class PostProcessor(Plugin):
    """Observes / modifies the world after an impact has been handled."""

    CATEGORY = "post"

    def on_fracture(self, ctx, result):
        pass

    def on_plastic(self, ctx, result):
        pass

    def on_none(self, ctx, result):
        pass


CATEGORIES = {
    cls.CATEGORY: cls for cls in (
        FractureGenerator, FaceWeighting, InitVectors,
        ElasticGenerator, PlasticGenerator, TetWeighting,
        ImpactModel, Router, FractureResponder, PlasticResponder,
        PostProcessor,
    )
}


# --------------------------------------------------------------------------
# Small shared helpers for plugin authors
# --------------------------------------------------------------------------

def distribute_force(weights, direction, magnitude):
    """Per-vertex force field whose total equals ``magnitude * direction``.

    ``weights`` is any non-negative per-vertex array; it is normalised to sum
    to one. Falls back to the largest weight if everything is zero.
    """
    w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    s = w.sum()
    if s <= 0.0:
        w = np.zeros_like(w)
        w[int(np.argmax(weights))] = 1.0
        s = 1.0
    d = np.asarray(direction, dtype=float).reshape(3)
    n = np.linalg.norm(d)
    d = d / n if n > 1e-12 else np.array([0.0, -1.0, 0.0])
    return (w / s)[:, None] * (float(magnitude) * d)[None, :]


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def bbox_diagonal(vertices):
    v = np.asarray(vertices, dtype=float)
    return float(np.linalg.norm(v.max(axis=0) - v.min(axis=0)))


def von_mises_from_strain(eps, E, nu):
    """Per-tet von-Mises stress from engineering-shear Voigt strain (T, 6)."""
    E = np.asarray(E, dtype=float)
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    tr = eps[:, 0] + eps[:, 1] + eps[:, 2]
    s = np.empty_like(eps)
    s[:, 0] = lam * tr + 2.0 * mu * eps[:, 0]
    s[:, 1] = lam * tr + 2.0 * mu * eps[:, 1]
    s[:, 2] = lam * tr + 2.0 * mu * eps[:, 2]
    s[:, 3] = mu * eps[:, 3]
    s[:, 4] = mu * eps[:, 4]
    s[:, 5] = mu * eps[:, 5]
    p = (s[:, 0] + s[:, 1] + s[:, 2]) / 3.0
    s[:, 0] -= p
    s[:, 1] -= p
    s[:, 2] -= p
    j = (s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2
         + 2.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))
    return np.sqrt(1.5 * j)
