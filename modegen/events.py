"""Plain data objects passed between the runtime and the plugins."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ImpactEvent:
    """One impact in WORLD coordinates.

    ``tool`` selects an impact model from the profile's ``runtime.tools``
    (``None`` = the profile's default tool). ``params`` is forwarded to the
    plugins as ``ctx.extra`` (e.g. a blade axis for slashes).
    """

    point: np.ndarray
    direction: np.ndarray
    magnitude: float = 1.0e6
    tool: str = None
    params: dict = field(default_factory=dict)
    time: float = 0.0


@dataclass
class ImpactContext:
    """Everything a runtime plugin may need about the impact on one piece.

    Geometry is in the hit instance's LOCAL frame (the frame the precomputed
    bundles live in). ``vertices`` are the current (plastically deformed)
    cage vertices, ``rest_vertices`` the precomputed ones; both share the
    indexing of the piece's cage, which is also the indexing of every force
    field and plastic displacement.
    """

    world: object
    event: ImpactEvent
    instance: object
    piece: object
    point: np.ndarray
    direction: np.ndarray
    magnitude: float
    rest_vertices: np.ndarray
    vertices: np.ndarray
    elements: np.ndarray
    materials: object
    plastic_state: dict = None
    extra: dict = field(default_factory=dict)
    scratch: dict = field(default_factory=dict)


@dataclass
class Decision:
    """Router output. ``fallback='fracture'`` asks the runtime to fracture
    when the plastic branch cannot deform any further."""

    branch: str
    reason: str = ""
    fallback: str = None
    params: dict = field(default_factory=dict)


@dataclass
class ImpactResult:
    branch: str
    reason: str
    decision: Decision = None
    analysis: dict = None
    instances: list = field(default_factory=list)
    hit_instance: object = None
    hit_piece: object = None
    plastic_result: dict = None
    fracture_info: dict = None
    tool: str = None
    missed: bool = False
