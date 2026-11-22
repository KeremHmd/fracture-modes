"""modegen - plugin-driven mode generation on top of the frozen fracture_utility baseline.

Only light (numpy-only) modules are imported here so that plugin discovery,
profile validation and the unit tests work without the heavy geometry stack.
The precompute / runtime modules import ``fracture_utility`` and are loaded
explicitly by the tools that need them::

    from modegen.precompute import build_tree
    from modegen.runtime import PluginWorld
"""

from ._version import __version__, PROFILE_SCHEMA_VERSION, TREE_FORMAT_VERSION
from .interfaces import (
    PLUGIN_API_VERSION, Param, Plugin, PluginError,
    FractureGenerator, FaceWeighting, InitVectors,
    ElasticGenerator, PlasticGenerator, TetWeighting,
    ImpactModel, Router, FractureResponder, PlasticResponder, PostProcessor,
    CATEGORIES,
)
from .events import ImpactEvent, ImpactContext, Decision, ImpactResult
from .registry import PluginRegistry, registry, register, get_registry
from .profile import Profile, DEFAULT_PROFILE

__all__ = [
    "__version__", "PROFILE_SCHEMA_VERSION", "TREE_FORMAT_VERSION",
    "PLUGIN_API_VERSION", "Param", "Plugin", "PluginError",
    "FractureGenerator", "FaceWeighting", "InitVectors",
    "ElasticGenerator", "PlasticGenerator", "TetWeighting",
    "ImpactModel", "Router", "FractureResponder", "PlasticResponder",
    "PostProcessor", "CATEGORIES",
    "ImpactEvent", "ImpactContext", "Decision", "ImpactResult",
    "PluginRegistry", "registry", "register", "get_registry",
    "Profile", "DEFAULT_PROFILE",
]
