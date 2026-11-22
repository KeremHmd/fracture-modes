"""Version checks for profiles and pickled trees (numpy-free).

Mismatches only warn: old profiles and trees must keep loading. Nothing here
raises unless a value is unreadable, and even then it warns and goes on.
"""

import warnings

from ._version import PROFILE_SCHEMA_VERSION, TREE_FORMAT_VERSION, __version__
from .interfaces import PLUGIN_API_VERSION


class ModegenVersionWarning(UserWarning):
    """A profile or tree was written by a different modegen version."""


def _as_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def profile_schema_version(data):
    """Schema version declared by a raw profile dict (missing means 1)."""
    if not isinstance(data, dict):
        return 1
    return _as_int(data.get("schema_version", 1), 1)


def check_profile_schema(data, source=None):
    """Warn when a profile declares a schema other than the current one.

    Returns the declared version.
    """
    v = profile_schema_version(data)
    if v != PROFILE_SCHEMA_VERSION:
        where = f" ({source})" if source else ""
        newer = "newer" if v > PROFILE_SCHEMA_VERSION else "older"
        warnings.warn(
            f"profile{where} declares schema_version {v}, this modegen "
            f"({__version__}) reads schema {PROFILE_SCHEMA_VERSION}; it was written "
            f"for a {newer} layout and is loaded best-effort",
            ModegenVersionWarning, stacklevel=3,
        )
    return v


def tree_version_info(tree):
    """Version metadata stored on a tree (defaults for pre-0.3 trees)."""
    return dict(
        tree_format=_as_int(getattr(tree, "modegen_tree_format", 1), 1),
        modegen_version=getattr(tree, "modegen_version", None),
        api_version=getattr(tree, "modegen_api_version", None),
        profile_schema=getattr(tree, "modegen_profile_schema_version", None),
        plugins=getattr(tree, "modegen_plugins", None),
    )


def check_tree(tree, source=None, registry=None):
    """Warn when a tree was written with a different tree format, plugin API
    major version, or (given a registry) a different major version of one of
    its precompute plugins. Returns ``tree_version_info(tree)``."""
    info = tree_version_info(tree)
    where = f" {source}" if source else ""
    if info["tree_format"] != TREE_FORMAT_VERSION:
        warnings.warn(
            f"tree{where} has format {info['tree_format']}, this modegen "
            f"({__version__}) writes format {TREE_FORMAT_VERSION}; loading best-effort",
            ModegenVersionWarning, stacklevel=2,
        )
    api = info["api_version"]
    if api is not None and str(api).split(".")[0] != str(PLUGIN_API_VERSION).split(".")[0]:
        warnings.warn(
            f"tree{where} was built for plugin API {api}, host is {PLUGIN_API_VERSION}",
            ModegenVersionWarning, stacklevel=2,
        )
    plugins = info["plugins"]
    if registry is not None and isinstance(plugins, dict):
        for slot, entries in plugins.items():
            for e in entries or ():
                try:
                    have = registry.version(e["category"], e["name"])
                except Exception:
                    have = None
                if have is None:
                    warnings.warn(f"tree{where}: {slot} plugin {e.get('name')!r} is not "
                                  f"registered; load it with --plugin-dir to use the tree",
                                  ModegenVersionWarning, stacklevel=2)
                elif str(have).split(".")[0] != str(e.get("version", "1")).split(".")[0]:
                    warnings.warn(f"tree{where}: {slot} plugin {e['name']} was "
                                  f"{e.get('version')} at build time, now {have}",
                                  ModegenVersionWarning, stacklevel=2)
    return info
