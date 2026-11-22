"""Single source of the modegen version numbers.

* ``__version__``            package version (SemVer). Written only here.
* ``PROFILE_SCHEMA_VERSION`` layout of profile JSON files (optional top-level
  ``"schema_version"``; a profile without it is schema 1).
* ``TREE_FORMAT_VERSION``    layout of the modegen attributes stored on pickled
  trees (``modegen_tree_format``; a tree without it is format 1).

The plugin contract version, ``PLUGIN_API_VERSION``, lives next to the
contracts in ``modegen/interfaces.py``.

Each compatibility number is bumped only on a breaking change to its surface.
"""

__version__ = "0.4.0"

PROFILE_SCHEMA_VERSION = 1
TREE_FORMAT_VERSION = 1
