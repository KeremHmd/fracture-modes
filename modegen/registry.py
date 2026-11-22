"""Plugin registry + discovery.

Plugins register with a decorator::

    from modegen import register, ImpactModel, Param

    @register("impact", "mystudio.hammer")
    class Hammer(ImpactModel):
        PARAMS = {"radius": Param(0.1, "face radius")}
        def force_field(self, ctx): ...

Discovery order (later sources may override earlier ones only when they
register with ``override=True``):

1. built-in plugins in ``modegen/plugins``
2. ``<project root>/user_plugins/*.py``
3. every directory in the ``MODEGEN_PLUGIN_PATH`` environment variable
4. directories passed explicitly (``--plugin-dir`` on every tool)
"""

import difflib
import hashlib
import importlib
import importlib.util
import os
import sys
import traceback

from .interfaces import CATEGORIES, PLUGIN_API_VERSION, Plugin, PluginError

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _major(version):
    return str(version).split(".")[0]


class PluginRegistry:
    def __init__(self):
        self._plugins = {c: {} for c in CATEGORIES}
        self._sources = {}
        self._versions = {}
        self.load_errors = []
        self._loaded_dirs = set()
        self._builtins_loaded = False

    # ------------------------------------------------------------ register
    def register(self, category, name, override=False):
        if category not in CATEGORIES:
            raise PluginError(
                f"Unknown plugin category {category!r}; known: {sorted(CATEGORIES)}"
            )
        base = CATEGORIES[category]

        def deco(cls):
            if not (isinstance(cls, type) and issubclass(cls, base)):
                raise PluginError(
                    f"{name}: plugin class must subclass {base.__name__} "
                    f"for category {category!r}"
                )
            if _major(cls.API_VERSION) != _major(PLUGIN_API_VERSION):
                raise PluginError(
                    f"{name}: built for plugin API {cls.API_VERSION}, "
                    f"host is {PLUGIN_API_VERSION}"
                )
            if name in self._plugins[category] and not override:
                raise PluginError(
                    f"{category}:{name} is already registered by "
                    f"{self._sources.get((category, name))}; pass override=True "
                    f"to replace it deliberately"
                )
            cls.NAME = name
            cls.CATEGORY = category
            self._plugins[category][name] = cls
            self._sources[(category, name)] = getattr(cls, "__module__", "?")
            self._versions[(category, name)] = str(getattr(cls, "VERSION", "1.0.0") or "1.0.0")
            return cls

        return deco

    # ------------------------------------------------------------ lookup
    def categories(self):
        return list(CATEGORIES)

    def names(self, category):
        return sorted(self._plugins[category])

    def get(self, category, name):
        if category not in self._plugins:
            raise PluginError(f"Unknown plugin category {category!r}")
        try:
            return self._plugins[category][name]
        except KeyError:
            close = difflib.get_close_matches(name, self._plugins[category], n=3)
            hint = f" Did you mean {close}?" if close else ""
            raise PluginError(
                f"No {category} plugin named {name!r}.{hint} "
                f"Available: {self.names(category)}"
            ) from None

    def source(self, category, name):
        return self._sources.get((category, name))

    def version(self, category, name):
        """The plugin's own ``VERSION`` as recorded at registration (or None)."""
        return self._versions.get((category, name))

    # ------------------------------------------------------------ create
    def create(self, category, spec):
        """Instantiate a plugin from a profile spec.

        ``spec`` may be ``None`` (returns None), a plugin name, a dict
        ``{"plugin": name, "params": {...}}`` or an already-built instance.
        Nested plugin parameters are instantiated recursively.
        """
        if spec is None or spec is False:
            return None
        if isinstance(spec, Plugin):
            return spec
        if isinstance(spec, str):
            name, params = spec, {}
        elif isinstance(spec, dict):
            if "plugin" not in spec:
                raise PluginError(f"{category} spec needs a 'plugin' key: {spec}")
            name = spec["plugin"]
            params = dict(spec.get("params") or {})
            extra = set(spec) - {"plugin", "params"}
            if extra:
                raise PluginError(
                    f"{category} spec for {name!r} has unexpected keys {sorted(extra)} "
                    f"(parameters go under 'params')"
                )
        else:
            raise PluginError(f"Bad {category} spec: {spec!r}")

        cls = self.get(category, name)
        unknown = sorted(set(params) - set(cls.PARAMS))
        if unknown:
            suggestions = {
                u: difflib.get_close_matches(u, cls.PARAMS, n=1) for u in unknown
            }
            raise PluginError(
                f"{category}:{name}: unknown parameter(s) {unknown} "
                f"(suggestions: {suggestions}); allowed: {sorted(cls.PARAMS)}"
            )
        for key, p in cls.PARAMS.items():
            if p.plugin is None:
                continue
            value = params.get(key, p.default)
            if p.many:
                value = [] if value is None else list(value)
                params[key] = [self.create(p.plugin, v) for v in value]
            else:
                params[key] = self.create(p.plugin, value)
        return cls(**params)

    # ------------------------------------------------------------ discovery
    def load_builtins(self):
        if self._builtins_loaded:
            return
        importlib.import_module("modegen.plugins")
        self._builtins_loaded = True

    def load_directory(self, path):
        path = os.path.abspath(path)
        if path in self._loaded_dirs or not os.path.isdir(path):
            return []
        self._loaded_dirs.add(path)
        loaded = []
        for fname in sorted(os.listdir(path)):
            if not fname.endswith(".py") or fname.startswith("_"):
                continue
            fpath = os.path.join(path, fname)
            tag = hashlib.md5(fpath.encode()).hexdigest()[:8]
            modname = f"modegen_user_{tag}_{fname[:-3]}"
            try:
                spec = importlib.util.spec_from_file_location(modname, fpath)
                module = importlib.util.module_from_spec(spec)
                sys.modules[modname] = module
                spec.loader.exec_module(module)
                loaded.append(fpath)
            except Exception as exc:
                self.load_errors.append((fpath, repr(exc), traceback.format_exc()))
        return loaded

    def discover(self, extra_dirs=(), include_user=True, include_env=True):
        self.load_builtins()
        dirs = []
        if include_user:
            dirs.append(os.path.join(PROJECT_ROOT, "user_plugins"))
        if include_env:
            env = os.environ.get("MODEGEN_PLUGIN_PATH", "")
            dirs.extend(d for d in env.split(os.pathsep) if d)
        dirs.extend(extra_dirs or ())
        for d in dirs:
            self.load_directory(d)
        return self

    def describe(self, category=None):
        cats = [category] if category else self.categories()
        out = []
        for c in cats:
            for n in self.names(c):
                info = self._plugins[c][n].describe()
                info["source"] = self.source(c, n)
                info["version"] = self.version(c, n) or info.get("version")
                out.append(info)
        return out


registry = PluginRegistry()


def register(category, name, override=False):
    """Decorator registering a plugin class in the global registry."""
    return registry.register(category, name, override=override)


def get_registry(extra_dirs=(), include_user=True):
    """Global registry with built-ins + user plugin directories loaded."""
    return registry.discover(extra_dirs=extra_dirs, include_user=include_user)
