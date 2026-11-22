"""Profiles: JSON (or YAML) documents selecting one plugin per extension point.

A profile has two halves. ``precompute`` is consumed once, when the tree is
built (it decides WHAT modes exist). ``runtime`` is consumed every time an
impact happens (it decides HOW impacts use those modes). The precompute half
is stored inside the tree pickle, so a tree always remembers how it was made;
the runtime half can be swapped freely on an existing tree.

Every profile is deep-merged on top of ``DEFAULT_PROFILE`` (the realistic
baseline), so a profile only needs to state what it changes. When a profile
names a DIFFERENT plugin than the default for some slot, that whole spec is
replaced rather than merged (so stale parameters never leak across plugins).
Setting a slot to ``null`` disables it (e.g. ``"plastic": null``).

A profile may declare ``"schema_version"`` (an integer); a profile without one is schema 1. A schema other than
``PROFILE_SCHEMA_VERSION`` produces a warning, never an error.
"""

import copy
import json
import os

from .compat import check_profile_schema, profile_schema_version
from .registry import PROJECT_ROOT, get_registry

DEFAULT_PROFILE = {
    "name": "realistic",
    "description": "Baseline: heterogeneous Sellan SOCP fracture + sparse plastic "
                   "modes + stress-gated routing (identical to the frozen "
                   "fracture_utility pipeline).",
    "precompute": {
        "fracture": {
            "plugin": "realistic.sellan_socp",
            "params": {
                "num_modes": 10,
                "weights": {"plugin": "realistic.uniform_faces"},
                "init": {"plugin": "realistic.laplacian"},
            },
        },
        "elastic": {
            "plugin": "realistic.linear_elastic",
            "params": {"num_modes": 30, "poisson_ratio": 0.3},
        },
        "plastic": {
            "plugin": "realistic.yield_modes",
            "params": {
                "num_modes": 16,
                "weights": {"plugin": "realistic.uniform_tets"},
            },
        },
    },
    "runtime": {
        "world": {
            "gravity": [0.0, -9.81, 0.0],
            "ground_y": None,
            "restitution": 0.3,
            "impact_kick": 0.5,
            "poisson_ratio": 0.3,
        },
        "default_tool": "contact",
        "tools": {
            "contact": {"plugin": "realistic.contact_patch",
                        "params": {"spread_radius": 0.05}},
        },
        "router": {"plugin": "realistic.stress_gate"},
        "fracture_response": {"plugin": "realistic.sellan_projection",
                              "params": {"threshold": 10.0, "mode": "contact"}},
        "plastic_response": {"plugin": "realistic.single_mode"},
        "post": [],
    },
}

_PRECOMPUTE_SLOTS = {
    "fracture": "fracture.generator",
    "elastic": "elastic.generator",
    "plastic": "plastic.generator",
}
_RUNTIME_SLOTS = {
    "router": "router",
    "fracture_response": "fracture.response",
    "plastic_response": "plastic.response",
}


def _is_spec(x):
    return isinstance(x, dict) and "plugin" in x


def deep_merge(base, override):
    """Merge ``override`` onto ``base`` (both left untouched)."""
    if _is_spec(base) and _is_spec(override) and base["plugin"] != override["plugin"]:
        return copy.deepcopy(override)
    if isinstance(base, dict) and isinstance(override, dict):
        out = copy.deepcopy(base)
        for k, v in override.items():
            out[k] = deep_merge(out[k], v) if k in out else copy.deepcopy(v)
        return out
    return copy.deepcopy(override)


def _parse_value(text):
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        lowered = str(text).lower()
        if lowered in ("none", "null"):
            return None
        if lowered in ("true", "false"):
            return lowered == "true"
        return text


class RuntimeConfig:
    """Instantiated runtime half of a profile."""

    def __init__(self, router, fracture_response, plastic_response, post,
                 tools, default_tool, world, render=None):
        self.router = router
        self.fracture_response = fracture_response
        self.plastic_response = plastic_response
        self.post = post
        self.tools = tools
        self.default_tool = default_tool
        self.world = world
        # (lift, smooth_iterations, smooth_weight); see modegen/lift.py (0.4.0).
        self.render = render or ("nearest", 0, 0.5)

    def tool(self, name=None):
        name = name or self.default_tool
        if name not in self.tools:
            raise KeyError(f"Unknown tool {name!r}; available: {sorted(self.tools)}")
        return name, self.tools[name]


class PrecomputeConfig:
    def __init__(self, fracture, elastic, plastic):
        self.fracture = fracture
        self.elastic = elastic
        self.plastic = plastic


class Profile:
    def __init__(self, data=None, name=None, source=None, merge_defaults=True):
        data = data or {}
        check_profile_schema(data, source)
        self.data = deep_merge(DEFAULT_PROFILE, data) if merge_defaults else copy.deepcopy(data)
        if name:
            self.data["name"] = name
        self.source = source

    # ---------------------------------------------------------- loading
    @staticmethod
    def resolve_path(name_or_path):
        if os.path.isfile(name_or_path):
            return name_or_path
        for ext in (".json", ".yaml", ".yml"):
            cand = os.path.join(PROJECT_ROOT, "profiles", name_or_path + ext)
            if os.path.isfile(cand):
                return cand
        raise FileNotFoundError(
            f"Profile {name_or_path!r} not found (looked for a file and in "
            f"{os.path.join(PROJECT_ROOT, 'profiles')})"
        )

    @classmethod
    def load(cls, name_or_path):
        path = cls.resolve_path(name_or_path)
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith((".yaml", ".yml")):
                try:
                    import yaml
                except ImportError as exc:
                    raise ImportError("YAML profiles need PyYAML (pip install pyyaml)") from exc
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        return cls(data, source=path)

    @classmethod
    def from_dict(cls, data):
        """Rebuild from a dict that is ALREADY merged (e.g. stored in a tree)."""
        return cls(data, merge_defaults=False)

    @staticmethod
    def available():
        d = os.path.join(PROJECT_ROOT, "profiles")
        if not os.path.isdir(d):
            return []
        return sorted(os.path.splitext(f)[0] for f in os.listdir(d)
                      if f.endswith((".json", ".yaml", ".yml")))

    # ---------------------------------------------------------- editing
    @property
    def name(self):
        return self.data.get("name", "unnamed")

    @property
    def schema_version(self):
        """Declared ``schema_version`` (profiles without one are schema 1)."""
        return profile_schema_version(self.data)

    def get(self, dotted, default=None):
        cur = self.data
        for part in dotted.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur

    def set(self, dotted, value):
        parts = dotted.split(".")
        cur = self.data
        for part in parts[:-1]:
            if not isinstance(cur.get(part), dict):
                cur[part] = {}
            cur = cur[part]
        cur[parts[-1]] = value

    def apply_overrides(self, overrides):
        """Apply ``key.path=value`` strings (value parsed as JSON when possible)."""
        for item in overrides or ():
            if "=" not in item:
                raise ValueError(f"Override must look like a.b.c=value, got {item!r}")
            key, raw = item.split("=", 1)
            self.set(key.strip(), _parse_value(raw.strip()))
        return self

    def to_dict(self):
        return copy.deepcopy(self.data)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    # ---------------------------------------------------------- building
    def build_precompute(self, registry=None):
        reg = registry or get_registry()
        pre = self.data.get("precompute") or {}
        built = {slot: reg.create(cat, pre.get(slot))
                 for slot, cat in _PRECOMPUTE_SLOTS.items()}
        if built["fracture"] is None:
            raise ValueError("profile.precompute.fracture must name a generator")
        return PrecomputeConfig(**built)

    def build_runtime(self, registry=None):
        reg = registry or get_registry()
        rt = self.data.get("runtime") or {}
        built = {slot: reg.create(cat, rt.get(slot))
                 for slot, cat in _RUNTIME_SLOTS.items()}
        if built["router"] is None:
            raise ValueError("profile.runtime.router must name a router")
        post = [reg.create("post", s) for s in (rt.get("post") or [])]
        tools = {name: reg.create("impact", spec)
                 for name, spec in (rt.get("tools") or {}).items() if spec}
        if not tools:
            raise ValueError("profile.runtime.tools must define at least one impact model")
        default_tool = rt.get("default_tool") or sorted(tools)[0]
        if default_tool not in tools:
            raise ValueError(f"default_tool {default_tool!r} not in tools {sorted(tools)}")
        world = dict(DEFAULT_PROFILE["runtime"]["world"])
        world.update(rt.get("world") or {})
        from .lift import make_lift
        return RuntimeConfig(post=[p for p in post if p is not None], tools=tools,
                             default_tool=default_tool, world=world,
                             render=make_lift(rt.get("render")), **built)

    def validate(self, registry=None):
        """Instantiate every plugin; return a list of error strings (empty = ok)."""
        errors = []
        for label, fn in (("precompute", self.build_precompute),
                          ("runtime", self.build_runtime)):
            try:
                fn(registry)
            except Exception as exc:
                errors.append(f"{label}: {exc}")
        return errors

    def __repr__(self):
        return f"Profile({self.name!r}, source={self.source!r})"
