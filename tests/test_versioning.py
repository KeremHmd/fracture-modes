"""numpy-only tests for the version numbers and compatibility checks.

Run with ``python tests/test_versioning.py`` or ``python tests/run_all.py``.
"""

import os
import re
import sys
import types
import warnings

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import modegen  # noqa: E402
from modegen import Profile, get_registry  # noqa: E402
from modegen._version import PROFILE_SCHEMA_VERSION, TREE_FORMAT_VERSION, __version__  # noqa: E402
from modegen.compat import ModegenVersionWarning, check_tree, tree_version_info  # noqa: E402
from modegen.precompute import precompute_plugin_versions  # noqa: E402  (numpy-only at import)

REG = get_registry()


def _caught(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = fn()
    return out, [x for x in w if issubclass(x.category, ModegenVersionWarning)]


def test_version_is_single_sourced_semver():
    assert modegen.__version__ == __version__
    assert re.match(r"^\d+\.\d+\.\d+((a|b|rc|\.dev)\d+)?$", __version__), __version__
    hits = []
    for folder in ("modegen", "tools"):
        for dp, _, files in os.walk(os.path.join(ROOT, folder)):
            for f in files:
                if f.endswith(".py") and f != "_version.py":
                    with open(os.path.join(dp, f), encoding="utf-8") as fh:
                        if f'"{__version__}"' in fh.read():
                            hits.append(f)
    assert not hits, f"version literal duplicated in {hits}"


def test_every_plugin_has_a_version():
    for info in REG.describe():
        assert re.match(r"^\d+\.\d+\.\d+", info["version"]), info
        assert REG.version(info["category"], info["name"]) == info["version"]


def test_profiles_without_schema_load_silently():
    for name in Profile.available():
        p, w = _caught(lambda: Profile.load(name))
        assert not w, (name, [str(x.message) for x in w])
        assert p.schema_version >= 1


def test_profile_schema_mismatch_warns_not_crashes():
    p, w = _caught(lambda: Profile({"schema_version": PROFILE_SCHEMA_VERSION + 1}))
    assert w and "schema_version" in str(w[0].message)
    assert not p.validate(REG)
    _, w = _caught(lambda: Profile({"schema_version": PROFILE_SCHEMA_VERSION}))
    assert not w


def test_old_tree_without_metadata_loads_with_defaults():
    old = types.SimpleNamespace(modegen_api_version="1.0", modegen_profile={})
    info, w = _caught(lambda: check_tree(old, registry=REG))
    assert not w
    assert info["tree_format"] == 1 and info["modegen_version"] is None
    baseline = types.SimpleNamespace()          # plain FractureTree from scripts/
    info, w = _caught(lambda: check_tree(baseline, registry=REG))
    assert not w and info["api_version"] is None


def test_tree_mismatches_warn():
    t = types.SimpleNamespace(modegen_tree_format=TREE_FORMAT_VERSION + 1,
                              modegen_api_version="9.0",
                              modegen_plugins={"plastic": [dict(category="plastic.generator",
                                                                name="nobody.missing",
                                                                version="1.0.0")]})
    _, w = _caught(lambda: check_tree(t, registry=REG))
    msgs = " | ".join(str(x.message) for x in w)
    assert "format" in msgs and "plugin API" in msgs and "not registered" in msgs, msgs


def test_precompute_plugin_versions_include_nested_hooks():
    gens = Profile().build_precompute(REG)
    rec = precompute_plugin_versions(gens)
    names = [e["name"] for e in rec["fracture"]]
    assert names[0] == "realistic.sellan_socp"
    assert "realistic.uniform_faces" in names and "realistic.laplacian" in names
    assert all(e["version"] for es in rec.values() for e in es)
    t = types.SimpleNamespace(modegen_tree_format=TREE_FORMAT_VERSION, modegen_api_version="1.0",
                              modegen_plugins=rec)
    _, w = _caught(lambda: check_tree(t, registry=REG))
    assert not w
    assert tree_version_info(t)["plugins"] == rec


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failed += 1
            import traceback
            print(f"FAIL {fn.__name__}: {exc!r}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
