# List registered plugins, inspect / validate profiles.
#
# Usage:
#   python tools\modegen_plugins.py [--category CAT] [--json]
#                                   [--profiles] [--profile NAME|PATH] [--set key=value ...]
#                                   [--plugin-dir DIR ...]

import argparse
import json

import _common  # noqa: F401

parser = argparse.ArgumentParser(description="Inspect modegen plugins and profiles.")
parser.add_argument("--category", default=None)
parser.add_argument("--json", action="store_true", help="Machine-readable output.")
parser.add_argument("--profiles", action="store_true", help="List shipped profiles.")
parser.add_argument("--profile", default=None, help="Validate + print a merged profile.")
parser.add_argument("--set", action="append", default=[])
_common.add_plugin_args(parser)
_common.add_version_arg(parser)
args = parser.parse_args()

from modegen import Profile  # noqa: E402

reg = _common.registry_from_args(args)

if args.profiles:
    for name in Profile.available():
        p = Profile.load(name)
        errs = p.validate(reg)
        print(f"{name:18s} {'OK ' if not errs else 'BAD'}  {p.data.get('description', '')}")
        for e in errs:
            print("    ", e)
    raise SystemExit(0)

if args.profile:
    p = Profile.load(args.profile).apply_overrides(args.set)
    print(json.dumps(p.to_dict(), indent=2))
    errs = p.validate(reg)
    print("\nVALID" if not errs else "\nINVALID:\n  " + "\n  ".join(errs))
    raise SystemExit(1 if errs else 0)

info = reg.describe(args.category)
if args.json:
    print(json.dumps(info, indent=2, default=str))
    raise SystemExit(0)
current = None
for d in info:
    if d["category"] != current:
        current = d["category"]
        print(f"\n=== {current} ===")
    print(f"\n  {d['name']}  v{d['version']}   [{d['source']}]")
    if d["description"]:
        print(f"    {d['description']}")
    for k, p in d["params"].items():
        kind = f" <{p['plugin']}{'[]' if p['many'] else ''}>" if p["plugin"] else ""
        ch = f" choices={list(p['choices'])}" if p["choices"] else ""
        print(f"    - {k}{kind} = {p['default']!r}{ch}  {p['doc']}")
