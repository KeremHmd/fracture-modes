"""Run every ``tests/test_*.py`` file (numpy-only unit tests).

Usage: ``python tests/run_all.py [-k SUBSTRING]``

Each file is imported as a module and its ``test_*`` functions are called in
name order, exactly as ``python tests/test_<x>.py`` would. Exit code 1 if any
test fails or a file cannot be imported.
"""

import argparse
import importlib.util
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))


def run_file(path, pattern=None):
    name = "modegen_test_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        print(f"ERROR importing {os.path.basename(path)}")
        traceback.print_exc()
        return 0, 1
    fns = [v for k, v in sorted(vars(module).items())
           if k.startswith("test_") and callable(v) and (not pattern or pattern in k)]
    passed = failed = 0
    for fn in fns:
        try:
            fn()
            passed += 1
            print(f"PASS {os.path.basename(path)}::{fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {os.path.basename(path)}::{fn.__name__}: {exc!r}")
            traceback.print_exc()
    return passed, failed


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-k", default=None, help="only run tests whose name contains this")
    args = ap.parse_args()
    files = sorted(f for f in os.listdir(HERE) if f.startswith("test_") and f.endswith(".py"))
    total_p = total_f = 0
    for f in files:
        p, fl = run_file(os.path.join(HERE, f), args.k)
        total_p += p
        total_f += fl
    print(f"\n{total_p}/{total_p + total_f} passed in {len(files)} file(s)")
    return 1 if total_f else 0


if __name__ == "__main__":
    sys.exit(main())
