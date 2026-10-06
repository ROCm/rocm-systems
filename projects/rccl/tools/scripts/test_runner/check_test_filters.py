#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""
Report gtest filter clauses in runner configs that match no test.

Every configuration is checked, including ones whose suites are disabled. Each
binary is listed once with --gtest_list_tests; a positive filter clause that
matches nothing exits non-zero.

Usage:
    check_test_filters.py --bin-dir BUILD_DIR [CONFIG.json ...]
"""

import argparse
import fnmatch
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.test_config import TestConfigProcessor  # noqa: E402


def parse_gtest_list(output):
    """Turn --gtest_list_tests output into full 'Suite.Test' names."""
    names, suite = [], None
    for line in output.splitlines():
        text = line.split("#", 1)[0].rstrip()
        if not text:
            continue
        if not text.startswith(" "):
            suite = text if text.endswith(".") else None
        elif suite:
            names.append(suite + text.strip())
    return names


def positive_clauses(test_filter):
    """Positive patterns of a gtest filter; an empty filter means '*'."""
    positive = (test_filter or "*").split("-", 1)[0]
    return [p for p in positive.split(":") if p] or ["*"]


def unmatched_clauses(test_filter, names):
    return [p for p in positive_clauses(test_filter)
            if not any(fnmatch.fnmatchcase(n, p) for n in names)]


def config_tests(config_file):
    """Yield (config name, test entry, binary) for every gtest entry in a file."""
    proc = TestConfigProcessor(str(config_file))
    for cname in proc.config.get("test_configurations", {}):
        combined = proc.combine_configs(cname)
        for test in combined.get("tests", []):
            if isinstance(test, str):
                test = {"name": test}
            if not test.get("is_gtest", combined.get("is_gtest", True)):
                continue
            binary = test.get("binary") or combined.get("binary")
            if binary:
                yield cname, test, os.path.basename(str(binary))


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin-dir", required=True, help="directory holding the built test binaries")
    ap.add_argument("--strict", action="store_true", help="fail when a referenced binary is not built")
    ap.add_argument("configs", nargs="*", help="config files (default: configs/*.json)")
    args = ap.parse_args()
    configs = [Path(c) for c in args.configs] or sorted((here / "configs").glob("*.json"))

    listings, missing, failures = {}, set(), 0
    for config_file in configs:
        for cname, test, binary in config_tests(config_file):
            if binary not in listings:
                path = Path(args.bin_dir) / binary
                if not path.is_file():
                    missing.add(binary)
                    listings[binary] = None
                    continue
                out = subprocess.run([str(path), "--gtest_list_tests"], capture_output=True,
                                     text=True, timeout=300, check=False)
                listings[binary] = parse_gtest_list(out.stdout) if out.returncode == 0 else None
                if listings[binary] is None:
                    print(f"ERROR: {binary} --gtest_list_tests exited {out.returncode}")
                    failures += 1
            if not listings[binary]:
                continue
            for clause in unmatched_clauses(test.get("test_filter"), listings[binary]):
                print(f"{config_file.name}: {cname}.{test.get('name')}: '{clause}' matches no test in {binary}")
                failures += 1
    for binary in sorted(missing):
        print(f"{'ERROR' if args.strict else 'WARN'}: {binary} not found in {args.bin_dir}; its filters were not checked")
    failures += len(missing) if args.strict else 0
    print(f"{failures} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
