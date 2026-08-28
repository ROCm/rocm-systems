#!/usr/bin/env python3
###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Parse an LLVM optimization-record YAML file (as produced by
`-Xoffload-linker --plugin-opt=opt-remarks-filename=...`) for the `inline`
pass into a normalized CSV, one row per remark.

See COMPILER_OPTIONS_REFERENCE.md for the full remark schema and the LLVM
inliner cost model (Cost/Threshold/the "last call to a static function"
bonus). In practice every remark's Pass is "inline" and Name is one of:
  Inlined      (!Passed, has Cost+Threshold)
  AlwaysInline (!Passed, neither -- always-inline attribute)
  NeverInline  (!Missed, has Reason -- e.g. noinline function attribute)
  TooCostly    (!Missed, has Cost+Threshold)

DebugLoc/Hotness are not emitted as columns: this is a whole-program LTO link
with no profile data, so both are always absent in practice for this pass.

Usage:
    python3 scripts/analysis/lto_inline_remarks_to_csv.py \\
        --yaml remarks.yaml --out remarks.csv \\
        --arch gfx950 --build-config ipc_single --commit d37891abaa4e

Each invocation parses one complete opt-remarks YAML and overwrites --out in
full -- unlike resource_usage_to_csv.py, there is no concatenated-per-TU-log
merge problem here to make idempotent-by-key loading worthwhile.
"""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

FIELDS = [
    "arch",
    "build_config",
    "commit",
    "kind",
    "pass",
    "name",
    "caller_mangled",
    "caller_demangled",
    "callee_mangled",
    "callee_demangled",
    "cost",
    "threshold",
    "reason",
    "message",
]


# LLVM's opt-record YAML tags each document with the remark kind
# (!Passed / !Missed / !Analysis / ...). PyYAML's SafeLoader has no
# constructor for unknown tags and raises; register one for every kind seen
# in practice and stash the kind itself (tag name minus "!") under a
# synthetic "_Kind" key so it survives into the parsed row.
class RemarkLoader(yaml.SafeLoader):
    pass


def _construct_remark(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mapping["_Kind"] = node.tag.lstrip("!")
    return mapping


for _tag in (
    "!Passed",
    "!Missed",
    "!Analysis",
    "!AnalysisFPCommute",
    "!AnalysisAliasing",
    "!Failure",
):
    RemarkLoader.add_constructor(_tag, _construct_remark)


def _cxxfilt_path():
    for name in ("llvm-cxxfilt", "c++filt"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _demangle_batch(names, cxxfilt):
    if not cxxfilt or not names:
        return {n: n for n in names}
    proc = subprocess.run(
        [cxxfilt], input="\n".join(names), capture_output=True, text=True, check=True
    )
    demangled = proc.stdout.splitlines()
    return dict(zip(names, demangled))


def _flatten_args(args, demangle):
    parts = []
    for arg in args or []:
        for key, value in arg.items():
            if key in ("Callee", "Caller", "Function"):
                value = demangle.get(value, value)
            parts.append(str(value))
    return "".join(parts)


def _row_from_doc(d, demangle):
    args_list = d.get("Args") or []
    args_by_key = {}
    for arg in args_list:
        for key, value in arg.items():
            if key in ("Callee", "Caller", "Cost", "Threshold", "Reason"):
                args_by_key.setdefault(key, value)

    caller_mangled = args_by_key.get("Caller", d.get("Function", ""))
    callee_mangled = args_by_key.get("Callee", "")

    return {
        "kind": d.get("_Kind", ""),
        "pass": d.get("Pass", ""),
        "name": d.get("Name", ""),
        "caller_mangled": caller_mangled,
        "caller_demangled": demangle.get(caller_mangled, caller_mangled),
        "callee_mangled": callee_mangled,
        "callee_demangled": demangle.get(callee_mangled, callee_mangled),
        "cost": args_by_key.get("Cost", ""),
        "threshold": args_by_key.get("Threshold", ""),
        "reason": args_by_key.get("Reason", ""),
        "message": _flatten_args(args_list, demangle),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--yaml", required=True, type=Path, help="opt-remarks-filename output to parse")
    ap.add_argument("--out", required=True, type=Path, help="CSV file to write")
    ap.add_argument("--arch", default="", help="GPU target, e.g. gfx950")
    ap.add_argument("--build-config", default="", help="build_configs script used, e.g. ipc_single")
    ap.add_argument("--commit", default="", help="git commit/ref this build came from")
    args = ap.parse_args()

    if not args.yaml.exists():
        sys.exit(f"error: yaml not found: {args.yaml}")

    with open(args.yaml) as f:
        docs = [d for d in yaml.load_all(f, Loader=RemarkLoader) if d]

    mangled_names = set()
    for d in docs:
        if "Function" in d:
            mangled_names.add(d["Function"])
        for arg in d.get("Args", []) or []:
            for key, value in arg.items():
                if key in ("Callee", "Caller"):
                    mangled_names.add(value)

    demangle = _demangle_batch(sorted(mangled_names), _cxxfilt_path())

    rows = []
    for d in docs:
        row = _row_from_doc(d, demangle)
        row["arch"] = args.arch
        row["build_config"] = args.build_config
        row["commit"] = args.commit
        rows.append(row)

    # Sort for determinism: a `--lto-partitions>1` link processes functions
    # across parallel backend workers, so raw remark order is not stable
    # build-to-build even for identical source -- an unsorted diff/join would
    # be dominated by that reordering noise rather than real content changes.
    rows.sort(
        key=lambda r: (
            r["kind"],
            r["name"],
            r["caller_demangled"],
            r["callee_demangled"],
            r["cost"],
            r["threshold"],
        )
    )

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({col: r.get(col, "") for col in FIELDS})

    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
