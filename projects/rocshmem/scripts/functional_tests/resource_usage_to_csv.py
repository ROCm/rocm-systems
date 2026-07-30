#!/usr/bin/env python3
###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Parse a resource_usage_summary.log (produced by a build compiled with
-Rpass-analysis=kernel-resource-usage) into a normalized CSV, one row per kernel.

Usage:
    python3 scripts/functional_tests/resource_usage_to_csv.py \\
        --log build/resource_usage_summary.log \\
        --arch gfx950 --build-config ipc_single \\
        --out resource_usage.csv

Rows are written to --out. If --out already exists, its rows are loaded and
updated: re-running with the same (arch, build-config, source_file, line,
mangled_name) tuple replaces the existing row instead of duplicating it, so
re-generating a build's numbers is idempotent.
"""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

FIELDS = [
    "arch",
    "build_config",
    "commit",
    "source_file",
    "line",
    "mangled_name",
    "demangled_name",
    "TotalSGPRs",
    "VGPRs",
    "AGPRs",
    "ScratchBytesPerLane",
    "DynamicStack",
    "OccupancyWavesPerSIMD",
    "SGPRsSpill",
    "VGPRsSpill",
    "LDSBytesPerBlock",
]

# Maps the label text before ':' (bracketed units stripped) to a CSV column.
KEY_TO_COLUMN = {
    "Function Name": "mangled_name",
    "TotalSGPRs": "TotalSGPRs",
    "VGPRs": "VGPRs",
    "AGPRs": "AGPRs",
    "ScratchSize": "ScratchBytesPerLane",
    "Dynamic Stack": "DynamicStack",
    "Occupancy": "OccupancyWavesPerSIMD",
    "SGPRs Spill": "SGPRsSpill",
    "VGPRs Spill": "VGPRsSpill",
    "LDS Size": "LDSBytesPerBlock",
}


def parse_log(log_path: Path):
    """Yield one dict per kernel found in a resource_usage_summary.log file."""
    row = None
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            # Expected: "<file>:<line>:<col>: <Key> [unit]: <value>"
            parts = line.split(":", 3)
            if len(parts) != 4:
                continue
            source_file, lineno, _col, rest = parts
            if not lineno.isdigit() or ":" not in rest.strip() and "Function Name" not in rest:
                pass
            if ": " not in rest.lstrip():
                continue
            key_part, _, value = rest.strip().partition(":")
            key = key_part.split(" [")[0].strip()
            value = value.strip()
            column = KEY_TO_COLUMN.get(key)
            if column is None:
                continue
            if column == "mangled_name":
                if row is not None:
                    yield row
                row = {"source_file": source_file.strip(), "line": lineno, column: value}
            else:
                if row is None:
                    continue
                row[column] = value
        if row is not None:
            yield row


def demangle_all(names):
    """Batch-demangle via a single c++filt invocation; falls back to raw names."""
    if not names or shutil.which("c++filt") is None:
        return {n: n for n in names}
    proc = subprocess.run(
        ["c++filt"], input="\n".join(names), capture_output=True, text=True, check=True
    )
    demangled = proc.stdout.splitlines()
    return dict(zip(names, demangled))


def load_existing(csv_path: Path):
    if not csv_path.exists():
        return {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return {
            (r["arch"], r["build_config"], r["source_file"], r["line"], r["mangled_name"]): r
            for r in reader
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, type=Path, help="path to resource_usage_summary.log")
    ap.add_argument("--arch", required=True, help="GPU target, e.g. gfx950")
    ap.add_argument("--build-config", required=True, help="build_configs script used, e.g. ipc_single")
    ap.add_argument("--commit", default="", help="optional git commit/ref this build came from")
    ap.add_argument("--out", required=True, type=Path, help="CSV file to write/append to")
    ap.add_argument("--top", type=int, default=10, help="print top-N kernels by VGPRs to stdout (0 to disable)")
    args = ap.parse_args()

    if not args.log.exists():
        sys.exit(f"error: log not found: {args.log}")

    parsed = list(parse_log(args.log))
    if not parsed:
        sys.exit(f"error: no 'Function Name:' blocks found in {args.log}")

    demangled = demangle_all(sorted({r["mangled_name"] for r in parsed}))

    existing = load_existing(args.out)
    for r in parsed:
        r["arch"] = args.arch
        r["build_config"] = args.build_config
        r["commit"] = args.commit
        r["demangled_name"] = demangled.get(r["mangled_name"], r["mangled_name"])
        for col in FIELDS:
            r.setdefault(col, "")
        key = (args.arch, args.build_config, r["source_file"], r["line"], r["mangled_name"])
        existing[key] = r

    # Sort by demangled_name (not line number) within each source file so that
    # renaming/retyping kernels (e.g. AMOStandardTest<int> -> AMOStandardTest_int<...>)
    # doesn't reshuffle row order -- related kernels stay adjacent alphabetically,
    # so two CSVs from before/after a rename can be opened side-by-side and scrolled
    # in tandem without any cross-CSV matching logic.
    rows = sorted(
        existing.values(),
        key=lambda r: (r["arch"], r["build_config"], r["source_file"], r["demangled_name"]),
    )
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({col: r.get(col, "") for col in FIELDS})

    print(f"Wrote {len(rows)} rows ({len(parsed)} from this log) to {args.out}")

    if args.top > 0:
        this_run = sorted(parsed, key=lambda r: int(r.get("VGPRs") or 0), reverse=True)
        print(f"\nTop {min(args.top, len(this_run))} kernels by VGPRs ({args.arch}/{args.build_config}):")
        for r in this_run[: args.top]:
            name = demangled.get(r["mangled_name"], r["mangled_name"])
            print(
                f"  VGPR={r.get('VGPRs',''):>4} SGPR={r.get('TotalSGPRs',''):>4} "
                f"Scratch={r.get('ScratchBytesPerLane','0'):>4} Occ={r.get('OccupancyWavesPerSIMD',''):>3} "
                f"  {name}  ({r['source_file']}:{r['line']})"
            )


if __name__ == "__main__":
    main()
