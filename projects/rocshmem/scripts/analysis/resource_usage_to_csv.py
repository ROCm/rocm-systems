#!/usr/bin/env python3
###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Parse a resource_usage_summary.log (produced by a build compiled with
-Rpass-analysis=kernel-resource-usage) into a normalized CSV, one row per kernel.

Usage:
    python3 scripts/analysis/resource_usage_to_csv.py \\
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
import re
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

# A real mangled (_Z...) or plain C (extern "C") symbol never contains
# whitespace, '/', '[', ']', or '%'. Block-buffered stdio doesn't guarantee a
# single write() per line for non-tty output, so two build processes sharing
# one fd (e.g. hipcc's remark output and cmake's own "[ 70%] Building ..."
# progress lines, both going through the same `tee`) can still splice
# mid-line even though each *complete* line is otherwise routed correctly by
# its location prefix -- this has been observed to silently merge a
# "Function Name:" line with a following, unrelated build-log line. Validate
# the value so that corruption produces a discarded row instead of a bogus
# kernel with a garbage name.
_VALID_SYMBOL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.$]*$")

# Every non-name metric value is either a plain integer or True/False.
_VALID_METRIC_VALUE = re.compile(r"^(?:\d+|True|False)$")

# Columns a real remark block always populates (everything but the identity
# fields set when the row is opened). A row that reaches finalization with
# none of these attached only happens when its "Function Name:" line's own
# location prefix was corrupted (see _VALID_SYMBOL_NAME above) rather than
# the value: the metric lines still carry the correct, uncorrupted prefix,
# so they route to a row keyed by that prefix that was never opened, and are
# silently dropped -- leaving this one an empty husk. Such a row must be
# discarded too, or it round-trips through the CSV as all-empty fields that
# downstream tooling coerces to 0, masquerading as a real (and dramatic)
# regression.
_METRIC_COLUMNS = frozenset(KEY_TO_COLUMN.values()) - {"mangled_name"}


def _finalize(open_rows, loc):
    row = open_rows.pop(loc)
    if not any(col in row for col in _METRIC_COLUMNS):
        print(
            f"warning: discarding kernel row with no metrics attached at "
            f"{loc[0]}:{loc[1]} (its 'Function Name:' line's location prefix "
            f"was likely corrupted by build-log splicing, orphaning its "
            f"metric lines): {row.get('mangled_name', '?')!r}",
            file=sys.stderr,
        )
        return None
    return row


def parse_log(log_path: Path):
    """Yield one dict per kernel found in a resource_usage_summary.log file.

    The log is a straight concatenation of `-Rpass-analysis=kernel-resource-usage`
    remark output from every translation unit the build compiles. When the build
    runs with `-j` > 1 (the normal case), multiple compiler processes emit their
    own multi-line remark blocks to the same shared log concurrently -- each
    individual line is written atomically, but there is no guarantee that one
    kernel's ~9-line block stays contiguous, so two (or more) kernels' blocks can
    interleave line-by-line. Every line in a remark block (not just the "Function
    Name" line) repeats that block's own "<file>:<line>:<col>:" location prefix,
    so a kernel's lines can always be routed back to it by that prefix alone --
    tracking one open row keyed by that location (instead of a single global
    "current row" that blindly accepts whatever line comes next) makes parsing
    immune to interleaving: a metric line can only ever update the row that was
    opened by a "Function Name" line at that exact same location.
    """
    open_rows = {}
    for log_line in _iter_lines(log_path):
        line = log_line.rstrip("\n")
        # Expected: "<file>:<line>:<col>: <Key> [unit]: <value>"
        parts = line.split(":", 3)
        if len(parts) != 4:
            continue
        source_file, lineno, _col, rest = parts
        if ": " not in rest.lstrip():
            continue
        key_part, _, value = rest.strip().partition(":")
        key = key_part.split(" [")[0].strip()
        value = value.strip()
        column = KEY_TO_COLUMN.get(key)
        if column is None:
            continue
        loc = (source_file.strip(), lineno.strip(), _col.strip())
        if column == "mangled_name":
            # A fresh "Function Name" at a location that already has an open row
            # means that row's block is complete (another instantiation sharing
            # the same source location has started) -- finalize the old one.
            if loc in open_rows:
                finalized = _finalize(open_rows, loc)
                if finalized is not None:
                    yield finalized
            if not _VALID_SYMBOL_NAME.match(value):
                print(
                    f"warning: discarding corrupted 'Function Name:' line at "
                    f"{loc[0]}:{loc[1]} (likely spliced with an unrelated build-log "
                    f"line): {value!r}",
                    file=sys.stderr,
                )
                continue
            open_rows[loc] = {"source_file": loc[0], "line": loc[1], column: value}
        else:
            row = open_rows.get(loc)
            if row is None:
                # A metric line with no open row at its own location means the
                # "Function Name" line that should have opened it was itself
                # lost to interleaving corruption -- nothing to attach this to.
                continue
            if not _VALID_METRIC_VALUE.match(value):
                # Splicing can also merge two metric lines (same location
                # prefix) into one physical line, not just corrupt "Function
                # Name:" -- drop just this field, not the whole row.
                print(
                    f"warning: discarding corrupted {key!r} value at "
                    f"{loc[0]}:{loc[1]} (likely spliced with a following "
                    f"build-log line): {value!r}",
                    file=sys.stderr,
                )
                continue
            row[column] = value
    for loc in list(open_rows):
        finalized = _finalize(open_rows, loc)
        if finalized is not None:
            yield finalized


def _iter_lines(log_path: Path):
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            yield line


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
            (
                r["arch"],
                r["build_config"],
                r["source_file"],
                r["line"],
                r["mangled_name"],
            ): r
            for r in reader
        }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--log", required=True, type=Path, help="path to resource_usage_summary.log"
    )
    ap.add_argument("--arch", required=True, help="GPU target, e.g. gfx950")
    ap.add_argument(
        "--build-config",
        required=True,
        help="build_configs script used, e.g. ipc_single",
    )
    ap.add_argument(
        "--commit", default="", help="optional git commit/ref this build came from"
    )
    ap.add_argument(
        "--out", required=True, type=Path, help="CSV file to write/append to"
    )
    ap.add_argument(
        "--top",
        type=int,
        default=10,
        help="print top-N kernels by VGPRs to stdout (0 to disable)",
    )
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
        key = (
            args.arch,
            args.build_config,
            r["source_file"],
            r["line"],
            r["mangled_name"],
        )
        existing[key] = r

    # Sort by demangled_name (not line number) within each source file so that
    # renaming/retyping kernels (e.g. AMOStandardTest<int> -> AMOStandardTest_int<...>)
    # doesn't reshuffle row order -- related kernels stay adjacent alphabetically,
    # so two CSVs from before/after a rename can be opened side-by-side and scrolled
    # in tandem without any cross-CSV matching logic.
    rows = sorted(
        existing.values(),
        key=lambda r: (
            r["arch"],
            r["build_config"],
            r["source_file"],
            r["demangled_name"],
        ),
    )
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({col: r.get(col, "") for col in FIELDS})

    print(f"Wrote {len(rows)} rows ({len(parsed)} from this log) to {args.out}")

    if args.top > 0:
        this_run = sorted(parsed, key=lambda r: int(r.get("VGPRs") or 0), reverse=True)
        print(
            f"\nTop {min(args.top, len(this_run))} kernels by VGPRs ({args.arch}/{args.build_config}):"
        )
        for r in this_run[: args.top]:
            name = demangled.get(r["mangled_name"], r["mangled_name"])
            print(
                f"  VGPR={r.get('VGPRs',''):>4} SGPR={r.get('TotalSGPRs',''):>4} "
                f"Scratch={r.get('ScratchBytesPerLane','0'):>4} Occ={r.get('OccupancyWavesPerSIMD',''):>3} "
                f"  {name}  ({r['source_file']}:{r['line']})"
            )


if __name__ == "__main__":
    main()
