#!/usr/bin/env python3
###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
#
# Summarizes a Ninja build log (.ninja_log) into a per-component time budget
# and a list of the slowest individual build edges (compiles/links). Ninja
# already timestamps every edge it runs, so this costs nothing extra at build
# time - it's just read after the fact. Run after a build:
#
#   python3 scripts/analysis/ninja_build_timing.py --ninja-log build/.ninja_log
#
# scripts/build_configs/all_backends calls this automatically.

import argparse
import collections
import os
import re
import sys

GROUP_PATTERNS = [
    (re.compile(r"tests/unit_tests/"), "tests/unit"),
    (re.compile(r"tests/functional_tests/"), "tests/functional"),
    (re.compile(r"tests/rocprofiler_register/"), "tests/rocprofiler_register"),
    (re.compile(r"(^|/)examples/"), "examples"),
    (re.compile(r"(^|/)tools/"), "tools"),
    (re.compile(r"src/gda/"), "src/gda"),
    (re.compile(r"src/reverse_offload/"), "src/reverse_offload"),
    (re.compile(r"src/ipc/"), "src/ipc"),
    (re.compile(r"src/sdma/"), "src/sdma"),
    (re.compile(r"src/net/"), "src/net"),
    (re.compile(r"src/host/"), "src/host"),
    (re.compile(r"src/containers/"), "src/containers"),
    (re.compile(r"src/memory/"), "src/memory"),
    (re.compile(r"src/sync/"), "src/sync"),
    (re.compile(r"src/bootstrap/"), "src/bootstrap"),
    (re.compile(r"src/memfabric/"), "src/memfabric"),
    (re.compile(r"src/device/"), "src/device"),
    (re.compile(r"CMakeFiles/rocshmem\.dir/"), "src/(top-level)"),
]


def classify(output):
    for pattern, group in GROUP_PATTERNS:
        if pattern.search(output):
            return group
    if output.endswith((".o", ".obj")):
        return "other-compile"
    return "link/other"


def parse_ninja_log(path):
    # Format: start_ms \t end_ms \t restat_mtime \t output \t command_hash
    # One line per edge run; a target rebuilt across incremental builds gets
    # multiple lines, so keep only the most recent (last) one per output.
    latest = {}
    with open(path) as f:
        first = f.readline()
        if not first.startswith("# ninja log"):
            f.seek(0)
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            try:
                start_ms = int(parts[0])
                end_ms = int(parts[1])
            except ValueError:
                continue
            output = parts[3]
            latest[output] = (start_ms, end_ms)
    return latest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ninja-log", default=".ninja_log", help="path to .ninja_log")
    ap.add_argument("--top", type=int, default=30, help="how many slowest edges to list")
    args = ap.parse_args()

    if not os.path.exists(args.ninja_log):
        print(f"no ninja log at {args.ninja_log} (did the build use -G Ninja?)", file=sys.stderr)
        return 1

    latest = parse_ninja_log(args.ninja_log)
    if not latest:
        print(f"{args.ninja_log} has no edges recorded yet", file=sys.stderr)
        return 1

    durations = sorted(
        ((end - start, out) for out, (start, end) in latest.items()), reverse=True
    )

    wall_start = min(start for start, _ in latest.values())
    wall_end = max(end for _, end in latest.values())

    group_totals = collections.Counter()
    group_counts = collections.Counter()
    for dur, out in durations:
        group = classify(out)
        group_totals[group] += dur
        group_counts[group] += 1

    print(f"ninja-tracked wall time: {(wall_end - wall_start) / 1000:.1f}s over {len(latest)} edges")
    print(f"sum of all edge durations (cpu-seconds, parallel work): {sum(group_totals.values()) / 1000:.1f}s")
    print()
    print(f"{'component':<24}{'#edges':>8}{'cpu-sum(s)':>12}{'avg(s)':>9}")
    for group, total in group_totals.most_common():
        count = group_counts[group]
        print(f"{group:<24}{count:>8}{total / 1000:>12.1f}{(total / count) / 1000:>9.2f}")

    print()
    print(f"slowest {args.top} edges:")
    for dur, out in durations[: args.top]:
        print(f"{dur / 1000:8.1f}s  {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
