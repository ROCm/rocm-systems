#!/usr/bin/env python3
###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
#
# Aggregates clang -ftime-trace JSON profiles (one per translation unit) into
# a cross-TU summary: where does total compile time go (parsing, template
# instantiation, codegen, optimization, ...), and which specific symbols
# (e.g. a particular rocSHMEM API x type instantiation) are the most expensive
# *summed across every TU that instantiates them*.
#
# Produce the input by configuring with -DROCSHMEM_TIME_TRACE=ON, then run:
#
#   python3 scripts/analysis/ftime_trace_summary.py --build-dir build
#
# Each compile job writes its own .json next to its object file under
# build/CMakeFiles/<target>.dir/..., so this walks the whole build tree.

import argparse
import collections
import json
import os
import sys

TRACE_SUFFIXES = (".json",)


def iter_trace_files(build_dir):
    for root, _dirs, files in os.walk(build_dir):
        for name in files:
            if name.endswith(TRACE_SUFFIXES):
                yield os.path.join(root, name)


def load_trace(path):
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or "traceEvents" not in data:
        return None
    return data["traceEvents"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-dir", default=".", help="build directory to scan for *.json time-trace files")
    ap.add_argument("--top", type=int, default=30, help="rows to show per table")
    args = ap.parse_args()

    category_totals = collections.Counter()
    category_counts = collections.Counter()
    detail_totals = collections.Counter()
    file_totals = collections.Counter()

    num_traces = 0
    for path in iter_trace_files(args.build_dir):
        events = load_trace(path)
        if events is None:
            continue
        num_traces += 1
        file_end = 0
        for ev in events:
            dur = ev.get("dur")
            name = ev.get("name")
            if not dur or not name:
                continue
            category_totals[name] += dur
            category_counts[name] += 1
            if name in ("InstantiateFunction", "InstantiateClass", "ParseTemplate"):
                detail = ev.get("args", {}).get("detail")
                if detail:
                    detail_totals[(name, detail)] += dur
            file_end = max(file_end, ev.get("ts", 0) + dur)
        file_totals[path] = file_end

    if num_traces == 0:
        print(f"no clang time-trace json files found under {args.build_dir}", file=sys.stderr)
        print("configure with -DROCSHMEM_TIME_TRACE=ON and rebuild first", file=sys.stderr)
        return 1

    print(f"parsed {num_traces} time-trace file(s)")
    print()
    print(f"{'category':<30}{'#events':>10}{'sum(s)':>12}")
    for name, total in category_totals.most_common(args.top):
        print(f"{name:<30}{category_counts[name]:>10}{total / 1e6:>12.2f}")

    print()
    print(f"hottest instantiated symbols (summed across all TUs), top {args.top}:")
    for (name, detail), total in detail_totals.most_common(args.top):
        print(f"{total / 1e6:8.2f}s  [{name}] {detail}")

    print()
    print(f"slowest {args.top} translation units (by compiler end timestamp):")
    for path, end in sorted(file_totals.items(), key=lambda kv: kv[1], reverse=True)[: args.top]:
        print(f"{end / 1e6:8.2f}s  {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
