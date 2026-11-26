#!/usr/bin/env python3
###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Parse a build dir's .ninja_log (+ build_timing.log, if present, as written
by build_time.sh) into a normalized CSV: one row per ninja edge plus three
summary "stage" rows (total / ninja_build / configure_install), for diffing
across commits with build_time_diff.py -- the build-time equivalent of
resource_usage_to_csv.py's per-kernel CSV.

Usage:
    python3 scripts/analysis/build_time_to_csv.py \\
        --build-dir build --gpu-target gfx950 --build-config all_backends \\
        --commit abc123 --out build_time_abc123.csv
"""

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_time_report as btr  # noqa: E402

FIELDS = ["gpu_target", "build_config", "commit", "category", "component", "key", "duration_s"]


def build_rows(build_dir, gpu_target, build_config, commit):
    ninja_log = os.path.join(build_dir, ".ninja_log")
    timing_log = os.path.join(build_dir, "build_timing.log")
    if not os.path.exists(ninja_log):
        sys.exit(f"error: no .ninja_log at {ninja_log} (did the build use -G Ninja?)")

    edges = btr.parse_ninja_log(ninja_log)
    if not edges:
        sys.exit(f"error: {ninja_log} has no edges recorded yet")

    wall_start = min(s for s, _ in edges.values())
    wall_end = max(e for _, e in edges.values())

    identity = {"gpu_target": gpu_target, "build_config": build_config, "commit": commit}
    rows = []
    for output, (start_ms, end_ms) in edges.items():
        rows.append({
            **identity,
            "category": "edge",
            "component": btr.classify_component(output),
            "key": output,
            "duration_s": f"{(end_ms - start_ms) / 1000.0:.3f}",
        })

    configure_s, install_s, total_s = btr.read_stage_timings(timing_log)
    ninja_build_s = (wall_end - wall_start) / 1000.0
    # total_s is 0.0 if build_timing.log is absent or predates the "total"
    # line (see read_stage_timings) -- fall back to the sum of the two
    # pieces this script can always measure on its own.
    stage_values = {
        "total": total_s if total_s else (ninja_build_s + configure_s + install_s),
        "ninja_build": ninja_build_s,
        "configure_install": configure_s + install_s,
    }
    for key, value in stage_values.items():
        rows.append({**identity, "category": "stage", "component": "", "key": key,
                     "duration_s": f"{value:.3f}"})
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--build-dir", required=True, help="build directory containing .ninja_log")
    ap.add_argument("--gpu-target", required=True, help="GPU target, e.g. gfx950")
    ap.add_argument(
        "--build-config", required=True, help="build_configs script used, e.g. all_backends"
    )
    ap.add_argument(
        "--commit", default="", help="git commit/ref (or cache key) this build came from"
    )
    ap.add_argument("--out", required=True, type=Path, help="CSV file to write")
    ap.add_argument(
        "--top", type=int, default=10, help="print top-N slowest edges to stdout (0 to disable)"
    )
    args = ap.parse_args()

    rows = build_rows(args.build_dir, args.gpu_target, args.build_config, args.commit)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    n_edges = sum(1 for r in rows if r["category"] == "edge")
    print(f"Wrote {len(rows)} rows ({n_edges} edges + 3 stage rows) to {args.out}")

    if args.top > 0:
        edges = sorted(
            (r for r in rows if r["category"] == "edge"),
            key=lambda r: float(r["duration_s"]),
            reverse=True,
        )
        print(f"\nTop {min(args.top, len(edges))} slowest edges ({args.gpu_target}/{args.build_config}):")
        for r in edges[: args.top]:
            print(f"  {float(r['duration_s']):7.2f}s  [{r['component']}]  {r['key']}")


if __name__ == "__main__":
    main()
