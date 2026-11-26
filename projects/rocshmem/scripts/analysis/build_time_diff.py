#!/usr/bin/env python3
###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Diff two build_time CSVs (see build_time_to_csv.py) produced from two
different commits/builds of the same project, matched on `key` (the stage
name, or the ninja edge's output path).

Wall-clock build time is noisy run-to-run (machine load, thermal throttling,
filesystem cache state) in a way GPU resource usage isn't -- a real
before/after decision needs a matched-fresh-pair rebuild of both sides (see
build_time_compare.sh's --force-rebuild), not a fresh build diffed against a
stale cached one from a different day/load.

Usage:
    python3 scripts/analysis/build_time_diff.py \\
        --baseline build_time_commitA.csv \\
        --branch   build_time_commitB.csv \\
        --out      build_time_diff.csv \\
        --chart    build_time_diff.png \\
        --top 30
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

STAGE_ORDER = ["total", "ninja_build", "configure_install"]
STAGE_LABELS = {
    "total": "total (wall)",
    "ninja_build": "build (ninja)",
    "configure_install": "configure+install (approx)",
}


def load(path: Path):
    stages, edges = {}, {}
    component_totals = defaultdict(float)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            dur = float(r["duration_s"])
            if r["category"] == "stage":
                stages[r["key"]] = dur
            else:
                edges[r["key"]] = dur
                component_totals[r["component"]] += dur
    return stages, edges, component_totals


def pct(delta, baseline):
    if baseline == 0:
        return None  # undefined (new, baseline was zero)
    return delta / baseline * 100.0


def _fmt_pct(p):
    return "" if p is None else f"{p:.1f}"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--branch", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="diff CSV to write")
    ap.add_argument("--chart", type=Path, help="PNG chart of the top per-component deltas")
    ap.add_argument("--top", type=int, default=30, help="edges to show in report/chart")
    args = ap.parse_args()

    base_stages, base_edges, base_components = load(args.baseline)
    branch_stages, branch_edges, branch_components = load(args.branch)

    rows = []  # (category, key, baseline_s, branch_s, delta_s, delta_pct)

    print("=== stage timing ===")
    print(f"{'stage':<28}{'baseline':>10}{'branch':>10}{'delta':>10}{'delta%':>9}")
    for key in STAGE_ORDER:
        b, r = base_stages.get(key, 0.0), branch_stages.get(key, 0.0)
        d = r - b
        p = pct(d, b)
        rows.append(("stage", key, b, r, d, p))
        print(f"{STAGE_LABELS[key]:<28}{b:>9.2f}s{r:>9.2f}s{d:>+9.2f}s{_fmt_pct(p):>8}%")

    print("\n=== per-component ninja time ===")
    comp_rows = []
    for comp in sorted(set(base_components) | set(branch_components)):
        b, r = base_components.get(comp, 0.0), branch_components.get(comp, 0.0)
        comp_rows.append((comp, b, r, r - b))
    comp_rows.sort(key=lambda t: abs(t[3]), reverse=True)
    print(f"{'component':<24}{'baseline':>10}{'branch':>10}{'delta':>10}")
    for comp, b, r, d in comp_rows:
        rows.append(("component", comp, b, r, d, pct(d, b)))
        print(f"{comp:<24}{b:>9.1f}s{r:>9.1f}s{d:>+9.1f}s")

    common_keys = set(base_edges) & set(branch_edges)
    edge_rows = sorted(
        ((key, base_edges[key], branch_edges[key], branch_edges[key] - base_edges[key])
         for key in common_keys),
        key=lambda t: abs(t[3]),
        reverse=True,
    )
    print(f"\n=== top {args.top} edges by |delta| (present in both) ===")
    for key, b, r, d in edge_rows[: args.top]:
        rows.append(("edge", key, b, r, d, pct(d, b)))
        print(f"{d:>+8.2f}s  ({b:6.2f}s -> {r:6.2f}s)  {key}")

    added = sorted(set(branch_edges) - set(base_edges), key=lambda k: branch_edges[k], reverse=True)
    removed = sorted(set(base_edges) - set(branch_edges), key=lambda k: base_edges[k], reverse=True)
    if added:
        print(f"\n=== {len(added)} edges only in branch (top {min(args.top, len(added))}) ===")
        for key in added[: args.top]:
            r = branch_edges[key]
            rows.append(("edge-added", key, 0.0, r, r, None))
            print(f"  +{r:6.2f}s  {key}")
    if removed:
        print(f"\n=== {len(removed)} edges only in baseline (top {min(args.top, len(removed))}) ===")
        for key in removed[: args.top]:
            b = base_edges[key]
            rows.append(("edge-removed", key, b, 0.0, -b, None))
            print(f"  -{b:6.2f}s  {key}")

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "key", "baseline_s", "branch_s", "delta_s", "delta_pct"])
        for category, key, b, r, d, p in rows:
            writer.writerow([category, key, f"{b:.3f}", f"{r:.3f}", f"{d:.3f}", _fmt_pct(p)])
    print(f"\nWrote {len(rows)} rows to {args.out}")

    if args.chart:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping chart", file=sys.stderr)
            return

        top_comp = comp_rows[: args.top]
        labels = [c for c, *_ in top_comp]
        deltas = [d for _, _, _, d in top_comp]
        colors = ["#d62728" if d > 0 else "#2ca02c" for d in deltas]

        fig_h = max(3.0, 0.32 * len(labels) + 1.5)
        fig, ax = plt.subplots(figsize=(10, fig_h))
        y = range(len(labels))
        ax.barh(y, deltas, color=colors)
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="#444444", linewidth=0.8)
        ax.set_xlabel("Δ ninja-tracked time (s), branch - baseline")
        ax.set_title("Per-component build time delta (red = slower, green = faster)")
        fig.tight_layout()
        fig.savefig(args.chart, dpi=150)
        print(f"Wrote {args.chart}")


if __name__ == "__main__":
    main()
