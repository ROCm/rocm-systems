#!/usr/bin/env python3
###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Single-commit dashboard for an LTO inline-remarks CSV (see
lto_inline_remarks_to_csv.py): a PNG with several panels, plus a long-format
summary CSV that an LLM/human can read without opening the chart.

Usage:
    python3 scripts/analysis/lto_inline_remarks_report.py \\
        --csv remarks.csv --out-chart dashboard.png --out-summary summary.csv

See COMPILER_OPTIONS_REFERENCE.md for what Cost/Threshold/Reason mean, and
LTO_INLINE_CALLER_COUNT_ISSUE.md for the caller-count "cliff" (a callee's
sole-remaining-caller status flipping a large inliner cost bonus on/off).
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import lto_inline_remarks_diff as lrd  # reuse PASSED_KINDS/MISSED_KINDS/COLOR_*/wrap_label

PASSED_KINDS = lrd.PASSED_KINDS
MISSED_KINDS = lrd.MISSED_KINDS
COLOR_GOOD = lrd.COLOR_GOOD
COLOR_BAD = lrd.COLOR_BAD
COLOR_NEUTRAL = lrd.COLOR_NEUTRAL
COLOR_STRIPE = lrd.COLOR_STRIPE

# Fixed, non-cycled color per remark name: Passed kinds are green-toned,
# Missed kinds are red-toned, regardless of plotting order.
KIND_COLOR = {
    "Inlined": COLOR_GOOD,
    "AlwaysInline": "#8fcf9d",
    "NeverInline": COLOR_BAD,
    "TooCostly": "#dd8b8e",
}

COST_BUCKETS = [
    (float("-inf"), -10000, "< -10000"),
    (-10000, -1000, "-10000..-1000"),
    (-1000, -100, "-1000..-100"),
    (-100, 0, "-100..0"),
    (0, 100, "0..100"),
    (100, 1000, "100..1000"),
    (1000, 10000, "1000..10000"),
    (10000, float("inf"), "> 10000"),
]


def to_num(v):
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def load(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def bucket_cost(cost):
    for lo, hi, label in COST_BUCKETS:
        if lo <= cost < hi:
            return label
    return COST_BUCKETS[-1][2]


def build_summary(rows):
    total = len(rows)
    kind_counts = Counter(r["name"] for r in rows)
    inline_rate = (
        (kind_counts.get("Inlined", 0) + kind_counts.get("AlwaysInline", 0)) / total
        if total
        else 0.0
    )

    # Keyed by mangled name (unique per symbol) rather than demangled text,
    # which can collide across distinct symbols (duplicate static/anonymous-
    # namespace functions across TUs, compiler clones like ".cold"/
    # ".constprop.N") and would otherwise silently merge unrelated
    # callers/callees. caller_label/callee_label carry the demangled text
    # for display only.
    caller_to_callees = defaultdict(set)
    callee_to_callers = defaultdict(set)
    caller_label = {}
    callee_label = {}
    for r in rows:
        if r["name"] in PASSED_KINDS and r["callee_mangled"]:
            caller_mangled = r["caller_mangled"]
            callee_mangled = r["callee_mangled"]
            caller_to_callees[caller_mangled].add(callee_mangled)
            callee_to_callers[callee_mangled].add(caller_mangled)
            caller_label.setdefault(caller_mangled, r["caller_demangled"])
            callee_label.setdefault(callee_mangled, r["callee_demangled"])

    top_callers = sorted(
        ((caller_label[name], len(callees)) for name, callees in caller_to_callees.items()),
        key=lambda t: t[1],
        reverse=True,
    )
    top_callees = sorted(
        ((callee_label[name], len(callers)) for name, callers in callee_to_callers.items()),
        key=lambda t: t[1],
        reverse=True,
    )

    never_inline_reasons = Counter(
        r["reason"] or "(no reason given)" for r in rows if r["name"] == "NeverInline"
    )

    near_misses = []
    for r in rows:
        if r["name"] != "TooCostly":
            continue
        cost, threshold = to_num(r["cost"]), to_num(r["threshold"])
        if cost is None or threshold is None:
            continue
        # TooCostly means Cost >= Threshold, so this gap is always >= 0 --
        # 0 is the exact boundary (cost == threshold), the closest possible miss.
        near_misses.append((r["caller_demangled"], r["callee_demangled"], cost - threshold))
    near_misses.sort(key=lambda t: t[2])

    return {
        "total": total,
        "kind_counts": kind_counts,
        "inline_rate": inline_rate,
        "top_callers": top_callers,
        "top_callees": top_callees,
        "never_inline_reasons": never_inline_reasons,
        "near_misses": near_misses,
        "_rows_for_cost_hist": [r for r in rows if r["name"] in ("Inlined", "TooCostly")],
    }


def write_summary_csv(summary, out_path, top_n):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric_group", "name", "value"])
        for kind in ("Inlined", "AlwaysInline", "NeverInline", "TooCostly"):
            writer.writerow(["totals", kind, summary["kind_counts"].get(kind, 0)])
        writer.writerow(["totals", "total_remarks", summary["total"]])
        writer.writerow(["totals", "inline_rate", f"{summary['inline_rate']:.4f}"])
        for name, count in summary["top_callers"][:top_n]:
            writer.writerow(["top_caller", name, count])
        for name, count in summary["top_callees"][:top_n]:
            writer.writerow(["top_callee", name, count])
        for reason, count in summary["never_inline_reasons"].most_common():
            writer.writerow(["never_inline_reason", reason, count])
        for caller, callee, margin in summary["near_misses"][:top_n]:
            writer.writerow(["too_costly_near_miss", f"{caller} -> {callee}", margin])
    print(f"Wrote summary to {out_path}")


def _style_axes(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", visible=False)


def _hbar_panel(ax, title, labels, values, color, value_fmt="{:d}"):
    n = len(labels)
    if n == 0:
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center", color=COLOR_NEUTRAL)
        ax.axis("off")
        return
    row_spacing = lrd.ROW_SPACING
    y = [i * row_spacing for i in range(n)]
    for yy in y[::2]:
        ax.axhspan(yy - row_spacing / 2, yy + row_spacing / 2, color=COLOR_STRIPE, zorder=0)
    ax.barh(y, values, height=0.6, color=color, zorder=2)
    xmax = max(values + [1])
    for yy, v in zip(y, values):
        ax.text(v + xmax * 0.02, yy, value_fmt.format(v), va="center", fontsize=7.5, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylim(-row_spacing / 2, y[-1] + row_spacing / 2)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax * 1.2)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    _style_axes(ax)


def make_dashboard(summary, chart_path, top_n, commit_label):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping chart", file=sys.stderr)
        return
    try:
        import seaborn as sns

        sns.set_theme(style="whitegrid", font_scale=0.95)
    except ImportError:
        pass

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)

    # Panel 1: overview -- counts by kind, fixed color per kind (never cycled).
    ax = axes[0][0]
    kinds = ["Inlined", "AlwaysInline", "NeverInline", "TooCostly"]
    counts = [summary["kind_counts"].get(k, 0) for k in kinds]
    colors = [KIND_COLOR[k] for k in kinds]
    for i in range(len(kinds)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color=COLOR_STRIPE, zorder=0)
    ax.bar(kinds, counts, color=colors, zorder=2, width=0.6)
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom", fontsize=8)
    ax.set_title(
        f"Overview — {summary['total']} remarks, "
        f"{summary['inline_rate']*100:.1f}% inlined (Inlined+AlwaysInline)",
        fontsize=10,
        fontweight="bold",
    )
    ax.tick_params(axis="x", labelsize=8, rotation=15)
    _style_axes(ax)
    ax.grid(axis="x", visible=False)

    # Panel 2: top callers by count of distinct callees inlined into them.
    top_callers = summary["top_callers"][:top_n]
    _hbar_panel(
        axes[0][1],
        "Top callers by inlined-callee count",
        [lrd.wrap_label(n) for n, _ in top_callers],
        [c for _, c in top_callers],
        COLOR_GOOD,
    )

    # Panel 3: top callees by distinct-caller count (duplication signal).
    top_callees = summary["top_callees"][:top_n]
    _hbar_panel(
        axes[0][2],
        "Top callees by distinct-caller count",
        [lrd.wrap_label(n) for n, _ in top_callees],
        [c for _, c in top_callees],
        COLOR_NEUTRAL,
    )

    # Panel 4: cost distribution over Inlined+TooCostly (the two kinds that carry Cost).
    ax = axes[1][0]
    labels = [b[2] for b in COST_BUCKETS]
    bucket_counts = {label: 0 for label in labels}
    for r in summary["_rows_for_cost_hist"]:
        cost = to_num(r["cost"])
        if cost is None:
            continue
        bucket_counts[bucket_cost(cost)] += 1
    values = [bucket_counts[label] for label in labels]
    x = list(range(len(labels)))
    for i in x:
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color=COLOR_STRIPE, zorder=0)
    ax.bar(x, values, color=COLOR_NEUTRAL, zorder=2, width=0.6)
    for i, v in enumerate(values):
        if v:
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_title("Cost distribution (Inlined + TooCostly)", fontsize=10, fontweight="bold")
    _style_axes(ax)
    ax.grid(axis="x", visible=False)

    # Panel 5: NeverInline reasons.
    reasons = summary["never_inline_reasons"].most_common(top_n)
    _hbar_panel(
        axes[1][1],
        "NeverInline reasons",
        [lrd.wrap_label(r) for r, _ in reasons],
        [c for _, c in reasons],
        COLOR_BAD,
    )

    # Panel 6: TooCostly near-miss gap (cost - threshold, always >= 0 -- 0 is
    # the exact boundary), smallest gap (closest to flipping to Inlined) first.
    near = summary["near_misses"][:top_n]
    _hbar_panel(
        axes[1][2],
        "TooCostly near-misses (cost − threshold, smallest gap first)",
        [lrd.wrap_label(f"{callee}") for _caller, callee, _m in near],
        [m for _c, _ce, m in near],
        KIND_COLOR["TooCostly"],
    )

    title = f"LTO inline remarks — {commit_label}" if commit_label else "LTO inline remarks"
    fig.suptitle(
        f"{title}\ngreen = Passed (Inlined/AlwaysInline), red = Missed (NeverInline/TooCostly), gray = neutral",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(chart_path, dpi=160, bbox_inches="tight")
    print(f"Wrote chart to {chart_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--csv", required=True, type=Path, help="CSV from lto_inline_remarks_to_csv.py")
    ap.add_argument("--out-chart", type=Path, default=None, help="PNG path (default: --csv with .png)")
    ap.add_argument("--out-summary", type=Path, default=None, help="summary CSV path (default: --csv with _summary.csv)")
    ap.add_argument("--no-chart", action="store_true", help="skip chart generation")
    ap.add_argument("--top", type=int, default=15, help="entries per panel/section")
    args = ap.parse_args()

    if not args.csv.exists():
        sys.exit(f"error: CSV not found: {args.csv}")

    rows = load(args.csv)
    if not rows:
        sys.exit(f"error: no rows in {args.csv}")

    summary = build_summary(rows)
    out_summary = args.out_summary or args.csv.with_name(args.csv.stem + "_summary.csv")
    write_summary_csv(summary, out_summary, args.top)

    if not args.no_chart:
        chart_path = args.out_chart or args.csv.with_suffix(".png")
        commit_label = next((r["commit"] for r in rows if r.get("commit")), "")
        make_dashboard(summary, chart_path, args.top, commit_label)


if __name__ == "__main__":
    main()
