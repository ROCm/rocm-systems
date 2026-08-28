#!/usr/bin/env python3
###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Diff two LTO inline-remarks CSVs (see lto_inline_remarks_to_csv.py) from two
different commits/builds, matched on (caller_mangled, callee_mangled).

Writes a pairs CSV (--out), a long-format summary CSV (--summary-out) meant
to be read by an LLM/human without opening the chart, and a PNG dashboard
(--chart).

Matching by mangled name (not source location -- LTO remarks never carry
DebugLoc in practice) with the same ".intern.<hash>" suffix stripped as
resource_usage_diff.py uses, since that disambiguation hash is not stable
build-to-build.

Usage:
    python3 scripts/analysis/lto_inline_remarks_diff.py \\
        --baseline remarks_commitA.csv --branch remarks_commitB.csv \\
        --out remarks_diff_pairs.csv --summary-out remarks_diff_summary.csv \\
        --chart remarks_diff_dashboard.png

See LTO_INLINE_CALLER_COUNT_ISSUE.md for the caller-count "cliff": a callee
crossing the 1<->2+ remaining-caller boundary can flip a large inliner cost
bonus on/off independent of any change to the callee itself. --top pins the
most notable such callees (by_callee_cliff in the summary CSV / panel 4 of
the chart) regardless of how their Cost/Threshold otherwise moved.
"""

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PASSED_KINDS = ("Inlined", "AlwaysInline")
MISSED_KINDS = ("NeverInline", "TooCostly")

COLOR_GOOD = "#55a868"
COLOR_BAD = "#c44e52"
COLOR_NEUTRAL = "#9aa5b1"
COLOR_STRIPE = "#f2f2f2"
COLOR_BASELINE = "#8fa8c7"
COLOR_BRANCH = "#4c72b0"

_INTERN_SUFFIX_RE = re.compile(r"\.intern\.[0-9a-f]+$")


def match_key_name(mangled_name):
    return _INTERN_SUFFIX_RE.sub("", mangled_name or "")


def to_num(v):
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def load(path: Path):
    """Group rows by (caller_mangled, callee_mangled) into a list -- a pair can
    legitimately recur (the same callsite text appearing more than once across
    an LTO partition split), so this keeps every occurrence rather than
    collapsing to one row per key.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    by_key = defaultdict(list)
    for r in rows:
        key = (match_key_name(r["caller_mangled"]), match_key_name(r["callee_mangled"]))
        by_key[key].append(r)
    return rows, by_key


def occurrence_sort_key(r):
    return (r["name"], to_num(r["cost"]) or 0, to_num(r["threshold"]) or 0)


def row_key(r):
    return (match_key_name(r["caller_mangled"]), match_key_name(r["callee_mangled"]))


def select_pinned(rows, match_re):
    if not match_re:
        return []
    return [
        r
        for r in rows
        if match_re.search(r.get("caller_demangled", ""))
        or match_re.search(r.get("callee_demangled", ""))
    ]


def build_pair_rows(baseline_by_key, branch_by_key):
    all_keys = sorted(set(baseline_by_key) | set(branch_by_key))
    pair_rows = []
    for key in all_keys:
        b_occs = sorted(baseline_by_key.get(key, []), key=occurrence_sort_key)
        n_occs = sorted(branch_by_key.get(key, []), key=occurrence_sort_key)
        nb, nn = len(b_occs), len(n_occs)
        count_mismatch = nb != nn
        for i in range(max(nb, nn)):
            b = b_occs[i] if i < nb else None
            n = n_occs[i] if i < nn else None
            status = "added" if b is None else "removed" if n is None else "common"
            sample = n or b
            cost_b, cost_n = (to_num(b["cost"]) if b else None), (to_num(n["cost"]) if n else None)
            thr_b, thr_n = (to_num(b["threshold"]) if b else None), (to_num(n["threshold"]) if n else None)
            name_b = b["name"] if b else ""
            name_n = n["name"] if n else ""
            regression = status == "common" and name_b in PASSED_KINDS and name_n in MISSED_KINDS
            improvement = status == "common" and name_b in MISSED_KINDS and name_n in PASSED_KINDS
            pair_rows.append(
                {
                    "caller_demangled": sample.get("caller_demangled", key[0]),
                    "caller_mangled": key[0],
                    "callee_demangled": sample.get("callee_demangled", key[1]),
                    "callee_mangled": key[1],
                    "status": status,
                    "occurrence_count_baseline": nb,
                    "occurrence_count_branch": nn,
                    "count_mismatch": count_mismatch,
                    "name_baseline": name_b,
                    "name_branch": name_n,
                    "name_changed": status == "common" and name_b != name_n,
                    "cost_baseline": cost_b if cost_b is not None else "",
                    "cost_branch": cost_n if cost_n is not None else "",
                    "cost_delta": (cost_n - cost_b) if cost_b is not None and cost_n is not None else "",
                    "threshold_baseline": thr_b if thr_b is not None else "",
                    "threshold_branch": thr_n if thr_n is not None else "",
                    "threshold_delta": (thr_n - thr_b) if thr_b is not None and thr_n is not None else "",
                    "reason_baseline": b["reason"] if b else "",
                    "reason_branch": n["reason"] if n else "",
                    "regression_signal": regression,
                    "improvement_signal": improvement,
                }
            )
    return pair_rows


def order_rows(pair_rows):
    def abs_cost_delta(r):
        d = r["cost_delta"]
        return abs(d) if d != "" else 0

    regressions = [r for r in pair_rows if r["regression_signal"]]
    improvements = [r for r in pair_rows if r["improvement_signal"]]
    flagged_keys = {(r["caller_mangled"], r["callee_mangled"]) for r in regressions + improvements}
    rest = sorted(
        (r for r in pair_rows if (r["caller_mangled"], r["callee_mangled"]) not in flagged_keys),
        key=abs_cost_delta,
        reverse=True,
    )
    # Inline-decision flips are the signal users care about most here (categorical,
    # not just a numeric shift), so they lead the CSV/report ahead of a pure
    # |cost_delta| ranking -- a deliberate deviation from resource_usage_diff.py's
    # single abs(delta) sort.
    return regressions + improvements + rest


PAIR_FIELDS = [
    "caller_demangled",
    "caller_mangled",
    "callee_demangled",
    "callee_mangled",
    "status",
    "occurrence_count_baseline",
    "occurrence_count_branch",
    "count_mismatch",
    "name_baseline",
    "name_branch",
    "name_changed",
    "cost_baseline",
    "cost_branch",
    "cost_delta",
    "threshold_baseline",
    "threshold_branch",
    "threshold_delta",
    "reason_baseline",
    "reason_branch",
    "regression_signal",
    "improvement_signal",
]


def write_pairs_csv(ordered, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        for r in ordered:
            writer.writerow({col: r.get(col, "") for col in PAIR_FIELDS})
    print(f"Wrote {len(ordered)} rows to {out_path}")


def build_callee_cliff(baseline_rows, branch_rows):
    """Per callee (by demangled name), the distinct-caller count on each side,
    for callees whose count crosses the 1<->2+ boundary -- see
    LTO_INLINE_CALLER_COUNT_ISSUE.md for why exactly that boundary is the one
    that can flip a large inliner cost bonus on/off.
    """
    def callers_by_callee(rows):
        m = defaultdict(set)
        names = {}
        for r in rows:
            callee_key = match_key_name(r["callee_mangled"])
            if not callee_key:
                continue
            m[callee_key].add(match_key_name(r["caller_mangled"]))
            names[callee_key] = r.get("callee_demangled", callee_key)
        return m, names

    b_map, b_names = callers_by_callee(baseline_rows)
    n_map, n_names = callers_by_callee(branch_rows)
    cliffs = []
    for callee_key in sorted(set(b_map) | set(n_map)):
        cb = len(b_map.get(callee_key, ()))
        cn = len(n_map.get(callee_key, ()))
        crossed = (cb <= 1) != (cn <= 1)
        if crossed:
            name = n_names.get(callee_key) or b_names.get(callee_key) or callee_key
            cliffs.append((name, cb, cn, cn - cb))
    cliffs.sort(key=lambda t: abs(t[3]), reverse=True)
    return cliffs


def build_transitions(ordered):
    counts = Counter(
        (r["name_baseline"], r["name_branch"])
        for r in ordered
        if r["status"] == "common" and r["name_baseline"] and r["name_branch"]
    )
    return counts.most_common()


def write_summary_csv(ordered, cliffs, transitions, baseline_rows, branch_rows, out_path, top_n):
    b_kind_counts = Counter(r["name"] for r in baseline_rows)
    n_kind_counts = Counter(r["name"] for r in branch_rows)
    kinds = ["Inlined", "AlwaysInline", "NeverInline", "TooCostly"]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric_group", "name", "value_baseline", "value_branch", "delta"])
        for kind in kinds:
            vb, vn = b_kind_counts.get(kind, 0), n_kind_counts.get(kind, 0)
            writer.writerow(["totals", kind, vb, vn, vn - vb])
        writer.writerow(
            ["totals", "total_remarks", len(baseline_rows), len(branch_rows), len(branch_rows) - len(baseline_rows)]
        )
        for (name_b, name_n), count in transitions:
            writer.writerow(["transition", f"{name_b} -> {name_n}", "", "", count])
        for name, cb, cn, delta in cliffs[:top_n]:
            writer.writerow(["by_callee_cliff", name, cb, cn, delta])
    print(f"Wrote summary to {out_path}")


ROW_SPACING = 1.6


def wrap_label(name, width=32, max_lines=2):
    """Wrap a long name onto up to max_lines lines, breaking on a natural
    boundary (::, (, ", ", space) near the width limit rather than a hard
    mid-token cut. Only the final line is hard-truncated (with an ellipsis)
    if it's still too long -- mirrors lto_inline_remarks_dashboard.py's JS
    wrapLabel() so the PNG and HTML dashboard treat long names the same way.
    """
    lines, remaining = [], name
    for _ in range(max_lines - 1):
        if len(remaining) <= width:
            break
        break_at = -1
        for sep in ("::", "(", ", ", " "):
            idx = remaining.rfind(sep, 0, width)
            if idx > width * 0.3:
                break_at = idx + len(sep)
                break
        if break_at == -1:
            break_at = width
        lines.append(remaining[:break_at])
        remaining = remaining[break_at:]
    if len(remaining) > width:
        remaining = remaining[: width - 1] + "…"
    lines.append(remaining)
    return "\n".join(lines)


def make_dashboard(ordered, cliffs, transitions, baseline_rows, branch_rows, chart_path, top_n, match_re=None):
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

    fig, axes = plt.subplots(2, 2, figsize=(16, 13), constrained_layout=True)

    # Panel 1: total remark counts by kind, baseline vs branch. Aggregate count
    # shift isn't inherently good/bad, so use neutral paired-blue bars, not red/green.
    ax = axes[0][0]
    kinds = ["Inlined", "AlwaysInline", "NeverInline", "TooCostly"]
    b_counts = Counter(r["name"] for r in baseline_rows)
    n_counts = Counter(r["name"] for r in branch_rows)
    x = list(range(len(kinds)))
    w = 0.35
    ax.bar([i - w / 2 for i in x], [b_counts.get(k, 0) for k in kinds], width=w, color=COLOR_BASELINE, label="baseline")
    ax.bar([i + w / 2 for i in x], [n_counts.get(k, 0) for k in kinds], width=w, color=COLOR_BRANCH, label="branch")
    ax.set_xticks(x)
    ax.set_xticklabels(kinds, fontsize=8, rotation=15)
    ax.set_title("Remark counts by kind", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", visible=False)

    # Panel 2: top flipped (caller, callee) pairs -- cost_baseline vs cost_branch.
    pinned = select_pinned(ordered, match_re)
    pinned_keys = {(r["caller_mangled"], r["callee_mangled"]) for r in pinned}
    flips = [r for r in ordered if r["regression_signal"] or r["improvement_signal"]]
    remaining_flips = [r for r in flips if (r["caller_mangled"], r["callee_mangled"]) not in pinned_keys]
    shown = (pinned + remaining_flips)[:top_n]
    ax = axes[0][1]
    if shown:
        shown = list(reversed(shown))
        n = len(shown)
        y = [i * ROW_SPACING for i in range(n)]
        for yy in y[::2]:
            ax.axhspan(yy - ROW_SPACING / 2, yy + ROW_SPACING / 2, color=COLOR_STRIPE, zorder=0)
        colors = [COLOR_BAD if r["regression_signal"] else COLOR_GOOD for r in shown]
        cost_b = [r["cost_baseline"] if r["cost_baseline"] != "" else 0 for r in shown]
        cost_n = [r["cost_branch"] if r["cost_branch"] != "" else 0 for r in shown]
        bar_h = 0.34
        ax.barh([yy + bar_h / 2 for yy in y], cost_b, height=bar_h, color=COLOR_NEUTRAL, zorder=2, label="baseline cost")
        ax.barh([yy - bar_h / 2 for yy in y], cost_n, height=bar_h, color=colors, zorder=2, label="branch cost")
        ax.set_yticks(y)
        ax.set_yticklabels(
            [wrap_label(f"{r['caller_demangled']} → {r['callee_demangled']}", width=40) for r in shown],
            fontsize=7.5,
        )
        ax.set_ylim(-ROW_SPACING / 2, y[-1] + ROW_SPACING / 2)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(axis="y", visible=False)
        # The LastCallToStaticBonus can push Cost to a large negative outlier
        # (tens of thousands) right next to small positive TooCostly values on
        # the same axis -- a linear scale flattens the small bars to invisible.
        # symlog keeps sign and a linear region near zero but compresses the
        # large-magnitude tail so both ends of a flip stay visible.
        ax.set_xscale("symlog", linthresh=100)
    else:
        ax.text(0.5, 0.5, "(no Passed<->Missed flips found)", ha="center", va="center", color=COLOR_NEUTRAL)
        ax.axis("off")
    ax.set_title("Top flipped caller→callee pairs (red=regressed, green=improved)", fontsize=10, fontweight="bold")

    # Panel 3: transition matrix (name_baseline x name_branch), single-hue heatmap.
    ax = axes[1][0]
    trans_map = dict(transitions)
    matrix = [[trans_map.get((rb, cb), 0) for cb in kinds] for rb in kinds]
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels(kinds, fontsize=8, rotation=15)
    ax.set_yticks(range(len(kinds)))
    ax.set_yticklabels(kinds, fontsize=8)
    ax.set_xlabel("branch", fontsize=8)
    ax.set_ylabel("baseline", fontsize=8)
    vmax = max((max(row) for row in matrix), default=0) or 1
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            ax.text(
                j, i, str(v), ha="center", va="center", fontsize=8,
                color="white" if v > vmax * 0.6 else "#333333",
            )
    ax.set_title("Transition matrix (common pairs)", fontsize=10, fontweight="bold")

    # Panel 4: top callees crossing the caller-count cliff.
    ax = axes[1][1]
    top_cliffs = cliffs[:top_n]
    if top_cliffs:
        top_cliffs = list(reversed(top_cliffs))
        n = len(top_cliffs)
        y = [i * ROW_SPACING for i in range(n)]
        for yy in y[::2]:
            ax.axhspan(yy - ROW_SPACING / 2, yy + ROW_SPACING / 2, color=COLOR_STRIPE, zorder=0)
        bar_h = 0.34
        ax.barh(
            [yy + bar_h / 2 for yy in y], [cb for _n, cb, _cn, _d in top_cliffs],
            height=bar_h, color=COLOR_NEUTRAL, zorder=2, label="baseline callers",
        )
        colors = [COLOR_BAD if d > 0 else COLOR_GOOD for _n, _cb, _cn, d in top_cliffs]
        ax.barh(
            [yy - bar_h / 2 for yy in y], [cn for _n, _cb, cn, _d in top_cliffs],
            height=bar_h, color=colors, zorder=2, label="branch callers",
        )
        ax.set_yticks(y)
        ax.set_yticklabels([wrap_label(name) for name, _cb, _cn, _d in top_cliffs], fontsize=7.5)
        ax.set_ylim(-ROW_SPACING / 2, y[-1] + ROW_SPACING / 2)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(axis="y", visible=False)
    else:
        ax.text(0.5, 0.5, "(no caller-count cliff crossings found)", ha="center", va="center", color=COLOR_NEUTRAL)
        ax.axis("off")
    ax.set_title("Callees crossing the 1↔2+ caller-count cliff", fontsize=10, fontweight="bold")

    fig.suptitle(
        "LTO inline remarks — baseline vs branch\n"
        "green = improved / Passed, red = regressed / Missed, blue = baseline vs branch counts",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(chart_path, dpi=160, bbox_inches="tight")
    print(f"Wrote chart to {chart_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--baseline", required=True, type=Path, help="CSV from the baseline commit")
    ap.add_argument("--branch", required=True, type=Path, help="CSV from the branch/candidate commit")
    ap.add_argument("--out", required=True, type=Path, help="pairs diff CSV to write")
    ap.add_argument("--summary-out", type=Path, default=None, help="summary CSV path (default: --out with _summary.csv)")
    ap.add_argument("--chart", type=Path, default=None, help="PNG dashboard path (default: --out with .png)")
    ap.add_argument("--no-chart", action="store_true", help="skip chart generation")
    ap.add_argument("--top", type=int, default=20, help="entries per panel/section")
    ap.add_argument(
        "--match",
        default=None,
        help="regex (case-insensitive) matched against caller/callee demangled name; "
        "matching pairs are pinned to the top of the flipped-pairs panel regardless of delta",
    )
    args = ap.parse_args()

    if not args.baseline.exists():
        sys.exit(f"error: baseline CSV not found: {args.baseline}")
    if not args.branch.exists():
        sys.exit(f"error: branch CSV not found: {args.branch}")

    match_re = None
    if args.match:
        try:
            match_re = re.compile(args.match, re.IGNORECASE)
        except re.error as e:
            sys.exit(f"error: invalid --match regex {args.match!r}: {e}")

    baseline_rows, baseline_by_key = load(args.baseline)
    branch_rows, branch_by_key = load(args.branch)

    pair_rows = build_pair_rows(baseline_by_key, branch_by_key)
    ordered = order_rows(pair_rows)
    write_pairs_csv(ordered, args.out)

    cliffs = build_callee_cliff(baseline_rows, branch_rows)
    transitions = build_transitions(ordered)

    summary_out = args.summary_out or args.out.with_name(args.out.stem + "_summary.csv")
    write_summary_csv(ordered, cliffs, transitions, baseline_rows, branch_rows, summary_out, args.top)

    if not args.no_chart:
        chart_path = args.chart or args.out.with_suffix(".png")
        make_dashboard(ordered, cliffs, transitions, baseline_rows, branch_rows, chart_path, args.top, match_re)


if __name__ == "__main__":
    main()
