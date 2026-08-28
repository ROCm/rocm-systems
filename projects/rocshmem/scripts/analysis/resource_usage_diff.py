#!/usr/bin/env python3
###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Diff two resource-usage CSVs (see resource_usage_to_csv.py) produced from
two different commits/builds, matched on (arch, build_config, mangled_name).

Matching by mangled_name (not source_file:line) so a function that moved
lines between commits is still compared correctly; template instantiations
keep distinct mangled names so they stay distinct rows.

Occupancy [waves/SIMD] is not an independent resource: it is the compiler's
resident-wavefront count per SIMD, capped by whichever of the VGPR file, the
SGPR file, or the LDS budget is tightest for that specific kernel. It can
move in a different direction than any single resource column — e.g. VGPRs
can go up while occupancy also goes up, if the same change relieves LDS or
SGPR pressure instead. For that reason this script colors the occupancy
panel by its own delta (more waves/SIMD = better) independently of how the
VGPR/SGPR/scratch/LDS panels are colored (less usage = better).

Usage:
    python3 scripts/analysis/resource_usage_diff.py \\
        --baseline resource_usage_commitA.csv \\
        --branch   resource_usage_commitB.csv \\
        --out      resource_usage_diff.csv \\
        --chart    resource_usage_diff.png \\
        --sort-by VGPRs

Use --match <regex> to pin specific kernels (matched against demangled_name or
mangled_name, case-insensitive) at the top of the text report and chart, even
if their delta is zero for every column -- useful when tracking one or two
kernels you care about instead of whatever happens to have the largest delta.

DynamicStack ("True"/"False", whether the kernel needs a dynamically-sized
stack frame) is a boolean compiler remark, not a resource count, so it isn't
part of NUMERIC_COLS/--sort-by. A baseline<->branch flip is still surfaced
per kernel in the text report (and counted in the "kernels changed" summary)
since a kernel gaining a dynamic stack is a real codegen change worth noticing
even when register/spill counts don't move.
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

NUMERIC_COLS = [
    "TotalSGPRs",
    "VGPRs",
    "AGPRs",
    "ScratchBytesPerLane",
    "OccupancyWavesPerSIMD",
    "SGPRsSpill",
    "VGPRsSpill",
    "LDSBytesPerBlock",
]

# Columns worth calling out in the text report / chart when they differ and
# are non-zero on at least one side. Occupancy is handled separately (see
# module docstring) since it's a derived result, not a resource you spend.
REPORT_COLS = [
    "VGPRs",
    "TotalSGPRs",
    "AGPRs",
    "VGPRsSpill",
    "SGPRsSpill",
    "ScratchBytesPerLane",
    "LDSBytesPerBlock",
]

RESOURCE_LABELS = {
    "VGPRs": "VGPRs",
    "TotalSGPRs": "SGPRs",
    "AGPRs": "AGPRs",
    "VGPRsSpill": "VGPR Spill",
    "SGPRsSpill": "SGPR Spill",
    "ScratchBytesPerLane": "Scratch [B/lane]",
    "LDSBytesPerBlock": "LDS [B/block]",
}
OCC_COL = "OccupancyWavesPerSIMD"
OCC_LABEL = "Occupancy [waves/SIMD]"

# DynamicStack is a boolean compiler remark ("True"/"False"), not a byte
# count -- it does NOT belong in NUMERIC_COLS (to_num() would coerce both
# "True" and "False" to 0 via its ValueError fallback, silently hiding every
# occurrence). Tracked separately as a flip (False<->True) per kernel instead.
DYNAMIC_STACK_COL = "DynamicStack"

COLOR_GOOD = "#2ca02c"
COLOR_BAD = "#d62728"
COLOR_NEUTRAL = "#9aa5b1"
COLOR_STRIPE = "#f2f2f2"


# Internal-linkage kernels (anonymous-namespace / local template instantiations,
# e.g. static `ipc_fadd<T>` helpers) get a compiler-generated ".intern.<hash>"
# suffix appended to their mangled name for disambiguation. That hash is not
# stable across separate compilations of the same source, so matching on the
# raw mangled name treats the *same* kernel as removed-in-baseline +
# added-in-branch whenever it recompiles with a different hash -- which was
# silently excluding most kernels from the diff (and could hide a real
# regression in one of them). Strip the suffix before using it as the match key.
_INTERN_SUFFIX_RE = re.compile(r"\.intern\.[0-9a-f]+$")


def match_key_name(mangled_name):
    return _INTERN_SUFFIX_RE.sub("", mangled_name)


def load(path: Path):
    """Group rows by (arch, build_config, mangled_name) into a list, not a single
    row. A name can legitimately recur (template instantiations, duplicate
    weak-symbol emission across TUs) with genuinely different resource shapes per
    occurrence -- collapsing to one row per name (last-write-wins) silently drops
    real data and can hide or fabricate a regression. build_diff_rows() below
    compares the full multiset of occurrences per name instead of a single pick.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    by_key = defaultdict(list)
    for r in rows:
        key = (r["arch"], r["build_config"], match_key_name(r["mangled_name"]))
        by_key[key].append(r)
    return by_key


def to_num(v):
    try:
        return int(v)
    except (ValueError, TypeError):
        return 0


def metric_tuple(r):
    return tuple(to_num(r[col]) for col in NUMERIC_COLS)


def short_commit(commit):
    return commit[:10] if commit else ""


def row_key(r):
    return (r["arch"], r["build_config"], r["mangled_name"])


def select_pinned(ordered, match_re):
    """Rows whose demangled or mangled name matches --match, in existing |delta|-sorted order."""
    if not match_re:
        return []
    return [
        r
        for r in ordered
        if match_re.search(r["demangled_name"]) or match_re.search(r["mangled_name"])
    ]


def build_diff_rows(baseline, branch):
    """Compare occurrences of each (arch, build_config, mangled_name) key as a
    multiset: sort each side's occurrences by their full metric tuple and pair
    them up positionally. This is order-invariant (unlike matching by emission
    order/line position, which -fgpu-rdc doesn't guarantee is stable across
    builds) and, for the common case of a single occurrence, reduces to a plain
    one-to-one comparison. When occurrence counts differ between baseline and
    branch, that is surfaced explicitly via count_mismatch/occurrence_count_*
    rather than silently comparing an arbitrary pair.
    """
    # Sorted, not a bare set iteration: str hashing is randomized per-process
    # (PYTHONHASHSEED), so iterating a set of string-keyed tuples gives a
    # different order every run. write_csv()'s final sort is stable, so ties
    # in |delta| (extremely common -- e.g. many zero-delta rows, or distinct
    # template instantiations with identical resource shapes) would otherwise
    # silently reorder -- and change which rows land in the top --top N --
    # from one invocation to the next on the exact same input CSVs.
    all_keys = sorted(set(baseline) | set(branch))
    diff_rows = []
    for key in all_keys:
        b_rows = sorted(baseline.get(key, []), key=metric_tuple)
        n_rows = sorted(branch.get(key, []), key=metric_tuple)
        nb, nn = len(b_rows), len(n_rows)
        count_mismatch = nb != nn
        for i in range(max(nb, nn)):
            b = b_rows[i] if i < nb else None
            n = n_rows[i] if i < nn else None
            status = "added" if b is None else "removed" if n is None else "common"
            row = {
                "arch": key[0],
                "build_config": key[1],
                "mangled_name": key[2],
                "demangled_name": (n or b).get("demangled_name", key[2]),
                "status": status,
                "occurrence_count_baseline": nb,
                "occurrence_count_branch": nn,
                "count_mismatch": count_mismatch,
            }
            for col in NUMERIC_COLS:
                bv = to_num(b[col]) if b else 0
                nv = to_num(n[col]) if n else 0
                row[f"{col}_baseline"] = bv if b else ""
                row[f"{col}_branch"] = nv if n else ""
                row[f"{col}_delta"] = nv - bv
            ds_baseline = b.get(DYNAMIC_STACK_COL, "") if b else ""
            ds_branch = n.get(DYNAMIC_STACK_COL, "") if n else ""
            row["DynamicStack_baseline"] = ds_baseline
            row["DynamicStack_branch"] = ds_branch
            row["DynamicStack_changed"] = (
                bool(ds_baseline) and bool(ds_branch) and ds_baseline != ds_branch
            )
            diff_rows.append(row)
    return diff_rows


def write_csv(diff_rows, out_path, sort_by):
    out_fields = [
        "arch",
        "build_config",
        "demangled_name",
        "mangled_name",
        "status",
        "occurrence_count_baseline",
        "occurrence_count_branch",
        "count_mismatch",
    ]
    for col in NUMERIC_COLS:
        out_fields += [f"{col}_baseline", f"{col}_branch", f"{col}_delta"]
    out_fields += [
        "DynamicStack_baseline",
        "DynamicStack_branch",
        "DynamicStack_changed",
    ]

    ordered = sorted(diff_rows, key=lambda r: abs(r[f"{sort_by}_delta"]), reverse=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for r in ordered:
            writer.writerow({col: r.get(col, "") for col in out_fields})
    return ordered


def print_kernel_block(r, sort_by, sort_label):
    d = r[f"{sort_by}_delta"]
    sign = "+" if d > 0 else ""
    status_note = f"  [{r['status']} vs baseline]" if r["status"] != "common" else ""
    if r.get("count_mismatch"):
        status_note += (
            f"  [occurrence count mismatch: {r['occurrence_count_baseline']} "
            f"baseline vs {r['occurrence_count_branch']} branch -- name recurs "
            f"a different number of times, comparison below is a best-effort pairing]"
        )
    print(
        f"\n  {sign}{d:>5} {sort_label}  {r['arch']}/{r['build_config']}  {r['demangled_name']}{status_note}"
    )
    printed_any = False
    for col in REPORT_COLS:
        delta = r[f"{col}_delta"]
        if delta == 0:
            continue
        bv, nv = r[f"{col}_baseline"], r[f"{col}_branch"]
        if bv == 0 and nv == 0:
            continue
        csign = "+" if delta > 0 else ""
        bv_disp = "-" if bv == "" else bv
        nv_disp = "-" if nv == "" else nv
        print(
            f"      {RESOURCE_LABELS[col]:22s} {bv_disp:>5} -> {nv_disp:<5} ({csign}{delta})"
        )
        printed_any = True
    occ_delta = r[f"{OCC_COL}_delta"]
    if occ_delta != 0:
        ob, on = r[f"{OCC_COL}_baseline"], r[f"{OCC_COL}_branch"]
        ob_disp = "-" if ob == "" else ob
        on_disp = "-" if on == "" else on
        direction = "DECREASED (worse)" if occ_delta < 0 else "INCREASED (better)"
        print(f"      {OCC_LABEL:22s} {ob_disp:>5} -> {on_disp:<5} {direction}")
        printed_any = True
    if r.get("DynamicStack_changed"):
        ds_b, ds_n = r["DynamicStack_baseline"], r["DynamicStack_branch"]
        print(
            f"      {'Dynamic Stack':22s} {ds_b:>5} -> {ds_n:<5} "
            f"(kernel {'gained' if ds_n == 'True' else 'lost'} a dynamic stack allocation)"
        )
        printed_any = True
    if not printed_any:
        print("      (no change in any resource column)")


def print_report(ordered, sort_by, top_n, match_re=None):
    sort_label = RESOURCE_LABELS.get(sort_by, OCC_LABEL)
    resource_changed = [
        r
        for r in ordered
        if r["status"] == "common" and r[f"{sort_by}_delta"] != 0
    ]
    dynamic_stack_only = [
        r
        for r in ordered
        if r["status"] == "common"
        and r[f"{sort_by}_delta"] == 0
        and r.get("DynamicStack_changed")
    ]
    added = [r for r in ordered if r["status"] == "added"]
    removed = [r for r in ordered if r["status"] == "removed"]
    print(
        f"{len(resource_changed)} kernels changed {sort_label}, "
        f"{len(dynamic_stack_only)} only flipped dynamic-stack usage, "
        f"{len(added)} added, {len(removed)} removed\n"
    )

    print(
        "Occupancy [waves/SIMD] is derived from whichever of VGPR/SGPR/LDS is the tightest\n"
        "constraint for a given kernel, so it can move independently of any single resource\n"
        "column below — it is reported per kernel, not implied by the other deltas.\n"
    )

    pinned = select_pinned(ordered, match_re)
    pinned_keys = {row_key(r) for r in pinned}
    if pinned:
        print(
            f"Pinned kernels matching --match {match_re.pattern!r} "
            f"({len(pinned)} match{'es' if len(pinned) != 1 else ''}, shown regardless of delta):"
        )
        for r in pinned:
            print_kernel_block(r, sort_by, sort_label)
        print()

    remaining = [r for r in resource_changed if row_key(r) not in pinned_keys]
    if remaining:
        print(
            f"Top {min(top_n, len(remaining))} kernels ranked by |{sort_label} delta| "
            f"(baseline -> branch, all non-zero deltas shown):"
        )
        for r in remaining[:top_n]:
            print_kernel_block(r, sort_by, sort_label)
    else:
        print(
            f"No non-zero {sort_label} deltas found in common kernels; see dynamic-stack summary above if applicable."
        )

    if dynamic_stack_only:
        print(
            f"\n{len(dynamic_stack_only)} kernel(s) only flipped dynamic-stack state without a {sort_label} delta."
        )


def wrap_label(row, width=42, pinned=False):
    name = row["demangled_name"]
    if len(name) > width:
        name = name[: width - 1] + "…"
    prefix = "★ " if pinned else ""
    return f"{prefix}{name}\n[{row['arch']}/{row['build_config']}]"


def make_chart(
    ordered, sort_by, chart_path, top_n, baseline_commit, branch_commit, match_re=None
):
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

    common = [r for r in ordered if r["status"] == "common"]
    pinned = select_pinned(common, match_re)
    pinned_keys = {row_key(r) for r in pinned}
    remaining = [
        r
        for r in common
        if r[f"{sort_by}_delta"] != 0 and row_key(r) not in pinned_keys
    ]
    changed = pinned + remaining[:top_n]
    if not changed:
        print("No changed kernels to chart; skipping chart", file=sys.stderr)
        return
    if pinned:
        print(
            f"Pinning {len(pinned)} kernel(s) matching --match {match_re.pattern!r} to the chart "
            f"(marked with ★), regardless of delta.",
            file=sys.stderr,
        )

    # Largest delta at the top of the chart; pinned kernels always rendered above everything else.
    changed = list(reversed(changed))
    n = len(changed)
    y = list(range(n))
    labels = [wrap_label(r, pinned=row_key(r) in pinned_keys) for r in changed]

    # Only render panels for resource types that actually changed among the displayed
    # kernels, plus occupancy always last (unless it's the sort column -- see below).
    # A pinned kernel with a non-zero baseline/branch value also activates its column
    # even with zero delta, since pinning means "I want to study this kernel" -- an
    # unchanged-but-nonzero resource is still worth showing for it.
    def col_active(c):
        for r in changed:
            if r[f"{c}_delta"] != 0:
                return True
            if row_key(r) in pinned_keys and (r[f"{c}_baseline"] or r[f"{c}_branch"]):
                return True
        return False

    # sort_by is always shown even if col_active(sort_by) is False -- that
    # happens when every displayed row was pulled in solely via the
    # DynamicStack_changed clause above (sort_by's own delta is 0 for all of
    # them), and hiding the very column the chart's title claims to rank by
    # is far more misleading than a panel full of "0->0" bars confirming
    # there's genuinely no change in that metric.
    active_cols = [c for c in REPORT_COLS if col_active(c) or c == sort_by]
    if sort_by in active_cols:
        active_cols.remove(sort_by)
    # OCC_COL isn't in REPORT_COLS (it's handled separately, see above), so when
    # sort_by is OccupancyWavesPerSIMD the "c == sort_by" check above never fires
    # and active_cols is just the other active REPORT_COLS -- appending OCC_COL
    # unconditionally here would then both duplicate it and, worse, leave it
    # stranded as the last panel even though it's the very column the chart is
    # ranked by (every other --sort-by choice gets promoted to the first/leftmost
    # panel below; occupancy deserves the same treatment instead of a silent
    # exception).
    panels = active_cols + ([OCC_COL] if sort_by != OCC_COL else [])
    panels.insert(0, sort_by)

    fig_height = max(3.7, 0.5 * n + 2.0)
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(3.0 * len(panels) + 2.2, fig_height),
        sharey=True,
    )
    if len(panels) == 1:
        axes = [axes]

    bar_h = 0.34
    for ax, col in zip(axes, panels):
        bvals = [r[f"{col}_baseline"] or 0 for r in changed]
        nvals = [r[f"{col}_branch"] or 0 for r in changed]
        deltas = [r[f"{col}_delta"] for r in changed]
        if col == OCC_COL:
            colors = [
                COLOR_GOOD if d > 0 else COLOR_BAD if d < 0 else COLOR_NEUTRAL
                for d in deltas
            ]
        else:
            colors = [
                COLOR_BAD if d > 0 else COLOR_GOOD if d < 0 else COLOR_NEUTRAL
                for d in deltas
            ]

        for i in y:
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color=COLOR_STRIPE, zorder=0)

        ax.barh(
            [i + bar_h / 2 for i in y],
            bvals,
            height=bar_h,
            color=COLOR_NEUTRAL,
            zorder=2,
            label="baseline",
        )
        ax.barh(
            [i - bar_h / 2 for i in y],
            nvals,
            height=bar_h,
            color=colors,
            zorder=2,
            label="branch",
        )

        xmax = max(bvals + nvals + [1])
        for i, (bv, nv) in enumerate(zip(bvals, nvals)):
            if bv == 0 and nv == 0:
                continue
            ax.text(
                max(bv, nv) + xmax * 0.03,
                i,
                f"{bv}→{nv}",
                va="center",
                fontsize=7.5,
                color="#333333",
            )

        ax.set_xlim(0, xmax * 1.28)
        ax.set_title(
            RESOURCE_LABELS.get(col, OCC_LABEL), fontsize=10, fontweight="bold"
        )
        ax.tick_params(axis="x", labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(axis="y", visible=False)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_ylim(-0.5, n - 0.5)

    baseline_label = short_commit(baseline_commit) or "baseline"
    branch_label = short_commit(branch_commit) or "branch"

    # suptitle/legend/axes are stacked in the header region above the panels. Their y
    # positions below are figure-fraction coordinates, but the figure height itself scales
    # with n (see fig_height above) -- a fixed fraction gap shrinks to nothing in absolute
    # terms for small n (few kernels), which is what caused the title/legend to overlap.
    # Reserving the header in fixed inches and converting to a fraction of fig_height keeps
    # the gap visually constant regardless of how many kernels are plotted.
    TOP_MARGIN_IN = 0.15
    TITLE_HEIGHT_IN = 0.45  # two lines at fontsize 12
    TITLE_LEGEND_GAP_IN = 0.18
    LEGEND_HEIGHT_IN = 0.2  # one line at fontsize 9
    LEGEND_AXES_GAP_IN = 0.15

    title_y = 1 - TOP_MARGIN_IN / fig_height
    legend_y = 1 - (TOP_MARGIN_IN + TITLE_HEIGHT_IN + TITLE_LEGEND_GAP_IN) / fig_height
    axes_top = (
        1
        - (
            TOP_MARGIN_IN
            + TITLE_HEIGHT_IN
            + TITLE_LEGEND_GAP_IN
            + LEGEND_HEIGHT_IN
            + LEGEND_AXES_GAP_IN
        )
        / fig_height
    )

    from matplotlib.patches import Patch

    legend_elems = [
        Patch(color=COLOR_NEUTRAL, label=f"{baseline_label}  (top bar)"),
        Patch(color=COLOR_BAD, label=f"{branch_label}  (bottom bar) — regressed"),
        Patch(color=COLOR_GOOD, label=f"{branch_label}  (bottom bar) — improved"),
    ]
    fig.legend(
        handles=legend_elems,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, legend_y),
        fontsize=9,
        frameon=False,
    )

    fig.suptitle(
        f"Resource usage — top {n} kernels by |{RESOURCE_LABELS.get(sort_by, sort_by)} delta|\n"
        f"baseline {baseline_label}  →  branch {branch_label}  (values shown as baseline→branch; "
        f"usage panels: lower=better, occupancy: higher=better)",
        fontsize=12,
        y=title_y,
    )
    fig.tight_layout(rect=[0, 0, 1, axes_top])
    fig.savefig(chart_path, dpi=160, bbox_inches="tight")
    print(f"Wrote chart to {chart_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--baseline", required=True, type=Path, help="CSV from the baseline commit"
    )
    ap.add_argument(
        "--branch",
        required=True,
        type=Path,
        help="CSV from the branch/candidate commit",
    )
    ap.add_argument("--out", required=True, type=Path, help="diff CSV to write")
    ap.add_argument(
        "--chart",
        type=Path,
        default=None,
        help="PNG path for the side-by-side comparison chart (default: --out with .png extension)",
    )
    ap.add_argument("--no-chart", action="store_true", help="skip chart generation")
    ap.add_argument(
        "--sort-by",
        default="VGPRs",
        choices=NUMERIC_COLS,
        help="column to rank kernels by (largest |delta| first)",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=20,
        help="number of kernels to show in the text report and chart",
    )
    ap.add_argument(
        "--match",
        default=None,
        help="regex (case-insensitive) matched against demangled/mangled kernel name; "
        "matching kernels are pinned to the top of the text report and chart "
        "regardless of delta, e.g. --match 'alltoall_test|wg_put_kernel'",
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

    baseline = load(args.baseline)
    branch = load(args.branch)
    baseline_commit = next(iter(baseline.values()))[0]["commit"] if baseline else ""
    branch_commit = next(iter(branch.values()))[0]["commit"] if branch else ""

    diff_rows = build_diff_rows(baseline, branch)
    ordered = write_csv(diff_rows, args.out, args.sort_by)
    print(f"Wrote {len(ordered)} rows to {args.out}")

    if match_re and not select_pinned(ordered, match_re):
        print(
            f"warning: --match {args.match!r} matched no kernels in either CSV",
            file=sys.stderr,
        )

    print_report(ordered, args.sort_by, args.top, match_re)

    if not args.no_chart:
        chart_path = args.chart or args.out.with_suffix(".png")
        make_chart(
            ordered,
            args.sort_by,
            chart_path,
            args.top,
            baseline_commit,
            branch_commit,
            match_re,
        )


if __name__ == "__main__":
    main()
