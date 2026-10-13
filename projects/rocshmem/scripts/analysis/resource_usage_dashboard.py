#!/usr/bin/env python3
###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Generate one self-contained, interactive HTML dashboard for a resource-usage
report: either a single commit's snapshot, or a baseline-vs-branch comparison
with both commits' data embedded in the same file (comparison view on top, an
in-page toggle to inspect either commit's own kernels below). See
resource_usage_to_csv.py / resource_usage_diff.py.

No external JS/CSS, no build step -- double-click the output file to view it,
matching this project's existing self-contained HTML report style (see
My-AMD-tips-and-tricks/Posts/2026-07-15-rocSHMEM-resource-chart.html).

Usage:
    python3 scripts/analysis/resource_usage_dashboard.py \\
        --baseline-csv res-abc123.csv --baseline-commit abc123 \\
        --out dashboard.html

    python3 scripts/analysis/resource_usage_dashboard.py \\
        --baseline-csv res-abc123.csv --baseline-commit abc123 \\
        --branch-csv   res-def456.csv --branch-commit def456 \\
        --diff-csv res_diff_VGPRs.csv \\
        --out dashboard.html
"""

import argparse
import csv
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import resource_usage_diff as rud  # reuse NUMERIC_COLS/labels/colors/to_num --
# keeps dashboard numbers in lockstep with the existing CSV/PNG report instead
# of a second, possibly-drifting implementation of the same matching logic.

NUMERIC_COLS = rud.NUMERIC_COLS
REPORT_COLS = rud.REPORT_COLS
RESOURCE_LABELS = rud.RESOURCE_LABELS
OCC_COL = rud.OCC_COL
OCC_LABEL = rud.OCC_LABEL
to_num = rud.to_num


def load_kernels(csv_path: Path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def compute_stats(rows):
    stats = {"kernel_count": len(rows)}
    for col in NUMERIC_COLS:
        vals = [to_num(r.get(col, 0)) for r in rows]
        stats[col] = {
            "max": max(vals) if vals else 0,
            "avg": round(sum(vals) / len(vals), 1) if vals else 0,
        }
    stats["spill_count"] = sum(
        1
        for r in rows
        if to_num(r.get("SGPRsSpill", 0)) > 0 or to_num(r.get("VGPRsSpill", 0)) > 0
    )
    stats["dynamic_stack_count"] = sum(
        1 for r in rows if r.get("DynamicStack") == "True"
    )
    return stats


def load_diff_rows(diff_csv_path: Path):
    """Read an already-generated res_diff_<Column>.csv (see resource_usage_diff.py).

    write_csv() there writes the same row set/fields for every --sort-by choice
    (only row order differs), so any one of the 8 per-metric CSVs carries the
    full multi-metric diff data -- metric selection in the dashboard is a
    client-side re-sort/re-color over this single embedded dataset, not 8
    separate fetches.
    """
    with open(diff_csv_path, newline="") as f:
        return list(csv.DictReader(f))


def render_stat_cards(stats):
    def card(label, value):
        return f'<div class="stat-card"><div class="stat-value">{value}</div><div class="stat-label">{html.escape(label)}</div></div>'

    cards = [
        card("Kernels", stats["kernel_count"]),
        card("Max VGPRs", stats["VGPRs"]["max"]),
        card("Max SGPRs", stats["TotalSGPRs"]["max"]),
        card("Avg Occupancy [waves/SIMD]", stats[OCC_COL]["avg"]),
        card("Kernels with spills", stats["spill_count"]),
        card("Kernels with dynamic stack", stats["dynamic_stack_count"]),
    ]
    return f'<div class="stat-grid">{"".join(cards)}</div>'


NOTES_HTML = f"""
<section class="notes">
  <h2>Reading this report</h2>
  <p><strong>{html.escape(OCC_LABEL)}</strong> is not an independent resource: it is the
  compiler's resident-wavefront count per SIMD, capped by whichever of the VGPR file, the
  SGPR file, or the LDS budget is tightest for that specific kernel. It can move in a
  different direction than any single resource column -- e.g. VGPRs can go up while
  occupancy also goes up, if the same change relieves LDS or SGPR pressure instead.</p>
  <p><strong>Dynamic stack</strong> is a boolean compiler remark (does this kernel need a
  dynamically-sized stack frame), not a resource count -- a kernel gaining one is a real
  codegen change worth noticing even when register/spill counts don't move.</p>
</section>
"""

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{
    --surface-1:    #fcfcfb;
    --surface-page: #f9f9f7;
    --surface-panel:#f4f4f2;
    --ink-primary:  #0b0b0b;
    --ink-secondary:#52514e;
    --ink-muted:    #898781;
    --gridline:     #e1e0d9;
    --accent:       #2a78d6;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --surface-1:    #1a1a19;
      --surface-page: #0d0d0d;
      --surface-panel:#242422;
      --ink-primary:  #ffffff;
      --ink-secondary:#c3c2b7;
      --ink-muted:    #898781;
      --gridline:     #2c2c2a;
      --accent:       #3987e5;
    }}
  }}
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--surface-page);
    color: var(--ink-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    font-size: 14px;
    padding: 24px;
  }}
  .dash-root {{ max-width: 1800px; margin: 0 auto; }}
  header {{
    background: var(--surface-1);
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 16px;
  }}
  header h1 {{ font-size: 18px; margin-bottom: 4px; }}
  header .sub {{ color: var(--ink-secondary); font-size: 13px; }}
  section {{
    background: var(--surface-1);
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 16px;
  }}
  section h2 {{ font-size: 14px; margin-bottom: 12px; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }}
  .stat-card {{ background: var(--surface-panel); border-radius: 6px; padding: 12px 14px; }}
  .stat-value {{ font-size: 22px; font-weight: 700; }}
  .stat-label {{ font-size: 11px; color: var(--ink-secondary); margin-top: 2px; }}
  select, input[type=text] {{
    font: inherit; color: var(--ink-primary); background: var(--surface-panel);
    border: 1px solid var(--gridline); border-radius: 4px; padding: 6px 10px;
  }}
  .controls {{ display: flex; gap: 10px; align-items: center; margin: 14px 0; flex-wrap: wrap; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
  th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--gridline); }}
  th {{ cursor: pointer; color: var(--ink-secondary); font-weight: 600; user-select: none; }}
  th:hover {{ color: var(--ink-primary); }}
  tbody tr:hover {{ background: var(--surface-panel); }}
  .scroll-box {{ max-height: 520px; overflow-y: auto; border: 1px solid var(--gridline); border-radius: 6px; }}
  .scroll-box table thead th {{ position: sticky; top: 0; background: var(--surface-1); z-index: 1; }}
  .scroll-box table {{ margin: 0; }}
  .notes p {{ color: var(--ink-secondary); font-size: 13px; line-height: 1.5; margin-bottom: 10px; }}
  .commit-toggle {{ display: flex; gap: 8px; margin-bottom: 16px; }}
  .commit-toggle button {{
    font: inherit; padding: 8px 16px; border-radius: 6px; border: 1px solid var(--gridline);
    background: var(--surface-panel); color: var(--ink-secondary); cursor: pointer;
  }}
  .commit-toggle button.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
  .diff-summary {{ color: var(--ink-secondary); font-size: 13px; margin-bottom: 10px; }}
  .legend {{ color: var(--ink-secondary); font-size: 12px; margin-bottom: 10px; }}
</style>
</head>
<body>
<div class="dash-root">
  <header>
    <h1>{title}</h1>
    <div class="sub">{arch} / {build_config}</div>
  </header>
  {notes}
  {comparison_section}
  <section class="commit-section">
    {commit_toggle}
    <div id="commitStats"></div>
    <div class="table-section">
      <h2>All kernels</h2>
      <p class="legend">Click a column header to sort.</p>
      <div class="controls">
        <input type="text" id="searchBox" placeholder="Filter by kernel name...">
      </div>
      <div class="scroll-box">
        <table id="kernelTable">
          <thead><tr></tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
  </section>
</div>
<script>
const KERNELS_BY_COMMIT = {kernels_by_commit_json};
const STATS_HTML_BY_COMMIT = {stats_html_by_commit_json};
const METRICS = {metrics_json};
const OCC_COL = "{occ_col}";
const COLOR_GOOD = "{color_good}", COLOR_BAD = "{color_bad}", COLOR_NEUTRAL = "#3a3f44";
const DIFF_ROWS = {diff_rows_json};

function toNum(v) {{ const n = parseInt(v, 10); return isNaN(n) ? 0 : n; }}
function deltaColor(metric, delta) {{
  if (delta === 0) return COLOR_NEUTRAL;
  const worse = metric !== OCC_COL ? delta > 0 : delta < 0;
  return worse ? COLOR_BAD : COLOR_GOOD;
}}

// --- commit toggle (per-commit section) -----------------------------------
let activeCommit = Object.keys(KERNELS_BY_COMMIT)[0];

function setActiveCommit(commit) {{
  activeCommit = commit;
  document.querySelectorAll(".commit-toggle button").forEach(b => {{
    b.classList.toggle("active", b.dataset.commit === commit);
  }});
  document.getElementById("commitStats").innerHTML = STATS_HTML_BY_COMMIT[commit];
  renderTable();
}}

document.querySelectorAll(".commit-toggle button").forEach(b => {{
  b.addEventListener("click", () => setActiveCommit(b.dataset.commit));
}});

// --- sortable / filterable kernel table (active commit only) -------------
const COLUMNS = ["demangled_name", ...METRICS.map(m => m.key), "DynamicStack"];
const HEAD_LABELS = ["Kernel", ...METRICS.map(m => m.label), "Dyn. Stack"];
let sortCol = METRICS[0].key, sortDesc = true;

function renderTable() {{
  const filter = document.getElementById("searchBox").value.toLowerCase();
  const kernels = KERNELS_BY_COMMIT[activeCommit];
  const rows = kernels.filter(k => k.demangled_name.toLowerCase().includes(filter));
  rows.sort((a, b) => {{
    const av = sortCol === "demangled_name" || sortCol === "DynamicStack" ? a[sortCol] : toNum(a[sortCol]);
    const bv = sortCol === "demangled_name" || sortCol === "DynamicStack" ? b[sortCol] : toNum(b[sortCol]);
    if (av < bv) return sortDesc ? 1 : -1;
    if (av > bv) return sortDesc ? -1 : 1;
    return 0;
  }});
  const tbody = document.querySelector("#kernelTable tbody");
  tbody.textContent = "";
  for (const k of rows) {{
    const tr = document.createElement("tr");
    for (const col of COLUMNS) {{
      const td = document.createElement("td");
      td.textContent = k[col] ?? "";
      tr.appendChild(td);
    }}
    tbody.appendChild(tr);
  }}
}}

function buildHeader() {{
  const tr = document.querySelector("#kernelTable thead tr");
  tr.textContent = "";
  COLUMNS.forEach((col, i) => {{
    const th = document.createElement("th");
    th.textContent = HEAD_LABELS[i];
    th.addEventListener("click", () => {{
      if (sortCol === col) sortDesc = !sortDesc; else {{ sortCol = col; sortDesc = true; }}
      renderTable();
    }});
    tr.appendChild(th);
  }});
}}

document.getElementById("searchBox").addEventListener("input", renderTable);

buildHeader();
setActiveCommit(activeCommit);

// --- comparison table (baseline vs branch, all kernels, no top-N cap) ----
if (DIFF_ROWS.length) {{
  let diffMetric = METRICS[0].key;
  const DIFF_COLUMNS = ["demangled_name", "status"];
  const DIFF_HEAD_LABELS = ["Kernel", "Status"];

  function diffMetricPicker() {{
    const picker = document.getElementById("diffMetricPicker");
    for (const m of METRICS) {{
      const opt = document.createElement("option");
      opt.value = m.key; opt.textContent = m.label;
      picker.appendChild(opt);
    }}
    picker.addEventListener("change", () => {{ diffMetric = picker.value; renderDiffTable(); }});
  }}

  function buildDiffHeader() {{
    const tr = document.querySelector("#diffTable thead tr");
    tr.textContent = "";
    const labels = [...DIFF_HEAD_LABELS, ...METRICS.map(m => m.label)];
    labels.forEach(label => {{
      const th = document.createElement("th");
      th.textContent = label;
      tr.appendChild(th);
    }});
  }}

  function renderDiffTable() {{
    const filter = document.getElementById("diffSearchBox").value.toLowerCase();
    const rows = DIFF_ROWS.filter(r => r.demangled_name.toLowerCase().includes(filter));
    rows.sort((a, b) => Math.abs(toNum(b[diffMetric + "_delta"])) - Math.abs(toNum(a[diffMetric + "_delta"])));

    const changed = rows.filter(r => r.status === "common" && toNum(r[diffMetric + "_delta"]) !== 0);
    const added = rows.filter(r => r.status === "added");
    const removed = rows.filter(r => r.status === "removed");
    document.getElementById("diffSummary").textContent =
      `${{changed.length}} changed, ${{added.length}} added, ${{removed.length}} removed (of ${{rows.length}} shown)`;

    const tbody = document.querySelector("#diffTable tbody");
    tbody.textContent = "";
    for (const r of rows) {{
      const tr = document.createElement("tr");
      const nameTd = document.createElement("td"); nameTd.textContent = r.demangled_name; tr.appendChild(nameTd);
      const statusTd = document.createElement("td"); statusTd.textContent = r.status; tr.appendChild(statusTd);
      for (const m of METRICS) {{
        const bv = r[m.key + "_baseline"] || "-";
        const nv = r[m.key + "_branch"] || "-";
        const delta = toNum(r[m.key + "_delta"]);
        const sign = delta > 0 ? "+" : "";
        const td = document.createElement("td");
        td.textContent = `${{nv}} (${{sign}}${{delta}})`;
        td.title = `${{bv}} → ${{nv}}`;
        td.style.backgroundColor = deltaColor(m.key, delta);
        td.style.color = "#fff";
        td.style.fontWeight = "600";
        tr.appendChild(td);
      }}
      tbody.appendChild(tr);
    }}
  }}

  document.getElementById("diffSearchBox").addEventListener("input", renderDiffTable);
  buildDiffHeader();
  diffMetricPicker();
  renderDiffTable();
}}
</script>
</body>
</html>
"""


def render_comparison_section(diff_rows, baseline_commit, branch_commit):
    if not diff_rows:
        return ""
    return f"""
  <section class="comparison-section">
    <h2>Baseline vs Branch &mdash; {html.escape(baseline_commit[:12])} &rarr; {html.escape(branch_commit[:12])}</h2>
    <p class="legend">Each cell shows the branch value with &Delta; from baseline; background: green = improved, red = regressed, gray = unchanged (lower is better except Occupancy, where higher is better).</p>
    <div class="controls">
      <select id="diffMetricPicker"></select>
      <input type="text" id="diffSearchBox" placeholder="Filter by kernel name...">
    </div>
    <div class="diff-summary" id="diffSummary"></div>
    <div class="scroll-box">
      <table id="diffTable">
        <thead><tr></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </section>
"""


def render_commit_toggle(baseline_commit, branch_commit):
    if not branch_commit:
        return ""
    b_short = baseline_commit[:12] or "baseline"
    n_short = branch_commit[:12] or "branch"
    return (
        '<div class="commit-toggle">'
        f'<button data-commit="baseline" class="active">Baseline ({html.escape(b_short)})</button>'
        f'<button data-commit="branch">Branch ({html.escape(n_short)})</button>'
        "</div>"
    )


def render_page(args, baseline_rows, branch_rows, diff_rows):
    metrics = [{"key": c, "label": RESOURCE_LABELS[c]} for c in REPORT_COLS] + [
        {"key": OCC_COL, "label": OCC_LABEL}
    ]

    def kernels_json(rows):
        return [
            {
                "demangled_name": r.get("demangled_name", ""),
                "DynamicStack": r.get("DynamicStack", ""),
                **{c: r.get(c, "") for c in NUMERIC_COLS},
            }
            for r in rows
        ]

    kernels_by_commit = {"baseline": kernels_json(baseline_rows)}
    stats_html_by_commit = {"baseline": render_stat_cards(compute_stats(baseline_rows))}
    if branch_rows is not None:
        kernels_by_commit["branch"] = kernels_json(branch_rows)
        stats_html_by_commit["branch"] = render_stat_cards(compute_stats(branch_rows))

    baseline_short = args.baseline_commit[:12] if args.baseline_commit else "(working tree)"
    if args.branch_commit:
        title = f"Resource usage — {baseline_short} vs {args.branch_commit[:12]}"
    else:
        title = f"Resource usage — {baseline_short}"

    return PAGE_TEMPLATE.format(
        title=html.escape(title),
        arch=html.escape(baseline_rows[0]["arch"]) if baseline_rows else "?",
        build_config=html.escape(baseline_rows[0]["build_config"]) if baseline_rows else "?",
        comparison_section=render_comparison_section(
            diff_rows, args.baseline_commit, args.branch_commit
        ),
        commit_toggle=render_commit_toggle(args.baseline_commit, args.branch_commit),
        notes=NOTES_HTML,
        kernels_by_commit_json=json.dumps(kernels_by_commit),
        stats_html_by_commit_json=json.dumps(stats_html_by_commit),
        metrics_json=json.dumps(metrics),
        top_n=args.top,
        occ_col=OCC_COL,
        color_good=rud.COLOR_GOOD,
        color_bad=rud.COLOR_BAD,
        color_neutral=rud.COLOR_NEUTRAL,
        diff_rows_json=json.dumps(diff_rows),
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--baseline-csv", required=True, type=Path, help="this commit's res-<sha>.csv"
    )
    ap.add_argument("--baseline-commit", default="", help="this commit's sha/ref")
    ap.add_argument(
        "--branch-csv",
        type=Path,
        default=None,
        help="the other commit's res-<sha>.csv (omit for a single-commit snapshot)",
    )
    ap.add_argument("--branch-commit", default="", help="the other commit's sha/ref")
    ap.add_argument(
        "--diff-csv",
        type=Path,
        default=None,
        help="an already-generated res_diff_<Column>.csv (see resource_usage_diff.py); "
        "any one metric's file carries the full multi-metric diff data. Only "
        "meaningful together with --branch-csv.",
    )
    ap.add_argument("--out", required=True, type=Path, help="output HTML path")
    ap.add_argument(
        "--top",
        type=int,
        default=20,
        help="unused (kept for compatibility with existing callers) -- the per-commit "
        "chart this used to size was removed since the sortable/scrollable 'All "
        "kernels' table already covers top-by-any-metric",
    )
    args = ap.parse_args()

    if not args.baseline_csv.exists():
        sys.exit(f"error: baseline csv not found: {args.baseline_csv}")
    baseline_rows = load_kernels(args.baseline_csv)
    if not baseline_rows:
        sys.exit(f"error: no kernel rows in {args.baseline_csv}")

    branch_rows = None
    if args.branch_csv:
        if not args.branch_csv.exists():
            sys.exit(f"error: branch csv not found: {args.branch_csv}")
        branch_rows = load_kernels(args.branch_csv)
        if not branch_rows:
            sys.exit(f"error: no kernel rows in {args.branch_csv}")

    diff_rows = []
    if args.diff_csv:
        if branch_rows is None:
            sys.exit("error: --diff-csv requires --branch-csv/--branch-commit")
        if not args.diff_csv.exists():
            sys.exit(f"error: diff csv not found: {args.diff_csv}")
        diff_rows = load_diff_rows(args.diff_csv)

    html_out = render_page(args, baseline_rows, branch_rows, diff_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_out)
    print(f"Wrote dashboard to {args.out}")


if __name__ == "__main__":
    main()
