#!/usr/bin/env python3
###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Generate a self-contained, interactive HTML dashboard for one commit's
resource-usage report (see resource_usage_to_csv.py / resource_usage_diff.py).

One dashboard per commit, not per comparison: --diff lets a commit's
dashboard also surface its delta vs. a paired commit, but the page itself is
always about a single commit's kernels. No external JS/CSS, no build step --
double-click the output file to view it, matching this project's existing
self-contained HTML report style (see
My-AMD-tips-and-tricks/Posts/2026-07-15-rocSHMEM-resource-chart.html).

Usage:
    python3 scripts/analysis/resource_usage_dashboard.py \\
        --csv resource_usage.csv --commit abc123 --role snapshot \\
        --out dashboard.html --top 20

    python3 scripts/analysis/resource_usage_dashboard.py \\
        --csv res-abc123.csv --commit abc123 --role baseline \\
        --counterpart-commit def456 \\
        --diff VGPRs:res_diff_VGPRs.csv --diff TotalSGPRs:res_diff_TotalSGPRs.csv \\
        --out dashboard-abc123.html --top 20
"""

import argparse
import csv
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import resource_usage_diff as rud  # reuse NUMERIC_COLS/labels/colors/to_num -- keeps

# dashboard numbers in lockstep with the existing CSV/PNG report instead of a
# second, possibly-drifting implementation of the same matching logic.

NUMERIC_COLS = rud.NUMERIC_COLS
REPORT_COLS = rud.REPORT_COLS
RESOURCE_LABELS = rud.RESOURCE_LABELS
OCC_COL = rud.OCC_COL
OCC_LABEL = rud.OCC_LABEL
to_num = rud.to_num

ROLE_LABELS = {
    "snapshot": "Snapshot",
    "baseline": "Baseline",
    "branch": "Branch",
}


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


def load_diff(metric, diff_csv_path, top_n):
    """Read an already-generated res_diff_<metric>.csv (see resource_usage_diff.py)
    and pull out the summary + top-|delta| rows for one metric's dashboard section.
    """
    with open(diff_csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    changed = [
        r for r in rows if r["status"] == "common" and to_num(r[f"{metric}_delta"]) != 0
    ]
    added = [r for r in rows if r["status"] == "added"]
    removed = [r for r in rows if r["status"] == "removed"]
    changed.sort(key=lambda r: abs(to_num(r[f"{metric}_delta"])), reverse=True)
    top_k = min(10, top_n)
    return {
        "label": RESOURCE_LABELS.get(metric, OCC_LABEL),
        "changed_count": len(changed),
        "added_count": len(added),
        "removed_count": len(removed),
        "top": changed[:top_k],
    }


def delta_color(metric, delta):
    if delta == 0:
        return rud.COLOR_NEUTRAL
    worse = delta > 0 if metric != OCC_COL else delta < 0
    return rud.COLOR_BAD if worse else rud.COLOR_GOOD


def render_diff_section(diffs):
    if not diffs:
        return ""
    blocks = []
    for metric, d in diffs.items():
        rows_html = []
        for r in d["top"]:
            delta = to_num(r[f"{metric}_delta"])
            color = delta_color(metric, delta)
            sign = "+" if delta > 0 else ""
            bv = r[f"{metric}_baseline"] or "-"
            nv = r[f"{metric}_branch"] or "-"
            rows_html.append(
                f"<tr><td>{html.escape(r['demangled_name'])}</td>"
                f"<td>{html.escape(bv)}</td><td>{html.escape(nv)}</td>"
                f'<td style="color:{color};font-weight:600">{sign}{delta}</td></tr>'
            )
        summary = (
            f"{d['changed_count']} changed, {d['added_count']} added, "
            f"{d['removed_count']} removed"
        )
        table = (
            f"<table class=\"diff-table\"><thead><tr><th>Kernel</th><th>Baseline</th>"
            f"<th>Branch</th><th>&Delta;</th></tr></thead><tbody>{''.join(rows_html) or '<tr><td colspan=4>(no non-zero deltas)</td></tr>'}</tbody></table>"
        )
        blocks.append(
            f"<details><summary>{html.escape(d['label'])} &mdash; {summary}</summary>{table}</details>"
        )
    return f'<section class="diff-section"><h2>Vs. counterpart</h2>{"".join(blocks)}</section>'


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
    return f'<section class="stat-grid">{"".join(cards)}</section>'


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
  .dash-root {{ max-width: 1100px; margin: 0 auto; }}
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
  table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; margin-top: 10px; }}
  th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--gridline); }}
  th {{ cursor: pointer; color: var(--ink-secondary); font-weight: 600; user-select: none; }}
  th:hover {{ color: var(--ink-primary); }}
  tbody tr:hover {{ background: var(--surface-panel); }}
  .notes p {{ color: var(--ink-secondary); font-size: 13px; line-height: 1.5; margin-bottom: 10px; }}
  details {{ margin-bottom: 8px; }}
  summary {{ cursor: pointer; font-size: 13px; color: var(--ink-secondary); padding: 6px 0; }}
  svg text {{ fill: var(--ink-secondary); font-size: 10px; }}
  .diff-table {{ margin-bottom: 4px; }}
</style>
</head>
<body>
<div class="dash-root">
  <header>
    <h1>{role_label} commit {commit_short}{counterpart_html}</h1>
    <div class="sub">{arch} / {build_config} &middot; {kernel_count} kernels</div>
  </header>
  {stat_cards}
  <section class="chart-section">
    <h2>Top kernels by resource</h2>
    <select id="metricPicker"></select>
    <div id="chartContainer"></div>
  </section>
  <section class="table-section">
    <h2>All kernels</h2>
    <input type="text" id="searchBox" placeholder="Filter by kernel name...">
    <table id="kernelTable">
      <thead><tr></tr></thead>
      <tbody></tbody>
    </table>
  </section>
  {notes}
  {diff_section}
</div>
<script>
const KERNELS = {kernels_json};
const METRICS = {metrics_json};
const TOP_N = {top_n};
const COLOR_GOOD = "{color_good}", COLOR_BAD = "{color_bad}", COLOR_NEUTRAL = "{color_neutral}";

function toNum(v) {{ const n = parseInt(v, 10); return isNaN(n) ? 0 : n; }}

// --- sortable / filterable kernel table -----------------------------------
const COLUMNS = ["demangled_name", ...METRICS.map(m => m.key), "DynamicStack"];
const HEAD_LABELS = ["Kernel", ...METRICS.map(m => m.label), "Dyn. Stack"];
let sortCol = METRICS[0].key, sortDesc = true;

function renderTable() {{
  const filter = document.getElementById("searchBox").value.toLowerCase();
  const rows = KERNELS.filter(k => k.demangled_name.toLowerCase().includes(filter));
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

// --- SVG bar chart for the selected metric --------------------------------
const ns = "http://www.w3.org/2000/svg";
function el(tag, attrs) {{
  const e = document.createElementNS(ns, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}}

function drawChart(metricKey) {{
  const container = document.getElementById("chartContainer");
  container.textContent = "";
  const metric = METRICS.find(m => m.key === metricKey);
  const ranked = [...KERNELS].sort((a, b) => toNum(b[metricKey]) - toNum(a[metricKey])).slice(0, TOP_N);
  if (!ranked.length) return;
  const marginL = 220, marginR = 50, rowH = 20, W = 760, plotW = W - marginL - marginR;
  const H = ranked.length * rowH + 20;
  const xmax = Math.max(...ranked.map(k => toNum(k[metricKey])), 1);
  const svg = el("svg", {{ viewBox: `0 0 ${{W}} ${{H}}`, width: "100%", height: H }});
  ranked.forEach((k, i) => {{
    const v = toNum(k[metricKey]);
    const barW = (v / xmax) * plotW;
    const y = i * rowH;
    svg.appendChild(el("rect", {{ x: marginL, y: y + 3, width: Math.max(barW, 1), height: rowH - 6, fill: COLOR_NEUTRAL }}));
    const label = el("text", {{ x: marginL - 8, y: y + rowH / 2 + 4, "text-anchor": "end" }});
    label.textContent = k.demangled_name.length > 32 ? k.demangled_name.slice(0, 31) + "\\u2026" : k.demangled_name;
    svg.appendChild(label);
    const val = el("text", {{ x: marginL + barW + 6, y: y + rowH / 2 + 4 }});
    val.textContent = v;
    svg.appendChild(val);
  }});
  container.appendChild(svg);
}}

function buildMetricPicker() {{
  const picker = document.getElementById("metricPicker");
  for (const m of METRICS) {{
    const opt = document.createElement("option");
    opt.value = m.key; opt.textContent = m.label;
    picker.appendChild(opt);
  }}
  picker.addEventListener("change", () => drawChart(picker.value));
}}

buildHeader();
buildMetricPicker();
renderTable();
drawChart(METRICS[0].key);
</script>
</body>
</html>
"""


def render_page(args, rows, stats, diffs):
    metrics = [{"key": c, "label": RESOURCE_LABELS[c]} for c in REPORT_COLS] + [
        {"key": OCC_COL, "label": OCC_LABEL}
    ]
    kernels_json = json.dumps(
        [
            {
                "demangled_name": r.get("demangled_name", ""),
                "DynamicStack": r.get("DynamicStack", ""),
                **{c: r.get(c, "") for c in NUMERIC_COLS},
            }
            for r in rows
        ]
    )
    commit_short = args.commit[:12] if args.commit else "(working tree)"
    counterpart_html = ""
    if args.counterpart_commit:
        verb = "vs branch" if args.role == "baseline" else "vs baseline"
        counterpart_html = f" <span class=\"sub\">({html.escape(verb)} {html.escape(args.counterpart_commit[:12])})</span>"
    title = f"Resource usage — {commit_short}"
    return PAGE_TEMPLATE.format(
        title=html.escape(title),
        role_label=html.escape(ROLE_LABELS.get(args.role, args.role)),
        commit_short=html.escape(commit_short),
        counterpart_html=counterpart_html,
        arch=html.escape(rows[0]["arch"]) if rows else "?",
        build_config=html.escape(rows[0]["build_config"]) if rows else "?",
        kernel_count=stats["kernel_count"],
        stat_cards=render_stat_cards(stats),
        notes=NOTES_HTML,
        diff_section=render_diff_section(diffs),
        kernels_json=kernels_json,
        metrics_json=json.dumps(metrics),
        top_n=args.top,
        color_good=rud.COLOR_GOOD,
        color_bad=rud.COLOR_BAD,
        color_neutral=rud.COLOR_NEUTRAL,
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--csv", required=True, type=Path, help="this commit's res-<sha>.csv")
    ap.add_argument("--commit", default="", help="this commit's sha/ref")
    ap.add_argument(
        "--role",
        default="snapshot",
        choices=["snapshot", "baseline", "branch"],
        help="snapshot: single-commit run; baseline/branch: part of a two-commit comparison",
    )
    ap.add_argument(
        "--counterpart-commit", default="", help="the other commit, when --role is baseline/branch"
    )
    ap.add_argument(
        "--diff",
        action="append",
        default=[],
        metavar="METRIC:PATH",
        help="an already-generated res_diff_<METRIC>.csv (see resource_usage_diff.py); "
        "repeatable, one per metric",
    )
    ap.add_argument("--out", required=True, type=Path, help="output HTML path")
    ap.add_argument(
        "--top", type=int, default=20, help="rows shown in the chart / diff sections"
    )
    args = ap.parse_args()

    if not args.csv.exists():
        sys.exit(f"error: csv not found: {args.csv}")

    rows = load_kernels(args.csv)
    if not rows:
        sys.exit(f"error: no kernel rows in {args.csv}")
    stats = compute_stats(rows)

    diffs = {}
    for spec in args.diff:
        metric, sep, path = spec.partition(":")
        if not sep:
            sys.exit(f"error: --diff must be METRIC:PATH, got {spec!r}")
        if metric not in NUMERIC_COLS:
            sys.exit(f"error: --diff metric {metric!r} is not one of {NUMERIC_COLS}")
        diff_path = Path(path)
        if not diff_path.exists():
            sys.exit(f"error: diff csv not found: {diff_path}")
        diffs[metric] = load_diff(metric, diff_path, args.top)

    html_out = render_page(args, rows, stats, diffs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_out)
    print(f"Wrote dashboard to {args.out}")


if __name__ == "__main__":
    main()
