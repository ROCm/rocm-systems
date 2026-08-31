#!/usr/bin/env python3
###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Generate one self-contained, interactive HTML dashboard for an LTO
inline-remarks report: either a single commit's snapshot, or a baseline-vs-
branch comparison with both commits' data embedded in the same file. Mirrors
resource_usage_dashboard.py's layout (comparison view on top, an in-page
commit toggle below). See lto_inline_remarks_to_csv.py / _diff.py / _report.py.

No external JS/CSS, no build step -- double-click the output file to view it.

Usage:
    python3 scripts/analysis/lto_inline_remarks_dashboard.py \\
        --baseline-csv remarks-abc123.csv --baseline-commit abc123 \\
        --out dashboard.html --top 20

    python3 scripts/analysis/lto_inline_remarks_dashboard.py \\
        --baseline-csv remarks-abc123.csv --baseline-commit abc123 \\
        --branch-csv   remarks-def456.csv --branch-commit def456 \\
        --pairs-csv remarks_diff_pairs.csv \\
        --out dashboard.html --top 20
"""

import argparse
import csv
import html
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import lto_inline_remarks_diff as lrd  # reuse PASSED_KINDS/MISSED_KINDS/COLOR_* --
# keeps this dashboard's Passed/Missed classification in lockstep with the
# existing CSV/PNG report instead of a second, possibly-drifting copy.

PASSED_KINDS = lrd.PASSED_KINDS
MISSED_KINDS = lrd.MISSED_KINDS
COLOR_GOOD = lrd.COLOR_GOOD
COLOR_BAD = lrd.COLOR_BAD
COLOR_NEUTRAL = lrd.COLOR_NEUTRAL

REMARK_FIELDS = [
    "kind",
    "pass",
    "name",
    "caller_demangled",
    "caller_mangled",
    "callee_demangled",
    "callee_mangled",
    "cost",
    "threshold",
    "reason",
    "message",
]

PAIR_FIELDS = lrd.PAIR_FIELDS


def to_num(v):
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def load_rows(csv_path: Path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


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
    ).most_common()

    near_misses = []
    for r in rows:
        if r["name"] != "TooCostly":
            continue
        cost, threshold = to_num(r["cost"]), to_num(r["threshold"])
        if cost is None or threshold is None:
            continue
        near_misses.append((f"{r['caller_demangled']} → {r['callee_demangled']}", cost - threshold))
    near_misses.sort(key=lambda t: t[1])

    return {
        "total": total,
        "kind_counts": {k: kind_counts.get(k, 0) for k in ("Inlined", "AlwaysInline", "NeverInline", "TooCostly")},
        "inline_rate": round(inline_rate * 100, 1),
        "top_callers": top_callers,
        "top_callees": top_callees,
        "never_inline_reasons": never_inline_reasons,
        "near_misses": near_misses,
    }


def render_stat_cards(summary):
    def card(label, value):
        return f'<div class="stat-card"><div class="stat-value">{value}</div><div class="stat-label">{html.escape(label)}</div></div>'

    kc = summary["kind_counts"]
    cards = [
        card("Total remarks", summary["total"]),
        card("Inline rate", f"{summary['inline_rate']}%"),
        card("Inlined", kc["Inlined"]),
        card("AlwaysInline", kc["AlwaysInline"]),
        card("NeverInline", kc["NeverInline"]),
        card("TooCostly", kc["TooCostly"]),
    ]
    return f'<div class="stat-grid">{"".join(cards)}</div>'


def render_comparison_stat_cards(pair_rows):
    added = sum(1 for r in pair_rows if r["status"] == "added")
    removed = sum(1 for r in pair_rows if r["status"] == "removed")
    common = sum(1 for r in pair_rows if r["status"] == "common")
    regressions = sum(1 for r in pair_rows if r["regression_signal"] in ("True", True))
    improvements = sum(1 for r in pair_rows if r["improvement_signal"] in ("True", True))

    def card(label, value):
        return f'<div class="stat-card"><div class="stat-value">{value}</div><div class="stat-label">{html.escape(label)}</div></div>'

    cards = [
        card("Pairs compared", len(pair_rows)),
        card("Common", common),
        card("Added", added),
        card("Removed", removed),
        card("Regressions (Passed→Missed)", regressions),
        card("Improvements (Missed→Passed)", improvements),
    ]
    return f'<div class="stat-grid">{"".join(cards)}</div>'


NOTES_HTML = """
<section class="notes">
  <h2>Reading this report</h2>
  <p><strong>Passed kinds</strong> (Inlined, AlwaysInline) mean the callsite was inlined;
  <strong>Missed kinds</strong> (NeverInline, TooCostly) mean it was not. A pair flipping
  from a Passed kind to a Missed kind between baseline and branch is a <em>regression</em>;
  the reverse is an <em>improvement</em> -- this is a categorical change in code shape, not
  just a numeric shift, so it is surfaced ahead of a plain |cost delta| ranking.</p>
  <p><strong>Cost</strong>/<strong>Threshold</strong> only apply to Inlined/TooCostly remarks
  (the inliner's actual cost-model comparison); NeverInline instead carries a
  <strong>reason</strong> (e.g. a <code>noinline</code> attribute) and AlwaysInline has
  neither -- it bypasses the cost model entirely.</p>
  <p>The <strong>caller-count cliff</strong>: a callee's distinct-caller count crossing the
  1&harr;2+ boundary can flip a large "last call to a static function" cost bonus on/off
  independent of any change to the callee itself -- see LTO_INLINE_CALLER_COUNT_ISSUE.md.</p>
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
  .scroll-box {{ max-height: 520px; overflow-y: auto; overflow-x: auto; border: 1px solid var(--gridline); border-radius: 6px; }}
  .scroll-box table thead th {{ position: sticky; top: 0; background: var(--surface-1); z-index: 1; }}
  .scroll-box table {{ margin: 0; }}
  .notes p {{ color: var(--ink-secondary); font-size: 13px; line-height: 1.5; margin-bottom: 10px; }}
  .commit-toggle {{ display: flex; gap: 8px; margin-bottom: 16px; }}
  .commit-toggle button {{
    font: inherit; padding: 8px 16px; border-radius: 6px; border: 1px solid var(--gridline);
    background: var(--surface-panel); color: var(--ink-secondary); cursor: pointer;
  }}
  .commit-toggle button.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
  .legend {{ color: var(--ink-secondary); font-size: 12px; margin-bottom: 10px; }}
  svg text {{ fill: var(--ink-secondary); font-size: 12px; }}
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
    <div class="chart-section">
      <h2>Callers / callees / reasons</h2>
      <p class="legend">Bar length = count/cost-gap for the selected view; green bars = negative value (e.g. cost under threshold).</p>
      <div class="controls">
        <select id="viewPicker"></select>
      </div>
      <div id="chartContainer" class="scroll-box"></div>
    </div>
    <div class="table-section">
      <h2>All remarks</h2>
      <p class="legend">Kind = raw LLVM remark tag (Passed/Missed/Analysis); Name = specific reason (Inlined, AlwaysInline, NeverInline, TooCostly).</p>
      <div class="controls">
        <input type="text" id="searchBox" placeholder="Filter by caller/callee name...">
        <select id="kindFilter"></select>
      </div>
      <div class="scroll-box">
        <table id="remarkTable">
          <thead><tr></tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
  </section>
</div>
<script>
const REMARKS_BY_COMMIT = {remarks_by_commit_json};
const STATS_HTML_BY_COMMIT = {stats_html_by_commit_json};
const VIEWS_BY_COMMIT = {views_by_commit_json};
const COLOR_GOOD = "{color_good}", COLOR_BAD = "{color_bad}", COLOR_NEUTRAL = "{color_neutral}";
const PAIR_ROWS = {pair_rows_json};

function toNum(v) {{ const n = parseInt(v, 10); return isNaN(n) ? null : n; }}

// --- commit toggle (per-commit section) -----------------------------------
let activeCommit = Object.keys(REMARKS_BY_COMMIT)[0];

function setActiveCommit(commit) {{
  activeCommit = commit;
  document.querySelectorAll(".commit-toggle button").forEach(b => {{
    b.classList.toggle("active", b.dataset.commit === commit);
  }});
  document.getElementById("commitStats").innerHTML = STATS_HTML_BY_COMMIT[commit];
  buildKindFilter();
  renderTable();
  drawChart(document.getElementById("viewPicker").value);
}}

document.querySelectorAll(".commit-toggle button").forEach(b => {{
  b.addEventListener("click", () => setActiveCommit(b.dataset.commit));
}});

// --- sortable / filterable raw-remarks table (active commit only) --------
const COLUMNS = ["kind", "name", "caller_demangled", "callee_demangled", "cost", "threshold", "reason"];
const HEAD_LABELS = ["Kind", "Name", "Caller", "Callee", "Cost", "Threshold", "Reason"];
let sortCol = "name", sortDesc = false;

function buildKindFilter() {{
  const picker = document.getElementById("kindFilter");
  picker.textContent = "";
  const kinds = Array.from(new Set(REMARKS_BY_COMMIT[activeCommit].map(r => r.kind))).sort();
  const allOpt = document.createElement("option");
  allOpt.value = "all"; allOpt.textContent = "All kinds";
  picker.appendChild(allOpt);
  for (const k of kinds) {{
    const opt = document.createElement("option");
    opt.value = k; opt.textContent = k;
    picker.appendChild(opt);
  }}
}}

function renderTable() {{
  const filter = document.getElementById("searchBox").value.toLowerCase();
  const kindFilter = document.getElementById("kindFilter").value;
  const rows = REMARKS_BY_COMMIT[activeCommit].filter(r =>
    (r.caller_demangled.toLowerCase().includes(filter) || r.callee_demangled.toLowerCase().includes(filter)) &&
    (kindFilter === "all" || r.kind === kindFilter)
  );
  rows.sort((a, b) => {{
    const av = a[sortCol] ?? "", bv = b[sortCol] ?? "";
    const an = toNum(av), bn = toNum(bv);
    const cmp = (an !== null && bn !== null) ? an - bn : String(av).localeCompare(String(bv));
    return sortDesc ? -cmp : cmp;
  }});
  const tbody = document.querySelector("#remarkTable tbody");
  tbody.textContent = "";
  for (const r of rows) {{
    const tr = document.createElement("tr");
    for (const col of COLUMNS) {{
      const td = document.createElement("td");
      td.textContent = r[col] ?? "";
      tr.appendChild(td);
    }}
    tbody.appendChild(tr);
  }}
}}

function buildHeader() {{
  const tr = document.querySelector("#remarkTable thead tr");
  tr.textContent = "";
  COLUMNS.forEach((col, i) => {{
    const th = document.createElement("th");
    th.textContent = HEAD_LABELS[i];
    th.addEventListener("click", () => {{
      if (sortCol === col) sortDesc = !sortDesc; else {{ sortCol = col; sortDesc = false; }}
      renderTable();
    }});
    tr.appendChild(th);
  }});
}}

document.getElementById("searchBox").addEventListener("input", renderTable);
document.getElementById("kindFilter").addEventListener("change", renderTable);

// --- SVG horizontal bar chart for the selected view (active commit only) -
const ns = "http://www.w3.org/2000/svg";
function el(tag, attrs) {{
  const e = document.createElementNS(ns, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}}

function wrapLabel(label, maxChars) {{
  if (label.length <= maxChars) return [label];
  let breakAt = -1;
  for (const sep of ["::", "(", ", ", " "]) {{
    const idx = label.lastIndexOf(sep, maxChars);
    if (idx > maxChars * 0.3) {{ breakAt = idx + sep.length; break; }}
  }}
  if (breakAt === -1) breakAt = maxChars;
  const line1 = label.slice(0, breakAt);
  let line2 = label.slice(breakAt);
  if (line2.length > maxChars) line2 = line2.slice(0, maxChars - 1) + "…";
  return [line1, line2];
}}

function drawChart(viewKey) {{
  const container = document.getElementById("chartContainer");
  container.textContent = "";
  const items = VIEWS_BY_COMMIT[activeCommit][viewKey] || [];
  if (!items.length) {{
    const p = document.createElement("p");
    p.textContent = "(no data)";
    p.style.color = COLOR_NEUTRAL;
    container.appendChild(p);
    return;
  }}
  const marginL = 360, marginR = 60, rowH = 40, plotW = 780, maxChars = 44;
  const W = marginL + plotW + marginR;
  const H = items.length * rowH + 20;
  const vmax = Math.max(...items.map(it => Math.abs(it[1])), 1);
  const svg = el("svg", {{ viewBox: `0 0 ${{W}} ${{H}}`, width: W, height: H }});
  items.forEach((it, i) => {{
    const [label, value] = it;
    const y = i * rowH;
    const barW = (Math.abs(value) / vmax) * plotW;
    const color = value < 0 ? COLOR_GOOD : COLOR_NEUTRAL;
    svg.appendChild(el("rect", {{ x: marginL, y: y + rowH / 2 - 7, width: Math.max(barW, 1), height: 14, fill: color }}));
    const lines = wrapLabel(label, maxChars);
    const lbl = el("text", {{ x: marginL - 8, y: y + rowH / 2 - (lines.length - 1) * 6 + 4, "text-anchor": "end" }});
    lines.forEach((line, li) => {{
      const tspan = document.createElementNS(ns, "tspan");
      tspan.setAttribute("x", marginL - 8);
      tspan.setAttribute("dy", li === 0 ? 0 : 13);
      tspan.textContent = line;
      lbl.appendChild(tspan);
    }});
    svg.appendChild(lbl);
    const val = el("text", {{ x: marginL + barW + 6, y: y + rowH / 2 + 4 }});
    val.textContent = value;
    svg.appendChild(val);
  }});
  container.appendChild(svg);
}}

function buildViewPicker() {{
  const picker = document.getElementById("viewPicker");
  const views = [
    ["top_callers", "Top callers (by inlined-callee count)"],
    ["top_callees", "Top callees (by distinct-caller count)"],
    ["never_inline_reasons", "NeverInline reasons"],
    ["near_misses", "TooCostly near-misses (cost - threshold)"],
  ];
  for (const [key, label] of views) {{
    const opt = document.createElement("option");
    opt.value = key; opt.textContent = label;
    picker.appendChild(opt);
  }}
  picker.addEventListener("change", () => drawChart(picker.value));
}}

buildHeader();
buildViewPicker();
setActiveCommit(activeCommit);
drawChart("top_callers");

// --- comparison table (baseline vs branch, all pairs, no top-N cap) ------
if (PAIR_ROWS.length) {{
  const PAIR_COLUMNS = ["callee_demangled", "caller_demangled", "status",
    "name_baseline", "cost_baseline", "threshold_baseline",
    "name_branch", "cost_branch", "threshold_branch",
    "cost_delta", "threshold_delta"];
  const PAIR_HEAD_LABELS = ["Callee", "Caller", "Status",
    "Name (baseline)", "Cost (baseline)", "Threshold (baseline)",
    "Name (branch)", "Cost (branch)", "Threshold (branch)",
    "Δ Cost", "Δ Threshold"];
  const STATUS_RANK = {{ removed: 0, added: 1, common: 2 }};
  const STATUS_CELL_COLOR = {{ removed: "#dba3a3", added: "#a9d1b2", common: "#a3bedb" }};
  const STATUS_CELL_TEXT = {{ removed: "#4a1414", added: "#173a22", common: "#15304a" }};
  let pairSortCol = "status", pairSortDesc = false;
  let pairView = "all";

  function isTrue(v) {{ return v === "True" || v === true; }}

  function viewFilter(r) {{
    if (pairView === "flips") return isTrue(r.regression_signal) || isTrue(r.improvement_signal);
    if (pairView === "added") return r.status === "added";
    if (pairView === "removed") return r.status === "removed";
    return true;
  }}

  function buildPairHeader() {{
    const tr = document.querySelector("#pairTable thead tr");
    tr.textContent = "";
    PAIR_COLUMNS.forEach((col, i) => {{
      const th = document.createElement("th");
      th.textContent = PAIR_HEAD_LABELS[i];
      th.addEventListener("click", () => {{
        if (pairSortCol === col) pairSortDesc = !pairSortDesc; else {{ pairSortCol = col; pairSortDesc = true; }}
        renderPairTable();
      }});
      tr.appendChild(th);
    }});
  }}

  function renderPairTable() {{
    const filter = document.getElementById("pairSearchBox").value.toLowerCase();
    const rows = PAIR_ROWS.filter(viewFilter).filter(r =>
      r.caller_demangled.toLowerCase().includes(filter) || r.callee_demangled.toLowerCase().includes(filter)
    );
    rows.sort((a, b) => {{
      if (pairSortCol === "status") {{
        const cmp = STATUS_RANK[a.status] - STATUS_RANK[b.status];
        return pairSortDesc ? -cmp : cmp;
      }}
      const av = a[pairSortCol] ?? "", bv = b[pairSortCol] ?? "";
      const an = toNum(av), bn = toNum(bv);
      const cmp = (an !== null && bn !== null) ? an - bn : String(av).localeCompare(String(bv));
      return pairSortDesc ? -cmp : cmp;
    }});
    const tbody = document.querySelector("#pairTable tbody");
    tbody.textContent = "";
    for (const r of rows) {{
      const tr = document.createElement("tr");
      for (const col of PAIR_COLUMNS) {{
        const td = document.createElement("td");
        td.textContent = r[col] ?? "";
        if (col === "status") {{
          td.style.backgroundColor = STATUS_CELL_COLOR[r.status] || "";
          td.style.color = STATUS_CELL_TEXT[r.status] || "";
          td.style.fontWeight = "600";
        }}
        if (col === "cost_delta" || col === "threshold_delta") {{
          if (isTrue(r.regression_signal)) td.style.color = COLOR_BAD;
          else if (isTrue(r.improvement_signal)) td.style.color = COLOR_GOOD;
          td.style.fontWeight = "600";
        }}
        tr.appendChild(td);
      }}
      tbody.appendChild(tr);
    }}
  }}

  document.getElementById("pairSearchBox").addEventListener("input", renderPairTable);
  document.getElementById("pairViewPicker").addEventListener("change", (e) => {{ pairView = e.target.value; renderPairTable(); }});
  buildPairHeader();
  renderPairTable();
}}
</script>
</body>
</html>
"""


def render_comparison_section(pair_rows, baseline_commit, branch_commit):
    if not pair_rows:
        return ""
    return f"""
  <section class="comparison-section">
    <h2>Baseline vs Branch &mdash; {html.escape(baseline_commit[:12])} &rarr; {html.escape(branch_commit[:12])}</h2>
    <p class="legend">Status cell color: green = added, red = removed, blue = common (present in both). &Delta; Cost/&Delta; Threshold text: green = improvement, red = regression.</p>
    {render_comparison_stat_cards(pair_rows)}
    <div class="controls">
      <select id="pairViewPicker">
        <option value="all">All pairs</option>
        <option value="flips">Regressions + improvements only</option>
        <option value="added">Added only</option>
        <option value="removed">Removed only</option>
      </select>
      <input type="text" id="pairSearchBox" placeholder="Filter by caller/callee name...">
    </div>
    <div class="scroll-box">
      <table id="pairTable">
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


def render_page(args, baseline_rows, branch_rows, pair_rows):
    def remarks_json(rows):
        return [{col: r.get(col, "") for col in REMARK_FIELDS} for r in rows]

    def views_json(summary):
        return {
            "top_callers": summary["top_callers"],
            "top_callees": summary["top_callees"],
            "never_inline_reasons": summary["never_inline_reasons"],
            "near_misses": summary["near_misses"],
        }

    baseline_summary = build_summary(baseline_rows)
    remarks_by_commit = {"baseline": remarks_json(baseline_rows)}
    stats_html_by_commit = {"baseline": render_stat_cards(baseline_summary)}
    views_by_commit = {"baseline": views_json(baseline_summary)}
    if branch_rows is not None:
        branch_summary = build_summary(branch_rows)
        remarks_by_commit["branch"] = remarks_json(branch_rows)
        stats_html_by_commit["branch"] = render_stat_cards(branch_summary)
        views_by_commit["branch"] = views_json(branch_summary)

    baseline_short = args.baseline_commit[:12] if args.baseline_commit else "(working tree)"
    if args.branch_commit:
        title = f"LTO inline remarks — {baseline_short} vs {args.branch_commit[:12]}"
    else:
        title = f"LTO inline remarks — {baseline_short}"

    return PAGE_TEMPLATE.format(
        title=html.escape(title),
        arch=html.escape(baseline_rows[0]["arch"]) if baseline_rows else "?",
        build_config=html.escape(baseline_rows[0]["build_config"]) if baseline_rows else "?",
        comparison_section=render_comparison_section(
            pair_rows, args.baseline_commit, args.branch_commit
        ),
        commit_toggle=render_commit_toggle(args.baseline_commit, args.branch_commit),
        notes=NOTES_HTML,
        remarks_by_commit_json=json.dumps(remarks_by_commit),
        stats_html_by_commit_json=json.dumps(stats_html_by_commit),
        views_by_commit_json=json.dumps(views_by_commit),
        top_n=args.top,
        color_good=COLOR_GOOD,
        color_bad=COLOR_BAD,
        color_neutral=COLOR_NEUTRAL,
        pair_rows_json=json.dumps(pair_rows),
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--baseline-csv", required=True, type=Path, help="this commit's remarks-<sha>.csv"
    )
    ap.add_argument("--baseline-commit", default="", help="this commit's sha/ref")
    ap.add_argument(
        "--branch-csv",
        type=Path,
        default=None,
        help="the other commit's remarks-<sha>.csv (omit for a single-commit snapshot)",
    )
    ap.add_argument("--branch-commit", default="", help="the other commit's sha/ref")
    ap.add_argument(
        "--pairs-csv",
        type=Path,
        default=None,
        help="an already-generated remarks_diff_pairs.csv (see lto_inline_remarks_diff.py); "
        "only meaningful together with --branch-csv.",
    )
    ap.add_argument("--out", required=True, type=Path, help="output HTML path")
    ap.add_argument(
        "--top",
        type=int,
        default=15,
        help="unused (kept for compatibility with existing callers) -- the "
        "per-commit chart now shows every entry in a scrollable container",
    )
    args = ap.parse_args()

    if not args.baseline_csv.exists():
        sys.exit(f"error: baseline csv not found: {args.baseline_csv}")
    baseline_rows = load_rows(args.baseline_csv)
    if not baseline_rows:
        sys.exit(f"error: no rows in {args.baseline_csv}")

    branch_rows = None
    if args.branch_csv:
        if not args.branch_csv.exists():
            sys.exit(f"error: branch csv not found: {args.branch_csv}")
        branch_rows = load_rows(args.branch_csv)
        if not branch_rows:
            sys.exit(f"error: no rows in {args.branch_csv}")

    pair_rows = []
    if args.pairs_csv:
        if branch_rows is None:
            sys.exit("error: --pairs-csv requires --branch-csv/--branch-commit")
        if not args.pairs_csv.exists():
            sys.exit(f"error: pairs csv not found: {args.pairs_csv}")
        pair_rows = load_rows(args.pairs_csv)

    html_out = render_page(args, baseline_rows, branch_rows, pair_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_out)
    print(f"Wrote dashboard to {args.out}")


if __name__ == "__main__":
    main()
