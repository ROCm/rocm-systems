#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
# ruff: noqa: E501
"""Build docs/plans/aiprofcomp-865-problem-decompose.html from live gfx942
inspector data.
"""

from __future__ import annotations

import html
import json
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_OUT = _ROOT / "docs/plans/aiprofcomp-865-problem-decompose.html"


def _metric_list_html(metrics: list[dict], *, css_class: str = "") -> str:
    if not metrics:
        return "<em>(none)</em>"
    lines = []
    for m in sorted(metrics, key=lambda x: x["id"]):
        bk = ", ".join(html.escape(b) for b in m["buckets"])
        lines.append(
            f"<li><code>{html.escape(m['id'])}</code> "
            f"{html.escape(m['name'])} "
            f"<span class='meta'>({m['bucket_count']} buckets: {bk})</span></li>"
        )
    cls = f" metric-list {css_class}".strip()
    return f"<ul class='{cls}'>{''.join(lines)}</ul>"


def _slot_dup_note_html(dups: dict) -> str:
    """Comment block for panel-mirrored SLOT_LIMIT metrics (same PMC set)."""
    if not dups["duplicate_groups"]:
        return (
            "Cannot pack all PMCs into one hardware bucket; decomposition required. "
            "No panel-mirrored duplicates in this set."
        )
    group_bits = []
    for g in dups["groups"]:
        ids = " ≡ ".join(
            f"<code>{html.escape(m['id'])}</code> {html.escape(m['name'])}"
            for m in g["metrics"]
        )
        group_bits.append(
            f"<li>×{g['count']} same PMC set ({g['pmc_count']} PMCs): {ids}</li>"
        )
    groups_html = "<ul class='dup-groups'>" + "".join(group_bits) + "</ul>"
    return (
        f"<strong>Panel duplicates:</strong> {dups['metric_count']} parent metrics → "
        f"<strong>{dups['unique_pmc_sets']}</strong> unique PMC sets "
        f"({dups['redundant_rows']} redundant rows in {dups['duplicate_groups']} groups). "
        f"Phase&nbsp;2 unique HW problems ≈ {dups['unique_pmc_sets']}, not "
        f"{dups['metric_count']}. Same counters shown on multiple panels "
        f"(e.g. System SoL vs CU pipeline)."
        f"{groups_html}"
    )


def _slot_metric_table_html(metrics: list[dict], dups: dict) -> str:
    """Flat SLOT_LIMIT table: one row per parent metric (no packing-method columns)."""
    note_by_id: dict[str, str] = {m["id"]: "Unique PMC set" for m in metrics}
    for g in dups.get("groups", []):
        members = g["metrics"]
        canonical = members[0]["id"]
        mirrors = ", ".join(f"<code>{html.escape(m['id'])}</code>" for m in members[1:])
        note_by_id[canonical] = f"Canonical; mirrored by {mirrors}"
        for m in members[1:]:
            note_by_id[m["id"]] = (
                f"Panel mirror of <code>{html.escape(canonical)}</code>"
            )

    rows = []
    for m in sorted(metrics, key=lambda x: x["id"]):
        rows.append(
            "<tr class='slot-limit'>"
            f"<td><code>{html.escape(m['id'])}</code></td>"
            f"<td>{html.escape(m['name'])}</td>"
            f"<td>{m['bucket_count']}</td>"
            f"<td>{m['pmc_count']}</td>"
            "<td><strong>No +passes</strong></td>"
            f"<td>{note_by_id[m['id']]}</td>"
            "</tr>"
        )
    return (
        '<table class="decompose slot-metrics">'
        "<thead><tr>"
        "<th>Metric id</th><th>Name</th><th>#Buckets</th><th>#PMCs</th>"
        "<th>Pass impact</th><th>Notes</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def main() -> None:
    data_script = _ROOT / "tools/aiprofcomp865_problem_decompose_data.py"
    raw = subprocess.check_output(
        [sys.executable, str(data_script)],
        cwd=_ROOT,
        env={**os.environ, "PYTHONPATH": str(_ROOT / "src")},
        text=True,
    )
    data = json.loads(raw)

    b = data["before"]
    a = data["after"]
    with_pmc = b["with_pmc"]
    no_profile_pmc = data["yaml_metric_total"] - with_pmc
    multi_before = len(b["multi"])
    multi_pct = round(100 * multi_before / with_pmc, 1) if with_pmc else 0.0
    multi_after = len(a["multi"])
    multi_after_pct = round(100 * multi_after / with_pmc, 1) if with_pmc else 0.0
    slot_dups = a["slot_limit_pmc_dups"]
    slot_dup_note = _slot_dup_note_html(slot_dups)
    slot_table = _slot_metric_table_html(a["slot_limit"], slot_dups)

    algo_comments = """
    <ol class='comments'>
      <li><strong>Two goals, not one objective:</strong> Round&nbsp;1 minimizes
      <em>pass count</em> (user cost). A second goal is <em>single-pass collection for
      packable metrics</em> (analyze correctness). Optimizing both equally in one
      global search is a bi-criteria rewrite—harder to reason about, riskier to ship,
      and not required for 865. An <em>online</em> joint search inside every
      <code>profile</code> is especially hard outside the default full-panel set:
      <code>--block</code> / <code>--set</code> subsets, other arches, TCC channel
      rules, and priority-policy mixes all change the instance, so a one-step solver
      that looks good on gfx942 default does not generalize cleanly.</li>
      <li><strong>Keep pass-count primary in round&nbsp;1:</strong> Today&apos;s
      heuristic + priority policy already targets (a). Folding (b) into the same
      greedy pack as an equal weight tends to trade extra passes or destabilize
      layouts users already depend on.</li>
      <li><strong>SPP replaced second-pass repair:</strong> Default allocate now
      guarantees packable unions in one bucket (PMC duplication allowed). Refill
      and CP-SAT paths are removed.</li>
      <li><strong>SLOT_LIMIT stays Phase&nbsp;2:</strong> The 16 parents that cannot
      fit one hardware bucket use <code>WEIGHTED_AVG</code>, not packing.</li>
      <li><strong>Named <code>*_ACCUM</code> PMCs:</strong> With rocprofiler-sdk,
      counters such as <code>SQ_LEVEL_WAVES_ACCUM</code> are
      <code>accumulate(LEVEL, HIGH_RES)</code> definitions and cost
      <strong>2</strong> block slots alone (BASE + HIGH_RES); if BASE is already
      in the same bucket, only +1 is charged. gfx942 <code>SQ</code> cap is
      <strong>8</strong>. Legacy <code>SQ_ACCUM_PREV_HIRES</code> pairing /
      dedicated accum buckets are not part of the allocator.</li>
    </ol>
    """

    weighted_avg_why_not_57 = f"""
    <ol class='comments'>
      <li><strong>Not slot-limited:</strong> For these {len(a["packable_multi"])}, <code>CounterFile</code> trial packing proves the full PMC set fits <em>one</em> hardware bucket; the error is cross-pass evaluation of a single ratio, not impossible HW layout.</li>
      <li><strong>WEIGHTED_AVG is for decomposition:</strong> Phase 2 merge applies when the parent <em>cannot</em> be collected in one bucket and you designed submetrics + proved <code>weight_counter</code> algebra—using it here would invent weights without a partition identity.</li>
      <li><strong>Wrong fix, new risk:</strong> Averaging sub-passes with ad hoc weights can change the metric definition (bias vs silent cap); packing fixes preserve the YAML formula on one pass.</li>
      <li><strong>Cost:</strong> {len(a["packable_multi"])} decompositions × sub-rows × analyze graph vs one coalesce/repack change that keeps parent formulas unchanged.</li>
    </ol>
    """

    packable_n = len(a["packable_multi"])
    slot_n = len(a["slot_limit"])
    spp = data["spp"]
    # Default single-pass packable (offline gfx942): all packable metrics get a
    # full-bucket guarantee; SLOT_LIMIT parents stay multi-pass (Phase 2).
    spp_single = with_pmc - slot_n  # product narrative: 374 − 16
    spp_passes = spp["buckets"]
    spp_slot_extra = spp["slot_additional_passes"]
    legacy_passes = data["buckets_after"]
    spp_delta = spp_passes - legacy_passes
    decomp_tree = f"""{data["yaml_metric_total"]} YAML metrics
├── {with_pmc} with profile PMCs
│   ├── <span class="hl">{a["single_count"]}</span> single-pass
│   │     ← Legacy heuristic + prioritized policy
│   └── {len(a["multi"])} multi-pass
│       ├── <span class="hl">{packable_n}</span> POLICY_GAP (packable leftovers)
│       │     ← additive +4 passes (est.) / policy / packing
│       └── <span class="hl">{slot_n}</span> SLOT_LIMIT
│             ← WEIGHTED_AVG collectables (Phase 2)
└── {no_profile_pmc} with no profile PMCs
      ← out of packing scope"""

    spp_decomp_tree = f"""{data["yaml_metric_total"]} YAML metrics
├── {with_pmc} with profile PMCs
│   ├── <span class="hl">{spp_single}</span> single-pass (all packable)
│   │     ← Default SPP + SLOT fill, {spp_passes} passes (Phase 1)
│   │     ← was {a["single_count"]} + {packable_n} POLICY_GAP under legacy
│   └── <span class="hl">{slot_n}</span> SLOT_LIMIT (parent still multi-pass)
│         ← WEIGHTED_AVG collectables (Phase 2)
│         ← PMC presence already in the {spp_passes} (+{spp_slot_extra} passes)
└── {no_profile_pmc} with no profile PMCs
      ← out of packing scope"""
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>AIPROFCOMP-865 — gfx942 problem decomposition</title>
  <style>
    :root {{
      --bg: #f8f9fb;
      --border: #c5cdd8;
      --text: #1a1d21;
      --muted: #5c6570;
      --slot: #e8d4ff;
      --slot-border: #9b6bd6;
      --ok: #e6f4ea;
      --warn: #fff8e6;
    }}
    html {{ font-size: 16px; }}
    body {{
      font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 1.5rem auto;
      padding: 0 1.5rem 2rem;
      line-height: 1.55;
      max-width: 1400px;
      font-size: 1rem;
    }}
    h1 {{ font-size: 1.5rem; margin-bottom: 0.35rem; font-weight: 600; }}
    section.doc-section p {{ margin: 0.5rem 0 0; max-width: 72ch; }}
    section.doc-section.target p {{
      max-width: none;
    }}
    section.doc-section {{
      background: #fff;
      border: 1px solid var(--border);
      padding: 1.25rem 1.5rem;
      margin: 1.25rem 0;
      box-shadow: 0 1px 3px rgba(0,0,0,.04);
    }}
    section.doc-section h2 {{
      font-size: 1.2rem;
      margin: 0 0 0.75rem;
      color: #1e3a5f;
      font-weight: 600;
    }}
    section.doc-section.status {{
      border-left: 4px solid #64748b;
    }}
    section.doc-section.terminology {{
      border-left: 4px solid #7c3aed;
    }}
    table.terms {{
      width: 100%;
      table-layout: fixed;
      border-collapse: collapse;
      margin: 0.5rem 0 0;
      font-size: 0.98rem;
      line-height: 1.5;
    }}
    table.terms th, table.terms td {{
      border: 1px solid var(--border);
      padding: 0.6rem 0.75rem;
      vertical-align: top;
      text-align: left;
    }}
    table.terms thead th {{
      background: #f3eefc;
      font-weight: 600;
      color: #0f172a;
    }}
    table.terms thead th:first-child,
    table.terms tbody th {{
      width: 8.5rem;
      max-width: 8.5rem;
      font-weight: 600;
      color: #0f172a;
      word-break: break-word;
    }}
    table.terms td {{
      color: #334155;
    }}
    section.doc-section.target {{
      border-left: 4px solid #2563eb;
    }}
    section.doc-section.solution {{
      border-left: 4px solid #059669;
    }}
    section.doc-section.faq {{
      border-left: 4px solid #d97706;
    }}
    section.doc-section.takeaway {{
      border-left: 4px solid #0f766e;
    }}
    section.doc-section.takeaway h3 {{
      font-size: 1.05rem;
      margin: 1.1rem 0 0.5rem;
      color: #1e3a5f;
      font-weight: 600;
    }}
    section.doc-section.takeaway h3:first-of-type {{
      margin-top: 0.35rem;
    }}
    .decomp-tree {{
      margin: 0.5rem 0 0;
      padding: 0.85rem 1rem;
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 6px;
      font-family: ui-monospace, "Cascadia Code", "SF Mono", Menlo, monospace;
      font-size: 0.92rem;
      line-height: 1.55;
      color: #0f172a;
      white-space: pre-wrap;
      overflow-x: auto;
    }}
    .decomp-tree .hl {{ font-weight: 700; color: #0f766e; }}
    .open-discuss {{
      margin: 0.5rem 0 0;
      padding-left: 1.25rem;
      color: #334155;
      font-size: 0.98rem;
      line-height: 1.55;
      max-width: none;
    }}
    .open-discuss li {{ margin: 0.4rem 0; }}
    ul.planned-todo {{
      margin: 0.5rem 0 0;
      padding-left: 1.25rem;
      color: #334155;
      font-size: 0.98rem;
      line-height: 1.55;
      max-width: none;
    }}
    ul.planned-todo li {{ margin: 0.4rem 0; }}
    section.doc-section.evaluation h2 {{
      margin-bottom: 0.85rem;
    }}
    section.doc-section.evaluation h3 {{
      font-size: 1.05rem;
      margin: 1.25rem 0 0.5rem;
      color: #1e3a5f;
      font-weight: 600;
    }}
    section.doc-section.evaluation h3:first-of-type {{
      margin-top: 0.35rem;
    }}
    section.doc-section.evaluation .table-caption-meta {{
      margin: 0 0 0.5rem;
      max-width: none;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    ol.faq-list {{
      list-style: none;
      margin: 0.75rem 0 0;
      padding: 0;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }}
    ol.faq-list > li {{
      margin: 0;
      padding: 0.85rem 1rem;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fffbf0;
    }}
    ol.faq-list .faq-q {{
      display: block;
      font-weight: 600;
      font-size: 1.02rem;
      margin: 0 0 0.45rem;
      color: #0f172a;
    }}
    ol.faq-list .faq-q .meta {{
      font-weight: 400;
    }}
    ol.faq-list ol.comments {{
      margin: 0;
      padding-left: 0;
      list-style: none;
      counter-reset: faq-idx;
      color: #334155;
      font-size: 0.98rem;
      line-height: 1.55;
    }}
    ol.faq-list ol.comments li {{
      margin: 0.35rem 0;
      padding-left: 2rem;
      position: relative;
      counter-increment: faq-idx;
    }}
    ol.faq-list ol.comments li::before {{
      content: "(" counter(faq-idx) ")";
      position: absolute;
      left: 0;
      font-weight: 600;
      color: #0f172a;
    }}
    .status-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 0.75rem 1.25rem;
      margin-top: 0.5rem;
    }}
    .status-grid dt {{ font-weight: 600; margin: 0; font-size: 0.95rem; }}
    .status-grid dd {{ margin: 0.2rem 0 0; color: var(--muted); font-size: 0.95rem; line-height: 1.45; }}
    ol.solutions {{
      list-style: none;
      margin: 0.75rem 0 0;
      padding: 0;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }}
    ol.solutions li {{
      margin: 0;
      padding: 0.85rem 1rem;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fafbfc;
    }}
    ol.solutions .solution-title {{
      display: block;
      font-weight: 600;
      font-size: 1.02rem;
      margin: 0 0 0.45rem;
      color: #0f172a;
    }}
    ol.solutions .solution-body {{
      margin: 0;
      max-width: none;
      font-size: 0.98rem;
      line-height: 1.55;
      color: #334155;
    }}
    .flow-diagram {{
      margin: 0.85rem 0 0;
      padding: 0.85rem 1rem 1rem;
      background: #1e1e1e;
      border: 1px solid #3a3a3a;
      border-radius: 6px;
      font-size: 0.92rem;
      line-height: 1.45;
      color: #e5e5e5;
      overflow-x: auto;
    }}
    .flow-diagram .flow-title {{
      font-weight: 600;
      margin: 0 0 0.55rem;
      color: #f3f4f6;
    }}
    .flow-diagram .flow-note {{
      margin: 0.65rem 0 0;
      color: #a3a3a3;
      font-size: 0.88rem;
    }}
    .flow-diagram .mermaid {{
      display: flex;
      justify-content: center;
      background: #1e1e1e;
      margin: 0.25rem 0 0;
    }}
    .flow-diagram .mermaid svg {{
      max-width: 100%;
      height: auto;
    }}
    .flow-diagram pre {{
      margin: 0;
      font-family: ui-monospace, "Cascadia Code", "SF Mono", Menlo, monospace;
      font-size: 0.84rem;
      line-height: 1.4;
      white-space: pre;
    }}
    .flow-diagram .hl-box {{
      background: #14532d;
      color: #bbf7d0;
      font-weight: 600;
    }}
    .walkthrough {{
      margin: 0.85rem 0 0;
      padding: 0.85rem 1rem;
      background: #fffbf0;
      border: 1px solid #e8d4a8;
      border-radius: 6px;
      font-size: 0.92rem;
      line-height: 1.5;
      color: #334155;
    }}
    .walkthrough .wt-title {{
      font-weight: 600;
      margin: 0 0 0.45rem;
      color: #0f172a;
    }}
    .walkthrough .wt-setup {{
      margin: 0 0 0.65rem;
      padding: 0.55rem 0.7rem;
      background: #fff;
      border: 1px solid #ecd9a8;
      border-radius: 4px;
      font-family: ui-monospace, "Cascadia Code", "SF Mono", Menlo, monospace;
      font-size: 0.84rem;
      line-height: 1.45;
      white-space: pre-wrap;
    }}
    .walkthrough ol.wt-steps {{
      margin: 0.35rem 0 0;
      padding-left: 1.25rem;
    }}
    .walkthrough ol.wt-steps li {{
      margin: 0.35rem 0;
      padding: 0;
      border: none;
      background: transparent;
      border-radius: 0;
    }}
    .walkthrough .wt-result {{
      margin: 0.65rem 0 0;
      padding: 0.55rem 0.7rem;
      background: #fff;
      border: 1px solid #ecd9a8;
      border-radius: 4px;
    }}
    .collection-highlight {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
      margin: 1rem 0 0.5rem;
    }}
    @media (max-width: 640px) {{
      .collection-highlight {{ grid-template-columns: 1fr; }}
    }}
    .stat-card {{
      border-radius: 6px;
      padding: 0.85rem 1rem;
      border: 1px solid var(--border);
    }}
    .stat-card.single {{
      background: #e8f5ee;
      border-color: #6bb892;
    }}
    .stat-card.multi {{
      background: #fef3e8;
      border-color: #e0a060;
    }}
    .stat-card .big {{
      font-size: 1.35rem;
      font-weight: 700;
      line-height: 1.25;
      margin: 0.25rem 0;
    }}
    .stat-card .big .pct {{ font-size: 1rem; font-weight: 600; color: var(--muted); }}
    .stat-card p {{ margin: 0.35rem 0 0; font-size: 0.9rem; color: var(--muted); }}
    section.doc-section.evaluation {{
      max-width: none;
      padding: 1.25rem 1rem;
      overflow-x: visible;
    }}
    section.doc-section.evaluation .table-wrap {{
      overflow-x: visible;
      margin-top: 0.5rem;
      width: 100%;
    }}
    table.decompose {{
      width: 100%;
      border-collapse: collapse;
      background: #fff;
      box-shadow: 0 1px 3px rgba(0,0,0,.06);
      font-size: 0.9rem;
      line-height: 1.4;
      table-layout: auto;
    }}
    table.decompose th, table.decompose td {{
      border: 1px solid var(--border);
      padding: 0.45rem 0.5rem;
      vertical-align: top;
      word-break: break-word;
    }}
    table.decompose th {{
      background: #eef1f6;
      text-align: left;
      width: auto;
    }}
    table.decompose td.topic {{
      width: 12rem;
      max-width: 12rem;
      font-weight: 600;
    }}
    table.decompose .metric-list {{
      max-height: 200px;
    }}
    table.decompose ul.dup-groups {{
      margin: 0.4rem 0 0;
      padding-left: 1.15rem;
      font-size: 0.88rem;
      line-height: 1.45;
    }}
    table.decompose ul.dup-groups li {{
      margin: 0.2rem 0;
    }}
    table.decompose.slot-metrics td:nth-child(3),
    table.decompose.slot-metrics td:nth-child(4),
    table.decompose.slot-metrics th:nth-child(3),
    table.decompose.slot-metrics th:nth-child(4) {{
      text-align: right;
      width: 5.5rem;
      white-space: nowrap;
    }}
    .slot-summary {{
      margin: 0.35rem 0 0.75rem;
      padding: 0.75rem 1rem;
      background: var(--slot);
      border: 1px solid var(--slot-border);
      border-radius: 6px;
      font-size: 0.95rem;
      line-height: 1.5;
      color: #334155;
    }}
    .slot-summary ul.dup-groups {{
      margin: 0.4rem 0 0;
      padding-left: 1.15rem;
      font-size: 0.9rem;
    }}
    .meta {{ color: var(--muted); font-size: 0.85em; }}
    .metric-list {{
      max-height: 220px;
      overflow: auto;
      margin: 0.35rem 0 0;
      padding-left: 1.2rem;
      font-size: 0.88rem;
    }}
    tr.slot-limit td {{ background: var(--slot); }}
    tr.slot-limit td.topic {{ border-left: 3px solid var(--slot-border); }}
    tr.improved td {{ background: var(--ok); }}
    code {{ font-size: 0.88em; }}
    .footnote {{ margin-top: 1rem; font-size: 0.85rem; color: var(--muted); }}
    .page-tabs {{
      display: flex;
      gap: 0.35rem;
      margin: 1rem 0 0;
      border-bottom: 2px solid var(--border);
      padding: 0;
    }}
    .page-tabs button {{
      appearance: none;
      border: 1px solid transparent;
      border-bottom: none;
      background: transparent;
      color: var(--muted);
      font: inherit;
      font-weight: 600;
      font-size: 0.95rem;
      padding: 0.55rem 1rem;
      margin-bottom: -2px;
      cursor: pointer;
      border-radius: 6px 6px 0 0;
    }}
    .page-tabs button:hover {{ color: var(--text); background: #eef2f7; }}
    .page-tabs button[aria-selected="true"] {{
      color: #1e3a5f;
      background: #fff;
      border-color: var(--border);
      border-bottom: 2px solid #fff;
    }}
    .tab-panel {{ display: none; padding-top: 0.25rem; }}
    .tab-panel.active {{ display: block; }}
  </style>
</head>
<body>
  <h1>AIPROFCOMP-865 — Problem decomposition (gfx942 default profile)</h1>
  <p class="meta">Generated from <code>tools/aiprofcomp865_problem_decompose_data.py</code>.
  gfx942 default profile; block 3000 excluded unless requested (same as impact report).</p>

  <div class="page-tabs" role="tablist" aria-label="Document views">
    <button type="button" role="tab" id="tab-overview" aria-controls="panel-overview"
      aria-selected="true" data-tab="overview">Overview</button>
    <button type="button" role="tab" id="tab-heuristic" aria-controls="panel-heuristic"
      aria-selected="false" data-tab="heuristic">Fewer passes (legacy)</button>
    <button type="button" role="tab" id="tab-single-pass" aria-controls="panel-single-pass"
      aria-selected="false" data-tab="single-pass">Single-pass packable (how it works)</button>
    <button type="button" role="tab" id="tab-spp-plan" aria-controls="panel-spp-plan"
      aria-selected="false" data-tab="spp-plan">Single-pass packable plan</button>
  </div>

  <div id="panel-overview" class="tab-panel active" role="tabpanel"
    aria-labelledby="tab-overview">

  <section class="doc-section terminology">
    <h2>Terminology</h2>
    <table class="terms">
      <colgroup>
        <col style="width: 8.5rem" />
        <col />
      </colgroup>
      <thead>
        <tr>
          <th>Term</th>
          <th>Meaning</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th scope="row"><a href="../design/analysis-config-redesign/lld-phase1-metric-library.md#layer-15----collectables">Collectable</a></th>
          <td>A <strong>single-pass fragment</strong>: a formula plus the PMC set that
          must be collected together in one perfmon pass (see Xuan&apos;s
          <a href="../design/analysis-config-redesign/lld-phase1-metric-library.md#layer-15----collectables">LLD Layer&nbsp;1.5</a>).
          On the legacy analysis path (865 scope), collectables are ordinary
          metric-table rows (optional <code>_collectable_id</code>). Parents that
          cannot be one collectable use <code>WEIGHTED_AVG</code> /
          <code>COLLECT_SUM</code> to recompose at analyze time from proved
          sub-collectables. Full LLD / SDK collectable registry is out of 865
          scope.
          <br/><br/>
          <strong>Example:</strong> pilot rows <code>hbm_read_sub</code> /
          <code>hbm_write_sub</code> (each a single-pass DRAM % formula), composed by
          parent <code>WEIGHTED_AVG(hbm_read_sub, hbm_write_sub)</code> with
          <code>weight_counter</code>s <code>TCC_EA0_RDREQ_sum</code> /
          <code>TCC_EA0_WRREQ_sum</code>
          (<code>tests/fixtures/weighted_avg/</code>).</td>
        </tr>
        <tr>
          <th scope="row"><code>SLOT_LIMIT</code></th>
          <td>The metric&apos;s <em>full</em> PMC set cannot pack into <strong>one</strong>
          hardware perfmon bucket under gfx942 <code>perfmon_config</code> IP-block slot
          caps—even after global repack. That is <strong>not</strong> the same as
          “multi-pass under today&apos;s layout” (those may still be packable). On gfx942
          default, <strong>16</strong> metrics fall in this category; they need
          decomposition (collectables + <code>WEIGHTED_AVG</code> / redesign), not more
          packing alone.
          <br/><br/>
          <strong>Example:</strong> <code>0200.201.0</code> VALU FLOPs (12 PMCs across 3
          buckets)—the whole-metric set exceeds one-bucket slot caps; split into
          single-pass sub-collectables then recompose (Phase&nbsp;2). Contrast:
          a packable POLICY_GAP metric whose PMCs fit one bucket but are split only
          by today&apos;s layout.</td>
        </tr>
        <tr>
          <th scope="row">Packable multi-pass (POLICY_GAP)</th>
          <td>Formula PMCs span 2+ buckets in the current implementation, but a
          <code>CounterFile</code> trial shows they <em>would</em> fit one bucket.
          Fix via packing / SPP (not <code>WEIGHTED_AVG</code>).</td>
        </tr>
      </tbody>
    </table>
  </section>

  <section class="doc-section status">
    <h2>1. Current status</h2>
    <p><strong>Architecture:</strong> gfx942 default analysis config.
    <strong>Default packing:</strong> Single-pass packable + SLOT fill
    (<strong>{spp_passes}</strong> passes, <code>packable_multi=0</code>).
    Overview tables below measure the <em>legacy</em> heuristic
    (coalesce + first-fit; no refill / CP-SAT) for historical comparison.</p>
    <dl class="status-grid">
      <div>
        <dt>{data["yaml_metric_total"]} YAML metrics total</dt>
        <dd><strong>{with_pmc}</strong> with profile PMCs ·
        <strong>{no_profile_pmc}</strong> peaks/derived (no profile PMCs)</dd>
      </div>
      <div>
        <dt>{data["pmc_total"]} unique PMC counters</dt>
        <dd>In the default profile collection set</dd>
      </div>
      <div>
        <dt>{spp_passes} perfmon passes (default SPP)</dt>
        <dd>Legacy heuristic (coalesce + first-fit):
        <strong>{legacy_passes}</strong> passes</dd>
      </div>
    </dl>
    <p class="meta" style="margin-top:0.75rem;">Legacy collection contract (of <strong>{with_pmc}</strong> metrics with profile PMCs; heuristic, no refill):</p>
    <div class="collection-highlight">
      <div class="stat-card single">
        <div class="meta">Single-pass collection</div>
        <p class="big"><strong>{b["single_count"]}</strong> / {with_pmc}
        <span class="pct">({b["single_pct"]}%)</span></p>
        <p>All in-profile PMCs for the metric land in one perfmon bucket—ratio formulas are evaluated on one pass.</p>
      </div>
      <div class="stat-card multi">
        <div class="meta">Multi-pass collection</div>
        <p class="big"><strong>{multi_before}</strong> / {with_pmc}
        <span class="pct">({multi_pct}%)</span></p>
        <p>Numerators and denominators split across 2+ buckets today—cross-pass ratio error risk.</p>
      </div>
    </div>
  </section>

  <section class="doc-section target">
    <h2>2. Target</h2>
    <p>Improve <em>metric accuracy</em> by eliminating incorrect ratio evaluation caused by
    <strong>multi-pass collection</strong>. For each ratio metric, numerators and denominators
    should be collected in the <strong>same perfmon pass</strong> when hardware slot limits allow—not
    merged or capped silently at analyze time.</p>
  </section>

  <section class="doc-section solution">
    <h2>3. Solution</h2>
    <ol class="solutions">
      <li>
        <span class="solution-title">1. Single-pass packable + SLOT_LIMIT fill (Phase 1 · Default)</span>
        <p class="solution-body">Default allocate path: every packable metric gets a bucket
        containing its full PMC set (counters may duplicate across passes), then
        <code>SLOT_LIMIT</code> PMCs are filled into those passes.
        Code:
        <code>_allocate_perfmon_counter_files</code> →
        <code>try_allocate_single_pass_packable</code> →
        <code>fill_slot_limit_into_existing_passes</code>.
        Offline gfx942: <strong>{spp_passes}</strong> passes,
        <code>packable_multi=0</code>, SLOT +{spp_slot_extra} passes.
        See the <strong>Single-pass packable</strong> tabs. Opt out with
        <code>ROCPROF_COMPUTE_PERFMON_LEGACY_HEURISTIC=1</code>.</p>
      </li>
      <li>
        <span class="solution-title">2. Legacy heuristic + prioritized grouping (historical)</span>
        <p class="solution-body">Previous shipping path: metric-aware coalesce plus
        <code>profiling_counter_grouping_policy.yaml</code>
        <em>same_bucket_priority_metric_ids</em>, then per-counter first-fit.
        Restored only via
        <code>ROCPROF_COMPUTE_PERFMON_LEGACY_HEURISTIC=1</code>
        (no refill / CP-SAT).
        Overview tables below measure this path. See the
        <strong>Fewer passes (legacy)</strong> tab.</p>
      </li>
      <li>
        <span class="solution-title">3. Additive passes for the {packable_n} packable leftovers
        (estimation · superseded by SPP)</span>
        <p class="solution-body">Under the legacy
        <strong>{legacy_passes}</strong>-pass legacy layout, the
        <strong>{a["single_count"]}</strong> single-pass metrics stay
        untouched. Adding buckets for leftovers was one alternative; SPP covers
        the same set with duplication instead (total
        <strong>{spp_passes}</strong> passes). Offline additive estimate on the
        legacy leftovers: <strong>+4</strong> (total
        <strong>{legacy_passes + 4}</strong>). See Table&nbsp;3.</p>
      </li>
      <li>
        <span class="solution-title">4. <code>WEIGHTED_AVG</code> collectables (Phase 2 · Planned)</span>
        <p class="solution-body">For the {len(a["slot_limit"])}
        <code>SLOT_LIMIT</code> metrics whose full PMC set cannot fit one hardware bucket: decompose
        into single-pass sub-collectables, collect submetrics in existing passes, and recompose the
        parent at analyze time with proved <code>weight_counter</code> weights—not for the
        former {len(a["packable_multi"])} POLICY_GAP leftovers (SPP covers those).</p>
      </li>
    </ol>
  </section>

  <section class="doc-section evaluation">
    <h2>4. Evaluation</h2>

    <h3>Table 1. Legacy heuristic layout (refill removed)</h3>
    <p class="table-caption-meta">Packing layout under legacy coalesce + first-fit (refill removed). Phase&nbsp;2
    <code>WEIGHTED_AVG</code> is not folded into these counts—see Table&nbsp;2.</p>
    <div class="table-wrap">
  <table class="decompose">
    <thead>
      <tr>
        <th>Topic</th>
        <th>Heuristic + prioritized policy<br/><span class="meta">(no refill)</span></th>
        <th>Legacy (same; refill removed)</th>
        <th>Notes / comments</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td class="topic">Single-pass collection<br/><span class="meta">(all in-profile PMCs in one bucket)</span></td>
        <td><strong>{b["single_count"]}</strong> / {with_pmc} ({b["single_pct"]}%)</td>
        <td><strong>{a["single_count"]}</strong> / {with_pmc} ({a["single_pct"]}%)</td>
        <td>Denominator: {with_pmc} metrics with profile PMCs.</td>
      </tr>
      <tr>
        <td class="topic">Multi-pass collection</td>
        <td><strong>{multi_before}</strong> / {with_pmc} ({multi_pct}%)</td>
        <td><strong>{multi_after}</strong> / {with_pmc} ({multi_after_pct}%)</td>
        <td>Formula PMCs map to 2+ buckets. Includes both <code>POLICY_GAP</code> and
        <code>SLOT_LIMIT</code>.</td>
      </tr>
      <tr>
        <td class="topic">Multi-pass metric list</td>
        <td>{_metric_list_html(b["multi"])}</td>
        <td>{_metric_list_html(a["multi"])}</td>
        <td>Scroll lists. Metric id = file.panel.idx.</td>
      </tr>
      <tr class="improved">
        <td class="topic">Refill (removed)</td>
        <td colspan="2">
          Net <strong>{multi_before - multi_after}</strong> fewer multi-pass metrics
          ({multi_before} → {multi_after}); single-pass {b["single_count"]} → {a["single_count"]}.
          Refill and CP-SAT are removed from allocate. Columns now both describe
          the legacy coalesce + first-fit layout (no second-pass repair).
        </td>
        <td>Use <code>--compare-spp</code> for legacy vs default SPP.</td>
      </tr>
      <tr>
        <td class="topic">Packable multi-pass leftover<br/><span class="meta">(POLICY_GAP — fix via packing)</span></td>
        <td><strong>{len(b["packable_multi"])}</strong></td>
        <td><strong>{len(a["packable_multi"])}</strong></td>
        <td>PMC set fits one <code>perfmon_config</code> bucket but layout splits across passes.</td>
      </tr>
      <tr>
        <td class="topic">Packable leftover list ({len(a["packable_multi"])} under legacy)</td>
        <td colspan="2">{_metric_list_html(a["packable_multi"])}</td>
        <td>
          <strong>Why still split:</strong> greedy pass-minimizing coalesce; priority policy covers only
          P0 ids; greedy coalesce; TCC channel rules constrain bucket sharing.
          <em>Not</em> <code>SLOT_LIMIT</code>.
        </td>
      </tr>
    </tbody>
  </table>
    </div>

    <h3>Table 2. SLOT_LIMIT metrics</h3>
    <p class="table-caption-meta">Cannot pack into one hardware bucket—Phase&nbsp;2
    <code>WEIGHTED_AVG</code> / redesign (not a packing-method comparison).</p>
    <div class="slot-summary">
      <strong>{len(a["slot_limit"])}</strong> parent metrics ·
      <strong>{slot_dups["unique_pmc_sets"]}</strong> unique PMC sets.<br/>
      <strong>Estimation (pass count):</strong> Creating bucket-aligned sub-collectables
      for these metrics does <strong>not</strong> increase collection passes. Offline
      gfx942 check: every parent&apos;s per-bucket PMC slice already fits
      <code>perfmon_config</code> slots in the current plan—same
      <strong>{data["pmc_total"]}</strong> PMCs and
      <strong>{data["buckets_after"]}</strong> passes; no new counters, no extra
      replays. Phase&nbsp;2 only adds analyze-time <code>WEIGHTED_AVG</code> /
      redesign over those existing single-pass slices.<br/>
      {slot_dup_note}
    </div>
    <div class="table-wrap">
      {slot_table}
    </div>

    <h3>Table 3. Additive passes to cover the {packable_n} packable leftovers</h3>
    <p class="table-caption-meta">Frozen current
    <strong>{data["buckets_after"]}</strong>-pass legacy plan (the
    <strong>{a["single_count"]}</strong> single-pass metrics stay untouched).
    New buckets may <em>duplicate</em> counters already collected in the base
    plan. Covers the {packable_n} <code>POLICY_GAP</code> metrics only—not the
    {slot_n} <code>SLOT_LIMIT</code> set.</p>
    <div class="table-wrap">
      <table class="decompose">
        <thead>
          <tr>
            <th>Model</th>
            <th>Extra passes</th>
            <th>Total</th>
            <th>Notes</th>
          </tr>
        </thead>
        <tbody>
          <tr class="improved">
            <td class="topic">Additive cover of the {packable_n}<br/>
              <span class="meta">(merge unique PMC unions into new buckets only)</span></td>
            <td><strong>+4</strong></td>
            <td><strong>{data["buckets_after"] + 4}</strong></td>
            <td>Offline: 43 unique unions pack into 4 new buckets; covers all
            {packable_n}. Preferred additive estimate.</td>
          </tr>
          <tr>
            <td class="topic">Naive: +1 per unique union</td>
            <td>+43</td>
            <td>{data["buckets_after"] + 43}</td>
            <td>One new bucket per distinct PMC set (43)—upper bound if no
            merging across unions.</td>
          </tr>
          <tr>
            <td class="topic">Naive: +1 per metric</td>
            <td>+{packable_n}</td>
            <td>{data["buckets_after"] + packable_n}</td>
            <td>One new bucket per leftover metric—ignores shared PMC sets.</td>
          </tr>
          <tr>
            <td class="topic">Old “68” story (13+55)</td>
            <td>—</td>
            <td>—</td>
            <td>Wrong model / old baseline (pre-SPP 13-pass + naive additive).
            Do not use.</td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>

  <section class="doc-section faq">
    <h2>5. FAQ</h2>
    <ol class="faq-list">
      <li>
        <span class="faq-q">Why packing stayed two-goal historically
          <span class="meta">(why not one-step multi-objective search)</span></span>
        {algo_comments}
      </li>
      <li>
        <span class="faq-q">Why not <code>WEIGHTED_AVG</code> for the
          {packable_n} packable leftovers</span>
        {weighted_avg_why_not_57}
      </li>
    </ol>
  </section>

  <section class="doc-section takeaway">
    <h2>6. Summary</h2>

    <h3>Decomposed tree (gfx942 legacy heuristic)</h3>
    <div class="decomp-tree">{decomp_tree}</div>
    <p style="max-width:none;margin-top:0.65rem;">
      Shorthand:
      <code>{data["yaml_metric_total"]}</code> →
      <code>{with_pmc}</code> + <code>{no_profile_pmc}</code>;
      <code>{with_pmc}</code> →
      <code>{a["single_count"]}</code> + <code>{len(a["multi"])}</code>;
      <code>{len(a["multi"])}</code> =
      <code>{packable_n}</code> + <code>{slot_n}</code>.
      The <code>{a["single_count"]}</code> are from legacy heuristic + prioritized policy (no refill).
      Under default SPP those plus the <code>{packable_n}</code> POLICY_GAP become
      single-pass; the <code>{slot_n}</code> go to Phase&nbsp;2 <code>WEIGHTED_AVG</code>.
    </p>

    <h3>Open discussion — how to improve the {packable_n}? (legacy leftovers)</h3>
    <p style="max-width:none;">Under the legacy path these are <code>POLICY_GAP</code>: the full PMC set
    <em>fits</em> one hardware bucket, but that layout still splits them. Default SPP
    covers them (may add passes / duplicate PMCs). Not
    <code>WEIGHTED_AVG</code> candidates. Historical options:</p>
    <ul class="open-discuss">
      <li><strong>Additive passes?</strong> Freeze the current
      {data["buckets_after"]}-pass layout and add buckets for the leftovers
      (Solution&nbsp;3 / Table&nbsp;3). Offline estimate: <strong>+4</strong>
      covers all {packable_n} (total {data["buckets_after"] + 4}); naive
      +43 / +{packable_n} are upper bounds. Trade-off: higher profile cost,
      no collateral on the {a["single_count"]} already single-pass.</li>
      <li><strong>Prioritized policy?</strong> Grow
      <code>same_bucket_priority_metric_ids</code> (more P0 / P1 ids) so greedy
      coalesce keeps those formulas together—trade-off: which metrics deserve
      priority, and does a larger policy destabilize pass count?</li>
      <li><strong>Improve packing?</strong> Stronger metric-aware coalesce / SPP,
      or offline global repack within a hard pass cap—still packing, not
      decomposition.</li>
      <li><strong>Something else?</strong> e.g. accept residual multi-pass for
      low-priority panels; change objective weights (pass count vs single-pass
      metric count); arch-specific rules (TCC channels).</li>
    </ul>

    <h3>Planned to-do</h3>
    <ul class="planned-todo">
      <li>Validate SPP with real workloads on MI300.</li>
      <li>Verify and mark the impact of the proposed solution on MI300
      <strong>CPX</strong> mode (partition/sysinfo correct; P0 HBM and
      related ratios vs SPX baseline).</li>
      <li>Re-baseline inspector counts after coalesce / policy changes land
      (e.g. post-#10912) before declaring Phase&nbsp;1 done.</li>
      <li>Phase&nbsp;2: pilot at least one <code>SLOT_LIMIT</code> parent as
      sub-collectables + <code>WEIGHTED_AVG</code> on gfx942 YAML (or written
      waiver).</li>
    </ul>
  </section>

  </div><!-- /panel-overview -->

  <div id="panel-heuristic" class="tab-panel" role="tabpanel"
    aria-labelledby="tab-heuristic" hidden>

  <section class="doc-section solution">
    <h2>Fewer passes — heuristic + prioritized grouping (legacy)</h2>
    <p class="solution-body">Historical path (opt-in via
    <code>ROCPROF_COMPUTE_PERFMON_LEGACY_HEURISTIC=1</code>). When
    <code>profiling_counter_grouping_policy.yaml</code> lists
    <em>same_bucket_priority_metric_ids</em>, profiling runs
    <strong>metric-aware coalesce</strong> first (keep a metric&apos;s PMCs in one
    bucket when possible), then <strong>per-counter first-fit</strong> for whatever
    is left. Code:
    <code>_allocate_perfmon_counter_files</code> →
    <code>_metric_aware_coalesce_pass</code> → first-fit
    (under <code>LEGACY_HEURISTIC=1</code>).
    Default allocate uses Single-pass packable instead.</p>

    <div class="flow-diagram">
      <div class="flow-title">Flow — minimize passes (metric-aware coalesce + first-fit)</div>
      <pre class="mermaid" id="heuristic-flow">
flowchart TD
  A[Profile PMC set] --> B{{Priority ids in<br/>profiling_counter_grouping_policy.yaml?}}
  B -->|no| FF[For each remaining PMC:<br/>per-counter first-fit<br/>into pmc_perf buckets]
  B -->|yes| C[Metric-aware coalesce]
  C --> O[Visit metrics in order:<br/>priority ids first,<br/>then metrics with more PMCs]
  O --> L[Next metric]
  L --> T{{Can put all still-unplaced PMCs<br/>for this metric into<br/>one existing bucket?}}
  T -->|yes| P[Place them in that bucket]
  T -->|no| U{{Can open a new bucket<br/>that holds all of them?}}
  U -->|yes| V[Open new bucket and place them]
  U -->|no| W[Leave those PMCs for<br/>per-counter first-fit]
  P --> M{{More metrics?}}
  V --> M
  W --> M
  M -->|yes| L
  M -->|no| FF
  FF --> G[pmc_perf buckets<br/>~{legacy_passes} passes on gfx942 legacy]
      </pre>
    </div>

    <div class="walkthrough">
      <div class="wt-title">Walk-through (toy, 3 counters per bucket)</div>
      <div class="wt-setup">Profile PMCs: A B C D E F
Priority metric (policy):  HBM-like = {{A, B}}
Other metrics:             M1 = {{C, D, E}}
                           M2 = {{A, F}}
                           M3 = {{B, D}}
Visit order: HBM-like → M1 → M2 → M3
(priority first, then more PMCs before fewer)</div>
      <ol class="wt-steps">
        <li><strong>HBM-like = {{A,B}}</strong> — open
          <code>bucket0 = {{A,B}}</code>.</li>
        <li><strong>M1 = {{C,D,E}}</strong> — does not fit in bucket0 → open
          <code>bucket1 = {{C,D,E}}</code>.</li>
        <li><strong>M2 = {{A,F}}</strong> — A already placed; only F left →
          add F to bucket0 → <code>{{A,B,F}}</code> (single-bucket).</li>
        <li><strong>M3 = {{B,D}}</strong> — B and D already in different buckets;
          coalesce does not move them → M3 stays multi-bucket
          (packable leftover / POLICY_GAP).</li>
        <li><strong>Per-counter first-fit</strong> — nothing left; done.</li>
      </ol>
      <div class="wt-result">
        <strong>Result:</strong>
        <code>bucket0 = {{A,B,F}}</code>,
        <code>bucket1 = {{C,D,E}}</code>.
        HBM-like, M1, M2 are single-bucket; M3 is the packable multi-pass leftover.
      </div>
    </div>
  </section>

  </div><!-- /panel-heuristic -->

  <div id="panel-single-pass" class="tab-panel" role="tabpanel"
    aria-labelledby="tab-single-pass" hidden>

  <section class="doc-section solution">
    <h2>Single-pass packable (default)</h2>
    <p class="solution-body">Default allocate path in
    <code>_allocate_perfmon_counter_files</code> (no env flag required).
    Restore the legacy coalesce / first-fit path with
    <code>ROCPROF_COMPUTE_PERFMON_LEGACY_HEURISTIC=1</code>
    (or <code>ROCPROF_COMPUTE_PERFMON_SINGLE_PASS_PACKABLE=0</code>).
    <strong>Goal:</strong> every packable metric (PMC set fits one hardware bucket)
    has <em>some</em> bucket containing its full PMC set; then fill
    <code>SLOT_LIMIT</code> PMCs into those passes (open new buckets only if a PMC
    is missing); then reduce passes where merges still preserve the packable
    guarantee. Named <code>*_ACCUM</code> PMCs cost 2 SQ slots (SQ ≤ 8 on gfx942).
    <strong>Trade-off:</strong> may use <em>more</em> passes and may
    <em>duplicate</em> counters across buckets—unlike legacy, which keeps each
    PMC in at most one pass. Offline gfx942 default: legacy
    <strong>{legacy_passes}</strong> passes → default SPP
    <strong>{spp_passes}</strong> after packable
    ({"+" if spp_delta >= 0 else ""}{spp_delta}), then
    <strong>+{spp_slot_extra}</strong> for <code>SLOT_LIMIT</code> fill (all
    SLOT_LIMIT PMCs already present in the packable layout). Code:
    <code>try_allocate_single_pass_packable</code> /
    <code>fill_slot_limit_into_existing_passes</code> in
    <code>counter_grouping_single_pass.py</code>.</p>

    <div class="flow-diagram">
      <div class="flow-title">Flow — single-pass packable + SLOT_LIMIT fill (default)</div>
      <pre class="mermaid" id="single-pass-flow">
flowchart TD
  A[Profile PMC set] --> B{{LEGACY_HEURISTIC=1<br/>or SINGLE_PASS_PACKABLE=0?}}
  B -->|yes| SH[Legacy path:<br/>heuristic coalesce + first-fit<br/>]
  B -->|no default| U[Unique packable PMC unions<br/>skip SLOT_LIMIT for now]
  U --> O[Order unions:<br/>largest PMC sets first]
  O --> L[Next packable union]
  L --> H{{Some bucket already<br/>contains full union?}}
  H -->|yes| M{{More unions?}}
  H -->|no| E{{Extend an existing bucket<br/>to hold full union?}}
  E -->|yes| X[Extend that bucket]
  E -->|no| N[Open new bucket<br/>with full union<br/>may duplicate PMCs]
  X --> M
  N --> M
  M -->|yes| L
  M -->|no| FF[First-fit PMCs<br/>not in any bucket yet]
  FF --> R[Merge bucket pairs<br/>when union still fits<br/>and packable guarantee holds]
  R --> S[SLOT_LIMIT fill:<br/>for each unique SLOT_LIMIT PMC set]
  S --> S1{{Every PMC already<br/>in some bucket?}}
  S1 -->|yes| G[pmc_perf buckets<br/>gfx942: {spp_passes} total, +{spp_slot_extra} for SLOT_LIMIT]
  S1 -->|no| S2{{Fit remaining PMCs<br/>into an existing bucket?}}
  S2 -->|yes| S3[Place into existing]
  S2 -->|no| S4[Open new bucket<br/>with largest fitting subset]
  S3 --> S1
  S4 --> S1
      </pre>
    </div>

    <div class="walkthrough">
      <div class="wt-title">Walk-through (same toy as legacy tab)</div>
      <div class="wt-setup">Profile PMCs: A B C D E F · 3 counters per bucket
Packable unions:  HBM-like = {{A, B}}
                  M1 = {{C, D, E}}
                  M2 = {{A, F}}
                  M3 = {{B, D}}
Visit order: largest unions first (M1, then HBM-like, M2, M3)</div>
      <ol class="wt-steps">
        <li><strong>M1 = {{C,D,E}}</strong> — open <code>bucket0 = {{C,D,E}}</code>.</li>
        <li><strong>HBM-like = {{A,B}}</strong> — open <code>bucket1 = {{A,B}}</code>.</li>
        <li><strong>M2 = {{A,F}}</strong> — extend bucket1 → <code>{{A,B,F}}</code>.</li>
        <li><strong>M3 = {{B,D}}</strong> — B and D live in different buckets;
          cannot merge without breaking M1/M2 → open
          <code>bucket2 = {{B,D}}</code> (B duplicated in bucket1 and bucket2).</li>
        <li><strong>First-fit / merge</strong> — all packable PMCs placed.</li>
        <li><strong>SLOT_LIMIT fill</strong> — if a SLOT_LIMIT metric only needs
          PMCs already in these buckets → <strong>+0</strong> passes (gfx942 case).
          Only missing PMCs open new buckets.</li>
      </ol>
      <div class="wt-result">
        <strong>Result:</strong> 3 passes vs legacy&apos;s 2 for the toy.
        Every packable union has a bucket with its full PMC set (including M3).
        Legacy leaves M3 as POLICY_GAP to save passes.
        gfx942 offline: <strong>{spp_passes}</strong> total after packable,
        <strong>+{spp_slot_extra}</strong> additional for SLOT_LIMIT fill
        (vs legacy <strong>{legacy_passes}</strong>).
      </div>
    </div>
  </section>

  </div><!-- /panel-single-pass -->

  <div id="panel-spp-plan" class="tab-panel" role="tabpanel"
    aria-labelledby="tab-spp-plan" hidden>

  <section class="doc-section takeaway">
    <h2>Single-pass packable plan (gfx942)</h2>
    <p style="max-width:none;">Default allocator
    (opt out with <code>ROCPROF_COMPUTE_PERFMON_LEGACY_HEURISTIC=1</code>):
    guarantee a full-bucket collection for every packable metric, then fill
    <code>SLOT_LIMIT</code> PMCs into those passes. May duplicate counters across
    buckets. See the <strong>how it works</strong> tab for the flowchart.
    Full implementation plan:
    <code>docs/plans/aiprofcomp-865-single-pass-packable-plan.md</code>
    (Phase&nbsp;1 = default SPP → {spp_single} single-pass / {spp_passes} passes;
    Phase&nbsp;2 <code>WEIGHTED_AVG</code> for {slot_n} <code>SLOT_LIMIT</code>;
    multi-arch validation; remove legacy heuristic + update inspector).</p>

    <h3>Decomposed tree (gfx942 default, single-pass packable)</h3>
    <div class="decomp-tree">{spp_decomp_tree}</div>
    <p style="max-width:none;margin-top:0.65rem;">
      Shorthand:
      <code>{data["yaml_metric_total"]}</code> →
      <code>{with_pmc}</code> + <code>{no_profile_pmc}</code>;
      <code>{with_pmc}</code> →
      <code>{spp_single}</code> + <code>{slot_n}</code>.
      The <code>{spp_single}</code> absorb legacy&apos;s
      <code>{a["single_count"]}</code> single-pass plus the
      <code>{packable_n}</code> POLICY_GAP leftovers.
      The <code>{slot_n}</code> still need Phase&nbsp;2
      <code>WEIGHTED_AVG</code> for the parent formula.
    </p>

    <h3>Pass count</h3>
    <div class="table-wrap">
      <table class="decompose">
        <thead>
          <tr>
            <th>Stage</th>
            <th>Passes</th>
            <th>Additional</th>
            <th>Notes</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td class="topic">Legacy (heuristic)</td>
            <td><strong>{legacy_passes}</strong></td>
            <td>—</td>
            <td>{a["single_count"]} single-pass · {packable_n} POLICY_GAP left</td>
          </tr>
          <tr class="improved">
            <td class="topic">Default SPP packable guarantee</td>
            <td><strong>{spp_passes}</strong></td>
            <td><strong>{"+" if spp_delta >= 0 else ""}{spp_delta}</strong></td>
            <td>All {spp_single} packable metrics have a full-bucket pass</td>
          </tr>
          <tr class="improved">
            <td class="topic">+ SLOT_LIMIT fill into existing</td>
            <td><strong>{spp_passes}</strong></td>
            <td><strong>+{spp_slot_extra}</strong></td>
            <td>All SLOT_LIMIT PMCs already present; no new buckets</td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>

  </div><!-- /panel-spp-plan -->

  <p class="footnote">
    See also <code>docs/plans/aiprofcomp-865-gfx942-single-pass-impact-report.md</code> and
    <code>aiprofcomp-865-stakeholder-qa.md</code>. Rebuild: <code>python3 tools/build_aiprofcomp865_problem_decompose_html.py</code>
  </p>
  <script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
  mermaid.initialize({{
    startOnLoad: false,
    theme: "dark",
    securityLevel: "loose",
    flowchart: {{ htmlLabels: true, curve: "basis" }}
  }});
  let heuristicRendered = false;
  let singlePassRendered = false;
  async function renderHeuristicFlow() {{
    if (heuristicRendered) return;
    const node = document.getElementById("heuristic-flow");
    if (!node) return;
    await mermaid.run({{ nodes: [node] }});
    heuristicRendered = true;
  }}
  async function renderSinglePassFlow() {{
    if (singlePassRendered) return;
    const node = document.getElementById("single-pass-flow");
    if (!node) return;
    await mermaid.run({{ nodes: [node] }});
    singlePassRendered = true;
  }}
  const tabs = document.querySelectorAll(".page-tabs [role=tab]");
  const panels = {{
    overview: document.getElementById("panel-overview"),
    heuristic: document.getElementById("panel-heuristic"),
    "single-pass": document.getElementById("panel-single-pass"),
    "spp-plan": document.getElementById("panel-spp-plan")
  }};
  function activate(name) {{
    tabs.forEach(function (btn) {{
      const on = btn.getAttribute("data-tab") === name;
      btn.setAttribute("aria-selected", on ? "true" : "false");
    }});
    Object.keys(panels).forEach(function (key) {{
      const panel = panels[key];
      const on = key === name;
      panel.classList.toggle("active", on);
      if (on) panel.removeAttribute("hidden");
      else panel.setAttribute("hidden", "");
    }});
    if (history.replaceState) {{
      history.replaceState(null, "", name === "overview" ? "#" : "#" + name);
    }}
    if (name === "heuristic") renderHeuristicFlow();
    if (name === "single-pass") renderSinglePassFlow();
  }}
  tabs.forEach(function (btn) {{
    btn.addEventListener("click", function () {{
      activate(btn.getAttribute("data-tab"));
    }});
  }});
  const hash = (location.hash || "").replace(/^#/, "");
  if (hash && panels[hash]) activate(hash);
  </script>
</body>
</html>
"""
    _OUT.write_text(doc, encoding="utf-8")
    print(f"Wrote {_OUT}")


if __name__ == "__main__":
    main()
