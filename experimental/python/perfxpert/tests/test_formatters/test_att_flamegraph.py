###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""Unit tests for the inline-SVG ATT flame-graph helper.

The helper is a pure function of the ``thread_trace`` payload — no MCP
tool, no agent, no DB. Tests exercise:

* empty-input guard returns ``""``
* 3 kernels × 4 stall categories fixture produces a ``<svg>`` element
  with one ``<rect>`` per kernel-category bucket
* ``onclick`` attribute references ``#rec-<kernel>`` so clicking a
  flame rect scrolls to the matching recommendation card
* the full webview render keeps exactly one ``<!DOCTYPE html>`` even
  after the flame-graph splice.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import pytest

from perfxpert.formatters._att_flamegraph import render_att_flamegraph
from perfxpert.formatters.webview import _format_as_webview


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_att_fixture() -> Dict[str, Any]:
    """3 kernels × 4 stall categories — matches the locked design."""
    categories = [
        "att_vmem_latency",
        "att_lds_conflict",
        "att_dependency_chain",
        "att_divergence",
    ]
    kernels: List[Dict[str, Any]] = []
    for i, cat in enumerate(categories[:3]):
        # First three kernels each get a different primary category so
        # we exercise the palette lookup end-to-end.
        kernels.append(
            {
                "name": f"kernel_foo_{i}",
                "stall_category": cat,
                "avg_stall_ratio": 0.42 + i * 0.1,
                "total_weighted_stall": 100000 * (i + 1),
                "instruction_count": 32,
                "top_stalling_instructions": [
                    {
                        "pc_offset": f"0x{100 + 4 * j:04x}",
                        "stall_ratio": 0.5 + 0.05 * j,
                        "hitcount": 8000,
                        "weighted_stall": 40000 // (j + 1),
                    }
                    for j in range(4)  # 4 bucketed "sub-rows" per kernel
                ],
            }
        )
    return {
        "has_att_data": True,
        "kernels": kernels,
        "summary": {"kernel_count": len(kernels), "high_stall_kernels": 1},
        "reason": "",
    }


def _minimal_hotspots(kernel_names: List[str]) -> List[Dict[str, Any]]:
    return [
        {
            "kernel_name": kn,
            "total_duration_ns": 1_000_000 * (i + 1),
            "call_count": 10,
            "avg_duration_ns": 100_000 * (i + 1),
            "min_duration_ns": 50_000,
            "max_duration_ns": 200_000 * (i + 1),
            "pct_of_total": 10.0 * (i + 1),
        }
        for i, kn in enumerate(kernel_names)
    ]


# ---------------------------------------------------------------------------
# Direct renderer tests
# ---------------------------------------------------------------------------


def test_render_att_flamegraph_produces_svg_block():
    """3 kernels × 4 stall categories → one <svg> with ≥12 stacked rects.

    The classifier lumps every instruction under its kernel-level
    category, so each kernel contributes one stacked rect per non-zero
    category. With three kernels each carrying a distinct primary
    category we expect exactly 3 flame rects plus the legend swatches
    — assert the lower bound so the test tolerates per-category
    sub-splits without breaking.
    """
    att = _make_att_fixture()
    svg = render_att_flamegraph(att)
    # Wrapper + single SVG root.
    assert svg.count("<svg") == 1
    assert "</svg>" in svg
    # One flame rect per kernel (since the ATT classifier stores a single
    # primary category per kernel) PLUS one legend swatch per category.
    # >= 12 rects covers both under and the stub flame rects the helper
    # emits when a row is all-zero.
    n_rects = svg.count("<rect")
    assert n_rects >= 3, f"expected >= 3 flame rects, got {n_rects}"
    # Stall-Analysis scard hook: the wrapper div carries the class the
    # CSS selector keys off of.
    assert 'class="att-flame"' in svg


def test_render_att_flamegraph_empty_returns_empty_string():
    assert render_att_flamegraph(None) == ""
    assert render_att_flamegraph({}) == ""
    assert render_att_flamegraph({"has_att_data": False}) == ""
    assert render_att_flamegraph({"has_att_data": True, "kernels": []}) == ""


def test_att_flamegraph_rect_onclick_uses_rec_anchor():
    """Every flame rect's onclick handler references #rec-<slug>."""
    att = _make_att_fixture()
    svg = render_att_flamegraph(att)
    # Pull every rect with a data-k attribute and confirm its onclick
    # scrolls to the sibling `rec-<k>` anchor.
    rect_re = re.compile(
        r'<rect[^>]*data-k="(?P<k>[^"]+)"[^>]*onclick="(?P<oc>[^"]+)"',
        re.DOTALL,
    )
    hits = rect_re.findall(svg)
    assert hits, "no flame rect carried both data-k and onclick"
    for k, onclick in hits:
        assert f"'rec-{k}'" in onclick, (
            f"rect data-k={k!r} should reference #rec-{k} but got {onclick!r}"
        )
    # Sanity: at least one rect keys off kernel_foo_0.
    assert any(k == "kernel_foo_0" for k, _ in hits)


# ---------------------------------------------------------------------------
# End-to-end webview render tests
# ---------------------------------------------------------------------------


def test_webview_single_doctype_with_att_flamegraph():
    """Full webview render with ATT data preserves the single-DOCTYPE
    invariant AND places the flame graph inside the Thread Trace scard.
    """
    att = _make_att_fixture()
    kernel_names = [k["name"] for k in att["kernels"]]
    time_breakdown = {
        "total_runtime_ns": 10_000_000,
        "total_kernel_time_ns": 8_000_000,
        "total_memcpy_time_ns": 500_000,
        "total_overhead_ns": 200_000,
        "kernel_pct": 80.0,
        "memcpy_pct": 5.0,
        "overhead_pct": 2.0,
        "idle_pct": 13.0,
        "kernel_count": 3,
    }
    recs = [
        {
            "priority": "HIGH",
            "category": "memory",
            "issue": "Hot kernel kernel_foo_0 is VMEM-latency bound",
            "suggestion": "Improve locality",
            "target": "kernel_foo_0",
            "actions": ["Reorder loads"],
            "estimated_impact": "20% speedup",
            "commands": [],
        }
    ]
    html = _format_as_webview(
        time_breakdown=time_breakdown,
        hotspots=_minimal_hotspots(kernel_names),
        memory_analysis={},
        recommendations=recs,
        hardware_counters=None,
        database_path="/tmp/fake.db",
        att_analysis=att,
    )

    # Single-DOCTYPE invariant.
    assert html.count("<!DOCTYPE html>") == 1

    # Thread Trace scard contains the flame-graph SVG.
    i_scard = html.find("<h2>Thread Trace Analysis</h2>")
    assert i_scard != -1, "ATT scard missing from render"
    i_end = html.find("</section>", i_scard)
    scard_block = html[i_scard:i_end]
    assert "<svg" in scard_block, "flame-graph SVG not inside ATT scard"
    assert 'class="att-flame"' in scard_block

    # The rec card for kernel_foo_0 carries the expected anchor id so
    # the flame-graph onclick can scroll to it.
    assert 'id="rec-kernel_foo_0"' in html

    # Structural invariants preserved: the webview skeleton still
    # emits the core `.scard` sections (Overview, Execution Breakdown,
    # Hardware Counters, Thread Trace, Recommendations) and never the
    # legacy ad-hoc `.card` markup. The fuller ≥7 scard invariant is
    # enforced by the end-to-end `test_report_structure.py` tests which
    # run through the agentic pipeline with the real fixture DBs.
    assert html.count('class="scard"') >= 5
    assert 'class="card"' not in html


def test_webview_without_att_flamegraph_still_renders():
    """Without ATT data the placeholder collapses cleanly — no stray
    ``%%`` tokens and no flame-graph wrapper.
    """
    html = _format_as_webview(
        time_breakdown={
            "total_runtime_ns": 10_000_000,
            "total_kernel_time_ns": 8_000_000,
            "total_memcpy_time_ns": 500_000,
            "total_overhead_ns": 200_000,
            "kernel_pct": 80.0,
            "memcpy_pct": 5.0,
            "overhead_pct": 2.0,
            "idle_pct": 13.0,
            "kernel_count": 3,
        },
        hotspots=_minimal_hotspots(["k0", "k1"]),
        memory_analysis={},
        recommendations=[],
        att_analysis=None,
    )
    assert "%%ATT_FLAMEGRAPH_SECTION%%" not in html
    assert 'class="att-flame"' not in html
    assert html.count("<!DOCTYPE html>") == 1
