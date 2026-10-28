# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Architecture-independent unit tests for the HTML memory chart report."""

import json
import re
from html.parser import HTMLParser
from pathlib import Path

from memory_chart.html.diagram import diagram_payload
from memory_chart.html.report import MODEL_ID, build_mem_chart_document
from memory_chart.html.views import (
    ChartView,
    ChartViewSet,
    KernelEntry,
    MembwDetail,
    SlotValue,
    StallAnnotation,
    TableView,
)
from memory_chart.loader import LayoutItem
from tests.unit.memory_chart.html.common import make_block, make_layout


def make_views(kernel_name: str) -> ChartViewSet:
    """Make two visibly different selections over an in-memory layout."""
    cu = make_block(
        "cu",
        items=(LayoutItem("Wavefront Occupancy", "Occupancy", "util", False),),
    )

    cache = make_block(
        "cache",
        column=1,
        items=(LayoutItem("Cache Hit", "Hit", "hit", True),),
    )

    layout = make_layout(columns=((cu,), (cache,)))

    aggregate = ChartView(
        key="aggregate",
        slots=(SlotValue("cu.0", "Wavefront Occupancy", "7.0", 7.0, None),),
        tables=(
            TableView(301, "Memory Chart", ("Metric", "Value"), (("Hits", 72.5),)),
        ),
        membw=MembwDetail(
            "full",
            None,
            (StallAnnotation("cu", "[!] Stall", "16.7%"),),
            ("Check the cache miss rate.",),
        ),
    )

    kernel_view = ChartView(
        key="kernel-0",
        slots=(
            SlotValue("cu.0", "Wavefront Occupancy", "9.0", 9.0, None),
            SlotValue("cache.0", "Cache Hit", "91.0%", 91.0, "█"),
        ),
        tables=(TableView(301, "Memory Chart", ("Metric", "Value"), (("Hits", 91),)),),
        membw=None,
    )

    return ChartViewSet(
        arch=layout.path.stem,
        diagram=diagram_payload(
            layout,
            {"Wavefront Occupancy": "Workgroups per CU", "Cache Hit": "Percent"},
        ),
        aggregate=aggregate,
        kernels=(KernelEntry(0, kernel_name),),
        kernel_views=(kernel_view,),
        initial_kernel=0,
    )


def embedded_model(document: str) -> dict:
    """Parse the JSON script exactly as a browser would."""
    match = re.search(
        rf'<script id="{MODEL_ID}" type="application/json">(.*?)</script>',
        document,
        re.DOTALL,
    )

    assert match is not None
    return json.loads(match.group(1))


def test_document_embeds_layout_and_selection_values() -> None:
    """Check that the report embeds layout and aggregate and kernel selections."""
    views = make_views("kernel zero")
    document = build_mem_chart_document(
        views,
        heading="3. Memory Chart (Normalization: per_kernel)",
        scope_note="All kernels · GPU 0",
    )

    model = embedded_model(document)

    assert model["heading"] == "3. Memory Chart (Normalization: per_kernel)"
    assert model["scopeNote"] == "All kernels · GPU 0"
    assert model["arch"] == views.arch
    assert model["layout"] == views.diagram

    assert model["aggregate"]["slots"]["cu.0"]["text"] == "7.0"
    assert model["aggregate"]["tables"][0]["rows"] == [["Hits", 72.5]]

    assert model["aggregate"]["membw"]["annotations"] == [
        {"blockId": "cu", "label": "[!] Stall", "value": "16.7%"}
    ]

    assert model["aggregate"]["membw"]["guidanceBlocks"] == [
        "Check the cache miss rate."
    ]

    kernel_slots = model["kernels"][0]["view"]["slots"]

    assert kernel_slots["cu.0"]["text"] == "9.0"
    assert kernel_slots["cu.0"]["percent"] is None
    assert kernel_slots["cache.0"]["percent"] == 91.0

    assert model["kernels"][0]["view"]["membw"] is None
    assert model["initialKernel"] == 0


def test_document_escapes_hostile_kernel_name_and_has_offline_assets() -> None:
    """Check that hostile names are escaped and the report uses offline assets."""
    hostile_name = '</script><script src="https://example.test/x.js">alert(1)</script>'
    document = build_mem_chart_document(
        make_views(hostile_name),
        heading="Memory <Chart>",
        scope_note="All kernels",
    )

    model = embedded_model(document)

    assert model["kernels"][0]["name"] == hostile_name
    assert hostile_name not in document
    assert "</script><script src=" not in document
    assert "<title>Memory &lt;Chart&gt;" in document

    class ScriptSources(HTMLParser):
        def handle_starttag(self, tag, attrs):
            """Reject script tags that load an external source."""
            assert not (tag == "script" and "src" in dict(attrs))

    ScriptSources().feed(document)
    assert "<noscript>" in document
    assert "needs JavaScript" in document


def test_page_script_uses_only_ids_rendered_by_body() -> None:
    """Check that script element lookups match IDs rendered by the report body."""
    assets = Path(__file__).resolve().parents[4] / "src/memory_chart/html/assets"
    body = (assets / "mem_chart_body.html").read_text(encoding="utf-8")
    script = (assets / "mem_chart.js").read_text(encoding="utf-8")
    ids = set(re.findall(r'getElementById\("([^"]+)"\)', script))

    required_ids = {
        "mem-chart-heading",
        "mem-chart-scope",
        "mem-chart-diagram",
        "mem-chart-legend",
        "mem-chart-tables",
        "mem-chart-guidance",
        "mem-chart-kernel-list",
        "mem-chart-kernel-count",
        "mem-chart-show-all",
        "mem-chart-theme-toggle",
        "mem-chart-selection",
        "mem-chart-normalization",
        "mem-chart-fit-diagram",
        "mem-chart-export-png",
        "mem-chart-export-status",
    }

    assert ids
    assert all(f'id="{element_id}"' in body for element_id in required_ids)
    assert all(f'id="{element_id}"' in body for element_id in ids)

    assert 'mode: "single"' in script
    assert "HtmlReport.createKernelList" in script
    assert "HtmlReport.initTheme" in script


def test_diagram_navigation_scrolls_page_and_uses_modified_wheel_for_zoom() -> None:
    """Keep wheel zoom modified and preserve pointer-anchored zoom and fit."""
    assets = Path(__file__).resolve().parents[4] / "src/memory_chart/html/assets"
    body = (assets / "mem_chart_body.html").read_text(encoding="utf-8")
    css = (assets / "mem_chart.css").read_text(encoding="utf-8")
    script = (assets / "mem_chart.js").read_text(encoding="utf-8")

    wheel_handler = script.split('diagram.addEventListener("wheel"', 1)[1].split(
        "var drag = null;", 1
    )[0]

    assert "if (!event.ctrlKey && !event.metaKey) { return; }" in wheel_handler
    assert wheel_handler.index("event.preventDefault()") > wheel_handler.index(
        "event.metaKey"
    )

    assert "camera.x = x - (x - camera.x) * scale / camera.scale" in wheel_handler
    assert "camera.y = y - (y - camera.y) * scale / camera.scale" in wheel_handler

    assert 'addEventListener("dblclick", fitDiagram)' in script
    fit_button = (
        'getElementById("mem-chart-fit-diagram").addEventListener("click", fitDiagram)'
    )
    assert fit_button in script

    assert "Hold Ctrl or Command while scrolling to zoom" in body
    assert 'read: "Read ←", write: "Write →", atomic: "Atomic ↔"' in script

    light_palette = re.search(r":root \{(.*?)\n\}", css, re.DOTALL)
    assert light_palette is not None

    colors = re.findall(
        r"--mem-(?:read|write|atomic|util|hit|stall|bw): (#[0-9a-f]+);",
        light_palette.group(1),
    )
    assert len(colors) == 7
    assert len(set(colors)) == len(colors)


def test_metrics_disclosure_starts_collapsed() -> None:
    """Keep the metric tables behind an initially closed native disclosure."""
    document = build_mem_chart_document(
        make_views("kernel"), heading="Memory Chart", scope_note=""
    )
    disclosure = re.search(r'<details\s+id="mem-chart-metrics"[^>]*>', document)

    assert disclosure is not None
    assert "open" not in disclosure.group()
    assert "<summary>Metrics</summary>" in document
