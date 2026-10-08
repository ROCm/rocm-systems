# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for the self-contained HTML memory chart report."""

import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from memory_chart import loader
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
from memory_chart.units import panel_units
from tests.unit.memory_chart.layout_cases import LAYOUT_ARCHS, panel_config


def make_views(arch: str, kernel_name: str) -> ChartViewSet:
    """Make two visibly different selections over a shipped layout."""
    layout = loader.Layouts.for_arch(arch)
    percent_slot_id = "lds.0" if arch == "gfx1250" else "vl1d.0"
    percent_metric = "LDS Utilization" if arch == "gfx1250" else "VL1 Hit"
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
            SlotValue(percent_slot_id, percent_metric, "91.0%", 91.0, "█"),
        ),
        tables=(TableView(301, "Memory Chart", ("Metric", "Value"), (("Hits", 91),)),),
        membw=None,
    )
    return ChartViewSet(
        arch=arch,
        diagram=diagram_payload(layout, panel_units(panel_config(arch))),
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


@pytest.mark.parametrize("arch", sorted({arch for _, arch in LAYOUT_ARCHS}))
def test_has_layout_accepts_every_loader_architecture(arch):
    assert loader.has_layout(arch)


@pytest.mark.parametrize("arch", ["gfx1201", "gfx1030"])
def test_has_layout_rejects_unsupported_architectures(arch):
    assert not loader.has_layout(arch)


def test_has_layout_preserves_loader_validation_errors(monkeypatch, tmp_path):
    path = tmp_path / "broken.json"
    path.write_text(json.dumps({"archs": ["gfx908"]}), encoding="utf-8")
    monkeypatch.setattr(loader, "layout_files", lambda: [path])
    monkeypatch.setattr(loader.Layouts, "_by_arch", None)
    with pytest.raises(SystemExit):
        loader.has_layout("gfx908")


@pytest.mark.parametrize("arch", ["gfx1150", "gfx1151", "gfx1152", "gfx1153"])
def test_has_layout_reuses_loader_architecture_mapping(arch):
    assert loader.has_layout(arch) == (loader.Layouts.for_arch(arch) is not None)


@pytest.mark.parametrize("arch", ["gfx1250", "gfx950"])
def test_document_embeds_layout_and_selection_values(arch: str) -> None:
    views = make_views(arch, "kernel zero")
    document = build_mem_chart_document(
        views,
        heading="3. Memory Chart (Normalization: per_kernel)",
        scope_note="All kernels · GPU 0",
    )
    model = embedded_model(document)

    assert model["heading"] == "3. Memory Chart (Normalization: per_kernel)"
    assert model["scopeNote"] == "All kernels · GPU 0"
    assert model["arch"] == arch
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
    percent_slot_id = "lds.0" if arch == "gfx1250" else "vl1d.0"
    assert kernel_slots[percent_slot_id]["percent"] == 91.0
    assert model["kernels"][0]["view"]["membw"] is None
    assert model["initialKernel"] == 0
    if arch == "gfx1250":
        assert [
            child["id"] for child in model["layout"]["gridBlocks"][1]["children"]
        ] == ["lds", "gl0"]
    else:
        assert [block["position"] for block in model["layout"]["ioBlocks"]] == [
            "above",
            "below",
        ]


def test_document_escapes_hostile_kernel_name_and_has_offline_assets() -> None:
    hostile_name = '</script><script src="https://example.test/x.js">alert(1)</script>'
    arch = "gfx950"
    document = build_mem_chart_document(
        make_views(arch, hostile_name),
        heading="Memory <Chart>",
        scope_note="All kernels",
    )
    model = embedded_model(document)

    assert model["kernels"][0]["name"] == hostile_name
    assert hostile_name not in document
    assert "</script><script src=" not in document
    assert "<title>Memory &lt;Chart&gt;" in document
    assert "Plotly" not in document

    class ScriptSources(HTMLParser):
        def handle_starttag(self, tag, attrs):
            assert not (tag == "script" and "src" in dict(attrs))

    ScriptSources().feed(document)
    assert "<noscript>" in document
    assert "needs JavaScript" in document


def test_page_script_uses_only_ids_rendered_by_body() -> None:
    assets = Path(__file__).resolve().parents[3] / "src/memory_chart/html/assets"
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
    assert "innerHTML" not in script


def test_metrics_disclosure_starts_collapsed() -> None:
    """Keep the metric tables behind an initially closed native disclosure."""
    document = build_mem_chart_document(
        make_views("gfx950", "kernel"), heading="Memory Chart", scope_note=""
    )
    disclosure = re.search(r'<details\s+id="mem-chart-metrics"[^>]*>', document)
    assert disclosure is not None
    assert "open" not in disclosure.group()
    assert "<summary>Metrics</summary>" in document
