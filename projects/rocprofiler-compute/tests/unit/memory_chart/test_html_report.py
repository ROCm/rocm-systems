# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for the self-contained HTML memory chart report."""

import hashlib
import json
import logging
import re
from argparse import Namespace
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from memory_chart import loader
from memory_chart.html import report
from memory_chart.html.diagram import diagram_payload
from memory_chart.html.report import (
    MODEL_ID,
    build_mem_chart_document,
    mem_chart_html_path,
    write_mem_chart_report,
)
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
from utils import schema


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


@pytest.mark.parametrize(
    ("gpu_ids", "dispatch_ids", "filename"),
    [
        ([], [], "mem_chart.html"),
        ("0", [], "mem_chart_gpu-0.html"),
        (0, [], "mem_chart_gpu-0.html"),
        (["1", "0", "1"], [], "mem_chart_gpu-0_1.html"),
        ([], ["> 5"], "mem_chart_d-gt5.html"),
        (["2", "0"], ["10", "3", "3"], "mem_chart_gpu-0_2_d-3_10.html"),
    ],
)
def test_mem_chart_html_path_normalizes_filters(
    tmp_path: Path, gpu_ids: object, dispatch_ids: object, filename: str
) -> None:
    assert mem_chart_html_path(tmp_path, gpu_ids, dispatch_ids) == tmp_path / filename


def test_mem_chart_html_path_hashes_long_dispatch_group(tmp_path: Path) -> None:
    ids = [str(index) for index in range(300)]
    path = mem_chart_html_path(tmp_path, ["0"], ids)
    joined = "_".join(ids)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:8]

    assert path.name == f"mem_chart_gpu-0_d-300ids-{digest}.html"
    assert len(path.name.encode("utf-8")) < 255
    assert path == mem_chart_html_path(tmp_path, ["0"], list(reversed(ids)))


def test_write_mem_chart_report_gates_and_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workload = schema.Workload(sys_info=pd.DataFrame([{"gpu_arch": "gfx950"}]))
    args = Namespace(normal_unit="per_kernel")
    build = Mock(return_value=make_views("gfx950", "kernel zero"))
    monkeypatch.setattr(report, "build_chart_views", build)

    assert write_mem_chart_report(workload, schema.ArchConfig(), args, tmp_path) is None
    assert build.call_count == 0
    assert not list(tmp_path.glob("mem_chart*.html"))

    workload.dfs[301] = pd.DataFrame()
    monkeypatch.setattr(loader, "has_layout", lambda _arch: False)
    caplog.set_level(logging.INFO)
    assert write_mem_chart_report(workload, schema.ArchConfig(), args, tmp_path) is None
    assert "no layout for gfx950" in caplog.text
    info_records = [
        record for record in caplog.records if record.levelno == logging.INFO
    ]
    assert len(info_records) == 1
    assert build.call_count == 0

    monkeypatch.setattr(loader, "has_layout", lambda _arch: True)
    path = write_mem_chart_report(workload, schema.ArchConfig(), args, tmp_path)
    assert path == tmp_path / "mem_chart.html"
    model = embedded_model(path.read_text(encoding="utf-8"))
    assert model["scopeNote"] == ""
    assert model["heading"] == "3. Memory Chart"
    assert model["normalization"] == "per_kernel"
    path.write_text("stale", encoding="utf-8")
    assert write_mem_chart_report(workload, schema.ArchConfig(), args, tmp_path) == path
    assert path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")

    workload.filter_gpu_ids = ["1", "0"]
    workload.filter_dispatch_ids = ["> 5"]
    filtered_path = write_mem_chart_report(
        workload, schema.ArchConfig(), args, tmp_path
    )
    assert filtered_path == tmp_path / "mem_chart_gpu-0_1_d-gt5.html"
    assert embedded_model(filtered_path.read_text(encoding="utf-8"))["scopeNote"] == (
        "GPU 0, 1 · dispatches > 5"
    )


def test_write_mem_chart_report_logs_write_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workload = schema.Workload(
        sys_info=pd.DataFrame([{"gpu_arch": "gfx950"}]),
        dfs={301: pd.DataFrame()},
    )
    monkeypatch.setattr(
        report, "build_chart_views", lambda *_: make_views("gfx950", "k")
    )

    def fail_write(_path: Path, _text: str, encoding: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", fail_write)
    assert (
        write_mem_chart_report(
            workload,
            schema.ArchConfig(),
            Namespace(normal_unit="per_kernel"),
            tmp_path,
        )
        is None
    )
    assert str(tmp_path / "mem_chart.html") in caplog.text
    assert "disk full" in caplog.text
    assert not list(tmp_path.glob("mem_chart*.html"))


def test_write_mem_chart_report_propagates_evaluation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workload = schema.Workload(
        sys_info=pd.DataFrame([{"gpu_arch": "gfx950"}]),
        dfs={301: pd.DataFrame()},
    )

    def fail_evaluation(*_args: object) -> ChartViewSet:
        raise OSError("evaluation failed")

    monkeypatch.setattr(report, "build_chart_views", fail_evaluation)
    with pytest.raises(OSError, match="evaluation failed"):
        write_mem_chart_report(
            workload,
            schema.ArchConfig(),
            Namespace(normal_unit="per_kernel"),
            tmp_path,
        )
