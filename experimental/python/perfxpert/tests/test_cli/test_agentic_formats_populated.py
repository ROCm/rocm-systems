###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""Regression-guard: agentic path populates EVERY section in every format.

Before this suite landed, `_execute_agentic` only threaded
``narrative + recommendations + primary_bottleneck + warnings + metadata``
into `_format_agentic_output`; the legacy deterministic dataset
(``time_breakdown``, ``hotspots``, ``memory_analysis``, ``hardware_counters``,
Tier-0 findings, …) was dropped so every format produced a
structurally-thin report.

These tests exercise the fix end-to-end using the airgap path so no LLM
credentials are required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from perfxpert import analyze as analyze_mod
from perfxpert import output_config
from perfxpert.connection import PerfxpertConnection as RocpdImportData


_FIXTURE_DB = (
    Path(__file__).resolve().parent.parent / "fixtures" / "compute_bound.db"
)


@pytest.fixture
def airgap(monkeypatch):
    """Force the agentic pipeline to take the deterministic airgap branch."""
    monkeypatch.setenv("PERFXPERT_AIRGAP", "1")


@pytest.fixture
def tiny_hip_src(tmp_path: Path) -> Path:
    """Create a minimal HIP source tree the Tier-0 scanner can detect."""
    (tmp_path / "kernel.hip").write_text(
        "#include <hip/hip_runtime.h>\n"
        "__global__ void add(float* a, float* b) { *a += *b; }\n"
        "int main() {\n"
        "  hipLaunchKernelGGL(add, dim3(1), dim3(1), 0, 0, nullptr, nullptr);\n"
        "  hipDeviceSynchronize();\n"
        "  return 0;\n"
        "}\n"
    )
    return tmp_path


def _run_airgap(
    *, input_db: Path | None, out_dir: Path, fmt: str, source_dir: Path | None = None
) -> Path:
    """Invoke ``_execute_agentic`` in airgap and return the written output file."""
    cfg = output_config.output_config(
        output_file="report", output_path=str(out_dir)
    )
    conn = RocpdImportData([str(input_db)]) if input_db else None
    analyze_mod._execute_agentic(
        conn,
        config=cfg,
        output_format=fmt,
        source_dir=str(source_dir) if source_dir else None,
    )
    ext = {"json": ".json", "markdown": ".md", "webview": ".html", "text": ".txt"}[fmt]
    return out_dir / f"report{ext}"


def test_json_format_populates_all_sections(airgap, tmp_path):
    """Airgap + real fixture DB → JSON must include every deterministic section
    plus the agent-brain keys. Asserts both presence and non-empty-where-DB-has-data.
    """
    out_dir = tmp_path / "out"
    written = _run_airgap(input_db=_FIXTURE_DB, out_dir=out_dir, fmt="json")
    assert written.is_file()
    doc = json.loads(written.read_text())

    # Deterministic sections — every key must be present.
    for key in (
        "time_breakdown",
        "hotspots",
        "memory_analysis",
        "hardware_counters",
        "recommendations",
    ):
        assert key in doc, f"missing key {key!r}: {list(doc.keys())}"

    # Agent-brain keys.
    for key in ("narrative", "primary_bottleneck", "warnings"):
        assert key in doc, f"missing agent key {key!r}"

    # Non-empty where DB has data.
    assert doc["time_breakdown"], "time_breakdown should be populated by the fixture"
    assert doc["time_breakdown"].get("total_runtime", 0) > 0
    assert len(doc["hotspots"]) >= 1, "fixture has kernels; hotspots should not be empty"
    assert doc["narrative"], "airgap narrative must not be empty"
    assert len(doc["recommendations"]) >= 1, "at least one deterministic rec expected"


def test_webview_renders_hotspot_table(airgap, tmp_path):
    """Airgap webview must render a hotspot `<table>` with at least one data row."""
    out_dir = tmp_path / "out"
    written = _run_airgap(input_db=_FIXTURE_DB, out_dir=out_dir, fmt="webview")
    html = written.read_text()
    assert "<!doctype" in html.lower() or "<html" in html.lower()
    # The deterministic hotspots table carries id="hs-tbl" in the legacy webview.
    assert 'id="hs-tbl"' in html, "expected hotspots table with id='hs-tbl' in HTML"
    # At least one data row beyond the header (either a <tbody><tr> pair).
    # Quick heuristic: the fixture has at least 1 kernel so the rendered HTML
    # should contain a '<tr>' that follows the '<tbody>' marker.
    body_idx = html.find("<tbody>")
    assert body_idx != -1, "webview missing <tbody> in hotspots table"
    assert html.find("<tr", body_idx) != -1, "webview hotspots <tbody> has no rows"


def test_markdown_has_tier0_section_when_source_dir_set(
    airgap, tiny_hip_src, tmp_path
):
    """Airgap + `--source-dir` must splice a Tier 0 section into the Markdown."""
    out_dir = tmp_path / "out"
    written = _run_airgap(
        input_db=_FIXTURE_DB, out_dir=out_dir, fmt="markdown", source_dir=tiny_hip_src
    )
    md = written.read_text()
    assert (
        "Tier 0 — Source Scan" in md or "Tier 0 \u2014 Source Scan" in md
    ), f"Markdown missing Tier 0 heading; got:\n{md[:400]}"


def test_recommendation_merge_dedupes_by_target():
    """One LLM rec + one deterministic rec for the same target → merged list has one entry."""
    from perfxpert.analysis.payload import merge_recommendations

    llm = [{"type": "compute", "target": "kernelA", "summary": "Use tensor cores"}]
    det = [
        {
            "type": "compute",
            "target": "kernelA",
            "category": "Compute",
            "issue": "Underutilised VALU",
            "suggestion": "Enable MFMA path",
            "citation": "heavy_kernel.cpp:42",
        }
    ]
    merged = merge_recommendations(llm, det)
    assert len(merged) == 1, f"expected 1 merged entry, got {len(merged)}: {merged}"
    # LLM summary wins…
    assert merged[0].get("summary") == "Use tensor cores"
    # …but deterministic citation is preserved.
    assert merged[0].get("citation") == "heavy_kernel.cpp:42"


def test_tier0_only_path_still_renders_full_scaffold(airgap, tiny_hip_src, tmp_path):
    """Airgap + `--source-dir` only (no ``-i``): every format writes a file
    with a Tier 0 section (`tier0_findings` present in JSON).
    """
    for fmt, ext in (("text", ".txt"), ("json", ".json"), ("markdown", ".md"), ("webview", ".html")):
        out_dir = tmp_path / f"out_{fmt}"
        written = _run_airgap(
            input_db=None, out_dir=out_dir, fmt=fmt, source_dir=tiny_hip_src
        )
        assert written.is_file(), f"{fmt}: expected output file"
        body = written.read_text()
        assert body, f"{fmt}: body is empty"
        if fmt == "json":
            doc = json.loads(body)
            assert "tier0_findings" in doc or "tier0" in doc, (
                f"json missing tier0 section: keys={list(doc.keys())}"
            )
