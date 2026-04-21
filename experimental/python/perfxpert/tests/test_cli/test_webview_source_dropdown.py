###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""Webview -- expandable per-row source-location panel on Top Kernel Hotspots.

Confluence row #5 (Source Code Line numbers). The Tier-0 scanner already
emits ``detected_kernels`` with ``{name, file, line, launch_type}``; this
test verifies the webview renders a VTune-style chevron-toggled panel on
each hotspot row showing the definition + launch site.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from perfxpert import analyze as analyze_mod
from perfxpert import output_config
from perfxpert.connection import PerfxpertConnection as RocpdImportData


_FIXTURE_DB = (
    Path(__file__).resolve().parent.parent / "fixtures" / "memory_bound.db"
)


@pytest.fixture
def airgap(monkeypatch):
    monkeypatch.setenv("PERFXPERT_AIRGAP", "1")


@pytest.fixture
def matching_hip_src(tmp_path: Path) -> Path:
    """A HIP tree with a kernel name that matches ``memory_bound.db``.

    The fixture DB's single dispatched kernel demangles to
    ``slow_memcpy_kernel``, so we define exactly that name + a launch
    site to exercise the definition + launch correlation pair.
    """
    (tmp_path / "kernel.hip").write_text(
        "#include <hip/hip_runtime.h>\n"
        "__global__ void slow_memcpy_kernel(const float* src, float* dst, int n, int k) {\n"
        "  for (int i = 0; i < n; ++i) { dst[i] = src[i]; }\n"
        "}\n"
        "int main() {\n"
        "  hipLaunchKernelGGL(slow_memcpy_kernel, dim3(1), dim3(1), 0, 0,\n"
        "                     nullptr, nullptr, 0, 0);\n"
        "  return 0;\n"
        "}\n"
    )
    return tmp_path


def _render_webview(tmp_path: Path, source_dir: Path | None) -> str:
    out_dir = tmp_path / "wv_out"
    out_dir.mkdir(exist_ok=True)
    cfg = output_config.output_config(
        output_file="report", output_path=str(out_dir)
    )
    conn = RocpdImportData([str(_FIXTURE_DB)])
    analyze_mod._execute_agentic(
        conn,
        config=cfg,
        output_format="webview",
        source_dir=str(source_dir) if source_dir else None,
    )
    return (out_dir / "report.html").read_text()


def _inject_via_formatter(detected_kernels, hotspots=None):
    """Render the webview directly from the formatter with injected data.

    Used when we want to assert on markup without depending on the
    fixture-DB kernel name overlapping with the synthetic source tree.
    """
    from perfxpert.formatters.webview import _format_as_webview

    hotspots = hotspots or [
        {
            "name": "ns::memory_bound_kernel(float*)",
            "calls": 5,
            "total_duration": 10_000_000,
            "avg_duration": 2_000_000,
            "min_duration": 1_500_000,
            "max_duration": 2_500_000,
            "percent_of_total": 55.0,
        }
    ]
    return _format_as_webview(
        time_breakdown={
            "total_runtime": 18_000_000,
            "total_kernel_time": 10_000_000,
            "total_memcpy_time": 2_000_000,
            "kernel_percent": 55.0,
            "memcpy_percent": 11.0,
            "overhead_percent": 34.0,
        },
        hotspots=hotspots,
        memory_analysis={},
        recommendations=[],
        hardware_counters=None,
        database_path="/tmp/fake.db",
        detected_kernels=detected_kernels,
    )


def test_webview_hotspot_row_has_source_chevron(airgap, matching_hip_src, tmp_path):
    """Each hotspot row carries an h-src-toggle button and a hidden
    sibling `<tr class="h-src-row">` containing the source panel.
    """
    html = _render_webview(tmp_path, source_dir=matching_hip_src)
    # Webview invariants remain intact.
    assert html.count("<!DOCTYPE html>") == 1
    assert 'class="card"' not in html
    # Top Kernel Hotspots still renders as an scard.
    assert "<h2>Top Kernel Hotspots</h2>" in html
    # Chevron toggle button + sibling source row exist.
    assert 'class="h-src-toggle"' in html
    assert 'class="h-src-row"' in html
    # Toggle helper wired in.
    assert "toggleHSrc(" in html


def test_webview_source_panel_has_file_and_line():
    """A hotspot with a matching detected_kernel renders the source
    panel with file:line and a launch-type badge.
    """
    detected = [
        {
            "name": "memory_bound_kernel",
            "file": "src/kernels.hip",
            "line": 42,
            "launch_type": "GLOBAL_KERNEL_DEF",
        },
        {
            "name": "memory_bound_kernel",
            "file": "src/main.hip",
            "line": 99,
            "launch_type": "HIP_KERNEL_LAUNCH",
        },
    ]
    html = _inject_via_formatter(detected)
    # Source panel contents.
    assert "src/kernels.hip:42" in html
    assert "src/main.hip:99" in html
    # Launch-type badge text.
    assert "__global__" in html
    assert "HIP_KERNEL_LAUNCH" in html
    # Kind label ("Definition:" / "Launch site:") appears on the panel.
    assert "Definition" in html
    assert "Launch site" in html


def test_webview_no_match_shows_hint():
    """When the hotspot has no matching source_locations and a source
    scan WAS attempted, the panel shows a clear empty-state hint.
    """
    detected = [
        {
            "name": "totally_unrelated_kernel",
            "file": "src/other.hip",
            "line": 1,
            "launch_type": "GLOBAL_KERNEL_DEF",
        }
    ]
    html = _inject_via_formatter(detected)
    assert "No matching source location detected" in html


def test_webview_no_source_scan_shows_hint():
    """When ``--source-dir`` was never supplied (detected_kernels is
    None), the expandable panel hints at how to enable correlation.
    """
    html = _inject_via_formatter(detected_kernels=None)
    assert "--source-dir" in html
    # The chevron is still rendered so the UX is consistent across modes.
    assert 'class="h-src-toggle"' in html


def test_webview_source_panel_inside_hotspots_scard():
    """All new collapsible content MUST live INSIDE the Top Kernel
    Hotspots scard (per task constraints).
    """
    detected = [
        {
            "name": "memory_bound_kernel",
            "file": "src/k.hip",
            "line": 12,
            "launch_type": "GLOBAL_KERNEL_DEF",
        }
    ]
    html = _inject_via_formatter(detected)
    # Locate the hotspots scard and verify the panel markup sits inside.
    m = re.search(
        r"<section class=\"scard\">\s*<div class=\"shdr\">.*?<h2>Top Kernel Hotspots</h2>"
        r".*?</section>",
        html,
        flags=re.DOTALL,
    )
    assert m is not None, "could not locate Top Kernel Hotspots scard"
    scard_html = m.group(0)
    assert 'class="h-src-toggle"' in scard_html
    assert 'class="h-src-row"' in scard_html
    assert "src/k.hip:12" in scard_html
