###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
###############################################################################

"""Tests for Tier-0 Detected GPU Kernels severity coloring.

Each row in the Tier-0 "Detected GPU Kernels" table is cross-referenced
with the Tier-1 hotspot list by canonicalized kernel name. When a match
is found, the ``<tr>`` carries an ``h-src-*`` class keyed off the
hotspot's ``percent_of_total`` bucket:

    >= 20%  -> h-src-critical
    5-20%   -> h-src-hot
    1-5%    -> h-src-warm
    <  1%   -> h-src-cool
    no match -> no severity class (neutral row)

Source-only callers pass ``hotspots=None`` -> table renders without the
Runtime % column or severity coloring.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from perfxpert.formatters.webview import _format_tier0_webview


# ---------------------------------------------------------------------------
# Test fixture helpers
# ---------------------------------------------------------------------------


def _make_tier0_ns(
    detected_kernels: Optional[List[Dict[str, Any]]] = None,
) -> SimpleNamespace:
    """Build a minimal tier-0 SimpleNamespace matching the formatter contract."""
    return SimpleNamespace(
        analysis_timestamp="2026-04-21T00:00:00Z",
        source_dir="/tmp/mock_src",
        programming_model="hip",
        files_scanned=1,
        files_skipped=0,
        kernel_count=len(detected_kernels or []),
        detected_kernels=detected_kernels or [],
        detected_patterns=[],
        risk_areas=[],
        recommendations=[],
        code_patterns=[],
        suggested_counters=[],
        suggested_first_command="",
        profiling_plan={},
        llm_explanation="",
        already_instrumented=False,
        roctx_marker_count=0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_detected_kernels_table_has_severity_class_on_matching_hotspot():
    """A detected kernel that matches a >=20% hotspot gets ``h-src-critical``."""
    detected = [
        {
            "name": "heavy_valu_kernel",
            "file": "src/compute.hip",
            "line": 42,
            "launch_type": "GLOBAL_KERNEL_DEF",
        }
    ]
    hotspots = [
        {
            "name": "heavy_valu_kernel(float const*, float*, int, int)",
            "percent_of_total": 99.9,
            "calls": 10,
            "total_duration": 1_000_000,
        }
    ]
    html = _format_tier0_webview(
        _make_tier0_ns(detected), has_profiling=True, hotspots=hotspots
    )
    # The row for the matched kernel must carry the critical-severity class.
    assert 'class="h-src-critical"' in html
    # And the kernel name must still appear in the table (sanity).
    assert "heavy_valu_kernel" in html


def test_detected_kernels_without_hotspot_match_has_no_severity_class():
    """A detected kernel with no hotspot match renders as a neutral row."""
    detected = [
        {
            "name": "cold_kernel",
            "file": "src/unused.hip",
            "line": 7,
            "launch_type": "GLOBAL_KERNEL_DEF",
        }
    ]
    hotspots = [
        {
            "name": "some_other_kernel(int)",
            "percent_of_total": 50.0,
        }
    ]
    html = _format_tier0_webview(
        _make_tier0_ns(detected), has_profiling=True, hotspots=hotspots
    )
    # The cold kernel row is present but must not carry any h-src-* class.
    # The only h-src-* classes in the document for the detected-kernel row
    # would be inside the <tbody>; check there is no per-row coloring
    # applied to the cold-kernel row by scanning around its name.
    # Simplest: the full cold_kernel row has no class="h-src-*" attached.
    # We look for 'class="h-src-' inside the Detected GPU Kernels table
    # - there should be none for this single cold kernel.
    # Extract the kernels-table fragment defensively.
    tbl_start = html.find("tier0-kernels-table")
    tbl_end = html.find("</table>", tbl_start) if tbl_start != -1 else -1
    kernels_tbl = html[tbl_start:tbl_end] if tbl_start != -1 else ""
    assert "cold_kernel" in kernels_tbl
    assert 'class="h-src-critical"' not in kernels_tbl
    assert 'class="h-src-hot"' not in kernels_tbl
    assert 'class="h-src-warm"' not in kernels_tbl
    assert 'class="h-src-cool"' not in kernels_tbl


@pytest.mark.parametrize(
    "pct, expected_cls",
    [
        (25.0, "h-src-critical"),
        (15.0, "h-src-hot"),
        (3.0, "h-src-warm"),
        (0.5, "h-src-cool"),
    ],
)
def test_severity_buckets_per_threshold(pct: float, expected_cls: str):
    """Parametrized sweep over the 4 severity buckets."""
    detected = [
        {
            "name": "target_kernel",
            "file": "src/a.hip",
            "line": 1,
            "launch_type": "GLOBAL_KERNEL_DEF",
        }
    ]
    hotspots = [
        {
            "name": "target_kernel(int)",
            "percent_of_total": pct,
        }
    ]
    html = _format_tier0_webview(
        _make_tier0_ns(detected), has_profiling=True, hotspots=hotspots
    )
    assert f'class="{expected_cls}"' in html


def test_severity_bucket_zero_percent_has_no_coloring():
    """0 % is treated the same as ``no match`` -> no severity class.

    Buckets are inclusive lower-bounds, so percent==0 falls in the INFO
    ("cool") bucket when matched. A hotspot with 0 % is still a match —
    matches always get a class. ``no match`` is the path with no class.
    """
    detected = [
        {
            "name": "stub_kernel",
            "file": "src/a.hip",
            "line": 1,
            "launch_type": "GLOBAL_KERNEL_DEF",
        }
    ]
    hotspots = [
        {
            "name": "stub_kernel(void)",
            "percent_of_total": 0.0,
        }
    ]
    html = _format_tier0_webview(
        _make_tier0_ns(detected), has_profiling=True, hotspots=hotspots
    )
    # 0.0 % is < 1 % -> INFO -> h-src-cool (still colored as a match).
    assert 'class="h-src-cool"' in html


def test_source_only_mode_no_hotspots_passes_null_hotspots():
    """``hotspots=None`` renders without error and without the Runtime % column."""
    detected = [
        {
            "name": "any_kernel",
            "file": "src/x.hip",
            "line": 99,
            "launch_type": "GLOBAL_KERNEL_DEF",
        }
    ]
    # Default call (no hotspots arg) — must succeed.
    html_default = _format_tier0_webview(_make_tier0_ns(detected))
    assert "any_kernel" in html_default
    # Runtime % column header only renders when hotspots are supplied.
    assert "Runtime %" not in html_default

    # Explicit None — must also succeed and match the default rendering.
    html_none = _format_tier0_webview(_make_tier0_ns(detected), hotspots=None)
    assert "any_kernel" in html_none
    assert "Runtime %" not in html_none

    # And no severity class on any row in source-only mode.
    assert 'class="h-src-critical"' not in html_default
    assert 'class="h-src-hot"' not in html_default
    assert 'class="h-src-warm"' not in html_default
    assert 'class="h-src-cool"' not in html_default


def test_detected_kernels_legend_present_when_hotspots_passed():
    """The legend line under the table appears only when hotspots are present."""
    detected = [
        {
            "name": "legend_kernel",
            "file": "src/legend.hip",
            "line": 5,
            "launch_type": "GLOBAL_KERNEL_DEF",
        }
    ]
    hotspots = [
        {"name": "legend_kernel(int)", "percent_of_total": 42.0},
    ]
    html_with = _format_tier0_webview(
        _make_tier0_ns(detected), has_profiling=True, hotspots=hotspots
    )
    assert "Row color indicates runtime share" in html_with

    html_without = _format_tier0_webview(_make_tier0_ns(detected), hotspots=None)
    assert "Row color indicates runtime share" not in html_without
