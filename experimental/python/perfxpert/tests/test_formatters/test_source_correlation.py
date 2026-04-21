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

"""Tests for the source-location correlation helper.

Confluence row #5 (Source Code Line numbers): cross-reference each Tier-1
hotspot kernel with its Tier-0 detected_kernels entry so the webview can
render an expandable per-row source-location panel.
"""

from perfxpert.formatters._source_correlation import (
    _classify_severity,
    _demangle_basename,
    correlate_hotspots_with_source,
    format_source_citation_inline,
)


def test_demangle_strips_namespace_templates_and_args():
    # Basic cases the matcher must handle.
    assert _demangle_basename("foo::bar(int)") == "bar"
    assert _demangle_basename("ns::inner::kernel<float>(int, int)") == "kernel"
    assert _demangle_basename("kernel") == "kernel"
    # Lowercase for comparison.
    assert _demangle_basename("MixedCase::Kernel") == _demangle_basename("kernel")
    # Nested template depth.
    assert _demangle_basename("ns::fn<A<B, C>, D>") == "fn"


def test_correlate_hotspots_with_detected_kernels_by_basename():
    hotspots = [
        {
            "name": "foo::bar(int)",
            "percent_of_total": 40.0,
            "calls": 3,
            "total_duration": 100,
        }
    ]
    detected = [
        {
            "name": "bar",
            "file": "src/kernels.hip",
            "line": 12,
            "launch_type": "GLOBAL_KERNEL_DEF",
        }
    ]
    out = correlate_hotspots_with_source(hotspots, detected)
    assert len(out) == 1
    locs = out[0]["source_locations"]
    assert len(locs) == 1
    assert locs[0]["file"] == "src/kernels.hip"
    assert locs[0]["line"] == 12
    assert locs[0]["kind"] == "definition"
    assert locs[0]["launch_type"] == "GLOBAL_KERNEL_DEF"


def test_correlate_hotspots_no_match_returns_empty_source_locations():
    hotspots = [{"name": "compute_kernel(int)"}]
    detected = [
        {
            "name": "unrelated",
            "file": "other.hip",
            "line": 5,
            "launch_type": "GLOBAL_KERNEL_DEF",
        }
    ]
    out = correlate_hotspots_with_source(hotspots, detected)
    assert out[0]["source_locations"] == []


def test_correlate_includes_both_def_and_launch():
    hotspots = [{"name": "foo::matmul<float>(float*, float*)"}]
    detected = [
        {
            "name": "matmul",
            "file": "src/ops.hip",
            "line": 42,
            "launch_type": "GLOBAL_KERNEL_DEF",
        },
        {
            "name": "matmul",
            "file": "src/main.hip",
            "line": 88,
            "launch_type": "HIP_KERNEL_LAUNCH",
        },
    ]
    out = correlate_hotspots_with_source(hotspots, detected)
    locs = out[0]["source_locations"]
    assert len(locs) == 2
    # Definitions must sort before launches (load-bearing for UI).
    assert locs[0]["kind"] == "definition"
    assert locs[1]["kind"] == "launch"
    assert locs[0]["line"] == 42
    assert locs[1]["line"] == 88


def test_correlate_handles_missing_detected_kernels():
    hotspots = [{"name": "foo"}]
    # None detected_kernels - field should still be present (empty list).
    out = correlate_hotspots_with_source(hotspots, None)
    assert out[0]["source_locations"] == []


def test_format_source_citation_inline_round_trip():
    locs = [
        {"file": "src/a.hip", "line": 10, "kind": "definition"},
        {"file": "src/b.hip", "line": 77, "kind": "launch"},
    ]
    s = format_source_citation_inline(locs)
    assert "src/a.hip:10 (definition)" in s
    assert "src/b.hip:77 (launch)" in s


def test_format_source_citation_empty():
    assert format_source_citation_inline(None) == ""
    assert format_source_citation_inline([]) == ""


# ---------------------------------------------------------------------------
# Severity buckets (Confluence row #5 refinement)
# ---------------------------------------------------------------------------


def test_classify_severity_buckets_exact_thresholds():
    """Thresholds: >=20 HIGH, >=5 MEDIUM, >=1 LOW, else INFO."""
    high_id, high_label, high_color = _classify_severity(25.0)
    assert high_id == "HIGH"
    assert high_label == "CRITICAL"
    assert high_color == "#e84040"

    med_id, med_label, _ = _classify_severity(10.0)
    assert med_id == "MEDIUM"
    assert med_label == "HOT"

    low_id, low_label, _ = _classify_severity(2.0)
    assert low_id == "LOW"
    assert low_label == "WARM"

    info_id, info_label, _ = _classify_severity(0.5)
    assert info_id == "INFO"
    assert info_label == "COOL"


def test_classify_severity_boundary_conditions():
    """Inclusive lower bounds: 20.0 is HIGH, 5.0 is MEDIUM, 1.0 is LOW."""
    assert _classify_severity(20.0)[0] == "HIGH"
    assert _classify_severity(19.999)[0] == "MEDIUM"
    assert _classify_severity(5.0)[0] == "MEDIUM"
    assert _classify_severity(4.999)[0] == "LOW"
    assert _classify_severity(1.0)[0] == "LOW"
    assert _classify_severity(0.999)[0] == "INFO"
    assert _classify_severity(0.0)[0] == "INFO"


def test_classify_severity_handles_garbage():
    """Non-numeric / None / negative inputs fall back to INFO."""
    assert _classify_severity(None)[0] == "INFO"
    assert _classify_severity("banana")[0] == "INFO"
    assert _classify_severity(-5.0)[0] == "INFO"


def test_source_correlation_severity_buckets():
    """Feed 4 hotspots at 25/10/2/0.5 pct → 4 distinct severities threaded
    through each ``source_locations`` entry."""
    hotspots = [
        {"name": "crit_kernel", "percent_of_total": 25.0},
        {"name": "hot_kernel", "percent_of_total": 10.0},
        {"name": "warm_kernel", "percent_of_total": 2.0},
        {"name": "cool_kernel", "percent_of_total": 0.5},
    ]
    detected = [
        {"name": "crit_kernel", "file": "a.hip", "line": 10,
         "launch_type": "GLOBAL_KERNEL_DEF"},
        {"name": "hot_kernel", "file": "a.hip", "line": 20,
         "launch_type": "GLOBAL_KERNEL_DEF"},
        {"name": "warm_kernel", "file": "a.hip", "line": 30,
         "launch_type": "GLOBAL_KERNEL_DEF"},
        {"name": "cool_kernel", "file": "a.hip", "line": 40,
         "launch_type": "GLOBAL_KERNEL_DEF"},
    ]
    annotated = correlate_hotspots_with_source(hotspots, detected)
    expected = [
        ("HIGH", "CRITICAL"),
        ("MEDIUM", "HOT"),
        ("LOW", "WARM"),
        ("INFO", "COOL"),
    ]
    for row, (sev_id, sev_label) in zip(annotated, expected):
        locs = row["source_locations"]
        assert len(locs) == 1
        assert locs[0]["severity"] == sev_id, row["name"]
        assert locs[0]["severity_label"] == sev_label, row["name"]
        # Color always present.
        assert locs[0]["severity_color"].startswith("#")


def test_format_source_citation_suffix_includes_severity_label():
    """Markdown + text formatters gain a ``[CRITICAL]`` / ``[HOT]`` /
    ``[WARM]`` / ``[COOL]`` suffix after the file:line list."""
    locs = [
        {"file": "src/a.hip", "line": 10, "kind": "definition",
         "severity": "HIGH", "severity_label": "CRITICAL"},
    ]
    s = format_source_citation_inline(locs)
    assert "src/a.hip:10 (definition)" in s
    assert s.endswith("[CRITICAL]")

    cool = [
        {"file": "src/b.hip", "line": 20, "kind": "launch",
         "severity": "INFO", "severity_label": "COOL"},
    ]
    assert format_source_citation_inline(cool).endswith("[COOL]")


def test_webview_h_src_row_uses_severity_color():
    """Render the webview and grep for the CRITICAL red border on a >=20%
    kernel's h-src-row panel."""
    from perfxpert.formatters.webview import _format_as_webview

    hotspots = [
        {
            "name": "crit_kernel",
            "percent_of_total": 25.0,
            "calls": 10,
            "total_duration": 100_000,
            "avg_duration": 10_000,
            "min_duration": 5_000,
        },
        {
            "name": "cool_kernel",
            "percent_of_total": 0.5,
            "calls": 10,
            "total_duration": 100,
            "avg_duration": 10,
            "min_duration": 5,
        },
    ]
    detected_kernels = [
        {"name": "crit_kernel", "file": "src/a.hip", "line": 10,
         "launch_type": "GLOBAL_KERNEL_DEF"},
        {"name": "cool_kernel", "file": "src/b.hip", "line": 20,
         "launch_type": "GLOBAL_KERNEL_DEF"},
    ]
    html = _format_as_webview(
        time_breakdown={
            "total_runtime": 1_000_000,
            "total_kernel_time": 900_000,
            "total_memcpy_time": 50_000,
            "kernel_percent": 90.0,
            "memcpy_percent": 5.0,
            "overhead_percent": 5.0,
        },
        hotspots=hotspots,
        memory_analysis={},
        recommendations=[],
        hardware_counters=None,
        database_path="dummy.db",
        detected_kernels=detected_kernels,
    )
    # CRITICAL (>=20%) → #e84040 red border.
    assert "border-left:3px solid #e84040" in html
    # COOL (<1%) → #4d8ef2 blue border.
    assert "border-left:3px solid #4d8ef2" in html
    # Badge labels round-trip too.
    assert ">CRITICAL<" in html
    assert ">COOL<" in html
