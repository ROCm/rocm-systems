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
