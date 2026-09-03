#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
validate.py -- context stop/start independence between two tools.

Both Tool A (early, never stopped) and Tool B (late, stopped during phase B) trace
KERNEL_DISPATCH over the same streams. The proof of per-client context independence:

  * Tool B captured strictly fewer dispatches than Tool A (it missed the stopped window).
  * The dispatches B missed form a single CONTIGUOUS window in time (the stop/start
    window) -- i.e. B has one large internal timestamp gap.
  * Tool A captured dispatches INSIDE that window -- the work B dropped was still recorded
    by the tool that was never stopped.
"""

import sys
import pytest


def _sdk(data):
    assert "rocprofiler-sdk-json-tool" in data
    return data["rocprofiler-sdk-json-tool"]


def _dispatch_starts(data):
    recs = _sdk(data)["buffer_records"]["kernel_dispatch"]
    return sorted(r["start_timestamp"] for r in recs)


def test_structure(tool_a_data, tool_b_data):
    for data in (tool_a_data, tool_b_data):
        sdk = _sdk(data)
        assert "metadata" in sdk
        for key in ("pid", "init_time", "fini_time"):
            assert key in sdk["metadata"], f"metadata missing {key}"
        assert "buffer_records" in sdk


def test_both_captured_dispatch(tool_a_data, tool_b_data):
    assert len(_dispatch_starts(tool_a_data)) > 0, "Tool A captured no dispatches"
    assert len(_dispatch_starts(tool_b_data)) > 0, "Tool B captured no dispatches"


def test_exact_dispatch_counts(tool_a_data, tool_b_data):
    """Exact dispatch accounting proves the stop/start window is the right size.

    The driver creates NUM_STREAMS(4) streams, each seeded with one hipMemsetAsync (a memset
    is itself a kernel dispatch), then launches NUM_STREAMS*PHASE_ITERS = 20 kernels in each of
    four phases: warm-up, A, B, C.

      Tool A (early, never stopped) captures the 4 stream-init memsets + all four phases:
          4 + 4*20 = 84.
      Tool B (registers after warm-up, stopped during phase B) captures neither the memsets
          nor warm-up, and drops phase B: only phases A and C: 2*20 = 40.
    """
    per_phase = 4 * 5  # NUM_STREAMS * PHASE_ITERS
    a_expected = 4 + 4 * per_phase  # 4 stream-init memsets + all four phases = 84
    b_expected = 2 * per_phase  # phases A + C only = 40
    a = len(_dispatch_starts(tool_a_data))
    b = len(_dispatch_starts(tool_b_data))
    assert a == a_expected, f"Tool A should capture {a_expected} dispatches, got {a}"
    assert b == b_expected, f"Tool B should capture {b_expected} dispatches, got {b}"


def _largest_gap_window(starts):
    """Return (gap_start, gap_end, gap, median_gap) for the largest inter-dispatch gap."""
    gaps = [(b - a, a, b) for a, b in zip(starts, starts[1:])]
    gap, gstart, gend = max(gaps)
    ordered = sorted(g for g, _, _ in gaps)
    median = ordered[len(ordered) // 2]
    return gstart, gend, gap, median


def test_b_has_stopped_window_gap(tool_b_data):
    """Tool B's missed work is one contiguous block: it has a single large internal gap."""
    starts = _dispatch_starts(tool_b_data)
    assert len(starts) >= 2
    _, _, gap, median = _largest_gap_window(starts)
    assert (
        gap > median * 5
    ), f"expected a large gap across the stopped window; gap={gap}, median={median}"


def test_a_captured_what_b_dropped(tool_a_data, tool_b_data):
    """The tool that was never stopped (A) captured dispatches inside B's stopped window."""
    b_starts = _dispatch_starts(tool_b_data)
    assert len(b_starts) >= 2
    gap_start, gap_end, _, _ = _largest_gap_window(b_starts)

    a_in_window = [t for t in _dispatch_starts(tool_a_data) if gap_start < t < gap_end]
    assert len(a_in_window) > 0, (
        "Tool A captured nothing inside Tool B's stopped window; expected A to record the "
        f"work B dropped (window {gap_start}..{gap_end})"
    )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
