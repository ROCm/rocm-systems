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
validate.py -- force_configure() idempotency (no double-registration).

The driver called json_tool_force_configure() TWICE with the same configure function. The
SDK de-duplicates the configure function, so the second call is a no-op registration. This
validates that the double call did NOT double-register Tool B:

  * Tool A (early, HIP-only) captured HIP records and NO kernel dispatches.
  * Tool B (late) appears EXACTLY ONCE -- a single tool object in the JSON, not two.
  * Tool B captured EXACTLY the phase-2 dispatches (not 2x), proving tool_init ran once and
    only one context was subscribed despite force_configure() being called twice.

If the second force_configure() had erroneously registered Tool B again, we would see
either a doubled dispatch count or duplicated dispatch ids.
"""

import sys
import pytest


def _sdk(data):
    assert "rocprofiler-sdk-json-tool" in data
    return data["rocprofiler-sdk-json-tool"]


def _has_structure(sdk):
    assert "metadata" in sdk
    for key in ("pid", "main_tid", "init_time", "fini_time"):
        assert key in sdk["metadata"], f"metadata missing {key}"
    assert "callback_records" in sdk
    assert "buffer_records" in sdk


def test_tool_a_structure(tool_a_data):
    _has_structure(_sdk(tool_a_data))


def test_tool_b_structure(tool_b_data):
    _has_structure(_sdk(tool_b_data))


def test_tool_b_single_object(tool_b_data):
    """The JSON contains exactly one json-tool object -- not two from a double-register."""
    assert isinstance(tool_b_data, dict)
    tools = [k for k in tool_b_data.keys() if k == "rocprofiler-sdk-json-tool"]
    assert len(tools) == 1, f"expected one json-tool object, got {tools}"


def test_tool_a_captured_hip_only(tool_a_data):
    sdk = _sdk(tool_a_data)
    assert len(sdk["buffer_records"]["hip_api_traces"]) > 0, "Tool A missing HIP records"
    assert len(sdk["buffer_records"]["kernel_dispatch"]) == 0, "Tool A leaked dispatches"


def test_tool_b_captured_dispatch_only(tool_b_data):
    sdk = _sdk(tool_b_data)
    assert len(sdk["buffer_records"]["kernel_dispatch"]) > 0, "Tool B missing dispatches"
    assert len(sdk["buffer_records"]["hip_api_traces"]) == 0, "Tool B leaked HIP records"


def test_tool_b_not_double_registered(tool_b_data, phase2_count):
    """The crux: a second force_configure() must NOT double the captured records.

    Tool B subscribed once, so it captured exactly the phase-2 dispatch count. A doubled
    count (2x phase2) would mean the second force_configure() registered a second context.
    """
    captured = len(_sdk(tool_b_data)["buffer_records"]["kernel_dispatch"])
    assert captured == phase2_count, (
        f"expected exactly {phase2_count} dispatches (single registration); got "
        f"{captured}. A doubled count means force_configure() double-registered the tool."
    )


def test_tool_b_dispatch_ids_unique(tool_b_data):
    """No duplicate dispatch ids -- a second subscription would re-emit the same dispatches."""
    ids = [
        itr["dispatch_info"]["dispatch_id"]
        for itr in _sdk(tool_b_data)["buffer_records"]["kernel_dispatch"]
    ]
    assert len(ids) == len(set(ids)), "duplicate dispatch ids: tool was registered twice"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
