#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Validates that hipStream_t pointer recycling does not cause stream ID collisions.

The test binary exercises multiple stream reuse patterns:
  Phase 1: 4x sequential create/use/destroy (LIFO reuse)
  Phase 2: 4x sequential create/use/destroy (recycled pointers)
  Phase 3: 4x bulk create/use/destroy, then 4x bulk re-create/use/destroy
  Phase 4: 3x paired create/use/destroy with alternating destroy order

Total: 22 kernel dispatches, each on a distinct stream.

The test binary also writes the raw hipStream_t pointer values to a file.
If HIP did not actually recycle any pointer values, the uniqueness assertion
is trivially satisfied and this test is NOT exercising the fix — a warning
is emitted in that case.
"""

import sys
import warnings
import pytest

NUM_EXPECTED_DISPATCHES = 22


def _pointers_were_recycled(pointer_data):
    """Return True if any raw pointer value appears more than once."""
    return len(pointer_data) != len(set(pointer_data))


def test_stream_ids_are_unique_after_reuse(json_data, pointer_data):
    """
    Every kernel dispatch must be attributed to a unique stream ID.
    Before the fix, recycled hipStream_t pointers would collide with stale
    map entries and dispatches would be misattributed to old stream lanes.
    """
    data = json_data["rocprofiler-sdk-tool"]
    buffer_records = data["buffer_records"]

    kernel_dispatch_data = buffer_records["kernel_dispatch"]
    assert len(kernel_dispatch_data) > 0, "Expected kernel dispatch records"

    assert len(kernel_dispatch_data) == NUM_EXPECTED_DISPATCHES, (
        f"Expected {NUM_EXPECTED_DISPATCHES} kernel dispatches, "
        f"got {len(kernel_dispatch_data)}"
    )

    stream_ids = []
    for node in kernel_dispatch_data:
        assert "stream_id" in node
        stream_id = node.stream_id.handle
        assert stream_id != 0, "Kernel dispatch should not be on the null stream"
        stream_ids.append(stream_id)

    unique_stream_ids = set(stream_ids)

    if not _pointers_were_recycled(pointer_data):
        warnings.warn(
            "HIP did not recycle any hipStream_t pointer values in this run. "
            "This test is NOT exercising the stream-reuse fix. "
            "The uniqueness assertion passes trivially.",
            stacklevel=1,
        )

    assert len(unique_stream_ids) == NUM_EXPECTED_DISPATCHES, (
        f"Expected {NUM_EXPECTED_DISPATCHES} unique stream IDs but got "
        f"{len(unique_stream_ids)}: {sorted(unique_stream_ids)}. "
        f"Stream ID collision detected (pointer recycling bug)."
    )


def test_stream_ids_monotonically_increase(json_data, pointer_data):
    """
    Stream IDs should be assigned in increasing order since streams are created
    sequentially in this test (no concurrent creation).
    """
    data = json_data["rocprofiler-sdk-tool"]
    buffer_records = data["buffer_records"]

    kernel_dispatch_data = buffer_records["kernel_dispatch"]
    assert len(kernel_dispatch_data) > 0

    stream_ids = [node.stream_id.handle for node in kernel_dispatch_data]

    if not _pointers_were_recycled(pointer_data):
        warnings.warn(
            "HIP did not recycle any hipStream_t pointer values in this run. "
            "Monotonicity check passes trivially without recycling.",
            stacklevel=1,
        )

    for i in range(1, len(stream_ids)):
        assert stream_ids[i] > stream_ids[i - 1], (
            f"Stream IDs should be monotonically increasing but "
            f"stream_id[{i}]={stream_ids[i]} <= stream_id[{i-1}]={stream_ids[i-1]}"
        )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
