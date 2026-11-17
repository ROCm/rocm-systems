#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

import sys

import pytest


def test_every_dispatch_traced(kernel_trace, min_dispatches):
    """The execute step completing already proves hipStreamCreate did not deadlock. This checks
    the queues HIP created through hsa_amd_queue_create were replaced by working InterceptQueues:
    every kernel of both phases shows up in the kernel trace."""
    names = [row["Kernel_Name"] for row in kernel_trace]
    assert (
        len(names) >= min_dispatches
    ), f"expected at least {min_dispatches} dispatch records, found {len(names)}"
    assert any(
        "spin_kernel" in name for name in names
    ), "poll phase kernels were not traced"
    assert any(
        "tiny_kernel" in name for name in names
    ), "burst phase kernels were not traced"


def test_dispatches_span_many_queues(kernel_trace):
    """Every stream got its own hardware queue (GPU_MAX_HW_QUEUES), so the dispatches must be
    spread over many queue ids; a single id would mean the churn never created new queues and the
    deadlock path was not exercised."""
    queues = {row["Queue_Id"] for row in kernel_trace}
    assert len(queues) > 1, f"all dispatches landed on a single queue: {queues}"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
