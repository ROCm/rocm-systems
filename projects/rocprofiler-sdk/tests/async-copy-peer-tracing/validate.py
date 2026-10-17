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

import re
import sys
import pytest

# rocprofiler_agent_type_t
AGENT_TYPE_CPU = 1
AGENT_TYPE_GPU = 2

# rocprofiler_memory_copy_operation_t
MEMORY_COPY_HOST_TO_HOST = 1
MEMORY_COPY_HOST_TO_DEVICE = 2
MEMORY_COPY_DEVICE_TO_HOST = 3
MEMORY_COPY_DEVICE_TO_DEVICE = 4

expected_direction = {
    (AGENT_TYPE_CPU, AGENT_TYPE_CPU): MEMORY_COPY_HOST_TO_HOST,
    (AGENT_TYPE_CPU, AGENT_TYPE_GPU): MEMORY_COPY_HOST_TO_DEVICE,
    (AGENT_TYPE_GPU, AGENT_TYPE_CPU): MEMORY_COPY_DEVICE_TO_HOST,
    (AGENT_TYPE_GPU, AGENT_TYPE_GPU): MEMORY_COPY_DEVICE_TO_DEVICE,
}

# kernels HIP dispatches when it performs a copy with shaders instead of a DMA engine
BLIT_KERN_REGEX = re.compile(r"__amd_rocclr_.*")


def get_agent_types(sdk_data):
    return dict([(itr.id.handle, itr.type) for itr in sdk_data.agents])


def get_memory_copies(sdk_data):
    """yields (record, src agent handle, dst agent handle) for buffer and callback records"""
    for itr in sdk_data.buffer_records.memory_copies:
        yield itr, itr.src_agent_id.handle, itr.dst_agent_id.handle
    for itr in sdk_data.callback_records.memory_copies:
        yield itr, itr.payload.src_agent_id.handle, itr.payload.dst_agent_id.handle


def get_hip_api_correlation_ids(sdk_data, name):
    buffer_records = sdk_data.buffer_records
    for kind, itr in enumerate(buffer_records.names):
        if itr.kind == "HIP_RUNTIME_API":
            op = itr.operations.index(name)
            return set(
                x.correlation_id.internal
                for x in buffer_records.hip_api_traces
                if x.kind == kind and x.operation == op
            )
    return set()


def get_blit_kernels(sdk_data, correlation_ids=None):
    """names of the blit kernels dispatched, only within the given API calls if provided"""
    kernel_names = dict(
        [
            (itr.payload.kernel_id, itr.payload.kernel_name)
            for itr in sdk_data.callback_records.kernel_symbols
        ]
    )
    blit_kernels = set()
    for itr in sdk_data.buffer_records.kernel_dispatch:
        name = kernel_names.get(itr.dispatch_info.kernel_id, "")
        if not BLIT_KERN_REGEX.search(name):
            continue
        corr_id = itr.correlation_id
        if (
            correlation_ids is None
            or corr_id.internal in correlation_ids
            or corr_id.ancestor in correlation_ids
        ):
            blit_kernels.add(name)
    return sorted(blit_kernels)


def is_peer_copy(agent_types, src, dst):
    return (
        src != dst
        and agent_types[src] == AGENT_TYPE_GPU
        and agent_types[dst] == AGENT_TYPE_GPU
    )


def test_memory_copy_direction(input_data):
    sdk_data = input_data["rocprofiler-sdk-json-tool"]
    agent_types = get_agent_types(sdk_data)

    num_copies = 0
    for itr, src, dst in get_memory_copies(sdk_data):
        num_copies += 1
        expected = expected_direction[(agent_types[src], agent_types[dst])]
        assert (
            itr.operation == expected
        ), f"expected direction {expected} for a copy from agent {src} to {dst}:\n{itr}"

    assert num_copies > 0, "no memory copies were traced"


def test_peer_memory_copy(input_data):
    sdk_data = input_data["rocprofiler-sdk-json-tool"]

    # the agent list includes GPUs hidden from the application by HIP_VISIBLE_DEVICES or
    # ROCR_VISIBLE_DEVICES, so skip based on what the workload actually did
    peer_copy_ids = get_hip_api_correlation_ids(sdk_data, "hipMemcpyPeer")
    if not peer_copy_ids:
        pytest.skip(
            "hipMemcpyPeer was not called, two peer-accessible devices are required"
        )

    blit_kernels = get_blit_kernels(sdk_data, peer_copy_ids)
    assert not blit_kernels, (
        f"HIP used blit kernel(s) {blit_kernels} for the peer copy instead of SDMA, "
        "so the test no longer exercises the async-copy path"
    )

    agent_types = get_agent_types(sdk_data)
    num_peer_copies = 0
    for itr, src, dst in get_memory_copies(sdk_data):
        if not is_peer_copy(agent_types, src, dst):
            continue
        num_peer_copies += 1
        assert (
            itr.operation == MEMORY_COPY_DEVICE_TO_DEVICE
        ), f"ROCM-30981: copy between GPU agents {src} and {dst} is not D2D:\n{itr}"

    # the other copies in the workload may also use blit kernels, so these are only a hint
    assert num_peer_copies > 0, (
        "no copy between two different GPU agents was traced, blit kernels dispatched "
        f"by the workload: {get_blit_kernels(sdk_data)}"
    )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
