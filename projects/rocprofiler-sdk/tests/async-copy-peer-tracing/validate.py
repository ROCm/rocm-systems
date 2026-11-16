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


def get_api_records(buffer_records, kind_name, operation_name, trace_name):
    for kind, itr in enumerate(buffer_records.names):
        if itr.kind != kind_name or operation_name not in itr.operations:
            continue

        operation = itr.operations.index(operation_name)
        return [
            x
            for x in buffer_records[trace_name]
            if x.kind == kind and x.operation == operation
        ]
    return []


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
    buffer_records = sdk_data.buffer_records

    # the agent list includes GPUs hidden from the application by HIP_VISIBLE_DEVICES or
    # ROCR_VISIBLE_DEVICES, so skip based on what the workload actually did
    peer_api_records = get_api_records(
        buffer_records, "HIP_RUNTIME_API", "hipMemcpyPeer", "hip_api_traces"
    )
    if not peer_api_records:
        pytest.skip(
            "hipMemcpyPeer was not called, two devices with bidirectional peer access are required"
        )

    peer_api_records.sort(key=lambda x: x.start_timestamp)
    assert len(peer_api_records) == 2, (
        "expected one A->B hipMemcpyPeer call with the destination device current "
        "and one with the source device current"
    )
    peer_copy_ids = {x.correlation_id.internal for x in peer_api_records}

    blit_kernels = get_blit_kernels(sdk_data, peer_copy_ids)
    assert not blit_kernels, (
        f"HIP used blit kernel(s) {blit_kernels} for a peer copy instead of SDMA, "
        "so the test no longer exercises the async-copy path"
    )

    agent_types = get_agent_types(sdk_data)
    h2d_destinations = {
        x.dst_agent_id.handle
        for x in buffer_records.memory_copies
        if x.operation == MEMORY_COPY_HOST_TO_DEVICE
        and agent_types[x.src_agent_id.handle] == AGENT_TYPE_CPU
        and agent_types[x.dst_agent_id.handle] == AGENT_TYPE_GPU
    }
    d2h_sources = {
        x.src_agent_id.handle
        for x in buffer_records.memory_copies
        if x.operation == MEMORY_COPY_DEVICE_TO_HOST
        and agent_types[x.src_agent_id.handle] == AGENT_TYPE_GPU
        and agent_types[x.dst_agent_id.handle] == AGENT_TYPE_CPU
    }
    assert (
        len(h2d_destinations) == 1
    ), f"expected H2D records to identify A on one GPU, got {h2d_destinations}"
    assert (
        len(d2h_sources) == 1
    ), f"expected D2H records to identify B on one GPU, got {d2h_sources}"
    agent_a = next(iter(h2d_destinations))
    agent_b = next(iter(d2h_sources))
    assert agent_a != agent_b, f"A and B must be on different GPU agents: {agent_a}"

    hsa_copy_records = []
    for operation_name in (
        "hsa_amd_memory_async_copy",
        "hsa_amd_memory_async_copy_on_engine",
    ):
        hsa_copy_records.extend(
            get_api_records(
                buffer_records,
                "HSA_AMD_EXT_API",
                operation_name,
                "hsa_api_traces",
            )
        )

    case_names = ("destination device current", "source device current")
    for case_name, peer_api in zip(case_names, peer_api_records):
        peer_id = peer_api.correlation_id.internal
        hsa_ids = {
            x.correlation_id.internal
            for x in hsa_copy_records
            if x.correlation_id.ancestor == peer_id
        }
        assert (
            hsa_ids
        ), f"{case_name}: no HSA async-copy call descended from HIP id {peer_id}"

        buffered_copies = [
            x
            for x in buffer_records.memory_copies
            if x.correlation_id.internal in hsa_ids
        ]
        callback_copies = [
            x
            for x in sdk_data.callback_records.memory_copies
            if x.correlation_id.internal in hsa_ids
        ]
        assert buffered_copies, f"{case_name}: no buffered memory-copy record"
        assert callback_copies, f"{case_name}: no callback memory-copy records"
        assert {x.correlation_id.internal for x in buffered_copies} == hsa_ids
        assert {x.correlation_id.internal for x in callback_copies} == hsa_ids
        assert len(callback_copies) == 2 * len(buffered_copies)

        for hsa_id in hsa_ids:
            phases = {
                x.phase for x in callback_copies if x.correlation_id.internal == hsa_id
            }
            assert phases == {1, 2}, (
                f"{case_name}: expected callback ENTER and EXIT for HSA id {hsa_id}, "
                f"got phases {phases}"
            )

        for itr in buffered_copies:
            assert (
                itr.operation == MEMORY_COPY_DEVICE_TO_DEVICE
            ), f"{case_name}: expected D2D buffered record:\n{itr}"
            assert (itr.src_agent_id.handle, itr.dst_agent_id.handle) == (
                agent_a,
                agent_b,
            ), f"{case_name}: expected agent {agent_a} -> {agent_b}:\n{itr}"

        for itr in callback_copies:
            assert (
                itr.operation == MEMORY_COPY_DEVICE_TO_DEVICE
            ), f"{case_name}: expected D2D callback record:\n{itr}"
            assert (
                itr.payload.src_agent_id.handle,
                itr.payload.dst_agent_id.handle,
            ) == (
                agent_a,
                agent_b,
            ), f"{case_name}: expected agent {agent_a} -> {agent_b}:\n{itr}"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
