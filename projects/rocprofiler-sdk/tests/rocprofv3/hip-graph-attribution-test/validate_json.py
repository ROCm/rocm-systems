#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

"""JSON-output smoke tests for HIP graph attribution.

Validates that rocprofv3's --output-format json carries through:
  - graph_exec_id / graph_node_id at the top level of KERNEL_DISPATCH records
    (siblings of stream_id, populated by the tool-layer ext record).
  - HIP_GRAPH summary records with the full field set.

The CSV validator (validate.py) does the heavier semantic checking.
"""

import sys
import pytest


def _buffer_records(json_input_data):
    return json_input_data["rocprofiler-sdk-tool"]["buffer_records"]


def test_kernel_dispatch_records_have_graph_fields(json_input_data):
    """Every KERNEL_DISPATCH JSON record must expose graph_exec_id and
    graph_node_id at the top level (siblings of stream_id, not under
    dispatch_info)."""
    kernel_dispatch = _buffer_records(json_input_data)["kernel_dispatch"]
    assert len(kernel_dispatch) > 0, "no kernel_dispatch records in JSON"
    for rec in kernel_dispatch:
        assert "graph_exec_id" in rec, f"record missing top-level graph_exec_id: {rec}"
        assert "graph_node_id" in rec, f"record missing top-level graph_node_id: {rec}"


def test_some_kernel_dispatches_have_nonzero_graph_exec_id(
    json_input_data,
    expected_iterations,
    expected_execs,
    expected_nodes_per_launch,
):
    """The graph dispatches in the workload must carry nonzero graph_exec_id.
    Count should match (iterations*execs + 1) * nodes_per_launch — the +1 is
    the one extra successful exec_b launch after the failed launch."""
    kernel_dispatch = _buffer_records(json_input_data)["kernel_dispatch"]
    graph_rows = [r for r in kernel_dispatch if int(r["graph_exec_id"]) != 0]
    expected_launches = expected_iterations * expected_execs + 1
    expected = expected_launches * expected_nodes_per_launch
    assert (
        len(graph_rows) == expected
    ), f"expected {expected} graph-attributed dispatches in JSON, got {len(graph_rows)}"


def test_graph_launch_records_present(
    json_input_data,
    expected_iterations,
    expected_execs,
):
    """HIP_GRAPH summary records must appear in the JSON output's
    buffer_records.graph_launch array. One per *successful* hipGraphLaunch."""
    buffer_records = _buffer_records(json_input_data)
    assert (
        "hip_graph" in buffer_records
    ), f"buffer_records missing 'hip_graph' key; got {list(buffer_records.keys())}"
    graph_launch = buffer_records["hip_graph"]
    expected_launches = expected_iterations * expected_execs + 1
    assert (
        len(graph_launch) == expected_launches
    ), f"expected {expected_launches} HIP_GRAPH records, got {len(graph_launch)}"


def test_graph_launch_record_shape(json_input_data, expected_nodes_per_launch):
    """Each HIP_GRAPH record carries the expected field set with sane values."""
    graph_launch = _buffer_records(json_input_data)["hip_graph"]
    required_fields = (
        "size",
        "kind",
        "operation",
        "thread_id",
        "correlation_id",
        "start_timestamp",
        "end_timestamp",
        "agent_id",
        "queue_id",
        "graph_exec_id",
        "kernel_dispatch_count",
    )
    for rec in graph_launch:
        for f in required_fields:
            assert f in rec, f"HIP_GRAPH record missing '{f}': {rec}"
        # graph_exec_id is a typed handle ({"handle": N}) on the HIP_GRAPH record.
        assert (
            int(rec["graph_exec_id"]["handle"]) > 0
        ), f"graph_exec_id must be nonzero: {rec}"
        assert (
            int(rec["kernel_dispatch_count"]) == expected_nodes_per_launch
        ), f"kernel_dispatch_count {rec['kernel_dispatch_count']} != {expected_nodes_per_launch}: {rec}"
        assert int(rec["end_timestamp"]) >= int(rec["start_timestamp"])
        assert (
            int(rec["correlation_id"]["internal"]) > 0
        ), f"HIP_GRAPH must have nonzero internal correlation_id: {rec}"


def test_graph_launch_exec_ids_match_kernel_dispatch_records(json_input_data):
    """The set of graph_exec_ids on HIP_GRAPH records must equal the set
    of nonzero graph_exec_ids on the KERNEL_DISPATCH records — the two views
    must agree."""
    buffer_records = _buffer_records(json_input_data)
    # HIP_GRAPH carries graph_exec_id as a typed handle; kernel_dispatch ext records
    # carry it as a raw uint64.
    launch_ids = {int(r["graph_exec_id"]["handle"]) for r in buffer_records["hip_graph"]}
    kernel_ids = {
        int(r["graph_exec_id"])
        for r in buffer_records["kernel_dispatch"]
        if int(r["graph_exec_id"]) != 0
    }
    assert (
        launch_ids == kernel_ids
    ), f"HIP_GRAPH ids {launch_ids} != graph-KERNEL_DISPATCH ids {kernel_ids}"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
