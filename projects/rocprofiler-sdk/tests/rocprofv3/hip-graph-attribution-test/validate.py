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

"""HIP graph attribution validators against rocprofv3 CSV output.

Direct CSV (and OTF2, Perfetto) output from rocprofv3 is deprecated for HIP
graph attribution: graph_exec_id / graph_node_id and the HIP_GRAPH summary
record flow only to JSON and the rocpd SQLite database. See validate_json.py
for the JSON-side semantic checks.

The tests in this file remain named as before (ctest IDs are stable) but now
verify that the deprecated CSV columns are absent.
"""

import sys
import pytest

# ---------------------------------------------------------------------------
# CSV column absence (deprecation guard)
# ---------------------------------------------------------------------------


def test_columns_present(kernel_input_data):
    """Graph_Exec_Id / Graph_Node_Id columns must NOT appear on kernel CSV.
    Direct CSV graph attribution was removed; use JSON or rocpd."""
    row = kernel_input_data[0]
    assert "Graph_Exec_Id" not in row, list(row.keys())
    assert "Graph_Node_Id" not in row, list(row.keys())


def test_non_graph_rows_render_empty(kernel_input_data):
    pytest.skip("graph attribution columns removed from kernel CSV; see JSON / rocpd")


def test_total_graph_dispatch_count(
    kernel_input_data,
    expected_iterations,
    expected_execs,
    expected_kernel_nodes_per_launch,
):
    pytest.skip("graph attribution columns removed from kernel CSV; see JSON / rocpd")


def test_two_distinct_exec_ids(kernel_input_data, expected_execs):
    pytest.skip("graph attribution columns removed from kernel CSV; see JSON / rocpd")


def test_graph_node_id_range_per_launch(
    kernel_input_data,
    expected_kernel_nodes_per_launch,
    expected_iterations,
    expected_execs,
):
    pytest.skip("graph attribution columns removed from kernel CSV; see JSON / rocpd")


def test_graph_node_id_stable_per_exec(kernel_input_data, expected_nodes_per_launch):
    pytest.skip("graph attribution columns removed from kernel CSV; see JSON / rocpd")


def test_distinct_kernel_nodes_remain_distinct(
    kernel_input_data, expected_distinct_kernels
):
    pytest.skip("graph attribution columns removed from kernel CSV; see JSON / rocpd")


# ---------------------------------------------------------------------------
# HIP_GRAPH summary record (CSV path removed)
# ---------------------------------------------------------------------------


def test_graph_launch_record_count(
    graph_launch_input_data, expected_iterations, expected_execs
):
    pytest.skip("HIP_GRAPH CSV output removed; see JSON / rocpd")


def test_graph_launch_dispatch_counts(
    graph_launch_input_data, expected_kernel_nodes_per_launch
):
    pytest.skip("HIP_GRAPH CSV output removed; see JSON / rocpd")


def test_graph_launch_exec_ids_match_kernel_csv(
    graph_launch_input_data, kernel_input_data, expected_execs
):
    pytest.skip("HIP_GRAPH CSV output removed; see JSON / rocpd")


def test_graph_launch_correlation_joins_to_hip_api(
    graph_launch_input_data, hip_api_input_data
):
    pytest.skip("HIP_GRAPH CSV output removed; see JSON / rocpd")


# ---------------------------------------------------------------------------
# Memory copy CSV (graph columns removed)
# ---------------------------------------------------------------------------


def test_memcpy_columns_present(memory_copy_input_data):
    """Graph_Exec_Id / Graph_Node_Id must NOT appear on memory_copy CSV."""
    row = memory_copy_input_data[0]
    assert "Graph_Exec_Id" not in row, list(row.keys())
    assert "Graph_Node_Id" not in row, list(row.keys())


def test_memcpy_graph_attribution_count(
    memory_copy_input_data,
    expected_iterations,
    expected_execs,
    expected_memcpy_nodes_per_launch,
):
    pytest.skip(
        "graph attribution columns removed from memory_copy CSV; see JSON / rocpd"
    )


def test_memcpy_exec_ids_match_kernel_csv(
    memory_copy_input_data, kernel_input_data, expected_execs
):
    pytest.skip(
        "graph attribution columns removed from memory_copy CSV; see JSON / rocpd"
    )


def test_non_graph_memcpy_rows_render_empty(memory_copy_input_data):
    pytest.skip(
        "graph attribution columns removed from memory_copy CSV; see JSON / rocpd"
    )


# ---------------------------------------------------------------------------
# Cross-CSV / per-exec checks (all CSV-based — now skipped)
# ---------------------------------------------------------------------------


def test_graph_node_id_range_per_launch_full_topology(
    kernel_input_data,
    request,
    expected_nodes_per_launch,
    expected_iterations,
    expected_execs,
):
    pytest.skip("graph attribution columns removed from kernel CSV; see JSON / rocpd")


def test_per_exec_kernel_launch_counts(
    kernel_input_data, expected_iterations, expected_kernel_nodes_per_launch
):
    pytest.skip("graph attribution columns removed from kernel CSV; see JSON / rocpd")


def test_per_launch_node_sequence_stable(kernel_input_data):
    pytest.skip("graph attribution columns removed from kernel CSV; see JSON / rocpd")


def test_graph_launch_per_exec_record_counts(
    graph_launch_input_data, expected_iterations, expected_execs
):
    pytest.skip("HIP_GRAPH CSV output removed; see JSON / rocpd")


def test_graph_launch_timestamps_sane(graph_launch_input_data):
    pytest.skip("HIP_GRAPH CSV output removed; see JSON / rocpd")


def test_graph_launch_agent_id_nonzero(graph_launch_input_data):
    pytest.skip("HIP_GRAPH CSV output removed; see JSON / rocpd")


def test_graph_launch_correlation_unique(graph_launch_input_data):
    pytest.skip("HIP_GRAPH CSV output removed; see JSON / rocpd")


def test_graph_kernels_carry_stream_id(kernel_input_data):
    pytest.skip("graph attribution columns removed from kernel CSV; see JSON / rocpd")


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
