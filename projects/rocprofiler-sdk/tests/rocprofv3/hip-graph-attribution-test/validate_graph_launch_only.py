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

"""HIP_GRAPH-only subscription smoke test against rocpd output."""

import sys
import pytest


def _graph_launch_rows(rocpd_connection):
    cur = rocpd_connection.execute("""
        SELECT graph_exec_id, kernel_dispatch_count, start, end
        FROM rocpd_graph_launch
        ORDER BY start ASC, end DESC
        """)
    return cur.fetchall()


def test_graph_launch_record_count(rocpd_connection, expected_iterations, expected_execs):
    rows = _graph_launch_rows(rocpd_connection)
    expected_launches = expected_iterations * expected_execs + 1
    assert len(rows) == expected_launches


def test_graph_launch_kernel_dispatch_count(rocpd_connection, expected_nodes_per_launch):
    rows = _graph_launch_rows(rocpd_connection)
    assert rows, "expected HIP_GRAPH records in rocpd"
    for _, kernel_dispatch_count, _, _ in rows:
        assert kernel_dispatch_count == expected_nodes_per_launch


def test_graph_launch_exec_ids_nonzero(rocpd_connection):
    rows = _graph_launch_rows(rocpd_connection)
    assert rows, "expected HIP_GRAPH records in rocpd"
    for graph_exec_id, _, start, end in rows:
        assert graph_exec_id > 0
        assert end >= start


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
