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
validate.py -- PyTest validation for the multiple anytime tool configuration test.

Verifies that both kinetoesque and protonesque tool libraries produced correct
JSON output with HIP API traces, and that both tools operated concurrently
without interference.
"""

import sys
import pytest


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _get_hip_traces(traces):
    """Return trace records whose name contains a HIP API call."""
    return [t for t in traces if "hip" in t.get("name", "").lower()]


def _get_unique_operations(traces):
    """Return the set of unique operation names found in traces."""
    return {t.get("name", "") for t in traces}


# ---------------------------------------------------------------------------
# Kinetoesque output validation
# ---------------------------------------------------------------------------
class TestKinetoesqueDataStructure:
    """Verify the kinetoesque JSON output has the expected structure."""

    def test_top_level_key(self, kinetoesque_data):
        assert "kinetoesque-traces" in kinetoesque_data

    def test_traces_is_list(self, kinetoesque_data):
        traces = kinetoesque_data["kinetoesque-traces"]
        assert isinstance(traces, list)

    def test_traces_not_empty(self, kinetoesque_data):
        traces = kinetoesque_data["kinetoesque-traces"]
        assert len(traces) > 0, "kinetoesque output has no traces"


class TestKinetoesqueTraceRecords:
    """Verify individual trace records have required fields."""

    def test_record_fields(self, kinetoesque_data):
        traces = kinetoesque_data["kinetoesque-traces"]
        required_fields = {
            "start_timestamp",
            "end_timestamp",
            "kind",
            "operation",
            "phase",
            "name",
        }
        for i, trace in enumerate(traces):
            missing = required_fields - set(trace.keys())
            assert not missing, f"Trace record {i} missing fields: {missing}"

    def test_timestamps_are_integers(self, kinetoesque_data):
        traces = kinetoesque_data["kinetoesque-traces"]
        for i, trace in enumerate(traces):
            assert isinstance(
                trace["start_timestamp"], int
            ), f"Trace {i}: start_timestamp is not int"
            assert isinstance(
                trace["end_timestamp"], int
            ), f"Trace {i}: end_timestamp is not int"

    def test_timestamps_non_negative(self, kinetoesque_data):
        traces = kinetoesque_data["kinetoesque-traces"]
        for i, trace in enumerate(traces):
            assert trace["start_timestamp"] >= 0, f"Trace {i}: negative start_timestamp"
            assert trace["end_timestamp"] >= 0, f"Trace {i}: negative end_timestamp"


class TestKinetoesqueHipTraces:
    """Verify kinetoesque captured HIP API traces."""

    def test_has_hip_traces(self, kinetoesque_data):
        traces = kinetoesque_data["kinetoesque-traces"]
        hip_traces = _get_hip_traces(traces)
        assert len(hip_traces) > 0, "kinetoesque output has no HIP API traces"

    def test_has_stream_operations(self, kinetoesque_data):
        """Verify stream creation/destruction APIs were captured (8 streams)."""
        traces = kinetoesque_data["kinetoesque-traces"]
        ops = _get_unique_operations(traces)
        stream_ops = [op for op in ops if "Stream" in op or "stream" in op]
        assert (
            len(stream_ops) > 0
        ), f"Expected stream-related HIP API calls, found operations: {sorted(ops)}"

    def test_has_memory_operations(self, kinetoesque_data):
        """Verify memory allocation APIs were captured."""
        traces = kinetoesque_data["kinetoesque-traces"]
        ops = _get_unique_operations(traces)
        mem_ops = [op for op in ops if "Malloc" in op or "Free" in op or "Memset" in op]
        assert (
            len(mem_ops) > 0
        ), f"Expected memory HIP API calls (Malloc/Free/Memset), found operations: {sorted(ops)}"

    def test_minimum_trace_count(self, kinetoesque_data):
        """Kinetoesque should have captured traces from 8-stream torchesque
        AND the 4-stream tritonesque phase (since it was restarted)."""
        traces = kinetoesque_data["kinetoesque-traces"]
        hip_traces = _get_hip_traces(traces)
        # 8 streams * 3 iterations of kernel launches + stream create/destroy
        # + 4 streams * 3 iterations (from the concurrent phase)
        # Each API call produces enter + exit = 2 records
        # Conservative minimum: at least 20 HIP trace records
        assert (
            len(hip_traces) >= 20
        ), f"Expected at least 20 HIP traces from kinetoesque, got {len(hip_traces)}"


# ---------------------------------------------------------------------------
# Protonesque output validation
# ---------------------------------------------------------------------------
class TestProtonesqueDataStructure:
    """Verify the protonesque JSON output has the expected structure."""

    def test_top_level_key(self, protonesque_data):
        assert "protonesque-traces" in protonesque_data

    def test_traces_is_list(self, protonesque_data):
        traces = protonesque_data["protonesque-traces"]
        assert isinstance(traces, list)

    def test_traces_not_empty(self, protonesque_data):
        traces = protonesque_data["protonesque-traces"]
        assert len(traces) > 0, "protonesque output has no traces"


class TestProtonesqueTraceRecords:
    """Verify individual trace records have required fields."""

    def test_record_fields(self, protonesque_data):
        traces = protonesque_data["protonesque-traces"]
        required_fields = {
            "start_timestamp",
            "end_timestamp",
            "kind",
            "operation",
            "phase",
            "name",
        }
        for i, trace in enumerate(traces):
            missing = required_fields - set(trace.keys())
            assert not missing, f"Trace record {i} missing fields: {missing}"

    def test_timestamps_are_integers(self, protonesque_data):
        traces = protonesque_data["protonesque-traces"]
        for i, trace in enumerate(traces):
            assert isinstance(
                trace["start_timestamp"], int
            ), f"Trace {i}: start_timestamp is not int"
            assert isinstance(
                trace["end_timestamp"], int
            ), f"Trace {i}: end_timestamp is not int"

    def test_timestamps_non_negative(self, protonesque_data):
        traces = protonesque_data["protonesque-traces"]
        for i, trace in enumerate(traces):
            assert trace["start_timestamp"] >= 0, f"Trace {i}: negative start_timestamp"
            assert trace["end_timestamp"] >= 0, f"Trace {i}: negative end_timestamp"


class TestProtonesqueHipTraces:
    """Verify protonesque captured HIP API traces."""

    def test_has_hip_traces(self, protonesque_data):
        traces = protonesque_data["protonesque-traces"]
        hip_traces = _get_hip_traces(traces)
        assert len(hip_traces) > 0, "protonesque output has no HIP API traces"

    def test_has_stream_operations(self, protonesque_data):
        """Verify stream creation/destruction APIs were captured (4 streams)."""
        traces = protonesque_data["protonesque-traces"]
        ops = _get_unique_operations(traces)
        stream_ops = [op for op in ops if "Stream" in op or "stream" in op]
        assert (
            len(stream_ops) > 0
        ), f"Expected stream-related HIP API calls, found operations: {sorted(ops)}"

    def test_has_memory_operations(self, protonesque_data):
        """Verify memory allocation APIs were captured."""
        traces = protonesque_data["protonesque-traces"]
        ops = _get_unique_operations(traces)
        mem_ops = [op for op in ops if "Malloc" in op or "Free" in op or "Memset" in op]
        assert (
            len(mem_ops) > 0
        ), f"Expected memory HIP API calls (Malloc/Free/Memset), found operations: {sorted(ops)}"

    def test_minimum_trace_count(self, protonesque_data):
        """Protonesque should have captured traces from 4-stream tritonesque phase only."""
        traces = protonesque_data["protonesque-traces"]
        hip_traces = _get_hip_traces(traces)
        # 4 streams * 3 iterations + stream create/destroy + memory ops
        # Each API call produces enter + exit = 2 records
        # Conservative minimum: at least 10 HIP trace records
        assert (
            len(hip_traces) >= 10
        ), f"Expected at least 10 HIP traces from protonesque, got {len(hip_traces)}"


# ---------------------------------------------------------------------------
# Concurrent tool validation
# ---------------------------------------------------------------------------
class TestConcurrentToolOperation:
    """Verify both tools operated concurrently without interference."""

    def test_both_tools_produced_output(self, kinetoesque_data, protonesque_data):
        """Both tools should have non-empty trace output."""
        k_traces = kinetoesque_data["kinetoesque-traces"]
        p_traces = protonesque_data["protonesque-traces"]
        assert len(k_traces) > 0
        assert len(p_traces) > 0

    def test_kinetoesque_has_more_traces(self, kinetoesque_data, protonesque_data):
        """Kinetoesque was active during both torchesque (8 streams) and
        tritonesque (4 streams) phases, so it should have more traces
        than protonesque which was only active during the tritonesque phase."""
        k_traces = kinetoesque_data["kinetoesque-traces"]
        p_traces = protonesque_data["protonesque-traces"]
        k_hip = _get_hip_traces(k_traces)
        p_hip = _get_hip_traces(p_traces)
        assert len(k_hip) > len(p_hip), (
            f"Expected kinetoesque ({len(k_hip)} HIP traces) to have more traces "
            f"than protonesque ({len(p_hip)} HIP traces) since it was active longer"
        )

    def test_protonesque_does_not_have_torchesque_only_traces(
        self, kinetoesque_data, protonesque_data
    ):
        """Protonesque was started AFTER the torchesque 8-stream phase, so the
        torchesque-only stream operations should NOT appear in protonesque output.

        We check that protonesque has fewer stream-create operations than kinetoesque,
        since kinetoesque saw both 8-stream + 4-stream creation while protonesque
        only saw 4-stream creation."""
        k_traces = kinetoesque_data["kinetoesque-traces"]
        p_traces = protonesque_data["protonesque-traces"]

        k_stream_creates = [t for t in k_traces if "hipStreamCreate" in t.get("name", "")]
        p_stream_creates = [t for t in p_traces if "hipStreamCreate" in t.get("name", "")]

        assert len(k_stream_creates) > len(p_stream_creates), (
            f"Expected kinetoesque to have more hipStreamCreate calls "
            f"({len(k_stream_creates)}) than protonesque ({len(p_stream_creates)})"
        )

    def test_tools_share_common_operations(self, kinetoesque_data, protonesque_data):
        """During the concurrent phase (tritonesque), both tools should have
        captured some of the same operation types."""
        k_traces = kinetoesque_data["kinetoesque-traces"]
        p_traces = protonesque_data["protonesque-traces"]

        k_ops = _get_unique_operations(k_traces)
        p_ops = _get_unique_operations(p_traces)

        common_ops = k_ops & p_ops
        assert len(common_ops) > 0, (
            f"Expected overlapping operations during concurrent phase. "
            f"Kinetoesque ops: {sorted(k_ops)}, Protonesque ops: {sorted(p_ops)}"
        )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
