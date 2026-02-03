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
Unit tests for rocprofiler.records module.
"""

import pytest
import sys

# Import the module under test
from rocprofiler.records import CounterRecord


class TestCounterRecord:
    """Tests for the CounterRecord dataclass."""

    def test_construction(self):
        """Test basic construction of CounterRecord."""
        record = CounterRecord(
            dispatch_id=1,
            counter_id=42,
            counter_name="SQ_WAVES",
            kernel_name="test_kernel",
            value=1234.5,
            agent_id=100,
            dimensions=[("shader_engine", 0)],
        )

        assert record.dispatch_id == 1
        assert record.counter_id == 42
        assert record.counter_name == "SQ_WAVES"
        assert record.kernel_name == "test_kernel"
        assert record.value == 1234.5
        assert record.agent_id == 100
        assert record.dimensions == [("shader_engine", 0)]

    def test_construction_empty_dimensions(self):
        """Test construction with empty dimensions list."""
        record = CounterRecord(
            dispatch_id=1,
            counter_id=42,
            counter_name="SQ_WAVES",
            kernel_name="test_kernel",
            value=1234.5,
            agent_id=100,
            dimensions=[],
        )

        assert record.dimensions == []

    def test_from_native(self, mock_counter_record):
        """Test conversion from native C++ CounterRecord."""
        native = mock_counter_record(
            dispatch_id=10,
            counter_id=20,
            counter_name="TCC_HIT",
            kernel_name="conv2d",
            value=5678.9,
            agent_id=200,
            dimensions=[("channel", 1), ("bank", 2)],
        )

        record = CounterRecord._from_native(native)

        assert record.dispatch_id == 10
        assert record.counter_id == 20
        assert record.counter_name == "TCC_HIT"
        assert record.kernel_name == "conv2d"
        assert record.value == 5678.9
        assert record.agent_id == 200
        assert record.dimensions == [("channel", 1), ("bank", 2)]

    def test_repr(self):
        """Test string representation of CounterRecord."""
        record = CounterRecord(
            dispatch_id=1,
            counter_id=42,
            counter_name="SQ_WAVES",
            kernel_name="test_kernel",
            value=1234.5,
            agent_id=100,
            dimensions=[],
        )

        repr_str = repr(record)

        assert "test_kernel" in repr_str
        assert "SQ_WAVES" in repr_str
        assert "1234.5" in repr_str

    def test_equality(self):
        """Test equality comparison of CounterRecords."""
        record1 = CounterRecord(
            dispatch_id=1,
            counter_id=42,
            counter_name="SQ_WAVES",
            kernel_name="test_kernel",
            value=1234.5,
            agent_id=100,
            dimensions=[],
        )
        record2 = CounterRecord(
            dispatch_id=1,
            counter_id=42,
            counter_name="SQ_WAVES",
            kernel_name="test_kernel",
            value=1234.5,
            agent_id=100,
            dimensions=[],
        )

        assert record1 == record2

    def test_inequality_different_values(self):
        """Test inequality when values differ."""
        record1 = CounterRecord(
            dispatch_id=1,
            counter_id=42,
            counter_name="SQ_WAVES",
            kernel_name="test_kernel",
            value=1234.5,
            agent_id=100,
            dimensions=[],
        )
        record2 = CounterRecord(
            dispatch_id=1,
            counter_id=42,
            counter_name="SQ_WAVES",
            kernel_name="test_kernel",
            value=9999.0,  # Different value
            agent_id=100,
            dimensions=[],
        )

        assert record1 != record2

    def test_is_dataclass(self):
        """Verify CounterRecord is a dataclass."""
        from dataclasses import is_dataclass

        assert is_dataclass(CounterRecord)

    def test_multiple_dimensions(self):
        """Test CounterRecord with multiple dimensions."""
        record = CounterRecord(
            dispatch_id=1,
            counter_id=42,
            counter_name="SQ_WAVES",
            kernel_name="test_kernel",
            value=1234.5,
            agent_id=100,
            dimensions=[
                ("shader_engine", 0),
                ("shader_array", 1),
                ("compute_unit", 5),
            ],
        )

        assert len(record.dimensions) == 3
        assert record.dimensions[0] == ("shader_engine", 0)
        assert record.dimensions[1] == ("shader_array", 1)
        assert record.dimensions[2] == ("compute_unit", 5)


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
