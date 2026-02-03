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
Unit tests for rocprofiler.counters module.
"""

import pytest
import sys

# Import the module under test
from rocprofiler.counters import CounterInfo, GPUAgent


class TestCounterInfo:
    """Tests for the CounterInfo dataclass."""

    def test_construction(self):
        """Test basic construction of CounterInfo."""
        info = CounterInfo(
            id=42,
            name="SQ_WAVES",
            description="Number of wavefronts dispatched",
            block="SQ",
            expression="",
            is_constant=False,
            is_derived=False,
        )

        assert info.id == 42
        assert info.name == "SQ_WAVES"
        assert info.description == "Number of wavefronts dispatched"
        assert info.block == "SQ"
        assert info.expression == ""
        assert info.is_constant is False
        assert info.is_derived is False

    def test_derived_counter(self):
        """Test construction of a derived counter."""
        info = CounterInfo(
            id=100,
            name="VALU_UTILIZATION",
            description="VALU utilization percentage",
            block="SQ",
            expression="(SQ_INSTS_VALU / (SQ_WAVES * 64)) * 100",
            is_constant=False,
            is_derived=True,
        )

        assert info.is_derived is True
        assert info.expression != ""

    def test_constant_counter(self):
        """Test construction of a constant counter."""
        info = CounterInfo(
            id=200,
            name="GPU_CLOCK_FREQUENCY",
            description="GPU clock frequency in MHz",
            block="SYS",
            expression="",
            is_constant=True,
            is_derived=False,
        )

        assert info.is_constant is True
        assert info.is_derived is False

    def test_from_native(self, mock_counter_info):
        """Test conversion from native C++ CounterInfo."""
        native = mock_counter_info(
            counter_id=50,
            name="TCC_HIT",
            description="L2 cache hits",
            block="TCC",
            expression="",
            is_constant=False,
            is_derived=False,
        )

        info = CounterInfo._from_native(native)

        assert info.id == 50
        assert info.name == "TCC_HIT"
        assert info.description == "L2 cache hits"
        assert info.block == "TCC"

    def test_repr(self):
        """Test string representation of CounterInfo."""
        info = CounterInfo(
            id=42,
            name="SQ_WAVES",
            description="Number of wavefronts",
            block="SQ",
            expression="",
            is_constant=False,
            is_derived=False,
        )

        repr_str = repr(info)
        assert "SQ_WAVES" in repr_str

    def test_is_dataclass(self):
        """Verify CounterInfo is a dataclass."""
        from dataclasses import is_dataclass

        assert is_dataclass(CounterInfo)


class TestGPUAgent:
    """Tests for the GPUAgent dataclass."""

    def test_construction(self):
        """Test basic construction of GPUAgent."""
        agent = GPUAgent(
            id=1234,
            name="gfx942",
            product_name="AMD Instinct MI300X",
            device_index=0,
            gfx_version=942,
        )

        assert agent.id == 1234
        assert agent.name == "gfx942"
        assert agent.product_name == "AMD Instinct MI300X"
        assert agent.device_index == 0
        assert agent.gfx_version == 942

    def test_from_native(self, mock_agent_info):
        """Test conversion from native C++ AgentInfo."""
        native = mock_agent_info(
            agent_id=5678,
            name="gfx90a",
            product_name="AMD Instinct MI210",
            device_index=1,
            gfx_version=90,
        )

        agent = GPUAgent._from_native(native)

        assert agent.id == 5678
        assert agent.name == "gfx90a"
        assert agent.product_name == "AMD Instinct MI210"
        assert agent.device_index == 1
        assert agent.gfx_version == 90

    def test_repr(self):
        """Test string representation of GPUAgent."""
        agent = GPUAgent(
            id=1234,
            name="gfx942",
            product_name="AMD Instinct MI300X",
            device_index=0,
            gfx_version=942,
        )

        repr_str = repr(agent)
        assert "gfx942" in repr_str
        assert "0" in repr_str  # device_index

    def test_is_dataclass(self):
        """Verify GPUAgent is a dataclass."""
        from dataclasses import is_dataclass

        assert is_dataclass(GPUAgent)

    def test_multiple_devices(self):
        """Test construction of multiple GPU agents."""
        agents = [
            GPUAgent(
                id=1000 + i,
                name=f"gfx942",
                product_name="AMD Instinct MI300X",
                device_index=i,
                gfx_version=942,
            )
            for i in range(4)
        ]

        assert len(agents) == 4
        for i, agent in enumerate(agents):
            assert agent.device_index == i
            assert agent.id == 1000 + i


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
