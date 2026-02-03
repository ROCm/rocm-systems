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
Pytest configuration and fixtures for rocprofiler Python bindings tests.
"""

import pytest


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU hardware")
    config.addinivalue_line("markers", "integration: mark test as an integration test")


def pytest_addoption(parser):
    """Add command line options."""
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="Run tests that require GPU hardware",
    )


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests unless --run-gpu is specified."""
    if config.getoption("--run-gpu"):
        # Run all tests including GPU tests
        return

    skip_gpu = pytest.mark.skip(reason="Need --run-gpu option to run")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture
def mock_counter_record():
    """Create a mock CounterRecord-like object for testing."""

    class MockNativeRecord:
        """Mock native C++ CounterRecord object."""

        def __init__(
            self,
            dispatch_id=1,
            counter_id=42,
            counter_name="SQ_WAVES",
            kernel_name="test_kernel",
            value=1234.5,
            agent_id=100,
            dimensions=None,
        ):
            self.dispatch_id = dispatch_id
            self.counter_id = counter_id
            self.counter_name = counter_name
            self.kernel_name = kernel_name
            self.value = value
            self.agent_id = agent_id
            self.dimensions = dimensions if dimensions is not None else []

    return MockNativeRecord


@pytest.fixture
def mock_counter_info():
    """Create a mock CounterInfo-like object for testing."""

    class MockNativeCounterInfo:
        """Mock native C++ CounterInfo object."""

        def __init__(
            self,
            counter_id=42,
            name="SQ_WAVES",
            description="Number of wavefronts",
            block="SQ",
            expression="",
            is_constant=False,
            is_derived=False,
        ):
            self.id = counter_id
            self.name = name
            self.description = description
            self.block = block
            self.expression = expression
            self.is_constant = is_constant
            self.is_derived = is_derived

    return MockNativeCounterInfo


@pytest.fixture
def mock_agent_info():
    """Create a mock AgentInfo-like object for testing."""

    class MockNativeAgentInfo:
        """Mock native C++ AgentInfo object."""

        def __init__(
            self,
            agent_id=1234,
            name="gfx942",
            product_name="AMD Instinct MI300X",
            device_index=0,
            gfx_version=942,
        ):
            self.id = agent_id
            self.name = name
            self.product_name = product_name
            self.device_index = device_index
            self.gfx_version = gfx_version

    return MockNativeAgentInfo


@pytest.fixture
def sample_counter_records(mock_counter_record):
    """Create a list of sample counter records for testing."""
    return [
        mock_counter_record(
            dispatch_id=1,
            counter_id=42,
            counter_name="SQ_WAVES",
            kernel_name="matmul",
            value=1000.0,
            agent_id=100,
        ),
        mock_counter_record(
            dispatch_id=1,
            counter_id=43,
            counter_name="TCC_HIT",
            kernel_name="matmul",
            value=5000.0,
            agent_id=100,
        ),
        mock_counter_record(
            dispatch_id=2,
            counter_id=42,
            counter_name="SQ_WAVES",
            kernel_name="vectorAdd",
            value=500.0,
            agent_id=100,
        ),
    ]
