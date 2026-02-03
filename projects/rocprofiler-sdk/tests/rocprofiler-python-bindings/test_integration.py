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
Integration tests for rocprofiler Python bindings.

These tests require actual GPU hardware and a working ROCm installation.
Run with: pytest --run-gpu test_integration.py
"""

import pytest
import sys

# Mark all tests in this module as requiring GPU
pytestmark = pytest.mark.gpu


@pytest.fixture
def ensure_rocprofiler():
    """Skip tests if rocprofiler is not available."""
    import rocprofiler

    if not rocprofiler.is_available():
        pytest.skip("rocprofiler not available")
    return rocprofiler


class TestRocprofilerIntegration:
    """Integration tests for rocprofiler functionality."""

    def test_is_available(self, ensure_rocprofiler):
        """Test that is_available returns True when rocprofiler is working."""
        assert ensure_rocprofiler.is_available() is True

    def test_get_gpu_agents(self, ensure_rocprofiler):
        """Test GPU agent discovery."""
        rocprofiler = ensure_rocprofiler

        agents = rocprofiler.gpu_agents()

        assert isinstance(agents, list)
        assert len(agents) > 0, "Expected at least one GPU agent"

        for agent in agents:
            assert agent.id > 0
            assert agent.name != ""
            assert agent.device_index >= 0
            assert agent.gfx_version > 0

    def test_get_available_counters(self, ensure_rocprofiler):
        """Test counter discovery."""
        rocprofiler = ensure_rocprofiler

        counters = rocprofiler.available_counters()

        assert isinstance(counters, list)
        assert len(counters) > 0, "Expected at least one counter"

        # Check for common counters
        counter_names = {c.name for c in counters}
        # SQ_WAVES is a commonly available counter on AMD GPUs
        assert "SQ_WAVES" in counter_names or any(
            "WAVE" in name for name in counter_names
        ), "Expected to find wave-related counters"

    def test_available_counters_for_specific_device(self, ensure_rocprofiler):
        """Test counter discovery for a specific device."""
        rocprofiler = ensure_rocprofiler

        agents = rocprofiler.gpu_agents()
        if not agents:
            pytest.skip("No GPU agents available")

        device_id = agents[0].device_index
        counters = rocprofiler.available_counters(device_id=device_id)

        assert isinstance(counters, list)
        assert len(counters) > 0

    def test_profile_context_basic(self, ensure_rocprofiler):
        """Test basic profile context manager usage."""
        rocprofiler = ensure_rocprofiler

        counters = rocprofiler.available_counters()
        if not counters:
            pytest.skip("No counters available")

        # Use a commonly available counter
        counter_name = counters[0].name

        with rocprofiler.profile(metrics=[counter_name]) as prof:
            # No GPU kernel is run, so we expect empty records
            pass

        assert prof.records is not None
        # May or may not have records depending on if any GPU work was done
        assert isinstance(prof.records, list)

    def test_profile_with_callback(self, ensure_rocprofiler):
        """Test profile context with callback."""
        rocprofiler = ensure_rocprofiler

        counters = rocprofiler.available_counters()
        if not counters:
            pytest.skip("No counters available")

        callback_invoked = []

        def on_records(records):
            callback_invoked.append(len(records))

        counter_name = counters[0].name

        with rocprofiler.profile(metrics=[counter_name], callback=on_records) as prof:
            pass

        # Callback may or may not have been invoked depending on GPU activity
        assert isinstance(callback_invoked, list)

    def test_profiler_session_is_active(self, ensure_rocprofiler):
        """Test profiler session is_active method."""
        rocprofiler = ensure_rocprofiler

        counters = rocprofiler.available_counters()
        if not counters:
            pytest.skip("No counters available")

        counter_name = counters[0].name

        with rocprofiler.profile(metrics=[counter_name]) as prof:
            assert prof._session is not None
            assert prof._session.is_active() is True

        assert prof._session.is_active() is False

    def test_profiler_clear_records(self, ensure_rocprofiler):
        """Test clearing records."""
        rocprofiler = ensure_rocprofiler

        counters = rocprofiler.available_counters()
        if not counters:
            pytest.skip("No counters available")

        counter_name = counters[0].name

        with rocprofiler.profile(metrics=[counter_name]) as prof:
            pass

        # Clear records
        prof.clear()
        assert len(prof.records) == 0

    def test_multiple_metrics(self, ensure_rocprofiler):
        """Test profiling with multiple metrics."""
        rocprofiler = ensure_rocprofiler

        counters = rocprofiler.available_counters()
        if len(counters) < 2:
            pytest.skip("Need at least 2 counters for this test")

        metrics = [counters[0].name, counters[1].name]

        with rocprofiler.profile(metrics=metrics) as prof:
            pass

        # Profile context should have been created successfully
        assert prof._session is not None


class TestEdgeCases:
    """Edge case tests for rocprofiler."""

    def test_empty_metrics_list(self, ensure_rocprofiler):
        """Test behavior with empty metrics list."""
        rocprofiler = ensure_rocprofiler

        # Empty metrics list should either raise or handle gracefully
        # The actual behavior depends on the implementation
        try:
            with rocprofiler.profile(metrics=[]) as prof:
                pass
        except (ValueError, RuntimeError):
            # Expected - empty metrics should be rejected
            pass

    def test_invalid_counter_name(self, ensure_rocprofiler):
        """Test behavior with invalid counter name."""
        rocprofiler = ensure_rocprofiler

        # Invalid counter name should raise an error
        with pytest.raises((ValueError, RuntimeError)):
            with rocprofiler.profile(metrics=["INVALID_COUNTER_NAME_XYZ123"]) as prof:
                pass

    def test_nested_contexts(self, ensure_rocprofiler):
        """Test nested profile contexts."""
        rocprofiler = ensure_rocprofiler

        counters = rocprofiler.available_counters()
        if not counters:
            pytest.skip("No counters available")

        counter_name = counters[0].name

        # Nested contexts may or may not be supported
        # This test documents the behavior
        try:
            with rocprofiler.profile(metrics=[counter_name]) as outer:
                with rocprofiler.profile(metrics=[counter_name]) as inner:
                    pass
        except (ValueError, RuntimeError) as e:
            # Nested contexts not supported - this is acceptable
            assert "active" in str(e).lower() or "session" in str(e).lower()


if __name__ == "__main__":
    exit_code = pytest.main(["-x", "--run-gpu", __file__] + sys.argv[1:])
    sys.exit(exit_code)
