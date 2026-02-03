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
Unit tests for rocprofiler.profiler module.
"""

import pytest
import sys

# Import the modules under test
from rocprofiler.profiler import ProfilerContext, profile


class TestProfilerContext:
    """Tests for the ProfilerContext class."""

    def test_construction(self):
        """Test basic construction of ProfilerContext."""
        ctx = ProfilerContext(metrics=["SQ_WAVES", "TCC_HIT"])

        assert ctx._metrics == ["SQ_WAVES", "TCC_HIT"]
        assert ctx._per_kernel is True  # default
        assert ctx._callback is None  # default
        assert ctx._session is None
        assert ctx._records == []

    def test_construction_with_per_kernel_false(self):
        """Test construction with per_kernel=False."""
        ctx = ProfilerContext(metrics=["SQ_WAVES"], per_kernel=False)

        assert ctx._per_kernel is False

    def test_construction_with_callback(self):
        """Test construction with a callback function."""
        callback_records = []

        def my_callback(records):
            callback_records.extend(records)

        ctx = ProfilerContext(metrics=["SQ_WAVES"], callback=my_callback)

        assert ctx._callback is my_callback

    def test_records_property_empty(self):
        """Test records property returns empty list initially."""
        ctx = ProfilerContext(metrics=["SQ_WAVES"])

        assert ctx.records == []

    def test_clear_empty(self):
        """Test clear on empty context."""
        ctx = ProfilerContext(metrics=["SQ_WAVES"])

        # Should not raise
        ctx.clear()
        assert ctx.records == []


class TestProfileFunction:
    """Tests for the profile() function."""

    def test_profile_returns_context(self):
        """Test that profile() returns a ProfilerContext."""
        ctx = profile(metrics=["SQ_WAVES"])

        assert isinstance(ctx, ProfilerContext)

    def test_profile_with_single_metric(self):
        """Test profile() with a single metric."""
        ctx = profile(metrics=["SQ_WAVES"])

        assert ctx._metrics == ["SQ_WAVES"]

    def test_profile_with_multiple_metrics(self):
        """Test profile() with multiple metrics."""
        ctx = profile(metrics=["SQ_WAVES", "TCC_HIT", "SQ_INSTS_VALU"])

        assert ctx._metrics == ["SQ_WAVES", "TCC_HIT", "SQ_INSTS_VALU"]

    def test_profile_with_per_kernel(self):
        """Test profile() with per_kernel parameter."""
        ctx1 = profile(metrics=["SQ_WAVES"], per_kernel=True)
        ctx2 = profile(metrics=["SQ_WAVES"], per_kernel=False)

        assert ctx1._per_kernel is True
        assert ctx2._per_kernel is False

    def test_profile_with_callback(self):
        """Test profile() with callback parameter."""

        def my_callback(records):
            pass

        ctx = profile(metrics=["SQ_WAVES"], callback=my_callback)

        assert ctx._callback is my_callback


class TestExceptions:
    """Tests for exception classes."""

    def test_rocprofiler_error_import(self):
        """Test that RocprofilerError can be imported."""
        from rocprofiler.exceptions import RocprofilerError

        assert issubclass(RocprofilerError, Exception)

    def test_profiler_not_available_error(self):
        """Test ProfilerNotAvailableError exception."""
        from rocprofiler.exceptions import ProfilerNotAvailableError

        assert issubclass(ProfilerNotAvailableError, Exception)

        with pytest.raises(ProfilerNotAvailableError):
            raise ProfilerNotAvailableError("Test error")

    def test_counter_not_found_error(self):
        """Test CounterNotFoundError exception."""
        from rocprofiler.exceptions import CounterNotFoundError

        assert issubclass(CounterNotFoundError, Exception)

        with pytest.raises(CounterNotFoundError):
            raise CounterNotFoundError("Counter XYZ not found")

    def test_session_error(self):
        """Test SessionError exception."""
        from rocprofiler.exceptions import SessionError

        assert issubclass(SessionError, Exception)

        with pytest.raises(SessionError):
            raise SessionError("Session already active")


@pytest.mark.gpu
class TestProfilerContextWithGPU:
    """
    Integration tests for ProfilerContext requiring GPU hardware.

    These tests are marked with @pytest.mark.gpu and will be skipped
    unless --run-gpu is specified.
    """

    def test_context_manager_enter_exit(self):
        """Test context manager __enter__ and __exit__."""
        import rocprofiler

        if not rocprofiler.is_available():
            pytest.skip("rocprofiler not available")

        with rocprofiler.profile(metrics=["SQ_WAVES"]) as prof:
            # Inside context, session should be started
            assert prof._session is not None
            assert prof._session.is_active()

        # After context, session should be stopped
        assert not prof._session.is_active()

    def test_context_manager_exception_handling(self):
        """Test that context manager properly handles exceptions."""
        import rocprofiler

        if not rocprofiler.is_available():
            pytest.skip("rocprofiler not available")

        class TestError(Exception):
            pass

        with pytest.raises(TestError):
            with rocprofiler.profile(metrics=["SQ_WAVES"]) as prof:
                raise TestError("test error")

        # Session should still be stopped after exception
        assert not prof._session.is_active()

    def test_available_counters(self):
        """Test that available_counters returns a list."""
        import rocprofiler

        if not rocprofiler.is_available():
            pytest.skip("rocprofiler not available")

        counters = rocprofiler.available_counters()

        assert isinstance(counters, list)
        if len(counters) > 0:
            # Verify first counter has expected attributes
            assert hasattr(counters[0], "name")
            assert hasattr(counters[0], "id")
            assert hasattr(counters[0], "block")

    def test_gpu_agents(self):
        """Test that gpu_agents returns a list."""
        import rocprofiler

        if not rocprofiler.is_available():
            pytest.skip("rocprofiler not available")

        agents = rocprofiler.gpu_agents()

        assert isinstance(agents, list)
        if len(agents) > 0:
            # Verify first agent has expected attributes
            assert hasattr(agents[0], "name")
            assert hasattr(agents[0], "id")
            assert hasattr(agents[0], "device_index")


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
