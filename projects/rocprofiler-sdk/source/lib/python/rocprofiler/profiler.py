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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
High-level profiler context manager API.

This module provides a PyTorch-profiler-like context manager interface
for collecting hardware counters during GPU kernel execution.
"""

from typing import List, Optional, Callable
from . import libpyrocprofiler
from .records import CounterRecord


class ProfilerContext:
    """
    Hardware counter profiling context.

    This class provides a context manager interface for collecting
    hardware counters during GPU kernel execution.

    Parameters
    ----------
    metrics : list of str
        List of counter names to collect (e.g., ["SQ_WAVES", "SQ_INSTS_VALU"])
    per_kernel : bool, optional
        If True, collect counters per-kernel dispatch (default True)
    callback : callable, optional
        Optional callback invoked when records are ready. The callback
        receives a list of CounterRecord objects.

    Example
    -------
    >>> with rocprofiler.profile(metrics=["SQ_WAVES"]) as prof:
    ...     my_gpu_kernel()
    >>> for record in prof.records:
    ...     print(f"{record.kernel_name}: {record.value}")
    """

    def __init__(
        self,
        metrics: List[str],
        per_kernel: bool = True,
        callback: Optional[Callable[[List[CounterRecord]], None]] = None,
    ):
        """Initialize profiler context."""
        self._metrics = metrics
        self._per_kernel = per_kernel
        self._callback = callback
        self._session: Optional[libpyrocprofiler.ProfilerSession] = None
        self._records: List[CounterRecord] = []

    def __enter__(self) -> "ProfilerContext":
        """Start profiling."""
        wrapped_callback = None
        if self._callback:

            def wrapped_callback(native_records):
                converted = [CounterRecord._from_native(r) for r in native_records]
                self._records.extend(converted)
                self._callback(converted)

        self._session = libpyrocprofiler.ProfilerSession(
            self._metrics, self._per_kernel, wrapped_callback
        )
        self._session.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling and collect remaining records."""
        if self._session:
            self._session.stop()
            # Collect any remaining records
            native_records = self._session.get_records()
            self._records.extend(CounterRecord._from_native(r) for r in native_records)
        return False  # Don't suppress exceptions

    @property
    def records(self) -> List[CounterRecord]:
        """
        Get collected counter records.

        Returns
        -------
        list of CounterRecord
            All counter records collected during profiling.
        """
        return self._records

    def clear(self):
        """Clear collected records."""
        self._records.clear()
        if self._session:
            self._session.clear_records()


def profile(
    metrics: List[str],
    per_kernel: bool = True,
    callback: Optional[Callable[[List[CounterRecord]], None]] = None,
) -> ProfilerContext:
    """
    Create a profiling context manager.

    This is the main entry point for counter collection. Use it as a
    context manager to profile GPU kernel execution.

    Parameters
    ----------
    metrics : list of str
        List of counter names to collect (e.g., ["SQ_WAVES", "TCC_HIT"])
    per_kernel : bool, optional
        Collect counters per-kernel dispatch (default True)
    callback : callable, optional
        Optional callback for streaming results. Receives a list of
        CounterRecord objects.

    Returns
    -------
    ProfilerContext
        A context manager that can be used with the 'with' statement.

    Example
    -------
    >>> import rocprofiler
    >>> with rocprofiler.profile(metrics=["SQ_WAVES", "TCC_HIT"]) as prof:
    ...     run_kernel()
    >>> for rec in prof.records:
    ...     print(f"{rec.kernel_name}: {rec.counter_name} = {rec.value}")
    """
    return ProfilerContext(
        metrics=metrics,
        per_kernel=per_kernel,
        callback=callback,
    )
