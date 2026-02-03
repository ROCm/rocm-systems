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
ROCProfiler SDK Python Bindings

This module provides Python bindings for hardware counter collection
using rocprofiler-sdk. It offers a PyTorch-profiler-like context manager
API for collecting GPU hardware counters.

Example
-------
>>> import rocprofiler
>>> with rocprofiler.profile(metrics=["SQ_WAVES", "TCC_HIT"]) as prof:
...     run_gpu_kernel()
>>> for record in prof.records:
...     print(f"{record.kernel_name}: {record.counter_name} = {record.value}")
"""

from . import libpyrocprofiler
from .profiler import profile, ProfilerContext
from .records import CounterRecord
from .counters import available_counters, gpu_agents, CounterInfo, GPUAgent
from .exceptions import RocprofilerError

__all__ = [
    "profile",
    "ProfilerContext",
    "CounterRecord",
    "CounterInfo",
    "GPUAgent",
    "available_counters",
    "gpu_agents",
    "is_available",
    "RocprofilerError",
]

# Version info - populated by CMake
version_info = {
    "version": "@PROJECT_VERSION@",
    "major": int("@PROJECT_VERSION_MAJOR@" or "0"),
    "minor": int("@PROJECT_VERSION_MINOR@" or "0"),
    "patch": int("@PROJECT_VERSION_PATCH@" or "0"),
}

__version__ = version_info["version"]


def is_available() -> bool:
    """
    Check if rocprofiler-sdk is available and initialized.

    Returns
    -------
    bool
        True if rocprofiler is available and can be used for profiling.
    """
    try:
        return libpyrocprofiler.is_available()
    except Exception:
        return False
