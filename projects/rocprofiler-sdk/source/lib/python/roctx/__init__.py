###############################################################################
# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
###############################################################################

"""
ROCTx Python bindings using ctypes - Version agnostic across Python 3.x

This module provides Python bindings to the ROCTx (ROCm Tracing) library
using ctypes, eliminating the need for version-specific compiled extensions.
"""

import ctypes
import os
import sys
from ctypes import c_char_p, c_int, c_uint64, POINTER

__all__ = [
    "mark",
    "profilerPause",
    "profilerResume",
    "getThreadId",
    "rangePush",
    "rangePop",
    "rangeStart",
    "rangeStop",
    "nameOsThread",
    "nameHipDevice",
    "context_decorators",
    "version_info",
]

# Type definitions matching roctx types.h
roctx_range_id_t = c_uint64
roctx_thread_id_t = c_uint64

# Library loading with multiple search strategies
_libroctx = None
_lib_load_error = None


def _find_and_load_library():
    """Find and load the ROCTx shared library"""
    global _libroctx, _lib_load_error

    # Strategy 1: Try loading from the same directory as this module
    # This works when the .so is packaged with the wheel
    module_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths = [
        # In the same directory
        os.path.join(module_dir, "librocprofiler-sdk-roctx.so"),
        os.path.join(module_dir, "lib", "librocprofiler-sdk-roctx.so"),
        # One level up (for site-packages layout)
        os.path.join(module_dir, "..", "lib", "librocprofiler-sdk-roctx.so"),
        # Standard ROCm installation paths
        "/opt/rocm/lib/librocprofiler-sdk-roctx.so",
        os.path.expanduser("~/rocm/lib/librocprofiler-sdk-roctx.so"),
    ]

    # Add ROCM_PATH if set
    if "ROCM_PATH" in os.environ:
        rocm_path = os.environ["ROCM_PATH"]
        search_paths.insert(
            0, os.path.join(rocm_path, "lib", "librocprofiler-sdk-roctx.so")
        )

    # Try each path
    for lib_path in search_paths:
        if os.path.exists(lib_path):
            try:
                _libroctx = ctypes.CDLL(lib_path)
                return
            except OSError as e:
                _lib_load_error = f"Found library at {lib_path} but failed to load: {e}"
                continue

    # Strategy 2: Try loading by name (will use LD_LIBRARY_PATH, ld.so.cache, etc.)
    try:
        _libroctx = ctypes.CDLL("librocprofiler-sdk-roctx.so")
        return
    except OSError:
        pass

    # Strategy 3: Try versioned name
    try:
        _libroctx = ctypes.CDLL("librocprofiler-sdk-roctx.so.0")
        return
    except OSError as e:
        _lib_load_error = f"Failed to load library: {e}\nSearched paths: {search_paths}"


# Load the library
_find_and_load_library()

if _libroctx is None:
    raise ImportError(
        f"Failed to load librocprofiler-sdk-roctx.so. {_lib_load_error}\n"
        "Please ensure ROCm is installed and ROCM_PATH or LD_LIBRARY_PATH is set correctly."
    )

# Define function signatures for all ROCTx API functions
# This ensures proper type checking and conversion at the C boundary

# void roctxMarkA(const char* message)
_libroctx.roctxMarkA.argtypes = [c_char_p]
_libroctx.roctxMarkA.restype = None

# int roctxProfilerPause(roctx_thread_id_t tid)
_libroctx.roctxProfilerPause.argtypes = [roctx_thread_id_t]
_libroctx.roctxProfilerPause.restype = c_int

# int roctxProfilerResume(roctx_thread_id_t tid)
_libroctx.roctxProfilerResume.argtypes = [roctx_thread_id_t]
_libroctx.roctxProfilerResume.restype = c_int

# int roctxGetThreadId(roctx_thread_id_t* tid)
_libroctx.roctxGetThreadId.argtypes = [POINTER(roctx_thread_id_t)]
_libroctx.roctxGetThreadId.restype = c_int

# int roctxRangePushA(const char* message)
_libroctx.roctxRangePushA.argtypes = [c_char_p]
_libroctx.roctxRangePushA.restype = c_int

# int roctxRangePop()
_libroctx.roctxRangePop.argtypes = []
_libroctx.roctxRangePop.restype = c_int

# roctx_range_id_t roctxRangeStartA(const char* message)
_libroctx.roctxRangeStartA.argtypes = [c_char_p]
_libroctx.roctxRangeStartA.restype = roctx_range_id_t

# void roctxRangeStop(roctx_range_id_t id)
_libroctx.roctxRangeStop.argtypes = [roctx_range_id_t]
_libroctx.roctxRangeStop.restype = None

# int roctxNameOsThread(const char* name)
_libroctx.roctxNameOsThread.argtypes = [c_char_p]
_libroctx.roctxNameOsThread.restype = c_int

# int roctxNameHipDevice(const char* name, int device_id)
_libroctx.roctxNameHipDevice.argtypes = [c_char_p, c_int]
_libroctx.roctxNameHipDevice.restype = c_int

# Version information - will be populated by CMake configure_file
version_info = {
    "version": "@PROJECT_VERSION@",
    "major": int("@PROJECT_VERSION_MAJOR@"),
    "minor": int("@PROJECT_VERSION_MINOR@"),
    "patch": int("@PROJECT_VERSION_PATCH@"),
    "git_revision": "@ROCPROFILER_SDK_GIT_REVISION@",
    "library_arch": "@CMAKE_LIBRARY_ARCHITECTURE@",
    "system_name": "@CMAKE_SYSTEM_NAME@",
    "system_processor": "@CMAKE_SYSTEM_PROCESSOR@",
    "system_version": "@CMAKE_SYSTEM_VERSION@",
    "compiler_id": "@CMAKE_CXX_COMPILER_ID@",
    "compiler_version": "@CMAKE_CXX_COMPILER_VERSION@",
    "rocm_version": "@rocm_version_FULL_VERSION@",
    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    "binding_type": "ctypes",
}


# Python wrapper functions providing a clean Pythonic API
def mark(msg):
    """
    Mark an event in any attached profiler.

    Args:
        msg (str): The message associated with the event.

    Example:
        >>> import roctx
        >>> roctx.mark("Starting computation")
    """
    if msg is not None:
        _libroctx.roctxMarkA(msg.encode("utf-8") if isinstance(msg, str) else msg)


def profilerPause(tid=0):
    """
    Request any currently running profiling tool to stop collecting data.

    Args:
        tid (int): Thread ID. Use 0 for all threads, or a specific thread ID.

    Returns:
        int: 0 on success, non-zero on failure or if not supported.

    Example:
        >>> import roctx
        >>> roctx.profilerPause()  # Pause profiling globally
    """
    return _libroctx.roctxProfilerPause(tid)


def profilerResume(tid=0):
    """
    Request any currently running profiling tool to resume collecting data.

    Args:
        tid (int): Thread ID. Use 0 for all threads, or a specific thread ID.

    Returns:
        int: 0 on success, non-zero on failure or if not supported.

    Example:
        >>> import roctx
        >>> roctx.profilerResume()  # Resume profiling globally
    """
    return _libroctx.roctxProfilerResume(tid)


def getThreadId():
    """
    Get the current thread ID compatible with ROCTx profiler.

    Returns:
        int: The current thread ID.

    Example:
        >>> import roctx
        >>> tid = roctx.getThreadId()
        >>> print(f"Current thread ID: {tid}")
    """
    tid = roctx_thread_id_t(0)
    _libroctx.roctxGetThreadId(ctypes.byref(tid))
    return tid.value


def rangePush(msg):
    """
    Start a new nested range.

    Nested ranges are stacked and local to the current CPU thread.

    Args:
        msg (str): The message associated with this range.

    Returns:
        int: The level this nested range is started at (0-based).

    Example:
        >>> import roctx
        >>> level = roctx.rangePush("Matrix multiplication")
        >>> # ... do work ...
        >>> roctx.rangePop()
    """
    return _libroctx.roctxRangePushA(msg.encode("utf-8") if isinstance(msg, str) else msg)


def rangePop():
    """
    Stop the current nested range.

    Returns:
        int: The level of the stopped range, or negative if no range was active.

    Example:
        >>> import roctx
        >>> roctx.rangePush("Computation")
        >>> # ... do work ...
        >>> roctx.rangePop()
    """
    return _libroctx.roctxRangePop()


def rangeStart(msg):
    """
    Start a process range.

    Start/stop ranges can be started and stopped in different threads.
    Each range is assigned a unique ID.

    Args:
        msg (str): The message associated with this range.

    Returns:
        int: The ID of the new range.

    Example:
        >>> import roctx
        >>> range_id = roctx.rangeStart("Async operation")
        >>> # ... do work (possibly in another thread) ...
        >>> roctx.rangeStop(range_id)
    """
    if msg is not None:
        return _libroctx.roctxRangeStartA(
            msg.encode("utf-8") if isinstance(msg, str) else msg
        )
    return None


def rangeStop(range_id=0):
    """
    Stop a process range.

    Args:
        range_id (int): The range ID returned from rangeStart().

    Example:
        >>> import roctx
        >>> range_id = roctx.rangeStart("Operation")
        >>> # ... do work ...
        >>> roctx.rangeStop(range_id)
    """
    if range_id is not None:
        _libroctx.roctxRangeStop(range_id)


def nameOsThread(name):
    """
    Label the current CPU OS thread with the provided name.

    This name will appear in profiler output where supported.

    Args:
        name (str): Name for the current OS thread.

    Returns:
        int: 0 on success, non-zero on failure or if not supported.

    Example:
        >>> import roctx
        >>> roctx.nameOsThread("Worker Thread 1")
    """
    return _libroctx.roctxNameOsThread(
        name.encode("utf-8") if isinstance(name, str) else name
    )


def nameHipDevice(name, device_id=0):
    """
    Label the given HIP device ID with the provided name.

    This name will appear in profiler output where supported.

    Args:
        name (str): Name for the specified device.
        device_id (int): HIP device ordinal.

    Returns:
        int: 0 on success, non-zero on failure or if not supported.

    Example:
        >>> import roctx
        >>> roctx.nameHipDevice("Primary GPU", 0)
    """
    return _libroctx.roctxNameHipDevice(
        name.encode("utf-8") if isinstance(name, str) else name, device_id
    )


# Import context decorators for convenience
from . import context_decorators
