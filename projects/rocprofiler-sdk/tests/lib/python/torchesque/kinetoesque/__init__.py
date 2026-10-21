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
kinetoesque -- Profiler submodule for torchesque.

Provides start/stop profiling controls backed by the kinetoesque
rocprofiler-sdk tool library, loaded via ctypes.
"""

import ctypes
import os
import sys

_lib = None
_initialized = False


def _find_library():
    """Locate the kinetoesque shared library."""
    # Check environment variable first
    lib_path = os.environ.get("KINETOESQUE_LIBRARY_PATH")
    if lib_path and os.path.isfile(lib_path):
        return lib_path

    # Search in common locations relative to this file and the build directory
    search_dirs = []

    # Relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs.append(script_dir)
    search_dirs.append(
        os.path.join(script_dir, "..", "..", "..", "..", "rocprofiler-sdk")
    )
    search_dirs.append(os.path.join(script_dir, "..", "..", "..", ".."))

    # From environment
    if "ROCPROFILER_SDK_TEST_LIB_DIR" in os.environ:
        search_dirs.append(os.environ["ROCPROFILER_SDK_TEST_LIB_DIR"])

    # LD_LIBRARY_PATH
    if "LD_LIBRARY_PATH" in os.environ:
        search_dirs.extend(os.environ["LD_LIBRARY_PATH"].split(":"))

    for d in search_dirs:
        candidate = os.path.join(d, "libkinetoesque-tool.so")
        if os.path.isfile(candidate):
            return os.path.realpath(candidate)

    return None


def _load_library():
    """Load the kinetoesque shared library via ctypes."""
    global _lib

    if _lib is not None:
        return _lib

    lib_path = _find_library()
    if lib_path is None:
        raise RuntimeError(
            "Could not find libkinetoesque-tool.so. "
            "Set KINETOESQUE_LIBRARY_PATH or ROCPROFILER_SDK_TEST_LIB_DIR."
        )

    print(f"[kinetoesque] Loading library: {lib_path}", file=sys.stderr)
    _lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_LOCAL)

    # Set up function signatures
    _lib.kinetoesque_init.restype = ctypes.c_int
    _lib.kinetoesque_init.argtypes = []

    _lib.kinetoesque_start.restype = ctypes.c_int
    _lib.kinetoesque_start.argtypes = []

    _lib.kinetoesque_stop.restype = ctypes.c_int
    _lib.kinetoesque_stop.argtypes = []

    _lib.kinetoesque_finalize.restype = ctypes.c_int
    _lib.kinetoesque_finalize.argtypes = []

    _lib.kinetoesque_set_output_file.restype = None
    _lib.kinetoesque_set_output_file.argtypes = [ctypes.c_char_p]

    _lib.kinetoesque_get_trace_count.restype = ctypes.c_int
    _lib.kinetoesque_get_trace_count.argtypes = []

    return _lib


def init():
    """Initialize the kinetoesque profiler tool via rocprofiler_force_configure."""
    global _initialized
    lib = _load_library()
    ret = lib.kinetoesque_init()
    if ret != 0:
        raise RuntimeError(f"kinetoesque_init() failed with code {ret}")
    _initialized = True


def start_profiler():
    """Start active trace collection."""
    if not _initialized:
        init()
    lib = _load_library()
    ret = lib.kinetoesque_start()
    if ret != 0:
        raise RuntimeError(f"kinetoesque_start() failed with code {ret}")


def stop_profiler():
    """Stop active trace collection."""
    lib = _load_library()
    ret = lib.kinetoesque_stop()
    if ret != 0:
        raise RuntimeError(f"kinetoesque_stop() failed with code {ret}")


def finalize():
    """Finalize and write output."""
    lib = _load_library()
    ret = lib.kinetoesque_finalize()
    if ret != 0:
        raise RuntimeError(f"kinetoesque_finalize() failed with code {ret}")


def set_output_file(filename):
    """Set the output filename for the JSON trace."""
    lib = _load_library()
    lib.kinetoesque_set_output_file(filename.encode("utf-8"))


def get_trace_count():
    """Get the number of trace records collected so far."""
    lib = _load_library()
    return lib.kinetoesque_get_trace_count()
