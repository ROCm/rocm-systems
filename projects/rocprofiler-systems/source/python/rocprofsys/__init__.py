#!/usr/bin/env python@_VERSION@
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import absolute_import

__author__ = "AMD ROCm"
__copyright__ = "Copyright 2026, Advanced Micro Devices, Inc."
__license__ = "MIT"
__version__ = "@PROJECT_VERSION@"
__maintainer__ = "AMD ROCm"
__status__ = "Development"

"""
This submodule imports the timemory Python function profiler
"""

try:
    import os
    import sys
    import importlib
    from pathlib import Path

    # Set up ROCPROFSYS environment variables. The package sits at
    # <root>/@CMAKE_INSTALL_PYTHONDIR@/rocprofsys in both the build and install trees;
    # export nothing rather than a wrong path if the package was relocated.
    rocprofsys_root = Path(__file__).resolve().parents[4]
    rocprofsys_libdir = rocprofsys_root / "@CMAKE_INSTALL_LIBDIR@"
    if (rocprofsys_libdir / "librocprof-sys-dl.so").exists():
        os.environ.update(
            {
                "ROCPROFSYS_ROOT": str(rocprofsys_root),
                "ROCPROFSYS_PATH": str(rocprofsys_libdir),
                "ROCPROFSYS_SCRIPT_PATH": str(
                    rocprofsys_root / "libexec/rocprofiler-systems"
                ),
            }
        )

    __all__ = [
        "initialize",
        "finalize",
        "is_initialized",
        "is_finalized",
        "Profiler",
        "Config",
        "FakeProfiler",
        "profiler_function",
        "profiler_init",
        "profiler_finalize",
        "config",
        "profile",
        "noprofile",
        "coverage",
    ]

    def _load_profiler_bindings():
        """Load the profiler API after command-line configuration is available."""
        bindings = importlib.import_module(".libpyrocprofsys", __name__)
        profiler = importlib.import_module(".profiler", __name__)
        native_profiler = importlib.import_module(".libpyrocprofsys.profiler", __name__)

        globals().update(
            {
                "coverage": bindings.coverage,
                "Profiler": profiler.Profiler,
                "FakeProfiler": profiler.FakeProfiler,
                "profiler_function": native_profiler.profiler_function,
                "profiler_init": native_profiler.profiler_init,
                "profiler_finalize": native_profiler.profiler_finalize,
                "initialize": bindings.initialize,
                "finalize": bindings.finalize,
                "is_initialized": bindings.is_initialized,
                "is_finalized": bindings.is_finalized,
                "Config": native_profiler.config,
                "config": native_profiler.config,
                "profile": profiler.Profiler,
                "noprofile": profiler.FakeProfiler,
            }
        )

    def __getattr__(name):
        if name not in __all__:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

        _load_profiler_bindings()
        return globals()[name]

    import atexit

    def _finalize_at_exit():
        bindings = sys.modules.get(f"{__name__}.libpyrocprofsys")
        if (
            bindings is not None
            and bindings.is_initialized()
            and not bindings.is_finalized()
        ):
            bindings.finalize()

    atexit.register(_finalize_at_exit)

except Exception as e:
    print("{}".format(e))
