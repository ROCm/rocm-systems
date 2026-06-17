#!/usr/bin/env python@_VERSION@
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import absolute_import

__author__ = "AMD ROCm"
__copyright__ = "Copyright 2025, Advanced Micro Devices, Inc."
__license__ = "MIT"
__version__ = "@PROJECT_VERSION@"
__maintainer__ = "AMD ROCm"
__status__ = "Development"

"""
This submodule imports the timemory Python function profiler
"""

try:
    import os
    from pathlib import Path

    # Set up ROCPROFSYS environment variables
    rocprofsys_root = Path(__file__).resolve().parents[4]
    os.environ.update(
        {
            "ROCPROFSYS_ROOT": str(rocprofsys_root),
            "ROCPROFSYS_PATH": str(rocprofsys_root / "lib"),
            "ROCPROFSYS_SCRIPT_PATH": str(
                rocprofsys_root / "libexec/rocprofiler-systems"
            ),
        }
    )

    from .libpyrocprofsys import coverage
    from . import user
    from .profiler import Profiler, FakeProfiler
    from .libpyrocprofsys.profiler import (
        profiler_function,
        profiler_init,
        profiler_finalize,
    )
    from .libpyrocprofsys import initialize
    from .libpyrocprofsys import finalize
    from .libpyrocprofsys import is_initialized
    from .libpyrocprofsys import is_finalized
    from .libpyrocprofsys.profiler import config as Config

    config = Config
    profile = Profiler
    noprofile = FakeProfiler

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
        "user",
    ]

    import atexit

    def _finalize_at_exit():
        if not is_finalized():
            finalize()

    atexit.register(_finalize_at_exit)

except Exception as e:
    print("{}".format(e))
