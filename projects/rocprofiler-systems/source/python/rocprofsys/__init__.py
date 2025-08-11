#!/usr/bin/env python@_VERSION@
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

from __future__ import absolute_import

__author__ = "AMD ROCm"
__copyright__ = "Copyright 2024, Advanced Micro Devices, Inc."
__license__ = "MIT"
__version__ = "@PROJECT_VERSION@"
__maintainer__ = "AMD ROCm"
__status__ = "Development"

"""
This submodule imports the timemory Python function profiler
"""

try:
    import os

    os.environ["ROCPROFSYS_PATH"] = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../..")
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
