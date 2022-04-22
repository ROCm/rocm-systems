#!/usr/bin/env python@_VERSION@
# MIT License
#
# Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

__author__ = "AMD Research"
__copyright__ = "Copyright 2022, Advanced Micro Devices, Inc."
__license__ = "MIT"
__version__ = "@PROJECT_VERSION@"
__maintainer__ = "AMD Research"
__status__ = "Development"

"""
This submodule imports the timemory Python function profiler
"""

try:
    from .profiler import Profiler, FakeProfiler
    from .libpyomnitrace.profiler import (
        profiler_function,
        profiler_init,
        profiler_finalize,
    )
    from .libpyomnitrace import initialize
    from .libpyomnitrace import finalize
    from .libpyomnitrace import is_initialized
    from .libpyomnitrace import is_finalized
    from .libpyomnitrace.profiler import config as Config

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
    ]

    import atexit

    def _finalize_at_exit():
        if not is_finalized():
            finalize()

    atexit.register(_finalize_at_exit)

except Exception as e:
    print("{}".format(e))
