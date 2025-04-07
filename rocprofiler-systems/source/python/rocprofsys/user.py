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

from functools import wraps

from . import libpyrocprofsys
from .libpyrocprofsys import user as _libuser
from .libpyrocprofsys.user import start_trace
from .libpyrocprofsys.user import start_thread_trace
from .libpyrocprofsys.user import stop_trace
from .libpyrocprofsys.user import stop_thread_trace
from .libpyrocprofsys.user import push_region
from .libpyrocprofsys.user import pop_region

from .common import _initialize
from .common import _file


__all__ = [
    "region",
    "Region",
    "start_trace",
    "start_thread_trace",
    "stop_trace",
    "stop_thread_trace",
    "push_region",
    "pop_region",
]


class Region:
    """Provides decorators and context-manager for the rocprof-sys user-defined regions"""

    # static variable
    _counter = 0

    def __init__(self, _label):
        """Stores the label"""
        self._active = False
        self._label = _label
        self._count = 0
        self._file = _file() if Region._counter == 0 else None

    def __del__(self):
        """Stops"""
        self.stop()

    def start(self):
        """Start the region"""

        if not self._active:
            self._active = True
            self._count = Region._counter
            if self._file is not None:
                _initialize(self._file)
            Region._counter += 1
            _libuser.push_region(self._label)

    def stop(self):
        """Stop the region"""

        if self._active:
            Region._counter -= 1
            _count = Region._counter
            self._active = False
            if _count != self._count:
                raise RuntimeError(
                    f"{self._label} was not popped in the order it was pushed. Current stack number: {_count}, expected stack number: {self._count}"
                )
            _libuser.pop_region(self._label)

    def __call__(self, func):
        """Decorator"""

        @wraps(func)
        def function_wrapper(*args, **kwargs):
            # start the region
            self.start()
            # execute the wrapped function
            result = func(*args, **kwargs)
            # stop the region
            self.stop()
            return result

        return function_wrapper

    def __enter__(self, *args, **kwargs):
        """Context manager start function"""

        self.start()

    def __exit__(self, exec_type, exec_value, exec_tb):
        """Context manager stop function"""

        self.stop()

        if exec_type is not None and exec_value is not None and exec_tb is not None:
            import traceback

            traceback.print_exception(exec_type, exec_value, exec_tb, limit=5)

    def runcall(self, func, *args, **kwargs):
        """Profile a single function call"""

        try:
            self.start()
            return func(*args, **kwargs)
        finally:
            self.stop()


region = Region
