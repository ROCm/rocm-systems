###############################################################################
# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
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


from . import libpyroctx
from functools import wraps


class RoctxRange:
    """Provides decorators and context-manager for roctx range"""

    def __init__(self, msg=None):
        """Initialize with a message"""
        self.msg = msg

    def __call__(self, func):
        """Decorator"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            libpyroctx.roctxRangePush(self.msg)
            try:
                return func(*args, **kwargs)
            finally:
                libpyroctx.roctxRangePop()

        return wrapper

    def __enter__(self):
        """Context manager start function"""
        if self.msg is not None:
            self.a = libpyroctx.roctxRangePush(self.msg)
            return self.a
        return self

    def __exit__(self, exc_type, exc_value, tb):
        """Context manager stop function"""
        if self.msg is not None:
            libpyroctx.roctxRangePop()

        if exc_type is not None and exc_value is not None and tb is not None:
            import traceback

            traceback.print_exception(exc_type, exc_value, tb, limit=5)


class RoctxProfiler:
    """Provides decorators and context-manager for roctx profiler"""

    def __init__(self, tid=0):
        """Initialize with a tid"""
        self.tid = tid

    def __call__(self, func):
        """Decorator"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            libpyroctx.roctxProfilerResume(self.tid)
            try:
                return func(*args, **kwargs)
            finally:
                libpyroctx.roctxProfilerPause(self.tid)

        return wrapper

    def __enter__(self):
        """Context manager start function"""
        self.a = libpyroctx.roctxProfilerResume(self.tid)
        return self.a

    def __exit__(self, exc_type, exc_value, tb):
        """Context manager stop function"""

        libpyroctx.roctxProfilerPause(self.tid)

        if exc_type is not None and exc_value is not None and tb is not None:
            import traceback

            traceback.print_exception(exc_type, exc_value, tb, limit=5)
