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
Context managers and decorators for ROCTx profiling.
"""

from functools import wraps

# Import from parent module (avoid circular import)
import sys

_module = sys.modules[__name__.rsplit(".", 1)[0]]


class RoctxRange:
    """
    Provides decorators and context-manager for roctx range.

    Can be used as a decorator or context manager to automatically
    push/pop ROCTx ranges around code blocks.

    Examples:
        As a decorator:
        >>> @RoctxRange("my_function")
        ... def my_function():
        ...     pass

        As a context manager:
        >>> with RoctxRange("my_operation"):
        ...     # do work
        ...     pass
    """

    def __init__(self, msg=None):
        """
        Initialize with a message.

        Args:
            msg (str): The message to associate with this range.
        """
        self.msg = msg

    def __call__(self, func):
        """
        Decorator usage.

        Args:
            func: The function to wrap.

        Returns:
            The wrapped function.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            _module.rangePush(self.msg)
            try:
                return func(*args, **kwargs)
            finally:
                _module.rangePop()

        return wrapper

    def __enter__(self):
        """
        Context manager start function.

        Returns:
            The range level.
        """
        if self.msg is not None:
            self.a = _module.rangePush(self.msg)
            return self.a
        return self

    def __exit__(self, exc_type, exc_value, tb):
        """
        Context manager stop function.

        Pops the range and prints exception info if an exception occurred.
        """
        if self.msg is not None:
            _module.rangePop()

        if exc_type is not None and exc_value is not None and tb is not None:
            import traceback

            traceback.print_exception(exc_type, exc_value, tb, limit=5)


class RoctxProfiler:
    """
    Provides decorators and context-manager for roctx profiler control.

    Can be used as a decorator or context manager to automatically
    pause/resume the profiler around code blocks.

    Examples:
        As a decorator:
        >>> @RoctxProfiler()
        ... def my_function():
        ...     # This function will be profiled
        ...     pass

        As a context manager:
        >>> with RoctxProfiler():
        ...     # This code will be profiled
        ...     pass
    """

    def __init__(self, tid=0):
        """
        Initialize with a thread ID.

        Args:
            tid (int): Thread ID (0 for all threads).
        """
        self.tid = tid

    def __call__(self, func):
        """
        Decorator usage.

        Args:
            func: The function to wrap.

        Returns:
            The wrapped function.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            _module.profilerResume(self.tid)
            try:
                return func(*args, **kwargs)
            finally:
                _module.profilerPause(self.tid)

        return wrapper

    def __enter__(self):
        """
        Context manager start function.

        Returns:
            The result of profilerResume.
        """
        self.a = _module.profilerResume(self.tid)
        return self.a

    def __exit__(self, exc_type, exc_value, tb):
        """
        Context manager stop function.

        Pauses the profiler and prints exception info if an exception occurred.
        """
        _module.profilerPause(self.tid)

        if exc_type is not None and exc_value is not None and tb is not None:
            import traceback

            traceback.print_exception(exc_type, exc_value, tb, limit=5)
