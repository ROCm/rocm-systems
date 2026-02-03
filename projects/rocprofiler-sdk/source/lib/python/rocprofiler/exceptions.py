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
Custom exception classes for rocprofiler.

This module defines exception classes used by the rocprofiler Python bindings.
"""


class RocprofilerError(Exception):
    """
    Base exception for rocprofiler errors.

    This exception is raised when rocprofiler-sdk operations fail.
    """

    pass


class ProfilerNotAvailableError(RocprofilerError):
    """
    Raised when rocprofiler is not available.

    This can happen if:
    - ROCm is not installed
    - No GPU devices are available
    - rocprofiler-sdk failed to initialize
    """

    pass


class CounterNotFoundError(RocprofilerError):
    """
    Raised when a requested counter is not found.

    This can happen if:
    - The counter name is misspelled
    - The counter is not supported on the target GPU
    """

    pass


class SessionError(RocprofilerError):
    """
    Raised when there is an error with a profiling session.

    This can happen if:
    - A session is already active
    - Failed to create profiling context
    - Failed to start/stop profiling
    """

    pass
