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
from . import context_decorators

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
]


def mark(msg):
    return libpyroctx.roctxMark(msg) if msg is not None else None


def profilerPause(tid=0):
    return libpyroctx.roctxProfilerPause(tid)


def profilerResume(tid=0):
    return libpyroctx.roctxProfilerResume(tid)


def getThreadId():
    return libpyroctx.roctxGetThreadId()


def rangePush(msg):
    return libpyroctx.roctxRangePush(msg)


def rangePop():
    return libpyroctx.roctxRangePop()


def rangeStart(msg):
    return libpyroctx.roctxRangeStart(msg) if msg is not None else None


def rangeStop(id=0):
    return libpyroctx.roctxRangeStop(id) if id is not None else None


def nameOsThread(name):
    return libpyroctx.roctxNameOsThread(name)


# def nameHsaAgent(name, agent):
#     return libpyroctx.roctxNameHsaAgent(name, agent)


def nameHipDevice(name, device_id=0):
    return libpyroctx.roctxNameHipDevice(name, device_id)


# def nameHipStream(name, stream):
#     return libpyroctx.roctxNameHipStream(name, stream)
