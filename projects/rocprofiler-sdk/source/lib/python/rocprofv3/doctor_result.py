# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""Core value types for the ``rocprofv3 --doctor`` check framework.

Deliberately dependency-free (no ``dataclasses``, no ``enum``) so the module
imports cleanly on the Python 3.6 floor supported by the SDK.
"""

from __future__ import absolute_import

# How users invoke the doctor, for messages that tell them what to run.
# SYNC: COMMAND in source/libexec/rocprofiler-sdk/rocprofv3-doctor/rocprofv3-doctor.py
# and the flag dispatched in source/bin/rocprofv3.py main().
DOCTOR_FLAG = "--doctor"
DOCTOR_COMMAND = "rocprofv3 " + DOCTOR_FLAG

# status values -- the outcome of running a single check
STATUS_PASS = "pass"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"
STATUS_SKIP = "skip"
# the check itself broke (raised, or returned a non-Result): a doctor bug,
# which says nothing about whether the system is healthy
STATUS_ERROR = "error"

STATUS_VALUES = (STATUS_PASS, STATUS_WARN, STATUS_FAIL, STATUS_SKIP, STATUS_ERROR)

# severity values -- how much a failure of this check matters, declared
# statically at registration time (independent of the run-time status)
SEV_ERROR = "error"
SEV_WARNING = "warning"
SEV_INFO = "info"

SEVERITY_VALUES = (SEV_ERROR, SEV_WARNING, SEV_INFO)

# probe categories -- what running a check does to the machine, declared at
# registration so the cost and effects of a run can be published
PROBE_PASSIVE = "passive"  # reads files and the environment; read-only queries
PROBE_PROCESS = "process"  # starts ROCm code (loads libraries, runs tools)
PROBE_GPU = "gpu"  # initializes the GPU runtime or runs GPU work

PROBE_VALUES = (PROBE_PASSIVE, PROBE_PROCESS, PROBE_GPU)


class Result(object):
    """Outcome of a single check.

    Attributes:
        status: one of the ``STATUS_*`` constants.
        detail: human-readable explanation of why this status was produced.
        remediation: copy-pasteable command(s), or "" when there is nothing
            actionable for the user to do.
        data: dict of structured information surfaced in JSON output and in
            ``--verbose`` text output.
    """

    __slots__ = ("status", "detail", "remediation", "data")

    def __init__(self, status, detail="", remediation="", data=None):
        if status not in STATUS_VALUES:
            raise ValueError("invalid status {!r}".format(status))
        self.status = status
        self.detail = detail
        self.remediation = remediation
        # copy so that a caller mutating the dict it passed in (or the dict it
        # reads back out) cannot retroactively change a recorded result
        self.data = dict(data) if data else {}

    def to_dict(self):
        return {
            "status": self.status,
            "detail": self.detail,
            "remediation": self.remediation,
            "data": dict(self.data),
        }

    def __repr__(self):
        return "Result(status={!r}, detail={!r})".format(self.status, self.detail)


def make_pass(detail="", remediation="", data=None):
    return Result(STATUS_PASS, detail, remediation, data)


def make_warn(detail="", remediation="", data=None):
    return Result(STATUS_WARN, detail, remediation, data)


def make_fail(detail="", remediation="", data=None):
    return Result(STATUS_FAIL, detail, remediation, data)


def make_skip(detail="", remediation="", data=None):
    return Result(STATUS_SKIP, detail, remediation, data)
