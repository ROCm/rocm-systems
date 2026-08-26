#!/usr/bin/env python3
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""GPU PTL (performance trace log) APIs."""

import unittest

import common.common as common

_VALID_FORMATS = [
    (member.name, member) for member in common.amdsmi.AmdSmiPtlData if member.name != "INVALID"
]


def _format_param(name):
    # AmdSmiPtlData.INVALID passes the isinstance guard but is rejected on its
    # own, so it belongs in the invalid set rather than the sweep.
    return common.Param(
        name,
        _VALID_FORMATS[0],
        [("bad-type", common.BAD_ENUM), ("INVALID", common.amdsmi.AmdSmiPtlData.INVALID)],
        sweep=_VALID_FORMATS,
    )


class TestGpuPtl(common.ApiTestCase):
    def test_get_gpu_ptl_state(self):
        self.both("amdsmi_get_gpu_ptl_state", self.handle)

    def test_get_gpu_ptl_formats(self):
        self.both("amdsmi_get_gpu_ptl_formats", self.handle)

    def test_set_gpu_ptl_state(self):
        # state is a ctypes c_bool: every value coerces to True, so driving it
        # here would enable tracing instead of being refused. Handle only.
        self.reject_only("amdsmi_set_gpu_ptl_state", self.handle)

    def test_set_gpu_ptl_formats(self):
        self.reject_only(
            "amdsmi_set_gpu_ptl_formats", self.handle, _format_param("fmt1"), _format_param("fmt2")
        )


if __name__ == "__main__":
    unittest.main()
