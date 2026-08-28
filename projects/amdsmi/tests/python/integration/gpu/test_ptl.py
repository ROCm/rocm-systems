#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU PTL (performance trace log) APIs."""

import unittest

import common.api_test as api
import common.common as common

_VALID_FORMATS = [
    (member.name, member) for member in common.amdsmi.AmdSmiPtlData if member.name != "INVALID"
]


def _format_param(name):
    # AmdSmiPtlData.INVALID passes the isinstance guard but is rejected on its
    # own, so it belongs in the invalid set rather than the sweep.
    return api.Param(
        name,
        _VALID_FORMATS[0],
        [("bad-type", api.BAD_ENUM), ("INVALID", common.amdsmi.AmdSmiPtlData.INVALID)],
        sweep=_VALID_FORMATS,
    )


class TestGpuPtl(api.ApiTestCase):
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
