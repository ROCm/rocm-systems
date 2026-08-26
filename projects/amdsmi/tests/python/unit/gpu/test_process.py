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
"""GPU process and isolation APIs."""

import unittest

import common.common as common


class TestGpuProcess(common.ApiTestCase):
    def test_get_gpu_process_list(self):
        self.both("amdsmi_get_gpu_process_list", self.handle)

    def test_get_gpu_process_isolation(self):
        self.both("amdsmi_get_gpu_process_isolation", self.handle)

    def test_get_gpu_compute_process_info(self):
        self.expect_only("amdsmi_get_gpu_compute_process_info")

    def test_get_gpu_process_list_by_pid(self):
        handles = common.Param(
            "processor_handles",
            ("[gpu=0]", [self.common.processors[0]]),
            [("bad-type", common.BAD_SEQUENCE), ("bad-element", [common.BAD_HANDLE])],
        )
        self.both("amdsmi_get_gpu_process_list_by_pid", handles)

    def test_get_gpu_compute_process_info_by_pid(self):
        # A positive read needs a PID that currently holds a compute context.
        self.reject_only("amdsmi_get_gpu_compute_process_info_by_pid", common.integer("pid", 1))

    def test_get_gpu_compute_process_gpus(self):
        self.reject_only("amdsmi_get_gpu_compute_process_gpus", common.integer("pid", 1))

    def test_set_gpu_process_isolation(self):
        self.reject_only(
            "amdsmi_set_gpu_process_isolation", self.handle, common.integer("pisolate", 0)
        )

    def test_clean_gpu_local_data(self):
        self.reject_only("amdsmi_clean_gpu_local_data", self.handle)


if __name__ == "__main__":
    unittest.main()
