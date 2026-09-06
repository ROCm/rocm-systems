#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU process and isolation APIs."""

import unittest

import common.api_test as api


class TestGpuProcess(api.ApiTestCase):
    def test_get_gpu_process_list(self):
        self.both("amdsmi_get_gpu_process_list", self.handle)

    def test_get_gpu_process_isolation(self):
        self.both("amdsmi_get_gpu_process_isolation", self.handle)

    def test_get_gpu_compute_process_info(self):
        self.expect_only("amdsmi_get_gpu_compute_process_info")

    def test_get_gpu_process_list_by_pid(self):
        handles = api.Param(
            "processor_handles",
            ("[gpu=0]", [self.common.processors[0]]),
            [("bad-type", api.BAD_SEQUENCE), ("bad-element", [api.BAD_HANDLE])],
        )
        self.both("amdsmi_get_gpu_process_list_by_pid", handles)

    def test_get_gpu_compute_process_info_by_pid(self):
        # A positive read needs a PID that currently holds a compute context.
        self.reject_only("amdsmi_get_gpu_compute_process_info_by_pid", api.integer("pid", 1))

    def test_get_gpu_compute_process_gpus(self):
        self.reject_only("amdsmi_get_gpu_compute_process_gpus", api.integer("pid", 1))

    def test_set_gpu_process_isolation(self):
        self.reject_only(
            "amdsmi_set_gpu_process_isolation", self.handle, api.integer("pisolate", 0)
        )

    def test_clean_gpu_local_data(self):
        self.reject_only("amdsmi_clean_gpu_local_data", self.handle)


if __name__ == "__main__":
    unittest.main()
