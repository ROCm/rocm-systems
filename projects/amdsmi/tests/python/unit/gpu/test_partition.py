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
"""GPU partition and reset APIs."""

import unittest

import common.api_test as api
import common.common as common


class TestGpuPartition(api.ApiTestCase):
    def test_get_gpu_compute_partition(self):
        self.both("amdsmi_get_gpu_compute_partition", self.handle)

    def test_get_gpu_compute_partition_mem_alloc_mode(self):
        self.both("amdsmi_get_gpu_compute_partition_mem_alloc_mode", self.handle)

    def test_get_gpu_accelerator_partition_mem_alloc_mode(self):
        self.both("amdsmi_get_gpu_accelerator_partition_mem_alloc_mode", self.handle)

    def test_get_gpu_accelerator_partition_profile(self):
        self.both("amdsmi_get_gpu_accelerator_partition_profile", self.handle)

    def test_get_gpu_accelerator_partition_profile_config(self):
        self.both("amdsmi_get_gpu_accelerator_partition_profile_config", self.handle)

    def test_get_gpu_memory_partition(self):
        self.both("amdsmi_get_gpu_memory_partition", self.handle)

    def test_get_gpu_memory_partition_config(self):
        self.both("amdsmi_get_gpu_memory_partition_config", self.handle)

    def test_set_gpu_compute_partition(self):
        self.reject_only(
            "amdsmi_set_gpu_compute_partition",
            self.handle,
            api.enum("compute_partition", common.COMPUTE_PARTITION_TYPES),
        )

    def test_set_gpu_compute_partition_mem_alloc_mode(self):
        self.reject_only(
            "amdsmi_set_gpu_compute_partition_mem_alloc_mode",
            self.handle,
            api.enum("mode", common.ACCELERATOR_PARTITION_MEM_ALLOC_MODE_TYPES),
        )

    def test_set_gpu_accelerator_partition_mem_alloc_mode(self):
        self.reject_only(
            "amdsmi_set_gpu_accelerator_partition_mem_alloc_mode",
            self.handle,
            api.enum("mode", common.ACCELERATOR_PARTITION_MEM_ALLOC_MODE_TYPES),
        )

    def test_set_gpu_accelerator_partition_profile(self):
        self.reject_only(
            "amdsmi_set_gpu_accelerator_partition_profile",
            self.handle,
            api.integer("profile_index", 0),
        )

    def test_set_gpu_memory_partition(self):
        self.reject_only(
            "amdsmi_set_gpu_memory_partition",
            self.handle,
            api.enum("memory_partition", common.MEMORY_PARTITION_TYPES),
        )

    def test_set_gpu_memory_partition_mode(self):
        self.reject_only(
            "amdsmi_set_gpu_memory_partition_mode",
            self.handle,
            api.enum("memory_partition", common.MEMORY_PARTITION_TYPES),
        )

    def test_reset_gpu(self):
        self.reject_only("amdsmi_reset_gpu", self.handle)


if __name__ == "__main__":
    unittest.main()
