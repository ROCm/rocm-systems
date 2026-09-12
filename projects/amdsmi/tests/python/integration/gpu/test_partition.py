#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
