#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU memory, VRAM, UMA carveout and TTM APIs."""

import unittest

import common.api_test as api
import common.common as common


class TestGpuMemory(api.ApiTestCase):
    def test_get_gpu_memory_total(self):
        self.both(
            "amdsmi_get_gpu_memory_total", self.handle, api.enum("mem_type", common.MEMORY_TYPES)
        )

    def test_get_gpu_memory_usage(self):
        self.both(
            "amdsmi_get_gpu_memory_usage", self.handle, api.enum("mem_type", common.MEMORY_TYPES)
        )

    def test_get_gpu_vram_usage(self):
        self.both("amdsmi_get_gpu_vram_usage", self.handle)

    def test_get_gpu_bad_page_info(self):
        self.both("amdsmi_get_gpu_bad_page_info", self.handle)

    def test_get_gpu_bad_page_threshold(self):
        self.both("amdsmi_get_gpu_bad_page_threshold", self.handle)

    def test_get_gpu_memory_reserved_pages(self):
        self.both("amdsmi_get_gpu_memory_reserved_pages", self.handle)

    def test_get_gpu_uma_carveout_info(self):
        self.both("amdsmi_get_gpu_uma_carveout_info", self.handle)

    def test_get_ttm_info(self):
        self.expect_only("amdsmi_get_ttm_info")

    def test_set_gpu_uma_carveout(self):
        self.reject_only("amdsmi_set_gpu_uma_carveout", self.handle, api.integer("option_index", 0))

    def test_set_ttm_pages_limit(self):
        # Zero pages is rejected by the library; the fixture's AMDSMI_DRY_RUN
        # keeps even an accepted value away from modprobe.d.
        self.reject_only(
            "amdsmi_set_ttm_pages_limit",
            api.Param("pages", ("1", 1), [("zero", 0), ("bad-type", api.BAD_INT)]),
        )

    def test_reset_ttm_pages_limit(self):
        # No invalid form and no payload; AMDSMI_DRY_RUN makes the write a no-op.
        self.expect_only("amdsmi_reset_ttm_pages_limit", validate=False)


if __name__ == "__main__":
    unittest.main()
