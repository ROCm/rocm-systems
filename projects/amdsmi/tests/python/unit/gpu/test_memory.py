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
"""GPU memory, VRAM, UMA carveout and TTM APIs."""

import unittest

import common.common as common


class TestGpuMemory(common.ApiTestCase):
    def test_get_gpu_memory_total(self):
        self.both(
            "amdsmi_get_gpu_memory_total", self.handle, common.enum("mem_type", common.MEMORY_TYPES)
        )

    def test_get_gpu_memory_usage(self):
        self.both(
            "amdsmi_get_gpu_memory_usage", self.handle, common.enum("mem_type", common.MEMORY_TYPES)
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
        self.reject_only(
            "amdsmi_set_gpu_uma_carveout", self.handle, common.integer("option_index", 0)
        )

    def test_set_ttm_pages_limit(self):
        # Zero pages is rejected by the library; the fixture's AMDSMI_DRY_RUN
        # keeps even an accepted value away from modprobe.d.
        self.reject_only(
            "amdsmi_set_ttm_pages_limit",
            common.Param("pages", ("1", 1), [("zero", 0), ("bad-type", common.BAD_INT)]),
        )

    def test_reset_ttm_pages_limit(self):
        # No invalid form and no payload; AMDSMI_DRY_RUN makes the write a no-op.
        self.expect_only("amdsmi_reset_ttm_pages_limit", validate=False)


if __name__ == "__main__":
    unittest.main()
