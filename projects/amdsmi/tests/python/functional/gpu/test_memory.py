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
"""GPU memory: UMA carveout info and set (dry run), TTM info and set (dry run)."""

import os
import unittest

import common.common as common
from common.common import amdsmi


class TestGpuMemory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common = common.Common(common.verbose)

    @classmethod
    def tearDownClass(cls):
        try:
            amdsmi.amdsmi_shut_down()
        except amdsmi.AmdSmiLibraryException:
            pass

    def setUp(self):
        self.raise_exception = None
        self.common.amdsmi_smart_init()
        self.common.processors = amdsmi.amdsmi_get_processor_handles()

    def tearDown(self):
        amdsmi.amdsmi_shut_down()

    def test_ttm_info(self):
        """Test TTM (GTT/shared memory) information retrieval"""
        self.common.print_func_name("")

        msg = "\t### amdsmi_get_ttm_info():"
        try:
            ttm_info = amdsmi.amdsmi_get_ttm_info()
            self.common.print(msg, ttm_info)
            self.common.check_ret("", "", self.common.PASS)
        except amdsmi.AmdSmiLibraryException as e:
            self.common.print(msg, e)
            self.assertEqual(
                e.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED, msg
            )
            return

        # Validate returned data structure
        self.assertIn("current_pages", ttm_info)

        # Validate that pages value is reasonable (> 0)
        self.assertGreater(ttm_info["current_pages"], 0)

        page_size = os.sysconf("SC_PAGESIZE")
        gb = (ttm_info["current_pages"] * page_size) / (1024**3)
        self.common.print(f"  TTM size: {gb:.2f} GB")
        return

    def test_ttm_set_dry_run(self):
        """Test TTM write operations in DRY_RUN mode"""
        self.common.print_func_name("")

        # Get current TTM info first
        msg = "\t### amdsmi_get_ttm_info():"
        try:
            ttm_info = amdsmi.amdsmi_get_ttm_info()
            self.common.print(msg, ttm_info)
            self.common.check_ret("", "", self.common.PASS)
        except amdsmi.AmdSmiLibraryException as e:
            self.common.print(msg, e)
            self.assertEqual(
                e.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED, msg
            )
            return

        # Enable DRY_RUN mode; ensure cleanup even if test fails
        os.environ["AMDSMI_DRY_RUN"] = "1"
        self.addCleanup(os.environ.pop, "AMDSMI_DRY_RUN", None)

        # Test setting TTM pages limit to current value
        msg = f"\t### amdsmi_set_ttm_pages_limit(pages={ttm_info['current_pages']}) (DRY_RUN):"
        try:
            ret = amdsmi.amdsmi_set_ttm_pages_limit(ttm_info["current_pages"])
            self.common.print(msg, ret)
            self.common.check_ret("", "", self.common.PASS)
        except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
            self.fail(f"Failed to set TTM to current value in DRY_RUN mode: {e}")

        # Test setting TTM to a different value
        test_pages = ttm_info["current_pages"] // 2
        if test_pages > 0:
            msg = f"\t### amdsmi_set_ttm_pages_limit(pages={test_pages}) (DRY_RUN):"
            try:
                ret = amdsmi.amdsmi_set_ttm_pages_limit(test_pages)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                self.fail(f"Failed to set TTM to different value in DRY_RUN mode: {e}")

        # Test setting TTM to 0 (should fail with AMDSMI_STATUS_INVAL)
        msg = "\t### amdsmi_set_ttm_pages_limit(pages=0) (DRY_RUN):"
        try:
            amdsmi.amdsmi_set_ttm_pages_limit(0)
            self.fail("Should have raised exception for pages=0")
        except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
            self.common.print(msg, e)
            self.assertEqual(e.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_INVAL, msg)

        # Test resetting TTM pages limit
        msg = "\t### amdsmi_reset_ttm_pages_limit() (DRY_RUN):"
        try:
            ret = amdsmi.amdsmi_reset_ttm_pages_limit()
            self.common.print(msg, ret)
            self.common.check_ret("", "", self.common.PASS)
        except amdsmi.AmdSmiLibraryException as e:
            self.fail(f"Failed to reset TTM in DRY_RUN mode: {e}")
        return

    def test_uma_carveout_info(self):
        """Test UMA carveout (VRAM) information retrieval"""
        self.common.print_func_name("")
        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), self.common.max_num_physical_devices)

        for i in range(0, len(processors)):
            self.common.print_device_header(i)
            bdf = amdsmi.amdsmi_get_gpu_device_bdf(processors[i])
            self.common.print(f"\n\n###Test Processor {i}, bdf: {bdf}")

            msg = f"\t### amdsmi_get_gpu_uma_carveout_info(gpu={i}):"
            try:
                uma_info = amdsmi.amdsmi_get_gpu_uma_carveout_info(processors[i])
                self.common.print(msg, uma_info)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                self.common.print(msg, e)
                self.assertEqual(
                    e.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED, msg
                )
                continue

            # Validate returned data structure
            self.assertIn("current_index", uma_info)
            self.assertIn("num_options", uma_info)
            self.assertIn("options", uma_info)

            # Validate that current_index is within valid range
            self.assertGreaterEqual(uma_info["current_index"], 0)
            self.assertLess(uma_info["current_index"], uma_info["num_options"])

            # Validate that we have at least one option
            self.assertGreater(uma_info["num_options"], 0)
            self.assertLessEqual(uma_info["num_options"], 16)

            # Validate options list
            self.assertEqual(len(uma_info["options"]), uma_info["num_options"])

            for j, opt in enumerate(uma_info["options"]):
                self.assertIn("index", opt)
                self.assertIn("description", opt)
                self.assertEqual(opt["index"], j)
                self.assertGreater(len(opt["description"]), 0)
                marker = "*" if opt["index"] == uma_info["current_index"] else " "
                self.common.print(f"  {marker} Option {opt['index']}: {opt['description']}")
        return

    def test_uma_carveout_set_dry_run(self):
        """Test UMA carveout write operations in DRY_RUN mode"""
        self.common.print_func_name("")
        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), self.common.max_num_physical_devices)

        # Enable DRY_RUN mode; ensure cleanup even if test fails
        os.environ["AMDSMI_DRY_RUN"] = "1"
        self.addCleanup(os.environ.pop, "AMDSMI_DRY_RUN", None)

        for i in range(0, len(processors)):
            self.common.print_device_header(i)
            bdf = amdsmi.amdsmi_get_gpu_device_bdf(processors[i])
            self.common.print(f"\n\n###Test Processor {i}, bdf: {bdf}")

            msg = f"\t### amdsmi_get_gpu_uma_carveout_info(gpu={i}):"
            try:
                uma_info = amdsmi.amdsmi_get_gpu_uma_carveout_info(processors[i])
                self.common.print(msg, uma_info)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                self.common.print(msg, e)
                self.assertEqual(
                    e.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED, msg
                )
                continue

            # Test setting to current value
            msg = f"\t### amdsmi_set_gpu_uma_carveout(gpu={i}, index={uma_info['current_index']}) (DRY_RUN):"
            try:
                ret = amdsmi.amdsmi_set_gpu_uma_carveout(processors[i], uma_info["current_index"])
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                self.fail(f"Failed to set UMA carveout to current value in DRY_RUN mode: {e}")

            # Test setting to different valid index if available
            if uma_info["num_options"] > 1:
                test_index = (uma_info["current_index"] + 1) % uma_info["num_options"]
                msg = f"\t### amdsmi_set_gpu_uma_carveout(gpu={i}, index={test_index}) (DRY_RUN):"
                try:
                    ret = amdsmi.amdsmi_set_gpu_uma_carveout(processors[i], test_index)
                    self.common.print(msg, ret)
                    self.common.check_ret("", "", self.common.PASS)
                except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                    self.fail(f"Failed to set UMA carveout to valid index in DRY_RUN mode: {e}")

            # Test setting to invalid index (should fail with AMDSMI_STATUS_INVAL)
            invalid_index = uma_info["num_options"] + 10
            msg = f"\t### amdsmi_set_gpu_uma_carveout(gpu={i}, index={invalid_index}) (DRY_RUN):"
            try:
                amdsmi.amdsmi_set_gpu_uma_carveout(processors[i], invalid_index)
                self.fail(f"Should have raised exception for invalid index {invalid_index}")
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                self.common.print(msg, e)
                self.assertEqual(e.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_INVAL, msg)
        return

    def test_get_gpu_bad_page_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_bad_page_info=amdsmi.amdsmi_get_gpu_bad_page_info
        )
        return

    def test_get_gpu_bad_page_threshold(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_bad_page_threshold=amdsmi.amdsmi_get_gpu_bad_page_threshold
        )
        return

    def test_get_gpu_memory_reserved_pages(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_memory_reserved_pages=amdsmi.amdsmi_get_gpu_memory_reserved_pages
        )
        return

    def test_get_gpu_memory_total(self):
        self.common.print_func_name("")
        self.common.Test_Per_GPU_With_One_Enum(
            amdsmi_get_gpu_memory_total=amdsmi.amdsmi_get_gpu_memory_total,
            memory_type=common.MEMORY_TYPES,
        )
        return

    def test_get_gpu_memory_usage(self):
        self.common.print_func_name("")
        self.common.Test_Per_GPU_With_One_Enum(
            amdsmi_get_gpu_memory_usage=amdsmi.amdsmi_get_gpu_memory_usage,
            memory_type=common.MEMORY_TYPES,
        )
        return

    def test_get_gpu_vram_usage(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_vram_usage=amdsmi.amdsmi_get_gpu_vram_usage)
        return
