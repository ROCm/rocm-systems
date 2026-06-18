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
"""Functional tests for fabric (UALoE) telemetry and fabric info."""

import unittest

import common.common as common
from common.common import amdsmi

# Union of all known fabric telemetry category masks.
ALL_CATEGORIES = amdsmi.amdsmi_wrapper.AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_ALL_KNOWN


class TestIfoeTelemetry(unittest.TestCase):
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

    def test_fabric_telemetry(self):
        """Exercise alloc/get/free + amdsmi_fabric_telem_id_to_string round-trip.

        On systems without an IFoE driver the API returns DRIVER_NOT_LOADED or
        NOT_SUPPORTED; treat either as a pass condition rather than a hard
        failure.
        """
        self.common.print_func_name("")

        expected_unavailable_statuses = (
            amdsmi.AmdSmiStatus.DRIVER_NOT_LOADED,
            amdsmi.AmdSmiStatus.NOT_SUPPORTED,
        )

        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)

            msg = f"\t### amdsmi_get_fabric_telemetry_data(gpu={i}):"
            try:
                telem = amdsmi.amdsmi_get_fabric_telemetry_data(gpu, ALL_CATEGORIES)
                self.common.print(msg, telem)
                self.common.check_ret("", "", self.common.PASS)
                self.assertIsInstance(telem, list)
                for category in telem:
                    self.assertIn("category", category)
                    self.assertIn("instances", category)
                    for instance in category["instances"]:
                        for item in instance["items"]:
                            self.assertIn("id", item)
                            self.assertIn("name", item)
                            self.assertIsInstance(item["name"], str)
                            # Every driver-provided id must resolve to a known
                            # name; a fallback of "UNKNOWN" indicates a gap.
                            self.assertNotEqual(item["name"], "UNKNOWN")
            except amdsmi.AmdSmiLibraryException as e:
                if e.get_error_code() in expected_unavailable_statuses:
                    self.common.print(msg, f"skipped: {e}")
                    self.common.check_ret("", "", self.common.PASS)
                elif self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
            except amdsmi.AmdSmiParameterException as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e

            msg = f"\t### amdsmi_get_gpu_fabric_info(gpu={i}):"
            try:
                info = amdsmi.amdsmi_get_gpu_fabric_info(gpu)
                self.common.print(msg, info)
                self.common.check_ret("", "", self.common.PASS)
                self.assertIsInstance(info, dict)
                self.assertIn("version", info)
                self.assertIn("fabric_type", info)
            except amdsmi.AmdSmiLibraryException as e:
                if e.get_error_code() in expected_unavailable_statuses:
                    self.common.print(msg, f"skipped: {e}")
                    self.common.check_ret("", "", self.common.PASS)
                elif self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
            except amdsmi.AmdSmiParameterException as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e

        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_fabric_telemetry_data(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_fabric_telemetry_data=amdsmi.amdsmi_get_fabric_telemetry_data,
            category_mask=ALL_CATEGORIES,
        )
        return

    def test_get_gpu_fabric_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_fabric_info=amdsmi.amdsmi_get_gpu_fabric_info)
        return


if __name__ == "__main__":
    unittest.main()
