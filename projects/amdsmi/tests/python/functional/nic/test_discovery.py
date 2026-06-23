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
"""NIC and switch device discovery: BDF and device ID enumeration."""

import unittest

import common.common as common
from common.common import amdsmi


class TestNicDiscovery(unittest.TestCase):
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

    def test_nic_bdf_device_id(self):
        self.common.print_func_name("")
        common.Common._skip_if_missing(
            self,
            [
                "amdsmi_get_nic_processor_handles",
                "amdsmi_get_nic_info",
                "amdsmi_get_nic_device_uuid",
            ],
        )
        processors = amdsmi.amdsmi_get_nic_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), self.common.max_num_physical_devices)
        for i in range(0, len(processors)):
            bdf = ""
            nic_info = amdsmi.amdsmi_get_nic_info(processors[i])
            if nic_info:
                bdf = nic_info["bdf"]
            print(f"\n\n###Test nic Processor {i}, bdf: {bdf}")
            print("\n###Test amdsmi_get_processor_handle_from_bdf\n")
            processor = amdsmi.amdsmi_get_processor_handle_from_bdf(bdf)
            print("\n###Test amdsmi_get_nic_device_uuid\n")
            uuid = amdsmi.amdsmi_get_nic_device_uuid(processor)
            print(f"  uuid is: {uuid}")
        print()
        return

    def test_switch_bdf_device_id(self):
        self.common.print_func_name("")
        common.Common._skip_if_missing(
            self,
            [
                "amdsmi_get_switch_processor_handles",
                "amdsmi_get_switch_device_bdf",
                "amdsmi_get_device_id",
            ],
        )
        processors = amdsmi.amdsmi_get_switch_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            bdf = amdsmi.amdsmi_get_switch_device_bdf(processors[i])
            print(f"\n\n###Test switch Processor {i}, bdf: {bdf}")
            print("\n###Test amdsmi_get_processor_handle_from_bdf\n")
            processor = amdsmi.amdsmi_get_processor_handle_from_bdf(bdf)
            print("\n###Test amdsmi_get_device_id\n")
            device_id = amdsmi.amdsmi_get_device_id(processor)
            print(f"  device_id is: {device_id}")
        print()
