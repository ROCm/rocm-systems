#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
