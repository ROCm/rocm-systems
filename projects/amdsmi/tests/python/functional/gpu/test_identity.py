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
"""GPU device identity: ASIC info, board info, IDs, BDF, UUID, firmware, VBIOS, enumeration."""

import unittest

import common.common as common
from common.common import amdsmi


class TestGpuIdentity(unittest.TestCase):
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

    def test_get_fw_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_fw_info=amdsmi.amdsmi_get_fw_info)
        return

    def test_get_gpu_asic_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_asic_info=amdsmi.amdsmi_get_gpu_asic_info)
        return

    def test_get_gpu_bdf_id(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_bdf_id=amdsmi.amdsmi_get_gpu_bdf_id)
        return

    def test_get_gpu_board_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_board_info=amdsmi.amdsmi_get_gpu_board_info)
        return

    def test_get_gpu_device_bdf(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_device_bdf=amdsmi.amdsmi_get_gpu_device_bdf)
        return

    def test_get_gpu_device_uuid(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_device_uuid=amdsmi.amdsmi_get_gpu_device_uuid)
        return

    def test_get_gpu_driver_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_driver_info=amdsmi.amdsmi_get_gpu_driver_info)
        return

    def test_get_gpu_enumeration_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_enumeration_info=amdsmi.amdsmi_get_gpu_enumeration_info
        )
        return

    def test_get_gpu_id(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_id=amdsmi.amdsmi_get_gpu_id)
        return

    def test_get_gpu_kfd_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_kfd_info=amdsmi.amdsmi_get_gpu_kfd_info)
        return

    def test_get_gpu_revision(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_revision=amdsmi.amdsmi_get_gpu_revision)
        return

    def test_get_gpu_subsystem_id(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_subsystem_id=amdsmi.amdsmi_get_gpu_subsystem_id)
        return

    def test_get_gpu_subsystem_name(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_subsystem_name=amdsmi.amdsmi_get_gpu_subsystem_name
        )
        return

    def test_get_gpu_vbios_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_vbios_info=amdsmi.amdsmi_get_gpu_vbios_info)
        return

    def test_get_gpu_vendor_name(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_vendor_name=amdsmi.amdsmi_get_gpu_vendor_name)
        return

    def test_get_gpu_virtualization_mode(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_virtualization_mode=amdsmi.amdsmi_get_gpu_virtualization_mode
        )
        return

    def test_get_gpu_vram_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_vram_info=amdsmi.amdsmi_get_gpu_vram_info)
        return

    def test_get_gpu_vram_vendor(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_vram_vendor=amdsmi.amdsmi_get_gpu_vram_vendor)
        return

    def test_get_lib_version(self):
        self.common.print_func_name("")
        self.common.Test_API(amdsmi_get_lib_version=amdsmi.amdsmi_get_lib_version)
        return

    def test_get_processor_count_from_handles(self):
        self.common.print_func_name("")
        self.common.Test_API(
            amdsmi_get_processor_count_from_handles=amdsmi.amdsmi_get_processor_count_from_handles,
            processors=self.common.processors,
        )
        return

    # print data issues

    def test_get_processor_handles(self):
        self.common.print_func_name("")
        msg = "\t### amdsmi_get_processor_handles():"
        try:
            procs = amdsmi.amdsmi_get_processor_handles()
            self.common.print(msg, [id(addr) for addr in procs])
            self.assertGreaterEqual(len(self.common.processors), 1)
            self.assertLessEqual(len(self.common.processors), self.common.max_num_physical_devices)
            self.common.check_ret("", "", self.common.PASS)
        except amdsmi.AmdSmiLibraryException as e:
            if self.common.check_ret(msg, e, self.common.PASS):
                self.raise_exception = e
        self.common.print("")
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_processor_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_processor_info=amdsmi.amdsmi_get_processor_info)
        return

    def test_get_processor_type(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_processor_type=amdsmi.amdsmi_get_processor_type)
        return

    # data print issues
