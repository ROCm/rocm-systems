#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU PCI: bandwidth, throughput, replay counter, PCIe info, link metrics, topology."""

import unittest

import common.common as common
from common.common import amdsmi


class TestGpuPci(unittest.TestCase):
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

    def test_get_gpu_pci_bandwidth(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_gpu_pci_bandwidth as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA)."
            self.common.print(msg)
            self.skipTest(msg)
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_pci_bandwidth=amdsmi.amdsmi_get_gpu_pci_bandwidth
        )
        return

    def test_get_link_metrics(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_link_metrics as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(amdsmi_get_link_metrics=amdsmi.amdsmi_get_link_metrics)
        return

    def test_get_pcie_info(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = (
                "\tSkipping test_get_pcie_info as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA)."
            )
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(amdsmi_get_pcie_info=amdsmi.amdsmi_get_pcie_info)
        return
