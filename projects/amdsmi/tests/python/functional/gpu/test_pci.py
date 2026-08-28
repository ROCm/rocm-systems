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

    def test_get_gpu_pci_replay_counter(self):
        self.common.print_func_name("")

        # TODO Check test_get_gpu_pci_replay_counter

        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_pci_replay_counter=amdsmi.amdsmi_get_gpu_pci_replay_counter
        )
        return

    def test_get_gpu_pci_throughput(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_pci_throughput=amdsmi.amdsmi_get_gpu_pci_throughput
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

    def test_get_link_topology_nearest(self):
        self.common.print_func_name("")
        self.common.Test_Per_GPU_With_One_Enum(
            amdsmi_get_link_topology_nearest=amdsmi.amdsmi_get_link_topology_nearest,
            link_type=common.LINK_TYPES,
        )
        return

    def test_get_minmax_bandwidth_between_processors(self):
        self.common.print_func_name("")
        self.common.Test_Per_GPU_With_GPU(
            amdsmi_get_minmax_bandwidth_between_processors=amdsmi.amdsmi_get_minmax_bandwidth_between_processors
        )
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
