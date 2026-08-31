#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU XGMI: XGMI info, link status, error status, topology, NUMA affinity, P2P."""

import unittest

import common.common as common
from common.common import amdsmi


class TestGpuXgmi(unittest.TestCase):
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

    def test_get_gpu_topo_numa_affinity(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_topo_numa_affinity=amdsmi.amdsmi_get_gpu_topo_numa_affinity
        )
        return

    def test_get_gpu_xgmi_link_status(self):
        self.common.print_func_name("")

        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_xgmi_link_status=amdsmi.amdsmi_get_gpu_xgmi_link_status
        )
        return

    def test_get_xgmi_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_xgmi_info=amdsmi.amdsmi_get_xgmi_info)
        return

    def test_gpu_xgmi_error_status(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_gpu_xgmi_error_status as it fails on MI300."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(
            amdsmi_gpu_xgmi_error_status=amdsmi.amdsmi_gpu_xgmi_error_status
        )
        return

    def test_is_P2P_accessible(self):
        self.common.print_func_name("")
        self.common.Test_Per_GPU_With_GPU(amdsmi_is_P2P_accessible=amdsmi.amdsmi_is_P2P_accessible)
        return

    def test_reset_gpu_xgmi_error(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_reset_gpu_xgmi_error as it fails on MI300."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(amdsmi_reset_gpu_xgmi_error=amdsmi.amdsmi_reset_gpu_xgmi_error)
        return

    def test_topo_get_link_type(self):
        self.common.print_func_name("")
        self.common.Test_Per_GPU_With_GPU(
            amdsmi_topo_get_link_type=amdsmi.amdsmi_topo_get_link_type
        )
        return

    def test_topo_get_link_weight(self):
        self.common.print_func_name("")
        self.common.Test_Per_GPU_With_GPU(
            amdsmi_topo_get_link_weight=amdsmi.amdsmi_topo_get_link_weight
        )
        return

    def test_topo_get_numa_node_number(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_topo_get_numa_node_number=amdsmi.amdsmi_topo_get_numa_node_number
        )
        return

    def test_topo_get_p2p_status(self):
        self.common.print_func_name("")
        self.common.Test_Per_GPU_With_GPU(
            amdsmi_topo_get_p2p_status=amdsmi.amdsmi_topo_get_p2p_status
        )
        return

    def test_topo_get_p2p_status_rejects_non_handle(self):
        self.common.print_func_name("")
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_topo_get_p2p_status("not_a_handle", "not_a_handle")
        return
