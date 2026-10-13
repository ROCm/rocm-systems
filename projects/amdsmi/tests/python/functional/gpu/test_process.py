#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU process: compute process info, process isolation, local data cleanup."""

import unittest

import common.common as common
from common.common import amdsmi


class TestGpuProcess(unittest.TestCase):
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

    def test_clean_gpu_local_data(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_clean_gpu_local_data=amdsmi.amdsmi_clean_gpu_local_data)
        return

    def test_get_gpu_compute_process_gpus(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_NOT_COMPLETE:
            msg = (
                "\tSkipping test_get_gpu_compute_process_gpus as it is not complete (Inval Error)."
            )
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_compute_process_gpus=amdsmi.amdsmi_get_gpu_compute_process_gpus
        )
        return

    def test_get_gpu_compute_process_info(self):
        self.common.print_func_name("")
        self.common.Test_API(
            amdsmi_get_gpu_compute_process_info=amdsmi.amdsmi_get_gpu_compute_process_info
        )
        return

    def test_get_gpu_compute_process_info_by_pid(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_NOT_COMPLETE:
            msg = "\tSkipping test_get_gpu_compute_process_info_by_pid as it not complete (Device not found)."
            self.common.print(msg)
            self.skipTest(msg)

        # TODO pid = 0
        pid = 0

        self.common.Test_API(
            amdsmi_get_gpu_compute_process_info_by_pid=amdsmi.amdsmi_get_gpu_compute_process_info_by_pid,
            pid=pid,
        )
        return

    def test_get_gpu_process_isolation(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_process_isolation=amdsmi.amdsmi_get_gpu_process_isolation
        )
        return

    def test_get_gpu_process_list(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_process_list=amdsmi.amdsmi_get_gpu_process_list)
        return

    def test_set_gpu_process_isolation(self):
        self.common.print_func_name("")
        pisolates = [1, 0]
        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            for pisolate in pisolates:
                msg = f"\t### amdsmi_set_gpu_process_isolation(gpu={i}, pisolate={pisolate}):"
                try:
                    amdsmi.amdsmi_set_gpu_process_isolation(gpu, pisolate)
                    self.common.print(msg)
                    self.common.check_ret("", "", self.common.PASS)
                except amdsmi.AmdSmiLibraryException as e:
                    if self.common.check_ret(msg, e, self.common.PASS):
                        self.raise_exception = e
                self.common.print("")
        if self.raise_exception:
            raise self.raise_exception
        return

    # handle error_map list
