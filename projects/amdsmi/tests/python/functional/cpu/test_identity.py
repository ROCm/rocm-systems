#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU identity: family, model, handles, SMU firmware version, threads per core."""

import unittest

import common.common as common
from common.common import amdsmi


class TestCpuIdentity(unittest.TestCase):
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

    def test_get_cpu_family(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_cpu_family as it fails (IO Error)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API(amdsmi_get_cpu_family=amdsmi.amdsmi_get_cpu_family)
        return

    def test_get_cpu_model(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_cpu_model as it fails (IO Error)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API(amdsmi_get_cpu_model=amdsmi.amdsmi_get_cpu_model)
        return

    def test_get_threads_per_core(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_threads_per_core as it fails (IO Error)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API(amdsmi_get_threads_per_core=amdsmi.amdsmi_get_threads_per_core)
        return
