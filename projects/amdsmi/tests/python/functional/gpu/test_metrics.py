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
"""GPU metrics: activity, busy percent, cache, PM metrics, partition metrics, utilization."""

import unittest

import common.common as common
from common.common import amdsmi


class TestGpuMetrics(unittest.TestCase):
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

    def test_get_gpu_activity(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_gpu_activity as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(amdsmi_get_gpu_activity=amdsmi.amdsmi_get_gpu_activity)
        return

    def test_get_gpu_busy_percent(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_busy_percent=amdsmi.amdsmi_get_gpu_busy_percent)
        return

    def test_get_gpu_cache_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_cache_info=amdsmi.amdsmi_get_gpu_cache_info)
        return

    def test_get_gpu_metrics_header_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_metrics_header_info=amdsmi.amdsmi_get_gpu_metrics_header_info
        )
        return

    def test_get_gpu_metrics_info(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_gpu_metrics_info as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA)."
            self.common.print(msg)
            self.skipTest(msg)
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_metrics_info=amdsmi.amdsmi_get_gpu_metrics_info)
        return

    def test_get_gpu_partition_metrics_info(self):
        self.common.print_func_name("")
        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            try:
                msg = f"gpu({i}): "
                ret = amdsmi.amdsmi_get_gpu_partition_metrics_info(gpu)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except amdsmi.AmdSmiLibraryException as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
            self.common.print("")
        if self.raise_exception:
            raise self.raise_exception

    def test_get_gpu_pm_metrics_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_pm_metrics_info=amdsmi.amdsmi_get_gpu_pm_metrics_info
        )
        return

    def test_get_gpu_xcd_counter(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_gpu_xcd_counter as it fails (MI350, AMDSMI_STATUS_UNEXPECTED_DATA)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(amdsmi_get_gpu_xcd_counter=amdsmi.amdsmi_get_gpu_xcd_counter)
        return

    def test_get_utilization_count(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_gpu_event as it fails (Data Read Error)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_Per_GPU_With_One_Enum(
            amdsmi_get_utilization_count=amdsmi.amdsmi_get_utilization_count,
            utilization_counter_type=common.UTILIZATION_COUNTER_TYPES,
        )
        return

    def test_get_violation_status(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_violation_status=amdsmi.amdsmi_get_violation_status)
        return
