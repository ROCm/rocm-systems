#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for CPU and CPU-core functional-test handle drivers."""

from unittest import TestCase
from unittest.mock import patch

from common.common import Common


class TestCpuHandleHelpers(TestCase):
    def _common(self):
        test_common = object.__new__(Common)
        test_common.processors = []
        test_common.bad_gpu = None
        test_common.PASS = "AMDSMI_STATUS_SUCCESS"
        test_common.FAIL = "AMDSMI_STATUS_INVAL"
        test_common.print_device_header = lambda _index: None
        test_common.print = lambda *_args: None
        test_common.check_ret = lambda *_args: None
        return test_common

    def test_cpu_helper_uses_cpu_socket_handles(self):
        test_common = self._common()
        seen = []

        def api(handle):
            seen.append(handle)

        with patch(
            "common.common.amdsmi.amdsmi_get_cpu_handles",
            return_value={"processor_handles": ["cpu-0", "cpu-1"]},
        ):
            test_common.Test_API_Per_CPU(cpu_api=api)

        self.assertEqual(seen, ["cpu-0", "cpu-1", None])

    def test_cpu_core_helper_uses_cpu_core_handles(self):
        test_common = self._common()
        seen = []

        def api(handle):
            seen.append(handle)

        with patch(
            "common.common.amdsmi.amdsmi_get_cpucore_handles", return_value=["core-0", "core-1"]
        ):
            test_common.Test_API_Per_CPU_Core(cpu_core_api=api)

        self.assertEqual(seen, ["core-0", "core-1", None])
