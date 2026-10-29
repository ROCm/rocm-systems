#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for CPU benchmark availability decisions."""

from unittest import SkipTest, TestCase
from unittest.mock import Mock, patch

# Import the module, not the class: a TestCase class in this module's namespace
# would be collected by unit_tests.py, running the whole functional benchmark suite.
import functional.cpu.test_benchmark as cpu_benchmark


class TestCpuBenchmarkAvailability(TestCase):
    def test_cpu_absence_does_not_skip_benchmark_setup(self):
        test_case = cpu_benchmark.TestCpuBenchmark("test_performance_cpu_apb_disable")
        test_case.common = Mock()
        test_case.common.amdsmi_smart_init.return_value = None

        try:
            with patch.object(cpu_benchmark.TestCpuBenchmark, "_has_cpu", False, create=True):
                with patch(
                    "functional.cpu.test_benchmark.amdsmi.amdsmi_get_cpu_handles",
                    return_value={"processor_handles": []},
                ):
                    test_case.setUp()
        except SkipTest as exc:
            self.fail(f"CPU absence should not skip benchmark setup: {exc}")

        self.assertEqual(test_case.processors, [])

    def test_cpu_handles_are_sufficient_for_benchmark_availability(self):
        self.assertTrue(
            cpu_benchmark.TestCpuBenchmark._cpu_handles_available({"processor_handles": ["cpu-0"]})
        )

    def test_missing_cpu_handles_do_not_enable_benchmark_availability(self):
        self.assertFalse(
            cpu_benchmark.TestCpuBenchmark._cpu_handles_available({"processor_handles": []})
        )
