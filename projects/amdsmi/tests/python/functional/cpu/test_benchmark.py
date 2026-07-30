#!/usr/bin/env python3
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import json
import statistics
import time
import unittest

import common.common as common
from common.common import FAIL, PASS, amdsmi


class TestCpuBenchmark(unittest.TestCase):
    """
    Standalone Performance testing class for AMDSMI Python APIs.

    This class does NOT inherit from TestAmdSmiPython to avoid running all regular tests.
    It only runs performance-specific tests.
    """

    @classmethod
    def setUpClass(cls):
        # Shared Common instance for logging parity with the unit/integration
        # tests (print_func_name, etc.). Created once per class, as they do.
        cls.common = common.Common(common.verbose)
        # Detect CPU presence once. Like the regular CPU tests, the CPU perf suite
        # is skipped on machines without a CPU instead of exercising CPU APIs, which
        # the C library logs noisily ("No CPU sockets on machine", "Failed to get
        # cpu family", ...) when no CPU/ESMI driver is present.
        cls._has_cpu = False
        try:
            cls.common.amdsmi_smart_init()
            cpus = amdsmi.amdsmi_get_cpu_handles()
            # amdsmi_get_cpu_handles() returns {"cpu_count", "processor_handles"};
            # use the handle list (an empty list means no CPU sockets present).
            handles = cpus["processor_handles"] if isinstance(cpus, dict) else cpus
            # Require a *working* CPU monitoring driver, not just a CPU handle (which
            # exists on any AMD host even without the ESMI/HSMP driver). Without the
            # driver the CPU APIs log C-library errors, so skip the suite — mirroring
            # the regular CPU tests, which skip when the driver is absent.
            if handles:
                amdsmi.amdsmi_get_cpu_hsmp_driver_version(handles[0])
                cls._has_cpu = True
        except Exception:
            cls._has_cpu = False
        finally:
            try:
                amdsmi.amdsmi_shut_down()
            except Exception:
                pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Do NOT initialize amdsmi or enumerate hardware here. __init__ runs at
        # test-discovery time (including for `--list`), and hardware init must
        # not happen during collection. setUp() performs the real amdsmi_init()
        # before each test.
        # Performance test configuration
        self.perf_iterations = 11  # Number of iterations for each performance test
        self.perf_warmup_iterations = 3  # Number of warmup iterations
        self.perf_results = {}  # Store performance results

    def setUp(self):
        """Setup for performance tests - minimal setup just for performance testing."""
        if not self.__class__._has_cpu:
            self.skipTest("No AMD CPU present on this machine.")

        self.time = time
        self.statistics = statistics

        # Add skip flags for consistency with original tests
        self.TODO_SKIP_FAIL = False
        self.TODO_SKIP_NOT_COMPLETE = False

        # Use global constants matching original test file
        self.PASS = PASS
        self.FAIL = FAIL

        # Initialize AMDSMI like the unit/integration tests: smart-init auto-detects
        # available drivers, so (unlike a forced amdsmi_init(INIT_AMD_CPUS)) it does
        # not make the C library print "ESMI Not initialized, drivers not found" on
        # machines without a CPU/ESMI driver.
        try:
            self.common.amdsmi_smart_init()
            self.processors = (
                amdsmi.amdsmi_get_processor_handles()
                if hasattr(amdsmi, "amdsmi_get_processor_handles")
                else []
            )
        except Exception as e:
            self.common.print(f"Warning: Failed to initialize AMDSMI: {e}")
            self.processors = []

    def tearDown(self):
        """Cleanup after performance tests."""
        try:
            if hasattr(amdsmi, "amdsmi_shut_down"):
                amdsmi.amdsmi_shut_down()
        except Exception:
            pass  # Ignore cleanup errors

    def _log_test_start(self, api_name, device_type, device_id, **kwargs):
        """Helper method to log the start of a test."""
        extra_info = " ".join([f"{k}={v}" for k, v in kwargs.items()])
        self.common.print(f"Testing {api_name} on {device_type} {device_id} {extra_info}".strip())

    def _log_test_end(self, api_name, device_type, device_id, stats, **kwargs):
        """Helper method to log the end of a test."""
        if stats:
            self.common.print(
                f"Completed {api_name} on {device_type} {device_id}: {stats.get('mean_time_ms', 0):.3f}ms avg"
            )

    def _print_performance_results(self, stats):
        """Helper method to print performance results."""
        self.common.print(
            f"  Performance: {stats['mean_time_ms']:.3f}ms avg, {stats['min_time_ms']:.3f}ms min, {stats['max_time_ms']:.3f}ms max"
        )

    def _log_test_completion(self, device_type, device_id, extra_info=""):
        """Helper method to log test completion."""
        extra_msg = f" - {extra_info}" if extra_info else ""
        self.common.print(f"  {device_type} {device_id}: Test completed{extra_msg}")

    def _log_performance_summary(self, api_name, device_type_plural, test_name):
        """Helper method to log performance summary."""
        self.common.print(f"Performance test completed for {api_name} on {device_type_plural}")
        self.common.print("")  # Add empty line for readability

    def _print(self, msg, result):
        """Helper method for printing test results."""
        if isinstance(result, dict) or isinstance(result, list):
            if msg:
                self.common.print(msg)
            self.common.print(json.dumps(result, sort_keys=False, indent=4, default=str))
        else:
            self.common.print(f"{msg}: {result}" if msg else result)

    def _print_api_result(self, api_func, processor_id, *args, label_prefix="gpu", **kwargs):
        """
        Helper method to call an API and print its result before performance measurement.

        Args:
            api_func: The API function to call
            processor_id: The processor/GPU ID
            *args: Arguments to pass to the API
            label_prefix: Prefix for the label (default: "gpu")
            **kwargs: Keyword arguments to pass to the API
        """
        if self.common.verbose <= common.VERBOSITY_QUIET:
            return

        func_name = api_func.__name__
        msg = f"### test {func_name}({label_prefix}={processor_id})"

        try:
            result = api_func(*args, **kwargs)
            self._print(msg, result)
        except Exception:
            self.common.print(msg)
            # Don't raise - let the performance test measure the errors
            pass

    def _run_performance_assertions(self, stats, api_name):
        """Helper method to run assertions on performance test statistics."""
        # Basic performance test assertions
        self.assertIsInstance(stats, dict, f"Stats should be a dictionary for {api_name}")
        self.assertIn("iterations", stats, f"Stats should contain iterations for {api_name}")
        self.assertIn(
            "successful_runs", stats, f"Stats should contain successful_runs for {api_name}"
        )
        self.assertIn("error_count", stats, f"Stats should contain error_count for {api_name}")

        # If there were successful runs, check performance metrics
        if stats.get("successful_runs", 0) > 0:
            self.assertIn(
                "mean_time_ms", stats, f"Stats should contain mean_time_ms for {api_name}"
            )
            self.assertIn("min_time_ms", stats, f"Stats should contain min_time_ms for {api_name}")
            self.assertIn("max_time_ms", stats, f"Stats should contain max_time_ms for {api_name}")
            self.assertGreaterEqual(
                stats["mean_time_ms"], 0, f"Mean time should be non-negative for {api_name}"
            )
            self.assertGreaterEqual(
                stats["min_time_ms"], 0, f"Min time should be non-negative for {api_name}"
            )
            self.assertGreaterEqual(
                stats["max_time_ms"], 0, f"Max time should be non-negative for {api_name}"
            )

    def _measure_api_performance(self, api_func, api_name, *args, **kwargs) -> dict:
        """
        Measure the performance of an AMDSMI API function.

        Args:
            api_func: The API function to measure
            api_name: Human-readable name of the API
            *args: Arguments to pass to the API function
            **kwargs: Keyword arguments to pass to the API function

        Returns:
            dict: Performance statistics including min, max, mean, median times
        """
        times = []
        errors = []
        error_count = 0

        # Warmup iterations
        for _ in range(self.perf_warmup_iterations):
            try:
                api_func(*args, **kwargs)
            except Exception:
                pass  # Ignore warmup errors

        # Measurement iterations
        for i in range(self.perf_iterations):
            start_time = self.time.perf_counter()
            try:
                api_func(*args, **kwargs)
                end_time = self.time.perf_counter()
                execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
                times.append(execution_time)
            except Exception as e:
                error_count += 1
                end_time = self.time.perf_counter()
                execution_time = (end_time - start_time) * 1000
                times.append(execution_time)  # Include error times in statistics
                errors.append({"iteration": i, "error_info": str(e)})

        # Calculate statistics
        if times:
            successful_runs = len(times) - error_count
            stats = {
                "api_name": api_name,
                "iterations": len(times),
                "errors": errors,
                "error_count": error_count,
                "successful_runs": successful_runs,
                "min_time_ms": min(times),
                "max_time_ms": max(times),
                "mean_time_ms": self.statistics.mean(times),
                "median_time_ms": self.statistics.median(times),
                "times_ms": times,
            }

            if len(times) > 1:
                stats["stdev_ms"] = self.statistics.stdev(times)

            # Store results
            self.perf_results[api_name] = stats

            self.common.print(
                f"Performance {api_name}: {stats['mean_time_ms']:.3f}ms avg, "
                f"{stats['min_time_ms']:.3f}ms min, {stats['max_time_ms']:.3f}ms max, "
                f"{error_count} errors"
            )

            return stats
        else:
            return {"api_name": api_name, "error": "No successful measurements"}

    def test_performance_cpu_apb_disable(self):
        self.common.print_func_name("")
        # Use pstate=0 from original test
        pstate = 0
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]
        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_cpu_apb_disable", "cpu", i, pstate=pstate)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_cpu_apb_disable,
                    f"cpu_apb_disable_processor_{i}",
                    processor,
                    pstate,
                )
                self.perf_results[f"cpu_apb_disable_processor_{i}"] = stats
                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                else:
                    self.common.print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )
            self._log_test_completion("Processor", i)
            i = i + 1
        self._log_performance_summary("amdsmi_cpu_apb_disable", "Processors", "cpu_apb_disable")

    def test_performance_cpu_apb_enable(self):
        self.common.print_func_name("")
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_cpu_apb_enable", "cpu", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_cpu_apb_enable, f"cpu_apb_enable_processor_{i}", processor
                )

                self.perf_results[f"cpu_apb_enable_processor_{i}"] = stats

                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                else:
                    self.common.print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary("amdsmi_cpu_apb_enable", "Processors", "cpu_apb_enable")

    def test_performance_first_online_core_on_cpu_socket(self):
        self.common.print_func_name("")
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_first_online_core_on_cpu_socket", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_first_online_core_on_cpu_socket,
                    f"first_online_core_processor_{i}",
                    processor,
                )

                self.perf_results[f"first_online_core_processor_{i}"] = stats

                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                    self._run_performance_assertions(
                        stats, "amdsmi_first_online_core_on_cpu_socket"
                    )
                else:
                    self.common.print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside the loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_first_online_core_on_cpu_socket", "Processors", "first_online_core"
        )

    def test_performance_get_cpu_cclk_limit(self):
        self.common.print_func_name("")
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_cclk_limit", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_cclk_limit, f"get_cpu_cclk_limit_processor_{i}", processor
                )

                self.perf_results[f"get_cpu_cclk_limit_processor_{i}"] = stats

                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                else:
                    self.common.print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_cclk_limit", "Processors", "get_cpu_cclk_limit"
        )

    def test_performance_get_cpu_core_current_freq_limit(self):
        self.common.print_func_name("")
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_core_current_freq_limit", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_core_current_freq_limit,
                    f"get_cpu_core_current_freq_limit_processor_{i}",
                    processor,
                )

                self.perf_results[f"get_cpu_core_current_freq_limit_processor_{i}"] = stats

                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                else:
                    self.common.print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_core_current_freq_limit",
            "Processors",
            "get_cpu_core_current_freq_limit",
        )

    def test_performance_get_cpu_core_energy(self):
        self.common.print_func_name("")
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_core_energy", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_core_energy,
                    f"get_cpu_core_energy_processor_{i}",
                    processor,
                )

                self.perf_results[f"get_cpu_core_energy_processor_{i}"] = stats

                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                else:
                    self.common.print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside the loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_core_energy", "Processors", "get_cpu_core_energy"
        )

    def test_performance_get_cpu_current_io_bandwidth(self):
        self.common.print_func_name("")
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                for encoding_name, encoding, encoding_cond in common.IO_BW_ENCODINGS:
                    self._log_test_start(
                        "amdsmi_get_cpu_current_io_bandwidth",
                        "Processor",
                        i,
                        encoding=encoding_name,
                    )
                    stats = self._measure_api_performance(
                        amdsmi.amdsmi_get_cpu_current_io_bandwidth,
                        f"get_cpu_current_io_bandwidth_processor_{i}_encoding_{encoding_name}",
                        processor,
                        encoding,
                        encoding_name,
                    )

                    self.perf_results[
                        f"get_cpu_current_io_bandwidth_processor_{i}_encoding_{encoding_name}"
                    ] = stats

                    if stats["successful_runs"] > 0:
                        self._print_performance_results(stats)
                    else:
                        self.common.print(
                            f"  Processor {i} encoding={encoding_name}: All calls failed - "
                            f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                        )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_current_io_bandwidth", "Processors", "get_cpu_current_io_bandwidth"
        )

    def test_performance_get_cpu_ddr_bw(self):
        self.common.print_func_name("")
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_ddr_bw", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_ddr_bw, f"get_cpu_ddr_bw_processor_{i}", processor
                )

                self.perf_results[f"get_cpu_ddr_bw_processor_{i}"] = stats

                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                else:
                    self.common.print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside the loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary("amdsmi_get_cpu_ddr_bw", "Processors", "get_cpu_ddr_bw")

    def test_performance_get_cpu_dimm_power_consumption(self):
        self.common.print_func_name("")
        i = 0
        dimm_addr = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start(
                    "amdsmi_get_cpu_dimm_power_consumption", "Processor", i, dimm_addr=dimm_addr
                )
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_dimm_power_consumption,
                    f"get_cpu_dimm_power_processor_{i}",
                    processor,
                    dimm_addr,
                )

                self.perf_results[f"get_cpu_dimm_power_processor_{i}"] = stats

                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                else:
                    self.common.print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_dimm_power_consumption", "Processors", "get_cpu_dimm_power"
        )

    def test_performance_get_cpu_dimm_temp_range_and_refresh_rate(self):
        self.common.print_func_name("")
        dimm_addr = 0
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start(
                    "amdsmi_get_cpu_dimm_temp_range_and_refresh_rate",
                    "Processor",
                    i,
                    dimm_addr=dimm_addr,
                )
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_dimm_temp_range_and_refresh_rate,
                    f"get_cpu_dimm_temp_range_processor_{i}",
                    processor,
                    dimm_addr,
                )

                self.perf_results[f"get_cpu_dimm_temp_range_processor_{i}"] = stats

                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                else:
                    self.common.print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_dimm_temp_range_and_refresh_rate",
            "Processors",
            "get_cpu_dimm_temp_range",
        )

    def test_performance_get_cpu_dimm_thermal_sensor(self):
        self.common.print_func_name("")
        dimm_addr = 0
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start(
                    "amdsmi_get_cpu_dimm_thermal_sensor", "Processor", i, dimm_addr=dimm_addr
                )
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_dimm_thermal_sensor,
                    f"get_cpu_dimm_thermal_sensor_processor_{i}",
                    processor,
                    dimm_addr,
                )

                self.perf_results[f"get_cpu_dimm_thermal_sensor_processor_{i}"] = stats

                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                else:
                    self.common.print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_dimm_thermal_sensor", "Processors", "get_cpu_dimm_thermal_sensor"
        )

    def test_performance_get_cpu_family(self):
        self.common.print_func_name("")
        self._log_test_start("amdsmi_get_cpu_family", "System", "global")

        stats = self._measure_api_performance(amdsmi.amdsmi_get_cpu_family, "get_cpu_family_system")

        self.perf_results["get_cpu_family_system"] = stats
        if stats["successful_runs"] > 0:
            self._print_performance_results(stats)
        else:
            self.common.print(
                f"  System: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
            )

        self._log_test_completion("System", "global")
        self._log_performance_summary("amdsmi_get_cpu_family", "System", "get_cpu_family")

    def test_performance_get_cpu_fclk_mclk(self):
        self.common.print_func_name("")
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_fclk_mclk", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_fclk_mclk, f"get_cpu_fclk_mclk_processor_{i}", processor
                )

                self.perf_results[f"get_cpu_fclk_mclk_processor_{i}"] = stats

                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                else:
                    self.common.print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary("amdsmi_get_cpu_fclk_mclk", "Processors", "get_cpu_fclk_mclk")

    def test_performance_get_cpu_handles(self):
        self.common.print_func_name("")
        self._log_test_start("amdsmi_get_cpu_handles", "System", "global")

        stats = self._measure_api_performance(
            amdsmi.amdsmi_get_cpu_handles, "get_cpu_handles_system"
        )

        self.perf_results["get_cpu_handles_system"] = stats
        if stats["successful_runs"] > 0:
            self._print_performance_results(stats)
        else:
            self.common.print(
                f"  System: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
            )

        self._log_test_completion("System", "global")
        self._log_performance_summary("amdsmi_get_cpu_handles", "System", "get_cpu_handles")

    def test_performance_get_cpu_hsmp_driver_version(self):
        self.common.print_func_name("")
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_hsmp_driver_version", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_hsmp_driver_version,
                    f"get_cpu_hsmp_driver_version_processor_{i}",
                    processor,
                )

                self.perf_results[f"get_cpu_hsmp_driver_version_processor_{i}"] = stats

                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                else:
                    self.common.print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_hsmp_driver_version", "Processors", "get_cpu_hsmp_driver_version"
        )

    def test_performance_get_cpu_hsmp_proto_ver(self):
        self.common.print_func_name("")
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_hsmp_proto_ver", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_hsmp_proto_ver,
                    f"get_cpu_hsmp_proto_ver_processor_{i}",
                    processor,
                )

                self.perf_results[f"get_cpu_hsmp_proto_ver_processor_{i}"] = stats

                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                else:
                    self.common.print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_hsmp_proto_ver", "Processors", "get_cpu_hsmp_proto_ver"
        )

    def test_performance_get_cpu_model(self):
        self.common.print_func_name("")
        self._log_test_start("amdsmi_get_cpu_model", "System", "global")

        stats = self._measure_api_performance(amdsmi.amdsmi_get_cpu_model, "get_cpu_model_system")

        self.perf_results["get_cpu_model_system"] = stats
        if stats["successful_runs"] > 0:
            self._print_performance_results(stats)
        else:
            self.common.print(
                f"  System: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
            )

        self._log_test_completion("System", "global")
        self._log_performance_summary("amdsmi_get_cpu_model", "System", "get_cpu_model")

    def test_performance_get_cpu_prochot_status(self):
        self.common.print_func_name("")
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_prochot_status", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_prochot_status,
                    f"get_cpu_prochot_status_processor_{i}",
                    processor,
                )

                self.perf_results[f"get_cpu_prochot_status_processor_{i}"] = stats

                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                else:
                    self.common.print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_prochot_status", "Processors", "get_cpu_prochot_status"
        )

    def test_performance_get_cpu_pwr_svi_telemetry_all_rails(self):
        self.common.print_func_name("")
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_pwr_svi_telemetry_all_rails", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_pwr_svi_telemetry_all_rails,
                    f"get_cpu_pwr_svi_telemetry_processor_{i}",
                    processor,
                )

                self.perf_results[f"get_cpu_pwr_svi_telemetry_processor_{i}"] = stats

                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                else:
                    self.common.print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_pwr_svi_telemetry_all_rails", "Processors", "get_cpu_pwr_svi_telemetry"
        )

    def test_performance_get_cpu_smu_fw_version(self):
        self.common.print_func_name("")
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_smu_fw_version", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_smu_fw_version,
                    f"get_cpu_smu_fw_version_processor_{i}",
                    processor,
                )

                self.perf_results[f"get_cpu_smu_fw_version_processor_{i}"] = stats

                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                else:
                    self.common.print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_smu_fw_version", "Processors", "get_cpu_smu_fw_version"
        )

    def test_performance_get_cpu_socket_c0_residency(self):
        self.common.print_func_name("")
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_socket_c0_residency", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_socket_c0_residency,
                    f"get_cpu_socket_c0_residency_processor_{i}",
                    processor,
                )

                self.perf_results[f"get_cpu_socket_c0_residency_processor_{i}"] = stats

                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                else:
                    self.common.print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_socket_c0_residency", "Processors", "get_cpu_socket_c0_residency"
        )

    def test_performance_get_cpu_socket_current_active_freq_limit(self):
        self.common.print_func_name("")
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start(
                    "amdsmi_get_cpu_socket_current_active_freq_limit", "Processor", i
                )
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_socket_current_active_freq_limit,
                    f"get_cpu_socket_current_active_freq_limit_processor_{i}",
                    processor,
                )

                self.perf_results[f"get_cpu_socket_current_active_freq_limit_processor_{i}"] = stats

                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                else:
                    self.common.print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_socket_current_active_freq_limit",
            "Processors",
            "get_cpu_socket_current_active_freq_limit",
        )

    def test_performance_get_cpu_socket_energy(self):
        self.common.print_func_name("")
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_socket_energy", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_socket_energy,
                    f"get_cpu_socket_energy_processor_{i}",
                    processor,
                )

                self.perf_results[f"get_cpu_socket_energy_processor_{i}"] = stats

                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                else:
                    self.common.print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_socket_energy", "Processors", "get_cpu_socket_energy"
        )

    def test_performance_get_cpu_socket_freq_range(self):
        self.common.print_func_name("")
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_socket_freq_range", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_socket_freq_range,
                    f"get_cpu_socket_freq_range_processor_{i}",
                    processor,
                )

                self.perf_results[f"get_cpu_socket_freq_range_processor_{i}"] = stats

                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                else:
                    self.common.print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_socket_freq_range", "Processors", "get_cpu_socket_freq_range"
        )

    def test_performance_get_cpu_socket_lclk_dpm_level(self):
        self.common.print_func_name("")
        nbio_id = 0
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start(
                    "amdsmi_get_cpu_socket_lclk_dpm_level", "Processor", i, nbio_id=nbio_id
                )
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_socket_lclk_dpm_level,
                    f"get_cpu_socket_lclk_dpm_processor_{i}",
                    processor,
                    nbio_id,
                )

                self.perf_results[f"get_cpu_socket_lclk_dpm_processor_{i}"] = stats

                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                else:
                    self.common.print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_socket_lclk_dpm_level", "Processors", "get_cpu_socket_lclk_dpm_level"
        )

    def test_performance_get_cpu_socket_power(self):
        self.common.print_func_name("")
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_socket_power", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_socket_power,
                    f"get_cpu_socket_power_processor_{i}",
                    processor,
                )

                self.perf_results[f"get_cpu_socket_power_processor_{i}"] = stats
                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                else:
                    self.common.print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i += 1  # increment inside the loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_socket_power", "Processors", "get_cpu_socket_power"
        )

    def test_performance_get_cpu_socket_power_cap_max(self):
        self.common.print_func_name("")
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_socket_power_cap_max", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_socket_power_cap_max,
                    f"get_cpu_socket_power_cap_max_processor_{i}",
                    processor,
                )

                self.perf_results[f"get_cpu_socket_power_cap_max_processor_{i}"] = stats
                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                else:
                    self.common.print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i += 1  # increment inside the loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_socket_power_cap_max", "Processors", "get_cpu_socket_power_cap_max"
        )

    def test_performance_get_cpu_socket_temperature(self):
        self.common.print_func_name("")
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_socket_temperature", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_socket_temperature,
                    f"get_cpu_socket_temperature_processor_{i}",
                    processor,
                )

                self.perf_results[f"get_cpu_socket_temperature_processor_{i}"] = stats
                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                else:
                    self.common.print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i += 1  # increment inside the loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_socket_temperature", "Processors", "get_cpu_socket_temperature"
        )

    def test_performance_get_esmi_err_msg(self):
        self.common.print_func_name("")

        for status_type_name, status_type, status_cond in common.STATUS_TYPES:
            self._log_test_start("amdsmi_get_esmi_err_msg", "Status", status_type_name)

            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_esmi_err_msg,
                f"get_esmi_err_msg_status_{status_type_name}",
                status_type,
            )

            self.perf_results[f"get_esmi_err_msg_status_{status_type_name}"] = stats
            if stats["successful_runs"] > 0:
                self._print_performance_results(stats)
            else:
                self.common.print(
                    f"  Status {status_type_name}: All calls failed - "
                    f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                )

            self._log_test_completion("Status", status_type_name)

        self._log_performance_summary("amdsmi_get_esmi_err_msg", "Status types", "get_esmi_err_msg")

    def test_performance_get_hsmp_metrics_table(self):
        self.common.print_func_name("")
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_hsmp_metrics_table", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_hsmp_metrics_table,
                    f"get_hsmp_metrics_table_processor_{i}",
                    processor,
                )

                self.perf_results[f"get_hsmp_metrics_table_processor_{i}"] = stats
                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                else:
                    self.common.print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                self._log_test_completion("Processor", i)
                i += 1

        self._log_performance_summary(
            "amdsmi_get_hsmp_metrics_table", "Processors", "get_hsmp_metrics_table"
        )

    def test_performance_get_hsmp_metrics_table_version(self):
        self.common.print_func_name("")
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_hsmp_metrics_table_version", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_hsmp_metrics_table_version,
                    f"get_hsmp_metrics_table_version_processor_{i}",
                    processor,
                )

                self.perf_results[f"get_hsmp_metrics_table_version_processor_{i}"] = stats

                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                else:
                    self.common.print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                self._log_test_completion("Processor", i)
                i += 1

        self._log_performance_summary(
            "amdsmi_get_hsmp_metrics_table_version", "Sockets", "get_hsmp_metrics_table_version"
        )

    def test_performance_get_lib_version(self):
        self.common.print_func_name("")

        self._log_test_start("amdsmi_get_lib_version", "System", "global")

        stats = self._measure_api_performance(amdsmi.amdsmi_get_lib_version, "get_lib_version")

        self.perf_results["get_lib_version"] = stats

        if stats["successful_runs"] > 0:
            self._print_performance_results(stats)
        else:
            self.common.print(
                f"  All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
            )

        self._log_test_completion("System", "global")
        self._log_performance_summary("amdsmi_get_lib_version", "System", "get_lib_version")

    def test_performance_set_cpu_pcie_link_rate(self):
        self.common.print_func_name("")
        # Test with different rate_ctrl values
        rate_ctrls = [0]  # Starting with 0 as in original test
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                for rate_ctrl in rate_ctrls:
                    self._log_test_start(
                        "amdsmi_set_cpu_pcie_link_rate", "Processor", i, rate_ctrl=rate_ctrl
                    )

                    stats = self._measure_api_performance(
                        amdsmi.amdsmi_set_cpu_pcie_link_rate,
                        f"set_cpu_pcie_link_rate_cpu_{i}_rate_{rate_ctrl}",
                        processor,
                        rate_ctrl,
                    )

                    self.perf_results[f"set_cpu_pcie_link_rate_cpu_{i}_rate_{rate_ctrl}"] = stats

                    if stats["successful_runs"] > 0:
                        self._print_performance_results(stats)
                        self._run_performance_assertions(stats, "amdsmi_set_cpu_pcie_link_rate")
                    else:
                        self.common.print(
                            f"  CPU {i} rate_ctrl {rate_ctrl}: All calls failed - "
                            f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                        )

                self._log_test_completion("CPU", i, f"rate_ctrl={rate_ctrl}")

        self._log_performance_summary(
            "amdsmi_set_cpu_pcie_link_rate", "CPUs", "set_cpu_pcie_link_rate"
        )

    def test_performance_get_processor_count_from_handles(self):
        self.common.print_func_name("")

        self._log_test_start("amdsmi_get_processor_count_from_handles", "Processors", "all")

        stats = self._measure_api_performance(
            amdsmi.amdsmi_get_processor_count_from_handles,
            "get_processor_count_from_handles",
            self.processors,
        )

        self.perf_results["get_processor_count_from_handles"] = stats

        if stats["successful_runs"] > 0:
            self._print_performance_results(stats)
            self._run_performance_assertions(stats, "amdsmi_get_processor_count_from_handles")
        else:
            self.common.print(
                f"  All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
            )

        self._log_test_completion("Processors", "all")
        self._log_performance_summary(
            "amdsmi_get_processor_count_from_handles",
            "Processors",
            "get_processor_count_from_handles",
        )

    def test_performance_get_processor_handle_from_bdf(self):
        self.common.print_func_name("")
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_processor_handle_from_bdf", "CPU", i)

                try:
                    # Get BDF for this processor first
                    bdf = amdsmi.amdsmi_get_gpu_device_bdf(processor)

                    stats = self._measure_api_performance(
                        amdsmi.amdsmi_get_processor_handle_from_bdf,
                        f"get_processor_handle_from_bdf_cpu_{i}",
                        bdf,
                    )

                    self.perf_results[f"get_processor_handle_from_bdf_cpu_{i}"] = stats
                    if stats["successful_runs"] > 0:
                        self._print_performance_results(stats)
                        self._run_performance_assertions(
                            stats, "amdsmi_get_processor_handle_from_bdf"
                        )

                    # Validate that the returned handle matches the original processor
                    ret = amdsmi.amdsmi_get_processor_handle_from_bdf(bdf)
                    if processor.value != ret.value:
                        self.common.print(
                            f"  WARNING: CPU {i} - Handle mismatch! Expected: {processor.value}, Received: {ret.value}"
                        )
                    else:
                        self.common.print(
                            f"  CPU {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                        )

                except Exception as e:
                    self.common.print(f"  CPU {i}: Error getting BDF - {e}")

                self._log_test_completion("CPU", i)
                i = i + 1

        self._log_performance_summary(
            "amdsmi_get_processor_handle_from_bdf", "CPUs", "get_processor_handle_from_bdf"
        )

    def test_performance_get_processor_handles(self):
        self.common.print_func_name("")

        self._log_test_start("amdsmi_get_processor_handles", "System", "global")

        stats = self._measure_api_performance(
            amdsmi.amdsmi_get_processor_handles, "get_processor_handles"
        )

        self.perf_results["get_processor_handles"] = stats

        if stats["successful_runs"] > 0:
            self._print_performance_results(stats)
            self._run_performance_assertions(stats, "amdsmi_get_processor_handles")
        else:
            self.common.print(
                f"  All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
            )

        self._log_test_completion("System", "global")
        self._log_performance_summary(
            "amdsmi_get_processor_handles", "System", "get_processor_handles"
        )

    def test_performance_get_processor_handles_by_type(self):
        self.common.print_func_name("")

        socket_handles = amdsmi.amdsmi_get_socket_handles()

        for index, socket_handle in enumerate(socket_handles):
            for processor_name, processor_type, processor_cond in common.PROCESSOR_TYPES:
                self._log_test_start(
                    "amdsmi_get_processor_handles_by_type",
                    "Socket",
                    index,
                    processor_type=processor_name,
                )

                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_processor_handles_by_type,
                    f"get_processor_handles_by_type_socket_{index}_type_{processor_name}",
                    socket_handle,
                    processor_type,
                )

                self.perf_results[
                    f"get_processor_handles_by_type_socket_{index}_type_{processor_name}"
                ] = stats

                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                    self._run_performance_assertions(stats, "amdsmi_get_processor_handles_by_type")
                else:
                    self.common.print(
                        f"  Socket {index} type {processor_name}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                self._log_test_completion("Socket", index, f"processor_type={processor_name}")

        self._log_performance_summary(
            "amdsmi_get_processor_handles_by_type", "Sockets", "get_processor_handles_by_type"
        )

    def test_performance_get_processor_info(self):
        self.common.print_func_name("")
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_processor_info", "CPU", i)

                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_processor_info, f"get_processor_info_cpu_{i}", processor
                )

                self.perf_results[f"get_processor_info_cpu_{i}"] = stats
                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                    self._run_performance_assertions(stats, "amdsmi_get_processor_info")
                else:
                    self.common.print(
                        f"  CPU {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                self._log_test_completion("CPU", i)
                i = i + 1

        self._log_performance_summary("amdsmi_get_processor_info", "CPUs", "get_processor_info")

    def test_performance_get_processor_type(self):
        self.common.print_func_name("")
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_processor_type", "CPU", i)

                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_processor_type, f"get_processor_type_cpu_{i}", processor
                )

                self.perf_results[f"get_processor_type_cpu_{i}"] = stats
                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                    self._run_performance_assertions(stats, "amdsmi_get_processor_type")
                else:
                    self.common.print(
                        f"  CPU {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                self._log_test_completion("CPU", i)
                i = i + 1

        self._log_performance_summary("amdsmi_get_processor_type", "CPUs", "get_processor_type")

    def test_performance_get_socket_handles(self):
        self.common.print_func_name("")

        self._log_test_start("amdsmi_get_socket_handles", "System", "global")

        stats = self._measure_api_performance(
            amdsmi.amdsmi_get_socket_handles, "get_socket_handles"
        )

        self.perf_results["get_socket_handles"] = stats

        if stats["successful_runs"] > 0:
            self._print_performance_results(stats)
            self._run_performance_assertions(stats, "amdsmi_get_socket_handles")
        else:
            self.common.print(
                f"  All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
            )

        self._log_test_completion("System", "global")
        self._log_performance_summary("amdsmi_get_socket_handles", "System", "get_socket_handles")

    def test_performance_get_socket_info(self):
        self.common.print_func_name("")

        sockets = amdsmi.amdsmi_get_socket_handles()

        for i, socket in enumerate(sockets):
            self._log_test_start("amdsmi_get_socket_info", "Socket", i)

            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_socket_info, f"get_socket_info_socket_{i}", socket
            )

            self.perf_results[f"get_socket_info_socket_{i}"] = stats

            if stats["successful_runs"] > 0:
                self._print_performance_results(stats)
                self._run_performance_assertions(stats, "amdsmi_get_socket_info")
            else:
                self.common.print(
                    f"  Socket {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                )

            self._log_test_completion("Socket", i)

        self._log_performance_summary("amdsmi_get_socket_info", "Sockets", "get_socket_info")

    def test_performance_get_temp_metric(self):
        self.common.print_func_name("")
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                for (
                    temperature_type_name,
                    temperature_type,
                    temperature_type_cond,
                ) in common.TEMPERATURE_TYPES:
                    for (
                        temperature_metric_name,
                        temperature_metric,
                        temperature_metric_cond,
                    ) in common.TEMPERATURE_METRICS:
                        self._log_test_start(
                            "amdsmi_get_temp_metric",
                            "CPU",
                            i,
                            temperature_type=temperature_type_name,
                            temperature_metric=temperature_metric_name,
                        )

                        stats = self._measure_api_performance(
                            amdsmi.amdsmi_get_temp_metric,
                            f"get_temp_metric_cpu_{i}_{temperature_type_name}_{temperature_metric_name}",
                            processor,
                            temperature_type,
                            temperature_metric,
                        )

                        self.perf_results[
                            f"get_temp_metric_cpu_{i}_{temperature_type_name}_{temperature_metric_name}"
                        ] = stats
                        if stats["successful_runs"] > 0:
                            self._print_performance_results(stats)
                            self._run_performance_assertions(stats, "amdsmi_get_temp_metric")
                        else:
                            self.common.print(
                                f"  CPU {i} {temperature_type_name}/{temperature_metric_name}: All calls failed - "
                                f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                            )

                        self._log_test_completion(
                            "CPU",
                            i,
                            f"temperature_type={temperature_type_name}, temperature_metric={temperature_metric_name}",
                        )
                i = i + 1

        self._log_performance_summary("amdsmi_get_temp_metric", "CPUs", "get_temp_metric")

    def test_performance_get_threads_per_core(self):
        self.common.print_func_name("")
        self._log_test_start("amdsmi_get_threads_per_core", "System", 0)

        stats = self._measure_api_performance(
            amdsmi.amdsmi_get_threads_per_core, "get_threads_per_core"
        )

        self.perf_results["get_threads_per_core"] = stats

        if stats["successful_runs"] > 0:
            self._print_performance_results(stats)
            self._run_performance_assertions(stats, "amdsmi_get_threads_per_core")
        else:
            self.common.print(
                f"  System: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
            )

        self._log_test_completion("System", 0)

        self._log_performance_summary(
            "amdsmi_get_threads_per_core", "System", "get_threads_per_core"
        )

    def test_performance_init(self):
        self.common.print_func_name("")

        self._log_test_start("amdsmi_init", "System", "global")

        # Note: We need to be careful with init/shutdown as they affect the entire library state
        stats = self._measure_api_performance(amdsmi.amdsmi_init, "init_system")

        self.perf_results["init_system"] = stats

        if stats["successful_runs"] > 0:
            self._print_performance_results(stats)
        else:
            self.common.print(
                f"  All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
            )

        self._log_test_completion("System", "global")
        self._log_performance_summary("amdsmi_init", "System", "init")

    def test_performance_shutdown(self):
        self.common.print_func_name("")

        self._log_test_start("amdsmi_shut_down", "System", "global")

        # Note: We need to be careful with init/shutdown as they affect the entire library state
        stats = self._measure_api_performance(amdsmi.amdsmi_shut_down, "shutdown_system")

        self.perf_results["shutdown_system"] = stats

        if stats["successful_runs"] > 0:
            self._print_performance_results(stats)
        else:
            self.common.print(
                f"  All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
            )

        self._log_test_completion("System", "global")
        self._log_performance_summary("amdsmi_shut_down", "System", "shutdown")

    def test_performance_cpu_core_boostlimit(self):
        self.common.print_func_name("")
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("cpu_core_boostlimit_workflow", "Processor", i)

                # Test Get CPU Core Boostlimit
                stats_get = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_core_boostlimit,
                    f"get_cpu_core_boostlimit_processor_{i}",
                    processor,
                )
                self.perf_results[f"get_cpu_core_boostlimit_processor_{i}"] = stats_get

                if stats_get["successful_runs"] > 0:
                    self._print_performance_results(stats_get)

                    # Try to get the boost limit value for setting test
                    try:
                        boost_limit = amdsmi.amdsmi_get_cpu_core_boostlimit(processor)

                        # Test Set CPU Core Boostlimit (with same value to avoid changing system state)
                        stats_set = self._measure_api_performance(
                            amdsmi.amdsmi_set_cpu_core_boostlimit,
                            f"set_cpu_core_boostlimit_processor_{i}",
                            processor,
                            boost_limit,
                        )
                        self.perf_results[f"set_cpu_core_boostlimit_processor_{i}"] = stats_set
                        self._print_performance_results(stats_set)

                    except amdsmi.AmdSmiLibraryException:
                        self.common.print(
                            f"  Processor {i}: Could not get boost_limit for set test"
                        )
                else:
                    self.common.print(
                        f"  Processor {i}: Get failed - "
                        f"{stats_get['errors'][0]['error_info'] if stats_get['errors'] else 'Unknown'}"
                    )

                self._log_test_completion("Processor", i)
                i = i + 1

        self._log_performance_summary(
            "cpu_core_boostlimit_workflow", "Processors", "cpu_core_boostlimit"
        )
        self._log_performance_summary(
            "amdsmi_cpu_core_boostlimit", "Sockets", "cpu_core_boostlimit"
        )

    def test_performance_set_cpu_df_pstate_range(self):
        self.common.print_func_name("")

        # Use TODO placeholder values like original test
        max_pstate = 0
        min_pstate = 0
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start(
                    "amdsmi_set_cpu_df_pstate_range",
                    "Processor",
                    i,
                    max_pstate=max_pstate,
                    min_pstate=min_pstate,
                )

                stats = self._measure_api_performance(
                    amdsmi.amdsmi_set_cpu_df_pstate_range,
                    f"set_cpu_df_pstate_range_processor_{i}",
                    processor,
                    max_pstate,
                    min_pstate,
                )

                self.perf_results[f"set_cpu_df_pstate_range_processor_{i}"] = stats
                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                else:
                    self.common.print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                self._log_test_completion("Processor", i)
                i = i + 1

        self._log_performance_summary(
            "amdsmi_set_cpu_df_pstate_range", "Processors", "set_cpu_df_pstate_range"
        )

    def test_performance_set_cpu_gmi3_link_width_range(self):
        self.common.print_func_name("")

        # Use TODO placeholder values like original test
        min_link_width = 0
        max_link_width = 0
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start(
                    "amdsmi_set_cpu_gmi3_link_width_range",
                    "Processor",
                    i,
                    min_link_width=min_link_width,
                    max_link_width=max_link_width,
                )

                stats = self._measure_api_performance(
                    amdsmi.amdsmi_set_cpu_gmi3_link_width_range,
                    f"set_cpu_gmi3_link_width_range_processor_{i}",
                    processor,
                    min_link_width,
                    max_link_width,
                )

                self.perf_results[f"set_cpu_gmi3_link_width_range_processor_{i}"] = stats
                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                else:
                    self.common.print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                self._log_test_completion("Processor", i)
                i = i + 1

        self._log_performance_summary(
            "amdsmi_set_cpu_gmi3_link_width_range", "Processors", "set_cpu_gmi3_link_width_range"
        )

    def test_performance_set_cpu_pwr_efficiency_mode(self):
        self.common.print_func_name("")

        # Use modes from original test
        modes = [0, 1, 2]
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                for mode in modes:
                    self._log_test_start(
                        "amdsmi_set_cpu_pwr_efficiency_mode", "Processor", i, mode=mode
                    )

                    stats = self._measure_api_performance(
                        amdsmi.amdsmi_set_cpu_pwr_efficiency_mode,
                        f"set_cpu_pwr_efficiency_mode_processor_{i}_mode_{mode}",
                        processor,
                        mode,
                    )

                    self.perf_results[f"set_cpu_pwr_efficiency_mode_processor_{i}_mode_{mode}"] = (
                        stats
                    )

                    if stats["successful_runs"] > 0:
                        self._print_performance_results(stats)
                    else:
                        self.common.print(
                            f"  Processor {i} mode {mode}: All calls failed - "
                            f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                        )

                    self._log_test_completion("Processor", i, f"mode={mode}")
                i = i + 1

        self._log_performance_summary(
            "amdsmi_set_cpu_pwr_efficiency_mode", "Processors", "set_cpu_pwr_efficiency_mode"
        )

    def test_performance_cpu_socket_boostlimit(self):
        self.common.print_func_name("")

        # Use TODO placeholder value like original test
        boost_limit = 0
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start(
                    "amdsmi_set_cpu_socket_boostlimit", "Processor", i, boost_limit=boost_limit
                )

                stats = self._measure_api_performance(
                    amdsmi.amdsmi_set_cpu_socket_boostlimit,
                    f"set_cpu_socket_boostlimit_processor_{i}",
                    processor,
                    boost_limit,
                )

                self.perf_results[f"set_cpu_socket_boostlimit_processor_{i}"] = stats
                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                else:
                    self.common.print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                self._log_test_completion("Processor", i)
                i = i + 1

        self._log_performance_summary(
            "amdsmi_set_cpu_socket_boostlimit", "Processors", "cpu_socket_boostlimit"
        )

    def test_performance_set_cpu_socket_lclk_dpm_level(self):
        self.common.print_func_name("")

        # Use TODO placeholder values like original test
        nbio_id = 0
        min_val = 0
        max_val = 0
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start(
                    "amdsmi_set_cpu_socket_lclk_dpm_level",
                    "Processor",
                    i,
                    nbio_id=nbio_id,
                    min_val=min_val,
                    max_val=max_val,
                )

                stats = self._measure_api_performance(
                    amdsmi.amdsmi_set_cpu_socket_lclk_dpm_level,
                    f"set_cpu_socket_lclk_dpm_level_processor_{i}",
                    processor,
                    nbio_id,
                    min_val,
                    max_val,
                )

                self.perf_results[f"set_cpu_socket_lclk_dpm_level_processor_{i}"] = stats
                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                else:
                    self.common.print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                self._log_test_completion("Processor", i)
                i = i + 1

        self._log_performance_summary(
            "amdsmi_set_cpu_socket_lclk_dpm_level", "Processors", "set_cpu_socket_lclk_dpm_level"
        )

    def test_performance_cpu_socket_power_cap(self):
        self.common.print_func_name("")
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("cpu_socket_power_cap_workflow", "Processor", i)

                # Test Get CPU Socket Power Cap
                stats_get = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_socket_power_cap,
                    f"get_cpu_socket_power_cap_processor_{i}",
                    processor,
                )
                self.perf_results[f"get_cpu_socket_power_cap_processor_{i}"] = stats_get
                if stats_get["successful_runs"] > 0:
                    self._print_performance_results(stats_get)

                    # Try to get the power cap value for setting test
                    try:
                        power_cap = amdsmi.amdsmi_get_cpu_socket_power_cap(processor)

                        # Test Set CPU Socket Power Cap (with same value to avoid changing system state)
                        stats_set = self._measure_api_performance(
                            amdsmi.amdsmi_set_cpu_socket_power_cap,
                            f"set_cpu_socket_power_cap_processor_{i}",
                            processor,
                            power_cap,
                        )
                        self.perf_results[f"set_cpu_socket_power_cap_processor_{i}"] = stats_set
                        self._print_performance_results(stats_set)

                    except amdsmi.AmdSmiLibraryException:
                        self.common.print(f"  Processor {i}: Could not get power_cap for set test")
                else:
                    self.common.print(
                        f"  Processor {i}: Get failed - "
                        f"{stats_get['errors'][0]['error_info'] if stats_get['errors'] else 'Unknown'}"
                    )

                self._log_test_completion("Processor", i)
                i = i + 1

        self._log_performance_summary(
            "cpu_socket_power_cap_workflow", "Processors", "cpu_socket_power_cap"
        )

    def test_performance_set_cpu_xgmi_width(self):
        self.common.print_func_name("")

        # Use TODO placeholder values like original test
        min_width = 0
        max_width = 0
        i = 0
        ret = amdsmi.amdsmi_get_cpu_handles()
        processor_handles = ret["processor_handles"]

        if len(processor_handles) == 0:
            self.common.print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start(
                    "amdsmi_set_cpu_xgmi_width",
                    "Processor",
                    i,
                    min_width=min_width,
                    max_width=max_width,
                )

                stats = self._measure_api_performance(
                    amdsmi.amdsmi_set_cpu_xgmi_width,
                    f"set_cpu_xgmi_width_processor_{i}",
                    processor,
                    min_width,
                    max_width,
                )

                self.perf_results[f"set_cpu_xgmi_width_processor_{i}"] = stats
                if stats["successful_runs"] > 0:
                    self._print_performance_results(stats)
                else:
                    self.common.print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                self._log_test_completion("Processor", i)
                i = i + 1

        self._log_performance_summary(
            "amdsmi_set_cpu_xgmi_width", "Processors", "set_cpu_xgmi_width"
        )


# =============================================================================
# PERFORMANCE TEST REPORTING SYSTEM (Converted from perf_test.sh)
# =============================================================================
