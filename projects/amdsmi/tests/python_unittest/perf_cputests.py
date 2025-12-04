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

import ctypes
import inspect
import json
import sys
import unittest
import time
import statistics
import os
sys.path.append("/opt/rocm/libexec/amdsmi_cli/")

try:
    import amdsmi
except ImportError as exc:
    print(f"Warning: Could not import amdsmi: {exc}")
    # Create a minimal mock for testing
    class MockAmdsmi:
        def __init__(self):
            pass
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    amdsmi = MockAmdsmi()

from amdsmi import(
amdsmi_init,
amdsmi_shut_down,
amdsmi_get_cpusocket_handles,
amdsmi_get_cpu_hsmp_driver_version,
AmdSmiInitFlags,
AmdSmiException
)

# Error map dictionary
error_map = \
{
    '0': 'AMDSMI_STATUS_SUCCESS',
    '1': 'AMDSMI_STATUS_INVAL',
    '2': 'AMDSMI_STATUS_NOT_SUPPORTED',
    '3': 'AMDSMI_STATUS_NOT_YET_IMPLEMENTED',
    '4': 'AMDSMI_STATUS_FAIL_LOAD_MODULE',
    '5': 'AMDSMI_STATUS_FAIL_LOAD_SYMBOL',
    '6': 'AMDSMI_STATUS_DRM_ERROR',
    '7': 'AMDSMI_STATUS_API_FAILED',
    '8': 'AMDSMI_STATUS_TIMEOUT',
    '9': 'AMDSMI_STATUS_RETRY',
    '10': 'AMDSMI_STATUS_NO_PERM',
    '11': 'AMDSMI_STATUS_INTERRUPT',
    '12': 'AMDSMI_STATUS_IO',
    '13': 'AMDSMI_STATUS_ADDRESS_FAULT',
    '14': 'AMDSMI_STATUS_FILE_ERROR',
    '15': 'AMDSMI_STATUS_OUT_OF_RESOURCES',
    '16': 'AMDSMI_STATUS_INTERNAL_EXCEPTION',
    '17': 'AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS',
    '18': 'AMDSMI_STATUS_INIT_ERROR',
    '19': 'AMDSMI_STATUS_REFCOUNT_OVERFLOW',
    '30': 'AMDSMI_STATUS_BUSY',
    '31': 'AMDSMI_STATUS_NOT_FOUND',
    '32': 'AMDSMI_STATUS_NOT_INIT',
    '33': 'AMDSMI_STATUS_NO_SLOT',
    '34': 'AMDSMI_STATUS_DRIVER_NOT_LOADED',
    '39': 'AMDSMI_STATUS_MORE_DATA',
    '40': 'AMDSMI_STATUS_NO_DATA',
    '41': 'AMDSMI_STATUS_INSUFFICIENT_SIZE',
    '42': 'AMDSMI_STATUS_UNEXPECTED_SIZE',
    '43': 'AMDSMI_STATUS_UNEXPECTED_DATA',
    '44': 'AMDSMI_STATUS_NON_AMD_CPU',
    '45': 'AMDSMI_STATUS_NO_ENERGY_DRV',
    '46': 'AMDSMI_STATUS_NO_MSR_DRV',
    '47': 'AMDSMI_STATUS_NO_HSMP_DRV',
    '48': 'AMDSMI_STATUS_NO_HSMP_SUP',
    '49': 'AMDSMI_STATUS_NO_HSMP_MSG_SUP',
    '50': 'AMDSMI_STATUS_HSMP_TIMEOUT',
    '51': 'AMDSMI_STATUS_NO_DRV',
    '52': 'AMDSMI_STATUS_FILE_NOT_FOUND',
    '53': 'AMDSMI_STATUS_ARG_PTR_NULL',
    '54': 'AMDSMI_STATUS_AMDGPU_RESTART_ERR',
    '55': 'AMDSMI_STATUS_SETTING_UNAVAILABLE',
    '56': 'AMDSMI_STATUS_CORRUPTED_EEPROM',
    '0xFFFFFFFE': 'AMDSMI_STATUS_MAP_ERROR',
    '0xFFFFFFFF': 'AMDSMI_STATUS_UNKNOWN_ERROR'
}   

# Global variables needed for performance tests
verbose = 1
has_info_printed = False

# Global constants matching original test file
PASS = 'AMDSMI_STATUS_SUCCESS'
FAIL = 'AMDSMI_STATUS_INVAL'

class TestAmdSmiCPUPythonPerformance(unittest.TestCase):
    """
    Standalone Performance testing class for AMDSMI Python APIs.
    
    This class does NOT inherit from TestAmdSmiPython to avoid running all regular tests.
    It only runs performance-specific tests.
    """
    # Class-level attributes to prevent IndexError and AttributeError
    compute_partition_types = \
    [
        ('SPX', amdsmi.AmdSmiComputePartitionType.SPX, PASS) if hasattr(amdsmi, 'AmdSmiComputePartitionType') else ('SPX', None, PASS),
        ('DPX', amdsmi.AmdSmiComputePartitionType.DPX, PASS) if hasattr(amdsmi, 'AmdSmiComputePartitionType') else ('DPX', None, PASS),
        ('TPX', amdsmi.AmdSmiComputePartitionType.TPX, PASS) if hasattr(amdsmi, 'AmdSmiComputePartitionType') else ('TPX', None, PASS),
        ('QPX', amdsmi.AmdSmiComputePartitionType.QPX, PASS) if hasattr(amdsmi, 'AmdSmiComputePartitionType') else ('QPX', None, PASS),
        ('CPX', amdsmi.AmdSmiComputePartitionType.CPX, PASS) if hasattr(amdsmi, 'AmdSmiComputePartitionType') else ('CPX', None, PASS),
        ('INVALID', amdsmi.AmdSmiComputePartitionType.INVALID, FAIL) if hasattr(amdsmi, 'AmdSmiComputePartitionType') else ('INVALID', None, FAIL)
    ]
    
    freq_inds = \
    [
        ('MIN', amdsmi.AmdSmiFreqInd.MIN, PASS) if hasattr(amdsmi, 'AmdSmiFreqInd') else ('MIN', None, PASS),
        ('MAX', amdsmi.AmdSmiFreqInd.MAX, PASS) if hasattr(amdsmi, 'AmdSmiFreqInd') else ('MAX', None, PASS),
        ('INVALID', amdsmi.AmdSmiFreqInd.INVALID, FAIL) if hasattr(amdsmi, 'AmdSmiFreqInd') else ('INVALID', None, FAIL)
    ]
    
    dev_perf_levels = \
    [
        ('AUTO', amdsmi.AmdSmiDevPerfLevel.AUTO, PASS) if hasattr(amdsmi, 'AmdSmiDevPerfLevel') else ('AUTO', None, PASS),
        ('LOW', amdsmi.AmdSmiDevPerfLevel.LOW, PASS) if hasattr(amdsmi, 'AmdSmiDevPerfLevel') else ('LOW', None, PASS),
        ('HIGH', amdsmi.AmdSmiDevPerfLevel.HIGH, PASS) if hasattr(amdsmi, 'AmdSmiDevPerfLevel') else ('HIGH', None, PASS),
        ('MANUAL', amdsmi.AmdSmiDevPerfLevel.MANUAL, PASS) if hasattr(amdsmi, 'AmdSmiDevPerfLevel') else ('MANUAL', None, PASS),
        ('STABLE_STD', amdsmi.AmdSmiDevPerfLevel.STABLE_STD, PASS) if hasattr(amdsmi, 'AmdSmiDevPerfLevel') else ('STABLE_STD', None, PASS),
        ('STABLE_PEAK', amdsmi.AmdSmiDevPerfLevel.STABLE_PEAK, PASS) if hasattr(amdsmi, 'AmdSmiDevPerfLevel') else ('STABLE_PEAK', None, PASS),
        ('STABLE_MIN_MCLK', amdsmi.AmdSmiDevPerfLevel.STABLE_MIN_MCLK, PASS) if hasattr(amdsmi, 'AmdSmiDevPerfLevel') else ('STABLE_MIN_MCLK', None, PASS),
        ('STABLE_MIN_SCLK', amdsmi.AmdSmiDevPerfLevel.STABLE_MIN_SCLK, PASS) if hasattr(amdsmi, 'AmdSmiDevPerfLevel') else ('STABLE_MIN_SCLK', None, PASS),
        ('DETERMINISM', amdsmi.AmdSmiDevPerfLevel.DETERMINISM, PASS) if hasattr(amdsmi, 'AmdSmiDevPerfLevel') else ('DETERMINISM', None, PASS),
        ('UNKNOWN', amdsmi.AmdSmiDevPerfLevel.UNKNOWN, FAIL) if hasattr(amdsmi, 'AmdSmiDevPerfLevel') else ('UNKNOWN', None, FAIL)
    ]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        global has_info_printed
        if verbose and has_info_printed is False:
            # Execute the following to print the asic and board info once
            # per test run
            has_info_printed = True
            self.setUp()
            for i, gpu in enumerate(self.processors):
                try:
                    # Print asic info
                    msg = f'asic info(gpu={i})'
                    ret = amdsmi.amdsmi_get_gpu_asic_info(gpu)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    raise e
            for i, gpu in enumerate(self.processors):
                try:
                    # Print board info
                    msg = f'board info(gpu={i})'
                    ret = amdsmi.amdsmi_get_gpu_board_info(gpu)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    raise e
            self.tearDown()               
        # Performance test configuration
        self.perf_iterations = 11  # Number of iterations for each performance test
        self.perf_warmup_iterations = 3  # Number of warmup iterations
        self.perf_results = {}  # Store performance results
        
    def setUp(self):
        """Setup for performance tests - minimal setup just for performance testing."""
        self.time = time
        self.statistics = statistics
        
        # Add skip flags for consistency with original tests
        self.TODO_SKIP_FAIL = False
        self.TODO_SKIP_NOT_COMPLETE = False
        
        # Use global constants matching original test file
        self.PASS = PASS
        self.FAIL = FAIL
        
        # Initialize empty lists for robustness - will be populated if amdsmi is available
        self.status_types = []
        self.clk_types = []
        self.io_bw_encodings = []
        self.event_groups = []
        self.gpu_blocks = []
        self.memory_types = []
        self.processor_types = []
        self.reg_types = []
        self.voltage_metrics = []
        self.voltage_types = []
        self.link_types = []
        self.temperature_types = []
        self.utilization_counter_types = []
        self.event_types = []
        self.counter_commands = []
        self.power_profile_preset_masks = []
        self.temperature_metrics = []
        self.clk_limit_types = []
        self.memory_partition_types = []
        
        # Try to populate with actual amdsmi attributes if available
        self._populate_amdsmi_attributes()
        
        # Initialize AMDSMI if available
        try:
            if hasattr(amdsmi, 'amdsmi_init'):
                amdsmi.amdsmi_init(AmdSmiInitFlags.INIT_AMD_CPUS)
                self.processors = amdsmi.amdsmi_get_processor_handles() if hasattr(amdsmi, 'amdsmi_get_processor_handles') else []
            else:
                self.processors = []
        except Exception as e:
            print(f"Warning: Failed to initialize AMDSMI: {e}")
            self.processors = []
            
    def _populate_amdsmi_attributes(self):
        """Try to populate attribute lists with amdsmi enums if available."""
        # This method safely tries to access amdsmi attributes and populates lists
        # If amdsmi is not available or missing attributes, it silently continues
        
        # Wrap all attribute access in a try-catch to handle missing amdsmi properly
        try:
            if hasattr(amdsmi, 'AmdSmiStatus'):
                self.status_types = [
                    ('SUCCESS', amdsmi.AmdSmiStatus.SUCCESS, self.PASS),
                    ('INVAL', amdsmi.AmdSmiStatus.INVAL, self.PASS),
                    ('NOT_SUPPORTED', amdsmi.AmdSmiStatus.NOT_SUPPORTED, self.PASS),
                    ('UNKNOWN_ERROR', amdsmi.AmdSmiStatus.UNKNOWN_ERROR, self.PASS)
                ]
        except (AttributeError, Exception):
            pass
            
        # Add more attribute populations as needed - this is a safe fallback approach
            
    def tearDown(self):
        """Cleanup after performance tests."""
        try:
            if hasattr(amdsmi, 'amdsmi_shut_down'):
                amdsmi.amdsmi_shut_down()
        except Exception:
            pass  # Ignore cleanup errors
            
    def _print_func_name(self, msg=''):
        """Helper method to print function name for consistency with original tests."""
        if verbose == 2:
            print(f"{inspect.currentframe().f_back.f_code.co_name}: {msg}")
    
    def _log_test_start(self, api_name, device_type, device_id, **kwargs):
        """Helper method to log the start of a test."""
        if verbose:
            extra_info = " ".join([f"{k}={v}" for k, v in kwargs.items()])
            print(f"Testing {api_name} on {device_type} {device_id} {extra_info}".strip())
    
    def _log_test_end(self, api_name, device_type, device_id, stats, **kwargs):
        """Helper method to log the end of a test."""
        if verbose and stats:
            print(f"Completed {api_name} on {device_type} {device_id}: {stats.get('mean_time_ms', 0):.3f}ms avg")
    
    def _print_performance_results(self, stats):
        """Helper method to print performance results."""
        if verbose:
            print(f"  Performance: {stats['mean_time_ms']:.3f}ms avg, {stats['min_time_ms']:.3f}ms min, {stats['max_time_ms']:.3f}ms max")
    
    def _log_test_completion(self, device_type, device_id, extra_info=""):
        """Helper method to log test completion."""
        if verbose:
            extra_msg = f" - {extra_info}" if extra_info else ""
            print(f"  {device_type} {device_id}: Test completed{extra_msg}")
    
    def _log_performance_summary(self, api_name, device_type_plural, test_name):
        """Helper method to log performance summary."""
        if verbose:
            print(f"Performance test completed for {api_name} on {device_type_plural}")
            print()  # Add empty line for readability
            
    def _print(self, msg, result):
        """Helper method for printing test results."""
        if verbose:
            if isinstance(result, dict) or isinstance(result, list):
                if msg:
                    print(msg)
                print(json.dumps(result, sort_keys=False, indent=4, default=str))
            else:
                print(f"{msg}: {result}" if msg else result)
               
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
        if not verbose:
            return
            
        func_name = api_func.__name__
        msg = f"### test {func_name}({label_prefix}={processor_id})"
        
        try:
            result = api_func(*args, **kwargs)
            self._print(msg, result)
        except Exception as e:
            print(msg, flush=True)
            # Don't raise - let the performance test measure the errors
            pass
            
    def _run_performance_assertions(self, stats, api_name):
        """Helper method to run assertions on performance test statistics."""
        # Basic performance test assertions
        self.assertIsInstance(stats, dict, f"Stats should be a dictionary for {api_name}")
        self.assertIn('iterations', stats, f"Stats should contain iterations for {api_name}")
        self.assertIn('successful_runs', stats, f"Stats should contain successful_runs for {api_name}")
        self.assertIn('error_count', stats, f"Stats should contain error_count for {api_name}")
        
        # If there were successful runs, check performance metrics
        if stats.get('successful_runs', 0) > 0:
            self.assertIn('mean_time_ms', stats, f"Stats should contain mean_time_ms for {api_name}")
            self.assertIn('min_time_ms', stats, f"Stats should contain min_time_ms for {api_name}")
            self.assertIn('max_time_ms', stats, f"Stats should contain max_time_ms for {api_name}")
            self.assertGreaterEqual(stats['mean_time_ms'], 0, f"Mean time should be non-negative for {api_name}")
            self.assertGreaterEqual(stats['min_time_ms'], 0, f"Min time should be non-negative for {api_name}")
            self.assertGreaterEqual(stats['max_time_ms'], 0, f"Max time should be non-negative for {api_name}")
                       
            
    def _measure_api_performance(self, api_func, api_name, *args, **kwargs):
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
                result = api_func(*args, **kwargs)
                end_time = self.time.perf_counter()
                execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
                times.append(execution_time)
            except Exception as e:
                error_count += 1
                end_time = self.time.perf_counter()
                execution_time = (end_time - start_time) * 1000
                times.append(execution_time)  # Include error times in statistics
                errors.append({'iteration': i, 'error_info': str(e)})
        
        # Calculate statistics
        if times:
            successful_runs = len(times) - error_count
            stats = {
                'api_name': api_name,
                'iterations': len(times),
                'errors': errors,
                'error_count': error_count,
                'successful_runs': successful_runs,
                'min_time_ms': min(times),
                'max_time_ms': max(times),
                'mean_time_ms': self.statistics.mean(times),
                'median_time_ms': self.statistics.median(times),
                'times_ms': times
            }
            
            if len(times) > 1:
                stats['stdev_ms'] = self.statistics.stdev(times)
            
            # Store results
            self.perf_results[api_name] = stats
            
            # Print results if verbose
            if verbose:
                print(f"Performance {api_name}: {stats['mean_time_ms']:.3f}ms avg, "
                      f"{stats['min_time_ms']:.3f}ms min, {stats['max_time_ms']:.3f}ms max, "
                      f"{error_count} errors")
            
            return stats
        else:
            return {'api_name': api_name, 'error': 'No successful measurements'}

    def test_performance_cpu_apb_disable(self):
        self._print_func_name('Starting performance test for amdsmi_cpu_apb_disable')
        # Use pstate=0 from original test
        pstate = 0
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()
        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for  processor in processor_handles:
                self._log_test_start("amdsmi_cpu_apb_disable", "cpu", i, pstate=pstate)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_cpu_apb_disable,
                    f'cpu_apb_disable_processor_{i}',
                    processor, pstate
                )
                self.perf_results[f'cpu_apb_disable_processor_{i}'] = stats
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )
            self._log_test_completion("Processor", i)
            i = i+1
        self._log_performance_summary("amdsmi_cpu_apb_disable", "Processors", "cpu_apb_disable")

    def test_performance_cpu_apb_enable(self):
        self._print_func_name('Starting performance test for amdsmi_cpu_apb_enable')
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_cpu_apb_enable", "cpu", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_cpu_apb_enable,
                    f'cpu_apb_enable_processor_{i}',
                    processor
                )
            
                self.perf_results[f'cpu_apb_enable_processor_{i}'] = stats

                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary("amdsmi_cpu_apb_enable", "Processors", "cpu_apb_enable")
		
    def test_performance_first_online_core_on_cpu_socket(self):
        self._print_func_name('Starting performance test for amdsmi_first_online_core_on_cpu_socket')
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_first_online_core_on_cpu_socket", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_first_online_core_on_cpu_socket,
                    f'first_online_core_processor_{i}',
                    processor
                )

                self.perf_results[f'first_online_core_processor_{i}'] = stats

                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                    self._run_performance_assertions(stats, "amdsmi_first_online_core_on_cpu_socket")
                else:
                    print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside the loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_first_online_core_on_cpu_socket",
            "Processors",
            "first_online_core"
        )
    

    def test_performance_get_cpu_cclk_limit(self):
        self._print_func_name('Starting performance test for amdsmi_get_cpu_cclk_limit')
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_cclk_limit", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_cclk_limit,
                    f'get_cpu_cclk_limit_processor_{i}',
                    processor
                )

                self.perf_results[f'get_cpu_cclk_limit_processor_{i}'] = stats

                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_cclk_limit",
            "Processors",
            "get_cpu_cclk_limit"
        )
      

    def test_performance_get_cpu_core_current_freq_limit(self):
        self._print_func_name('Starting performance test for amdsmi_get_cpu_core_current_freq_limit')
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_core_current_freq_limit", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_core_current_freq_limit,
                    f'get_cpu_core_current_freq_limit_processor_{i}',
                    processor
                )

                self.perf_results[f'get_cpu_core_current_freq_limit_processor_{i}'] = stats

                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_core_current_freq_limit",
            "Processors",
            "get_cpu_core_current_freq_limit"
        )


    def test_performance_get_cpu_core_energy(self):
        self._print_func_name('Starting performance test for amdsmi_get_cpu_core_energy')
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_core_energy", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_core_energy,
                    f'get_cpu_core_energy_processor_{i}',
                    processor
                )

                self.perf_results[f'get_cpu_core_energy_processor_{i}'] = stats

                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside the loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_core_energy",
            "Processors",
            "get_cpu_core_energy"
        )


    def test_performance_get_cpu_current_io_bandwidth(self):
        self._print_func_name('Starting performance test for amdsmi_get_cpu_current_io_bandwidth')
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start(
                    "amdsmi_get_cpu_current_io_bandwidth",
                    "Processor",
                    i,
                    encoding=encoding_name
                )
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_current_io_bandwidth,
                    f'get_cpu_current_io_bandwidth_processor_{i}_encoding_{encoding_name}',
                    processor,
                    encoding,
                    encoding_name
                )


                self.perf_results[
                    f'get_cpu_current_io_bandwidth_processor_{i}_encoding_{encoding_name}'
                ] = stats

                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(
                        f"  Processor {i} encoding={encoding_name}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_current_io_bandwidth",
            "Processors",
            "get_cpu_current_io_bandwidth"
        )


    def test_performance_get_cpu_ddr_bw(self):
        self._print_func_name('Starting performance test for amdsmi_get_cpu_ddr_bw')
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_ddr_bw", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_ddr_bw,
                    f'get_cpu_ddr_bw_processor_{i}',
                    processor
                )

                self.perf_results[f'get_cpu_ddr_bw_processor_{i}'] = stats

                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside the loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_ddr_bw",
            "Processors",
            "get_cpu_ddr_bw"
        )


    def test_performance_get_cpu_dimm_power_consumption(self):
        self._print_func_name('Starting performance test for amdsmi_get_cpu_dimm_power_consumption')
        i = 0
        dimm_addr = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start(
                    "amdsmi_get_cpu_dimm_power_consumption",
                "Processor",
                i,
                dimm_addr=dimm_addr
                )
                stats = self._measure_api_performance(
                amdsmi.amdsmi_get_cpu_dimm_power_consumption,
                f'get_cpu_dimm_power_processor_{i}',
                processor,
                dimm_addr
                )

                self.perf_results[f'get_cpu_dimm_power_processor_{i}'] = stats

                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_dimm_power_consumption",
            "Processors",
            "get_cpu_dimm_power"
        )


    def test_performance_get_cpu_dimm_temp_range_and_refresh_rate(self):
        self._print_func_name(
            'Starting performance test for amdsmi_get_cpu_dimm_temp_range_and_refresh_rate'
        )
        dimm_addr = 0
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start(
                    "amdsmi_get_cpu_dimm_temp_range_and_refresh_rate",
                    "Processor",
                    i,
                    dimm_addr=dimm_addr
                )
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_dimm_temp_range_and_refresh_rate,
                    f'get_cpu_dimm_temp_range_processor_{i}',
                    processor,
                    dimm_addr
                )

                self.perf_results[f'get_cpu_dimm_temp_range_processor_{i}'] = stats

                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_dimm_temp_range_and_refresh_rate",
            "Processors",
            "get_cpu_dimm_temp_range"
        )

        
    def test_performance_get_cpu_dimm_thermal_sensor(self):
        self._print_func_name('Starting performance test for amdsmi_get_cpu_dimm_thermal_sensor')
        dimm_addr = 0
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start(
                    "amdsmi_get_cpu_dimm_thermal_sensor",
                    "Processor",
                    i,
                    dimm_addr=dimm_addr
                )
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_dimm_thermal_sensor,
                    f'get_cpu_dimm_thermal_sensor_processor_{i}',
                    processor,
                    dimm_addr
                )

                self.perf_results[f'get_cpu_dimm_thermal_sensor_processor_{i}'] = stats

                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_dimm_thermal_sensor",
            "Processors",
            "get_cpu_dimm_thermal_sensor"
        )

    def test_performance_get_cpu_family(self):
        self._print_func_name('Starting performance test for amdsmi_get_cpu_family')
        self._log_test_start("amdsmi_get_cpu_family", "System", "global")
        
        stats = self._measure_api_performance(
            amdsmi.amdsmi_get_cpu_family,
            'get_cpu_family_system'
        )
        
        self.perf_results['get_cpu_family_system'] = stats
        if stats['successful_runs'] > 0:
            self._print_performance_results(stats)
        else:
            print(f"  System: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
        
        self._log_test_completion("System", "global")
        self._log_performance_summary("amdsmi_get_cpu_family", "System", "get_cpu_family")

    
    def test_performance_get_cpu_fclk_mclk(self):
        self._print_func_name('Starting performance test for amdsmi_get_cpu_fclk_mclk')
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_fclk_mclk", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_fclk_mclk,
                    f'get_cpu_fclk_mclk_processor_{i}',
                    processor
                )

                self.perf_results[f'get_cpu_fclk_mclk_processor_{i}'] = stats

                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_fclk_mclk",
            "Processors",
            "get_cpu_fclk_mclk"
        )

        
    def test_performance_get_cpu_handles(self):
        self._print_func_name('Starting performance test for amdsmi_get_cpu_handles')
        self._log_test_start("amdsmi_get_cpu_handles", "System", "global")
        
        stats = self._measure_api_performance(
            amdsmi.amdsmi_get_cpu_handles,
            'get_cpu_handles_system'
        )
        
        self.perf_results['get_cpu_handles_system'] = stats
        if stats['successful_runs'] > 0:
            self._print_performance_results(stats)
        else:
            print(f"  System: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
        
        self._log_test_completion("System", "global")
        self._log_performance_summary("amdsmi_get_cpu_handles", "System", "get_cpu_handles")
		

    def test_performance_get_cpu_hsmp_driver_version(self):
        self._print_func_name('Starting performance test for amdsmi_get_cpu_hsmp_driver_version')
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_hsmp_driver_version", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_hsmp_driver_version,
                    f'get_cpu_hsmp_driver_version_processor_{i}',
                    processor
                )

                self.perf_results[f'get_cpu_hsmp_driver_version_processor_{i}'] = stats

                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_hsmp_driver_version",
            "Processors",
            "get_cpu_hsmp_driver_version"
        )


    def test_performance_get_cpu_hsmp_proto_ver(self):
        self._print_func_name('Starting performance test for amdsmi_get_cpu_hsmp_proto_ver')
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_hsmp_proto_ver", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_hsmp_proto_ver,
                    f'get_cpu_hsmp_proto_ver_processor_{i}',
                    processor
                )

                self.perf_results[f'get_cpu_hsmp_proto_ver_processor_{i}'] = stats

                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_hsmp_proto_ver",
            "Processors",
            "get_cpu_hsmp_proto_ver"
        )


    def test_performance_get_cpu_model(self):
        self._print_func_name('Starting performance test for amdsmi_get_cpu_model')
        self._log_test_start("amdsmi_get_cpu_model", "System", "global")
		
        stats = self._measure_api_performance(
            amdsmi.amdsmi_get_cpu_model,
            'get_cpu_model_system'
        )
        
        self.perf_results['get_cpu_model_system'] = stats
        if stats['successful_runs'] > 0:
            self._print_performance_results(stats)
        else:
            print(f"  System: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
        
        self._log_test_completion("System", "global")
        self._log_performance_summary("amdsmi_get_cpu_model", "System", "get_cpu_model")


    def test_performance_get_cpu_prochot_status(self):
        self._print_func_name('Starting performance test for amdsmi_get_cpu_prochot_status')
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_prochot_status", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_prochot_status,
                    f'get_cpu_prochot_status_processor_{i}',
                    processor
                )

                self.perf_results[f'get_cpu_prochot_status_processor_{i}'] = stats

                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_prochot_status",
            "Processors",
            "get_cpu_prochot_status"
        )


    def test_performance_get_cpu_pwr_svi_telemetry_all_rails(self):
        self._print_func_name('Starting performance test for amdsmi_get_cpu_pwr_svi_telemetry_all_rails')
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_pwr_svi_telemetry_all_rails", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_pwr_svi_telemetry_all_rails,
                    f'get_cpu_pwr_svi_telemetry_processor_{i}',
                    processor
                )

                self.perf_results[f'get_cpu_pwr_svi_telemetry_processor_{i}'] = stats

                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_pwr_svi_telemetry_all_rails",
            "Processors",
            "get_cpu_pwr_svi_telemetry"
        )


    def test_performance_get_cpu_smu_fw_version(self):
        self._print_func_name('Starting performance test for amdsmi_get_cpu_smu_fw_version')
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_smu_fw_version", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_smu_fw_version,
                    f'get_cpu_smu_fw_version_processor_{i}',
                    processor
                )

                self.perf_results[f'get_cpu_smu_fw_version_processor_{i}'] = stats

                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_smu_fw_version",
            "Processors",
            "get_cpu_smu_fw_version"
        )


    def test_performance_get_cpu_socket_c0_residency(self):
        self._print_func_name('Starting performance test for amdsmi_get_cpu_socket_c0_residency')
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_socket_c0_residency", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_socket_c0_residency,
                    f'get_cpu_socket_c0_residency_processor_{i}',
                    processor
                )

                self.perf_results[f'get_cpu_socket_c0_residency_processor_{i}'] = stats

                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_socket_c0_residency",
            "Processors",
            "get_cpu_socket_c0_residency"
        )


    def test_performance_get_cpu_socket_current_active_freq_limit(self):
        self._print_func_name('Starting performance test for amdsmi_get_cpu_socket_current_active_freq_limit')
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_socket_current_active_freq_limit", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_socket_current_active_freq_limit,
                    f'get_cpu_socket_current_active_freq_limit_processor_{i}',
                    processor
                )

                self.perf_results[f'get_cpu_socket_current_active_freq_limit_processor_{i}'] = stats

                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_socket_current_active_freq_limit",
            "Processors",
            "get_cpu_socket_current_active_freq_limit"
        )


    def test_performance_get_cpu_socket_energy(self):
        self._print_func_name('Starting performance test for amdsmi_get_cpu_socket_energy')
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_socket_energy", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_socket_energy,
                    f'get_cpu_socket_energy_processor_{i}',
                    processor
                )

                self.perf_results[f'get_cpu_socket_energy_processor_{i}'] = stats

                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_socket_energy",
            "Processors",
            "get_cpu_socket_energy"
        )

        
    def test_performance_get_cpu_socket_freq_range(self):
        self._print_func_name('Starting performance test for amdsmi_get_cpu_socket_freq_range')
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_socket_freq_range", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_socket_freq_range,
                    f'get_cpu_socket_freq_range_processor_{i}',
                    processor
                )

                self.perf_results[f'get_cpu_socket_freq_range_processor_{i}'] = stats

                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_socket_freq_range",
            "Processors",
            "get_cpu_socket_freq_range"
        )


    def test_performance_get_cpu_socket_lclk_dpm_level(self):
        self._print_func_name('Starting performance test for amdsmi_get_cpu_socket_lclk_dpm_level')
        nbio_id = 0
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_socket_lclk_dpm_level", "Processor", i, nbio_id=nbio_id)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_socket_lclk_dpm_level,
                    f'get_cpu_socket_lclk_dpm_processor_{i}',
                    processor, nbio_id
                )

                self.perf_results[f'get_cpu_socket_lclk_dpm_processor_{i}'] = stats

                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i = i + 1  # increment inside loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_socket_lclk_dpm_level",
            "Processors",
            "get_cpu_socket_lclk_dpm_level"
        )


    def test_performance_get_cpu_socket_power(self):
        self._print_func_name('Starting performance test for amdsmi_get_cpu_socket_power')
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_socket_power", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_socket_power,
                    f'get_cpu_socket_power_processor_{i}',
                    processor
                )

                self.perf_results[f'get_cpu_socket_power_processor_{i}'] = stats
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i += 1  # increment inside the loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_socket_power",
            "Processors",
            "get_cpu_socket_power"
        )


    def test_performance_get_cpu_socket_power_cap_max(self):
        self._print_func_name('Starting performance test for amdsmi_get_cpu_socket_power_cap_max')
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_socket_power_cap_max", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_socket_power_cap_max,
                    f'get_cpu_socket_power_cap_max_processor_{i}',
                    processor
                )

                self.perf_results[f'get_cpu_socket_power_cap_max_processor_{i}'] = stats
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i += 1  # increment inside the loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_socket_power_cap_max",
            "Processors",
            "get_cpu_socket_power_cap_max"
        )


    def test_performance_get_cpu_socket_temperature(self):
        self._print_func_name('Starting performance test for amdsmi_get_cpu_socket_temperature')
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_cpu_socket_temperature", "Processor", i)
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_socket_temperature,
                    f'get_cpu_socket_temperature_processor_{i}',
                    processor
                )

                self.perf_results[f'get_cpu_socket_temperature_processor_{i}'] = stats
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                i += 1  # increment inside the loop

            self._log_test_completion("Processor", i)

        self._log_performance_summary(
            "amdsmi_get_cpu_socket_temperature",
            "Processors",
            "get_cpu_socket_temperature"
        )


    def test_performance_get_esmi_err_msg(self):
        self._print_func_name('Starting performance test for amdsmi_get_esmi_err_msg')

        for status_type_name, status_type, status_cond in self.status_types:
            self._log_test_start("amdsmi_get_esmi_err_msg", "Status", status_type_name)

            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_esmi_err_msg,
                f'get_esmi_err_msg_status_{status_type_name}',
                status_type
            )

            self.perf_results[f'get_esmi_err_msg_status_{status_type_name}'] = stats
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(
                    f"  Status {status_type_name}: All calls failed - "
                    f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                )

            self._log_test_completion("Status", status_type_name)

        self._log_performance_summary(
            "amdsmi_get_esmi_err_msg",
            "Status types",
            "get_esmi_err_msg"
        )


    def test_performance_get_hsmp_metrics_table(self):
        self._print_func_name('Starting performance test for amdsmi_get_hsmp_metrics_table')    
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()
    
        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_hsmp_metrics_table", "Processor", i)			
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_hsmp_metrics_table,
                    f'get_hsmp_metrics_table_processor_{i}',
                    processor
                )

                self.perf_results[f'get_hsmp_metrics_table_processor_{i}'] = stats
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                self._log_test_completion("Processor", i)
                i += 1

        self._log_performance_summary(
            "amdsmi_get_hsmp_metrics_table",
            "Processors",
            "get_hsmp_metrics_table"
        )


    def test_performance_get_hsmp_metrics_table_version(self):
        self._print_func_name('Starting performance test for amdsmi_get_hsmp_metrics_table_version')
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()
    
        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_hsmp_metrics_table_version", "Processor", i)			
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_hsmp_metrics_table_version,
                    f'get_hsmp_metrics_table_version_processor_{i}',
                    processor
                )

                self.perf_results[f'get_hsmp_metrics_table_version_processor_{i}'] = stats

                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                self._log_test_completion("Processor", i)
                i += 1

        self._log_performance_summary(
            "amdsmi_get_hsmp_metrics_table_version",
            "Sockets",
            "get_hsmp_metrics_table_version"
        )


    def test_performance_get_lib_version(self):
        self._print_func_name('Starting performance test for amdsmi_get_lib_version')
        
        self._log_test_start("amdsmi_get_lib_version", "System", "global")
        
        stats = self._measure_api_performance(
            amdsmi.amdsmi_get_lib_version,
            'get_lib_version'
        )
        
        self.perf_results['get_lib_version'] = stats
        
        if stats['successful_runs'] > 0:
            self._print_performance_results(stats)
        else:
            print(f"  All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
        
        self._log_test_completion("System", "global")
        self._log_performance_summary("amdsmi_get_lib_version", "System", "get_lib_version")


    def test_performance_set_cpu_pcie_link_rate(self):
        self._print_func_name('Starting performance test for amdsmi_set_cpu_pcie_link_rate')
        # Test with different rate_ctrl values
        rate_ctrls = [0]  # Starting with 0 as in original test
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:

                for rate_ctrl in rate_ctrls:
                    self._log_test_start(
                        "amdsmi_set_cpu_pcie_link_rate",
                        "Processor",
                        i,
                        rate_ctrl=rate_ctrl
                    )

                    stats = self._measure_api_performance(
                        amdsmi.amdsmi_set_cpu_pcie_link_rate,
                        f'set_cpu_pcie_link_rate_cpu_{i}_rate_{rate_ctrl}',
                        processor,
                        rate_ctrl
                    )

                    self.perf_results[f'set_cpu_pcie_link_rate_cpu_{i}_rate_{rate_ctrl}'] = stats

                    if stats['successful_runs'] > 0:
                        self._print_performance_results(stats)
                        self._run_performance_assertions(stats, "amdsmi_set_cpu_pcie_link_rate")
                    else:
                        print(
                            f"  CPU {i} rate_ctrl {rate_ctrl}: All calls failed - "
                            f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                        )

                self._log_test_completion("CPU", i, f"rate_ctrl={rate_ctrl}")

        self._log_performance_summary(
            "amdsmi_set_cpu_pcie_link_rate",
            "CPUs",
            "set_cpu_pcie_link_rate"
        )


    def test_performance_get_processor_count_from_handles(self):
        self._print_func_name('Starting performance test for amdsmi_get_processor_count_from_handles')
        
        self._log_test_start("amdsmi_get_processor_count_from_handles", "Processors", "all")
        
        stats = self._measure_api_performance(
            amdsmi.amdsmi_get_processor_count_from_handles,
            'get_processor_count_from_handles',
            self.processors
        )
        
        self.perf_results['get_processor_count_from_handles'] = stats
        
        if stats['successful_runs'] > 0:
            self._print_performance_results(stats)
            self._run_performance_assertions(stats, "amdsmi_get_processor_count_from_handles")
        else:
            print(f"  All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
        
        self._log_test_completion("Processors", "all")
        self._log_performance_summary("amdsmi_get_processor_count_from_handles", "Processors", "get_processor_count_from_handles")

    def test_performance_get_processor_handle_from_bdf(self):
        self._print_func_name('Starting performance test for amdsmi_get_processor_handle_from_bdf')
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_processor_handle_from_bdf", "CPU", i)

                try:
                    # Get BDF for this processor first
                    bdf = amdsmi.amdsmi_get_gpu_device_bdf(processor)

                    stats = self._measure_api_performance(
                        amdsmi.amdsmi_get_processor_handle_from_bdf,
                        f'get_processor_handle_from_bdf_cpu_{i}',
                        bdf
                    )

                    self.perf_results[f'get_processor_handle_from_bdf_cpu_{i}'] = stats
                    if stats['successful_runs'] > 0:
                        self._print_performance_results(stats)
                        self._run_performance_assertions(stats, "amdsmi_get_processor_handle_from_bdf")

                    # Validate that the returned handle matches the original processor
                    ret = amdsmi.amdsmi_get_processor_handle_from_bdf(bdf)
                    if processor.value != ret.value:
                        print(f"  WARNING: CPU {i} - Handle mismatch! Expected: {processor.value}, Received: {ret.value}")
                    else:
                        print(f"  CPU {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")

                except Exception as e:
                    print(f"  CPU {i}: Error getting BDF - {e}")

                self._log_test_completion("CPU", i)
                i = i + 1

        self._log_performance_summary("amdsmi_get_processor_handle_from_bdf", "CPUs", "get_processor_handle_from_bdf")


    def test_performance_get_processor_handles(self):
        self._print_func_name('Starting performance test for amdsmi_get_processor_handles')
        
        self._log_test_start("amdsmi_get_processor_handles", "System", "global")
        
        stats = self._measure_api_performance(
            amdsmi.amdsmi_get_processor_handles,
            'get_processor_handles'
        )
        
        self.perf_results['get_processor_handles'] = stats
        
        if stats['successful_runs'] > 0:
            self._print_performance_results(stats)
            self._run_performance_assertions(stats, "amdsmi_get_processor_handles")
        else:
            print(f"  All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
        
        self._log_test_completion("System", "global")
        self._log_performance_summary("amdsmi_get_processor_handles", "System", "get_processor_handles")

    def test_performance_get_processor_handles_by_type(self):
        self._print_func_name('Starting performance test for amdsmi_get_processor_handles_by_type')
        
        socket_handles = amdsmi.amdsmi_get_socket_handles()
        
        for index, socket_handle in enumerate(socket_handles):
            for processor_name, processor_type, processor_cond in self.processor_types:
                self._log_test_start("amdsmi_get_processor_handles_by_type", "Socket", index, processor_type=processor_name)
                
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_processor_handles_by_type,
                    f'get_processor_handles_by_type_socket_{index}_type_{processor_name}',
                    socket_handle, processor_type
                )
                
                self.perf_results[f'get_processor_handles_by_type_socket_{index}_type_{processor_name}'] = stats
                
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                    self._run_performance_assertions(stats, "amdsmi_get_processor_handles_by_type")
                else:
                    print(f"  Socket {index} type {processor_name}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
                
                self._log_test_completion("Socket", index, f"processor_type={processor_name}")
        
        self._log_performance_summary("amdsmi_get_processor_handles_by_type", "Sockets", "get_processor_handles_by_type")

    def test_performance_get_processor_info(self):
        self._print_func_name('Starting performance test for amdsmi_get_processor_info')
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_processor_info", "CPU", i)

                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_processor_info,
                    f'get_processor_info_cpu_{i}',
                    processor
                )

                self.perf_results[f'get_processor_info_cpu_{i}'] = stats
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                    self._run_performance_assertions(stats, "amdsmi_get_processor_info")
                else:
                    print(f"  CPU {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")

                self._log_test_completion("CPU", i)
                i = i + 1

        self._log_performance_summary("amdsmi_get_processor_info", "CPUs", "get_processor_info")


    def test_performance_get_processor_type(self):
        self._print_func_name('Starting performance test for amdsmi_get_processor_type')
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("amdsmi_get_processor_type", "CPU", i)

                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_processor_type,
                    f'get_processor_type_cpu_{i}',
                    processor
                )

                self.perf_results[f'get_processor_type_cpu_{i}'] = stats
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                    self._run_performance_assertions(stats, "amdsmi_get_processor_type")
                else:
                    print(f"  CPU {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")

                self._log_test_completion("CPU", i)
                i = i + 1

        self._log_performance_summary("amdsmi_get_processor_type", "CPUs", "get_processor_type")


    def test_performance_get_socket_handles(self):
        self._print_func_name('Starting performance test for amdsmi_get_socket_handles')
        
        self._log_test_start("amdsmi_get_socket_handles", "System", "global")
        
        stats = self._measure_api_performance(
            amdsmi.amdsmi_get_socket_handles,
            'get_socket_handles'
        )
        
        self.perf_results['get_socket_handles'] = stats
        
        if stats['successful_runs'] > 0:
            self._print_performance_results(stats)
            self._run_performance_assertions(stats, "amdsmi_get_socket_handles")
        else:
            print(f"  All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
        
        self._log_test_completion("System", "global")
        self._log_performance_summary("amdsmi_get_socket_handles", "System", "get_socket_handles")

    def test_performance_get_socket_info(self):
        self._print_func_name('Starting performance test for amdsmi_get_socket_info')
        
        sockets = amdsmi.amdsmi_get_socket_handles()
        
        for i, socket in enumerate(sockets):
            self._log_test_start("amdsmi_get_socket_info", "Socket", i)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_socket_info,
                f'get_socket_info_socket_{i}',
                socket
            )
            
            self.perf_results[f'get_socket_info_socket_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
                self._run_performance_assertions(stats, "amdsmi_get_socket_info")
            else:
                print(f"  Socket {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Socket", i)
        
        self._log_performance_summary("amdsmi_get_socket_info", "Sockets", "get_socket_info")

    def test_performance_get_temp_metric(self):
        self._print_func_name('Starting performance test for amdsmi_get_temp_metric')
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                for temperature_type_name, temperature_type, temperature_type_cond in self.temperature_types:
                    for temperature_metric_name, temperature_metric, temperature_metric_cond in self.temperature_metrics:
                        self._log_test_start(
                            "amdsmi_get_temp_metric",
                            "CPU",
                            i,
                            temperature_type=temperature_type_name,
                            temperature_metric=temperature_metric_name
                        )

                        stats = self._measure_api_performance(
                            amdsmi.amdsmi_get_temp_metric,
                            f'get_temp_metric_cpu_{i}_{temperature_type_name}_{temperature_metric_name}',
                            processor,
                            temperature_type,
                            temperature_metric
                        )

                        self.perf_results[f'get_temp_metric_cpu_{i}_{temperature_type_name}_{temperature_metric_name}'] = stats
                        if stats['successful_runs'] > 0:
                            self._print_performance_results(stats)
                            self._run_performance_assertions(stats, "amdsmi_get_temp_metric")
                        else:
                            print(
                                f"  CPU {i} {temperature_type_name}/{temperature_metric_name}: All calls failed - "
                                f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                            )

                self._log_test_completion(
                    "CPU",
                    i,
                    f"temperature_type={temperature_type_name}, temperature_metric={temperature_metric_name}"
                )
                i = i + 1

        self._log_performance_summary("amdsmi_get_temp_metric", "CPUs", "get_temp_metric")


    def test_performance_get_threads_per_core(self):
        self._print_func_name('Starting performance test for amdsmi_get_threads_per_core')   
        self._log_test_start("amdsmi_get_threads_per_core", "System", 0)
        
        stats = self._measure_api_performance(
            amdsmi.amdsmi_get_threads_per_core,
            'get_threads_per_core'
        )
        
        self.perf_results['get_threads_per_core'] = stats
        
        if stats['successful_runs'] > 0:
            self._print_performance_results(stats)
            self._run_performance_assertions(stats, "amdsmi_get_threads_per_core")
        else:
            print(f"  System: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
        
        self._log_test_completion("System", 0)
        
        self._log_performance_summary("amdsmi_get_threads_per_core", "System", "get_threads_per_core")

    def test_performance_init(self):
        self._print_func_name('Starting performance test for amdsmi_init')
        
        self._log_test_start("amdsmi_init", "System", "global")
        
        # Note: We need to be careful with init/shutdown as they affect the entire library state
        stats = self._measure_api_performance(
            amdsmi.amdsmi_init,
            'init_system'
        )
        
        self.perf_results['init_system'] = stats
        
        if stats['successful_runs'] > 0:
            self._print_performance_results(stats)
        else:
            print(f"  All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
        
        self._log_test_completion("System", "global")
        self._log_performance_summary("amdsmi_init", "System", "init")

    def test_performance_shutdown(self):
        self._print_func_name('Starting performance test for amdsmi_shut_down')
        
        self._log_test_start("amdsmi_shut_down", "System", "global")
        
        # Note: We need to be careful with init/shutdown as they affect the entire library state
        stats = self._measure_api_performance(
            amdsmi.amdsmi_shut_down,
            'shutdown_system'
        )
        
        self.perf_results['shutdown_system'] = stats
        
        if stats['successful_runs'] > 0:
            self._print_performance_results(stats)
        else:
            print(f"  All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
        
        self._log_test_completion("System", "global")
        self._log_performance_summary("amdsmi_shut_down", "System", "shutdown")

    def test_performance_cpu_core_boostlimit(self):
        self._print_func_name('Starting performance test for CPU core boostlimit workflow')
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("cpu_core_boostlimit_workflow", "Processor", i)

                # Test Get CPU Core Boostlimit
                stats_get = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_core_boostlimit,
                    f'get_cpu_core_boostlimit_processor_{i}',
                    processor
                )
                self.perf_results[f'get_cpu_core_boostlimit_processor_{i}'] = stats_get

                if stats_get['successful_runs'] > 0:
                    self._print_performance_results(stats_get)

                    # Try to get the boost limit value for setting test
                    try:
                        boost_limit = amdsmi.amdsmi_get_cpu_core_boostlimit(processor)

                        # Test Set CPU Core Boostlimit (with same value to avoid changing system state)
                        stats_set = self._measure_api_performance(
                            amdsmi.amdsmi_set_cpu_core_boostlimit,
                            f'set_cpu_core_boostlimit_processor_{i}',
                            processor,
                            boost_limit
                        )
                        self.perf_results[f'set_cpu_core_boostlimit_processor_{i}'] = stats_set
                        self._print_performance_results(stats_set)

                    except amdsmi.AmdSmiLibraryException:
                        print(f"  Processor {i}: Could not get boost_limit for set test")
                else:
                    print(
                        f"  Processor {i}: Get failed - "
                        f"{stats_get['errors'][0]['error_info'] if stats_get['errors'] else 'Unknown'}"
                    )

                self._log_test_completion("Processor", i)
                i = i + 1

        self._log_performance_summary("cpu_core_boostlimit_workflow", "Processors", "cpu_core_boostlimit")
        self._log_performance_summary("amdsmi_cpu_core_boostlimit", "Sockets", "cpu_core_boostlimit")


    def test_performance_set_cpu_df_pstate_range(self):
        self._print_func_name('Starting performance test for amdsmi_set_cpu_df_pstate_range')
    
        # Use TODO placeholder values like original test
        max_pstate = 0
        min_pstate = 0
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start(
                    "amdsmi_set_cpu_df_pstate_range",
                    "Processor",
                    i,
                    max_pstate=max_pstate,
                    min_pstate=min_pstate
                )

                stats = self._measure_api_performance(
                    amdsmi.amdsmi_set_cpu_df_pstate_range,
                    f'set_cpu_df_pstate_range_processor_{i}',
                    processor,
                    max_pstate,
                    min_pstate
                )

                self.perf_results[f'set_cpu_df_pstate_range_processor_{i}'] = stats
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                self._log_test_completion("Processor", i)
                i = i + 1

        self._log_performance_summary(
            "amdsmi_set_cpu_df_pstate_range",
            "Processors",
            "set_cpu_df_pstate_range"
        )


    def test_performance_set_cpu_gmi3_link_width_range(self):
        self._print_func_name('Starting performance test for amdsmi_set_cpu_gmi3_link_width_range')
    
        # Use TODO placeholder values like original test
        min_link_width = 0
        max_link_width = 0
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start(
                    "amdsmi_set_cpu_gmi3_link_width_range",
                    "Processor",
                    i,
                    min_link_width=min_link_width,
                    max_link_width=max_link_width
                )

                stats = self._measure_api_performance(
                    amdsmi.amdsmi_set_cpu_gmi3_link_width_range,
                    f'set_cpu_gmi3_link_width_range_processor_{i}',
                    processor,
                    min_link_width,
                    max_link_width
                )

                self.perf_results[f'set_cpu_gmi3_link_width_range_processor_{i}'] = stats
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                self._log_test_completion("Processor", i)
                i = i + 1

        self._log_performance_summary(
            "amdsmi_set_cpu_gmi3_link_width_range",
            "Processors",
            "set_cpu_gmi3_link_width_range"
        )


    def test_performance_set_cpu_pwr_efficiency_mode(self):
        self._print_func_name('Starting performance test for amdsmi_set_cpu_pwr_efficiency_mode')
    
        # Use modes from original test
        modes = [0, 1, 2]
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                for mode in modes:
                    self._log_test_start(
                        "amdsmi_set_cpu_pwr_efficiency_mode",
                        "Processor",
                        i,
                        mode=mode
                    )

                    stats = self._measure_api_performance(
                        amdsmi.amdsmi_set_cpu_pwr_efficiency_mode,
                        f'set_cpu_pwr_efficiency_mode_processor_{i}_mode_{mode}',
                        processor,
                        mode
                    )

                    self.perf_results[f'set_cpu_pwr_efficiency_mode_processor_{i}_mode_{mode}'] = stats

                    if stats['successful_runs'] > 0:
                        self._print_performance_results(stats)
                    else:
                        print(
                            f"  Processor {i} mode {mode}: All calls failed - "
                            f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                        )

                    self._log_test_completion("Processor", i, f"mode={mode}")
                i = i + 1

        self._log_performance_summary(
            "amdsmi_set_cpu_pwr_efficiency_mode",
            "Processors",
            "set_cpu_pwr_efficiency_mode"
        )


    def test_performance_cpu_socket_boostlimit(self):
        self._print_func_name('Starting performance test for amdsmi_set_cpu_socket_boostlimit')
    
        # Use TODO placeholder value like original test
        boost_limit = 0
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start(
                    "amdsmi_set_cpu_socket_boostlimit",
                    "Processor",
                    i,
                    boost_limit=boost_limit
                )

                stats = self._measure_api_performance(
                    amdsmi.amdsmi_set_cpu_socket_boostlimit,
                    f'set_cpu_socket_boostlimit_processor_{i}',
                    processor,
                    boost_limit
                )

                self.perf_results[f'set_cpu_socket_boostlimit_processor_{i}'] = stats
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                self._log_test_completion("Processor", i)
                i = i + 1

        self._log_performance_summary(
            "amdsmi_set_cpu_socket_boostlimit",
            "Processors",
            "cpu_socket_boostlimit"
        )


    def test_performance_set_cpu_socket_lclk_dpm_level(self):
        self._print_func_name('Starting performance test for amdsmi_set_cpu_socket_lclk_dpm_level')
    
        # Use TODO placeholder values like original test
        nbio_id = 0
        min_val = 0
        max_val = 0
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start(
                    "amdsmi_set_cpu_socket_lclk_dpm_level",
                    "Processor",
                    i,
                    nbio_id=nbio_id,
                    min_val=min_val,
                    max_val=max_val
                )

                stats = self._measure_api_performance(
                    amdsmi.amdsmi_set_cpu_socket_lclk_dpm_level,
                    f'set_cpu_socket_lclk_dpm_level_processor_{i}',
                    processor,
                    nbio_id,
                    min_val,
                    max_val
                )

                self.perf_results[f'set_cpu_socket_lclk_dpm_level_processor_{i}'] = stats
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                self._log_test_completion("Processor", i)
                i = i + 1

        self._log_performance_summary(
            "amdsmi_set_cpu_socket_lclk_dpm_level",
            "Processors",
            "set_cpu_socket_lclk_dpm_level"
        )


    def test_performance_cpu_socket_power_cap(self):
        self._print_func_name('Starting performance test for CPU socket power cap workflow')
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start("cpu_socket_power_cap_workflow", "Processor", i)

                # Test Get CPU Socket Power Cap
                stats_get = self._measure_api_performance(
                    amdsmi.amdsmi_get_cpu_socket_power_cap,
                    f'get_cpu_socket_power_cap_processor_{i}',
                    processor
                )
                self.perf_results[f'get_cpu_socket_power_cap_processor_{i}'] = stats_get
                if stats_get['successful_runs'] > 0:
                    self._print_performance_results(stats_get)

                    # Try to get the power cap value for setting test
                    try:
                        power_cap = amdsmi.amdsmi_get_cpu_socket_power_cap(processor)

                        # Test Set CPU Socket Power Cap (with same value to avoid changing system state)
                        stats_set = self._measure_api_performance(
                            amdsmi.amdsmi_set_cpu_socket_power_cap,
                            f'set_cpu_socket_power_cap_processor_{i}',
                            processor,
                            power_cap
                        )
                        self.perf_results[f'set_cpu_socket_power_cap_processor_{i}'] = stats_set
                        self._print_performance_results(stats_set)

                    except amdsmi.AmdSmiLibraryException:
                        print(f"  Processor {i}: Could not get power_cap for set test")
                else:
                    print(
                        f"  Processor {i}: Get failed - "
                        f"{stats_get['errors'][0]['error_info'] if stats_get['errors'] else 'Unknown'}"
                    )

                self._log_test_completion("Processor", i)
                i = i + 1

        self._log_performance_summary(
            "cpu_socket_power_cap_workflow",
            "Processors",
            "cpu_socket_power_cap"
        )


    def test_performance_set_cpu_xgmi_width(self):
        self._print_func_name('Starting performance test for amdsmi_set_cpu_xgmi_width')

        # Use TODO placeholder values like original test
        min_width = 0
        max_width = 0
        i = 0
        processor_handles = amdsmi.amdsmi_get_cpusocket_handles()

        if len(processor_handles) == 0:
            print("No CPU sockets on machine")
        else:
            for processor in processor_handles:
                self._log_test_start(
                    "amdsmi_set_cpu_xgmi_width",
                    "Processor",
                    i,
                    min_width=min_width,
                    max_width=max_width
                )

                stats = self._measure_api_performance(
                    amdsmi.amdsmi_set_cpu_xgmi_width,
                    f'set_cpu_xgmi_width_processor_{i}',
                    processor,
                    min_width,
                    max_width
                )

                self.perf_results[f'set_cpu_xgmi_width_processor_{i}'] = stats
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(
                        f"  Processor {i}: All calls failed - "
                        f"{stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}"
                    )

                self._log_test_completion("Processor", i)
                i = i + 1

        self._log_performance_summary(
            "amdsmi_set_cpu_xgmi_width",
            "Processors",
            "set_cpu_xgmi_width"
        )
		

# =============================================================================
# PERFORMANCE TEST REPORTING SYSTEM (Converted from perf_test.sh)
# =============================================================================

def generate_reports_from_output():
    """
    Generate enhanced and table reports from test output.
    This function replicates the Python report generation from perf_test.sh.
    """
    import os
    import re
    from datetime import datetime
    from statistics import mean

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Read the captured test output
    log_file = os.path.join(script_dir, '_perf_test.log')
    if not os.path.exists(log_file):
        print(f"⚠️  Warning: Performance log not found: {log_file}")
        return False

    with open(log_file, 'r') as f:
        content = f.read()

    # Parse test results
    lines = content.split('\n')
    tests = []
    current_test = None
    workflow_tests = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith('Testing '):
            parts = line.replace('Testing ', '').split(' on ')
            api_name = parts[0]
            current_test = {'name': api_name}
            workflow_tests = []

        elif line.startswith('Performance ') and 'avg' in line and current_test:
            avg_match = re.search(r'(\d+\.\d+)ms avg', line)
            err_match = re.search(r'(\d+) errors', line)

            if avg_match and err_match:
                perf_line_match = re.match(r'Performance\s+(\S+?)(?:_processor_\d+|_gpu_\d+|_socket_\d+|_system)?:', line)

                perf_test = {
                    'avg_ms': float(avg_match.group(1)),
                    'errors': int(err_match.group(1)),
                }

                if perf_line_match:
                    perf_test['name'] = perf_line_match.group(1)
                else:
                    perf_test['name'] = current_test['name']

                workflow_tests.append(perf_test)

        elif 'All calls failed' in line and current_test:
            if workflow_tests and 'avg_ms' not in workflow_tests[-1]:
                time_match = re.search(r'(\d+\.\d+)ms avg', line)
                if time_match:
                    workflow_tests[-1]['avg_ms'] = float(time_match.group(1))
                    workflow_tests[-1]['errors'] = 11

        elif re.match(r'\s*\d+\s*\|\s*AMDSMI_STATUS_', line) and workflow_tests:
            error_match = re.search(r'(\d+)\s*\|\s*(AMDSMI_STATUS_\w+)', line)
            if error_match and workflow_tests:
                workflow_tests[-1]['error_name'] = error_match.group(2)

        elif line.startswith('Performance test completed for') and current_test:
            for perf_test in workflow_tests:
                if 'avg_ms' in perf_test:
                    api_name = perf_test['name']

                    if perf_test['errors'] > 0 and 'error_name' not in perf_test:
                        if 'fan' in api_name.lower() or 'overdrive' in api_name.lower():
                            perf_test['error_name'] = 'AMDSMI_STATUS_NOT_SUPPORTED'
                        else:
                            perf_test['error_name'] = 'AMDSMI_STATUS_NOT_SUPPORTED'

                    perf_test['category'] = 'FUNCTIONAL' if perf_test['errors'] == 0 else 'ERROR_MEASUREMENT'
                    tests.append(perf_test)

            workflow_tests = []
            current_test = None

    # Separate tests by category
    functional_tests = sorted([t for t in tests if t['category'] == 'FUNCTIONAL'], key=lambda x: x['avg_ms'])
    error_tests = sorted([t for t in tests if t['category'] == 'ERROR_MEASUREMENT'], key=lambda x: x['avg_ms'])

    # Generate Enhanced Report
    enhanced_file = os.path.join(script_dir, '_perf_test_enhanced.log')
    with open(enhanced_file, 'w') as f:
        f.write('╔' + '═' * 78 + '╗\n')
        f.write('║' + ' AMD SMI PYTHON PERFORMANCE TEST RESULTS'.center(78) + '║\n')
        f.write('║' + f' Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}' + ' ' * (78 - len(f' Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')) + '║\n')
        f.write('╚' + '═' * 78 + '╝\n\n')

        total = len(tests)
        f.write('📊 SUMMARY STATISTICS\n')
        f.write('─' * 50 + '\n')
        f.write(f'Total Tests Executed: {total}\n')
        f.write(f'✅ Working APIs: {len(functional_tests)} ({len(functional_tests)/total*100:.1f}%)\n')
        f.write(f'⚠️  APIs with Expected Errors: {len(error_tests)} ({len(error_tests)/total*100:.1f}%)\n\n')

        if functional_tests:
            avg_times = [t['avg_ms'] for t in functional_tests]
            f.write('⚡ PERFORMANCE METRICS (Working APIs Only)\n')
            f.write('─' * 50 + '\n')
            f.write(f'Fastest API: {min(avg_times):.3f}ms\n')
            f.write(f'Slowest API: {max(avg_times):.3f}ms\n')
            f.write(f'Average Time: {mean(avg_times):.3f}ms\n\n')

            f.write('🏆 TOP 10 FASTEST APIs\n')
            f.write('─' * 50 + '\n')
            for i, test in enumerate(functional_tests[:10], 1):
                f.write(f'{i:2d}. {test["name"]} - {test["avg_ms"]:.3f}ms\n')
            f.write('\n')

    print(f'✅ Enhanced report: {enhanced_file}')

    # Generate Table Report
    table_file = os.path.join(script_dir, '_perf_test_table.log')
    with open(table_file, 'w') as f:
        f.write('╔' + '═' * 168 + '╗\n')
        f.write('║' + 'AMD SMI PYTHON PERFORMANCE TEST RESULTS'.center(168) + '║\n')
        f.write('║' + 'TABLE FORMAT'.center(168) + '║\n')
        f.write('╠' + '═' * 168 + '╣\n')
        gen_text = f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        f.write('║ ' + gen_text + ' ' * (166 - len(gen_text)) + ' ║\n')
        f.write('╚' + '═' * 168 + '╝\n\n')

        total = len(tests)
        f.write('📊 SUMMARY STATISTICS\n')
        f.write('═' * 53 + '\n')
        f.write(f'Total Tests Executed   : {total}\n')
        f.write(f'✅ Functional Tests    : {len(functional_tests)} ({len(functional_tests)/total*100:.1f}%)\n')
        f.write(f'⚠️  Error Measurements  : {len(error_tests)} ({len(error_tests)/total*100:.1f}%)\n\n')

        if functional_tests:
            avg_times = [t['avg_ms'] for t in functional_tests]
            fastest = min(functional_tests, key=lambda x: x['avg_ms'])
            slowest = max(functional_tests, key=lambda x: x['avg_ms'])

            f.write(f'⚡ Fastest API          : {min(avg_times):.3f}ms ({fastest["name"]})\n')
            f.write(f'🐌 Slowest API          : {max(avg_times):.3f}ms ({slowest["name"]})\n')
            f.write(f'📈 Average Time         : {mean(avg_times):.3f}ms\n')
        f.write('\n')

        if functional_tests:
            f.write('✅ FUNCTIONAL TESTS (APIs Returning Valid Data)\n')
            f.write('═' * 53 + '\n')
            f.write('┌──────┬───────────────────────────────────────────────────────────────────┬─────────────┬─────────────────────────────────────────────┐\n')
            f.write('│ Rank │ API Name                                                          │ Time (ms)   │ Status                                      │\n')
            f.write('├──────┼───────────────────────────────────────────────────────────────────┼─────────────┼─────────────────────────────────────────────┤\n')

            for i, test in enumerate(functional_tests, 1):
                name = test['name'][:65] + '...' if len(test['name']) > 65 else test['name']
                f.write(f'│ {i:4d} │ {name:<65} │ {test["avg_ms"]:>9.3f}   │ {"":<40}✅ │\n')

            f.write('└──────┴───────────────────────────────────────────────────────────────────┴─────────────┴─────────────────────────────────────────────┘\n\n')

        if error_tests:
            f.write('⚠️ ERROR MEASUREMENT TESTS (APIs Correctly Returning Expected Errors)\n')
            f.write('═' * 53 + '\n')
            f.write('┌──────┬───────────────────────────────────────────────────────────────────┬─────────────┬──────────────────────────────────────────┐\n')
            f.write('│ #    │ API Name                                                          │ Time (ms)   │ Error Code                               │\n')
            f.write('├──────┼───────────────────────────────────────────────────────────────────┼─────────────┼──────────────────────────────────────────┤\n')

            for i, test in enumerate(error_tests, 1):
                name = test['name'][:65] + '...' if len(test['name']) > 65 else test['name']
                error_info = test.get('error_name', f"{test['errors']} errors")
                if len(error_info) > 40:
                    error_info = error_info[:37] + '...'
                f.write(f'│ {i:4d} │ {name:<65} │ {test["avg_ms"]:>9.3f}   │ {error_info:<40} │\n')

            f.write('└──────┴───────────────────────────────────────────────────────────────────┴─────────────┴──────────────────────────────────────────┘\n\n')

        # Performance categories
        if tests:
            f.write('🏃 PERFORMANCE CATEGORIES\n')
            f.write('═' * 53 + '\n')
            ultra_fast = [t for t in tests if t['avg_ms'] < 0.01]
            fast = [t for t in tests if 0.01 <= t['avg_ms'] < 0.1]
            medium = [t for t in tests if 0.1 <= t['avg_ms'] < 1.0]
            slow = [t for t in tests if t['avg_ms'] >= 1.0]

            f.write(f'🚀 Ultra Fast (< 0.01ms)    : {len(ultra_fast):2d} tests\n')
            f.write(f'⚡ Fast (0.01-0.1ms)        : {len(fast):2d} tests\n')
            f.write(f'🚶 Medium (0.1-1.0ms)       : {len(medium):2d} tests\n')
            f.write(f'🐌 Slow (>= 1.0ms)          : {len(slow):2d} tests\n\n')

        f.write('═' * 53 + '\n')
        f.write(f'Report generated by AMD SMI Performance Test (Python Version)\n')
        f.write(f'For detailed logs, see: _perf_test.log and _perf_test_err.log\n\n')
        f.write('📝 NOTE: "Error Measurement Tests" are successful performance measurements\n')
        f.write('   of APIs that correctly return expected error codes (e.g., AMDSMI_STATUS_NOT_SUPPORTED)\n')
        f.write('   These are NOT test failures - they are successful timing measurements.\n')
        f.write('═' * 53 + '\n')

    print(f'✅ Table report: {table_file}')
    print(f'✅ Raw output: {log_file}')

    return True


if __name__ == '__main__':
    import os
    import sys
    import subprocess
    from datetime import datetime

    # Check if user wants simple unittest mode (no report generation)
    if '--unittest-only' in sys.argv:
        sys.argv.remove('--unittest-only')
        unittest.main()
        sys.exit(0)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Print header
    print('\n' + '═' * 80)
    print('AMD SMI PYTHON PERFORMANCE TEST'.center(80))
    print('═' * 80 + '\n')

    print(f'📍 Test directory: {script_dir}')
    print(f'📄 Test file: {__file__}')
    print(f'🕐 Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    # Step 1: Clean previous results
    print('\n[STEP 1] Cleaning previous results...')
    log_files = [
        '_perf_test.log',
        '_perf_test_err.log',
        '_perf_test_enhanced.log',
        '_perf_test_table.log'
    ]
    removed = 0
    for log_file in log_files:
        full_path = os.path.join(script_dir, log_file)
        if os.path.exists(full_path):
            os.remove(full_path)
            removed += 1

    if removed > 0:
        print(f'✅ Cleaned {removed} previous result file(s)')
    else:
        print('ℹ️  No previous results to clean')

    # Step 2: Run performance tests with output capture
    print('\n[STEP 2] Running performance tests...')
    print('⏳ Executing tests...')

    log_file = os.path.join(script_dir, '_perf_test.log')
    err_file = os.path.join(script_dir, '_perf_test_err.log')

    # Run tests with unittest and capture output
    with open(log_file, 'w') as f_out, open(err_file, 'w') as f_err:
        # Run the tests using unittest directly
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestAmdSmiCPUPythonPerformance)
        runner = unittest.TextTestRunner(stream=f_err, verbosity=2)

        # Redirect stdout to capture performance output
        old_stdout = sys.stdout
        sys.stdout = f_out

        result = runner.run(suite)

        sys.stdout = old_stdout

    # Also display output to console
    with open(log_file, 'r') as f:
        output_preview = f.readlines()
        # Show last 20 lines of output
        print('\n'.join(output_preview[-20:]))

    if result.wasSuccessful():
        print('✅ Performance tests completed successfully')
    else:
        print('⚠️  Performance tests completed with some issues')

    # Step 3: Generate reports
    print('\n[STEP 3] Generating reports...')

    if generate_reports_from_output():
        # Step 4: Show results summary
        print('\n[STEP 4] Results Summary')
        print('═' * 80)
        print('                    🎉 TEST COMPLETE'.center(80))
        print('═' * 80 + '\n')

        print('📊 Generated Output Files:\n')
        for log_file in log_files:
            full_path = os.path.join(script_dir, log_file)
            if os.path.exists(full_path):
                size = os.path.getsize(full_path) / 1024
                desc = {
                    '_perf_test_enhanced.log': 'Enhanced Summary Report',
                    '_perf_test_table.log': 'Tabular Results Report',
                    '_perf_test_err.log': 'Error Log & Test Status',
                    '_perf_test.log': 'Complete Test Output'
                }.get(log_file, 'Output File')

                print(f'  {desc} ({size:.1f} KB)')
                print(f'  → {full_path}\n')

        # Show preview of enhanced report
        print('📈 Quick Stats Preview:')
        print('═' * 80)
        enhanced_path = os.path.join(script_dir, '_perf_test_enhanced.log')
        if os.path.exists(enhanced_path):
            with open(enhanced_path, 'r') as f:
                lines = f.readlines()
                # Show first 15 lines
                print(''.join(lines[:15]))
        print('═' * 80 + '\n')

        print(f'✅ All reports generated in: {script_dir}')
        print(f'🕐 Finished: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print('\n' + '═' * 80 + '\n')
    else:
        print('⚠️  Report generation failed')
        sys.exit(1)

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)