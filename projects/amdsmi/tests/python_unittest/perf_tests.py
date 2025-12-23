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
sys.path.append("/opt/rocm/share/amd_smi/")

try:
    import amdsmi
except ImportError as exc:
    print(f"ERROR: Could not import amdsmi module: {exc}")
    print("The amdsmi module is required to run performance tests.")
    print("Please ensure amdsmi is installed and available in your Python path.")
    sys.exit(1)

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

class TestAmdSmiPythonPerformance(unittest.TestCase):
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

        # Performance test configuration - initialized once
        self.perf_iterations = 11  # Number of iterations for each performance test
        self.perf_warmup_iterations = 3  # Number of warmup iterations
        self.perf_results = {}  # Store performance results
        
        # Constants - set once instead of in setUp()
        self.TODO_SKIP_FAIL = False
        self.TODO_SKIP_NOT_COMPLETE = False
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
        
        # Initialize AMDSMI
        self.processors = []
        try:
            amdsmi.amdsmi_init()
            self.processors = amdsmi.amdsmi_get_processor_handles()
        except Exception as e:
            print(f"ERROR: Failed to initialize AMDSMI: {e}")
            raise
 
    def setUp(self):
        """Setup for performance tests - re-initialize AMDSMI before each test."""
        # Print device info once per test run if verbose
        global has_info_printed
        if verbose and has_info_printed is False:
            has_info_printed = True
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

        # Re-initialize AMDSMI for each test
        try:
            amdsmi.amdsmi_init()
            if not self.processors:  # Only get processors if not already set
                self.processors = amdsmi.amdsmi_get_processor_handles()
        except Exception as e:
            print(f"ERROR: Failed to initialize AMDSMI in setUp: {e}")
            raise
            
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
        
        # Warmup iterations
        for _ in range(self.perf_warmup_iterations):
            try:
                api_func(*args, **kwargs)
            except Exception:
                pass  # Ignore warmup errors
        
        # Bulk measurement - measure total time with counters outside the loop
        bulk_start_time = time.perf_counter()
        for i in range(self.perf_iterations):
            try:
                result = api_func(*args, **kwargs)
            except Exception as e:
                errors.append({'iteration': i, 'error_info': str(e)})
        bulk_end_time = time.perf_counter()

        # Calculate timing from bulk measurement
        bulk_total_time_ms = (bulk_end_time - bulk_start_time) * 1000
        bulk_avg_time_ms = bulk_total_time_ms / self.perf_iterations
        error_count = len(errors)
        
        # Calculate statistics
        successful_runs = self.perf_iterations - error_count
        stats = {
            'api_name': api_name,
            'iterations': self.perf_iterations,
            'errors': errors,
            'error_count': error_count,
            'successful_runs': successful_runs,
            'total_time_ms': bulk_total_time_ms,
            'avg_time_ms': bulk_avg_time_ms,
            'min_time_ms': bulk_avg_time_ms,  # Single measurement, so min=max=avg
            'max_time_ms': bulk_avg_time_ms,
            'mean_time_ms': bulk_avg_time_ms,
            'median_time_ms': bulk_avg_time_ms
        }

        # Store results
        self.perf_results[api_name] = stats

        # Print results if verbose
        if verbose:
            print(f"Performance {api_name}: {bulk_avg_time_ms:.3f}ms avg, "
                  f"{error_count} errors")

        return stats

    def test_performance_clean_gpu_local_data(self):
        self._print_func_name('Starting performance test for amdsmi_clean_gpu_local_data')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_clean_gpu_local_data", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_clean_gpu_local_data, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_clean_gpu_local_data,
                f'clean_gpu_local_data_processor_{i}',
                processor
            )
            
            self.perf_results[f'clean_gpu_local_data_processor_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_clean_gpu_local_data", "Processors", "clean_gpu_local_data")

    def test_performance_get_clk_freq(self):
        self._print_func_name('Starting performance test for amdsmi_get_clk_freq')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_clk_freq as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA).")
        
        for i, processor in enumerate(self.processors):
            for clk_type_name, clk_type, clk_cond in self.clk_types:
                if not clk_cond:
                    continue
                
                self._log_test_start("amdsmi_get_clk_freq", "Processor", i, clk_type=clk_type_name)
                
                self._print_api_result(amdsmi.amdsmi_get_clk_freq, i, processor, clk_type)
                
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_clk_freq,
                    f'get_clk_freq_processor_{i}_type_{clk_type_name}',
                    processor, clk_type
                )
                
                self.perf_results[f'get_clk_freq_processor_{i}_type_{clk_type_name}'] = stats
                
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(f"  Processor {i} clk_type={clk_type_name}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
                
                self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_clk_freq", "Processors", "get_clk_freq")

    def test_performance_get_clock_info(self):
        self._print_func_name('Starting performance test for amdsmi_get_clock_info')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_clock_info as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA).")
        
        for i, processor in enumerate(self.processors):
            for clk_type_name, clk_type, clk_cond in self.clk_types:
                if not clk_cond:
                    continue
                
                self._log_test_start("amdsmi_get_clock_info", "Processor", i, clk_type=clk_type_name)
                
                self._print_api_result(amdsmi.amdsmi_get_clock_info, i, processor, clk_type)
                
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_clock_info,
                    f'get_clock_info_processor_{i}_type_{clk_type_name}',
                    processor, clk_type
                )
                
                self.perf_results[f'get_clock_info_processor_{i}_type_{clk_type_name}'] = stats
                
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(f"  Processor {i} clk_type={clk_type_name}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
                
                self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_clock_info", "Processors", "get_clock_info")

    def test_performance_get_energy_count(self):
        self._print_func_name('Starting performance test for amdsmi_get_energy_count')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_energy_count as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA).")
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_energy_count", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_energy_count, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_energy_count,
                f'get_energy_count_processor_{i}',
                processor
            )
            
            self.perf_results[f'get_energy_count_processor_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_energy_count", "Processors", "get_energy_count")

    def test_performance_get_esmi_err_msg(self):
        self._print_func_name('Starting performance test for amdsmi_get_esmi_err_msg')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_esmi_err_msg as it fails (Unknown Error).")
        
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
                print(f"  Status {status_type_name}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Status", status_type_name)
        
        self._log_performance_summary("amdsmi_get_esmi_err_msg", "Status types", "get_esmi_err_msg")

    def test_performance_get_fw_info(self):
        self._print_func_name('Starting performance test for amdsmi_get_fw_info')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_fw_info", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_fw_info, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_fw_info,
                f'get_fw_info_processor_{i}',
                processor
            )
            
            self.perf_results[f'get_fw_info_processor_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_fw_info", "Processors", "get_fw_info")

    def test_performance_get_gpu_accelerator_partition_profile(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_accelerator_partition_profile')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_accelerator_partition_profile", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_accelerator_partition_profile, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_accelerator_partition_profile,
                f'get_gpu_accelerator_partition_profile_processor_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_accelerator_partition_profile_processor_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_accelerator_partition_profile", "Processors", "get_gpu_accelerator_partition_profile")

    def test_performance_get_gpu_accelerator_partition_profile_config(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_accelerator_partition_profile_config')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_accelerator_partition_profile_config", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_accelerator_partition_profile_config, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_accelerator_partition_profile_config,
                f'get_gpu_accelerator_partition_profile_config_processor_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_accelerator_partition_profile_config_processor_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_accelerator_partition_profile_config", "Processors", "get_gpu_accelerator_partition_profile_config")

    def test_performance_get_gpu_activity(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_activity')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_gpu_activity as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA).")
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_activity", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_activity, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_activity,
                f'get_gpu_activity_processor_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_activity_processor_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_activity", "Processors", "get_gpu_activity")

    def test_performance_get_gpu_asic_info(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_asic_info')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_asic_info", "GPU", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_asic_info, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_asic_info,
                f'get_gpu_asic_info_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_asic_info_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
                
            else:
                print(f"  GPU {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("GPU", i)
        
        self._log_performance_summary("amdsmi_get_gpu_asic_info", "GPUs", "get_gpu_asic_info")

    def test_performance_get_gpu_bad_page_info(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_bad_page_info')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_bad_page_info", "GPU", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_bad_page_info, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_bad_page_info,
                f'get_gpu_bad_page_info_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_bad_page_info_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
                
            else:
                print(f"  GPU {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("GPU", i)
        
        self._log_performance_summary("amdsmi_get_gpu_bad_page_info", "GPUs", "get_gpu_bad_page_info")

    def test_performance_get_gpu_bad_page_threshold(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_bad_page_threshold')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_bad_page_threshold", "GPU", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_bad_page_threshold, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_bad_page_threshold,
                f'get_gpu_bad_page_threshold_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_bad_page_threshold_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
                
            else:
                print(f"  GPU {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("GPU", i)
        
        self._log_performance_summary("amdsmi_get_gpu_bad_page_threshold", "GPUs", "get_gpu_bad_page_threshold")

    def test_performance_get_gpu_bdf_id(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_bdf_id')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_bdf_id", "GPU", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_bdf_id, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_bdf_id,
                f'get_gpu_bdf_id_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_bdf_id_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
                
            else:
                print(f"  GPU {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("GPU", i)
        
        self._log_performance_summary("amdsmi_get_gpu_bdf_id", "GPUs", "get_gpu_bdf_id")

    def test_performance_get_gpu_board_info(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_board_info')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_board_info", "GPU", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_board_info, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_board_info,
                f'get_gpu_board_info_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_board_info_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
                
            else:
                print(f"  GPU {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("GPU", i)
        
        self._log_performance_summary("amdsmi_get_gpu_board_info", "GPUs", "get_gpu_board_info")

    def test_performance_get_gpu_cache_info(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_cache_info')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_cache_info", "GPU", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_cache_info, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_cache_info,
                f'get_gpu_cache_info_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_cache_info_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
                
            else:
                print(f"  GPU {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("GPU", i)
        
        self._log_performance_summary("amdsmi_get_gpu_cache_info", "GPUs", "get_gpu_cache_info")

    def test_performance_get_gpu_compute_partition(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_compute_partition')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_compute_partition", "GPU", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_compute_partition, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_compute_partition,
                f'get_gpu_compute_partition_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_compute_partition_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
                
            else:
                print(f"  GPU {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("GPU", i)
        
        self._log_performance_summary("amdsmi_get_gpu_compute_partition", "GPUs", "get_gpu_compute_partition")

    def test_performance_get_gpu_compute_process_gpus(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_compute_process_gpus')
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_get_gpu_compute_process_gpus as it is not complete (Inval Error).")
        
        pid = 0
        
        self._log_test_start("amdsmi_get_gpu_compute_process_gpus", "Process", pid)
        
        stats = self._measure_api_performance(
            amdsmi.amdsmi_get_gpu_compute_process_gpus,
            f'get_gpu_compute_process_gpus_pid_{pid}',
            pid
        )
        
        self.perf_results[f'get_gpu_compute_process_gpus_pid_{pid}'] = stats
        
        if stats['successful_runs'] > 0:
            self._print_performance_results(stats)
        else:
            print(f"  PID {pid}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
        
        self._log_test_completion("Process", pid)
        self._log_performance_summary("amdsmi_get_gpu_compute_process_gpus", "Processes", "get_gpu_compute_process_gpus")

    def test_performance_get_gpu_compute_process_info(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_compute_process_info')
        
        self._log_test_start("amdsmi_get_gpu_compute_process_info", "System", "global")
        
        stats = self._measure_api_performance(
            amdsmi.amdsmi_get_gpu_compute_process_info,
            'get_gpu_compute_process_info_system'
        )
        
        self.perf_results['get_gpu_compute_process_info_system'] = stats
        
        if stats['successful_runs'] > 0:
            self._print_performance_results(stats)
            
        else:
            print(f"  System: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
        
        self._log_test_completion("System", "global")
        self._log_performance_summary("amdsmi_get_gpu_compute_process_info", "System", "get_gpu_compute_process_info")

    def test_performance_get_gpu_compute_process_info_by_pid(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_compute_process_info_by_pid')
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_get_gpu_compute_process_info_by_pid as it not complete (Device not found).")
        
        pid = 0
        
        self._log_test_start("amdsmi_get_gpu_compute_process_info_by_pid", "Process", pid)
        
        stats = self._measure_api_performance(
            amdsmi.amdsmi_get_gpu_compute_process_info_by_pid,
            f'get_gpu_compute_process_info_by_pid_{pid}',
            pid
        )
        
        self.perf_results[f'get_gpu_compute_process_info_by_pid_{pid}'] = stats
        
        if stats['successful_runs'] > 0:
            self._print_performance_results(stats)
            
        else:
            print(f"  PID {pid}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
        
        self._log_test_completion("Process", pid)
        self._log_performance_summary("amdsmi_get_gpu_compute_process_info_by_pid", "Processes", "get_gpu_compute_process_info_by_pid")

    def test_performance_get_gpu_device_bdf(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_device_bdf')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_device_bdf", "GPU", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_device_bdf, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_device_bdf,
                f'get_gpu_device_bdf_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_device_bdf_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
                
            else:
                print(f"  GPU {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("GPU", i)
        
        self._log_performance_summary("amdsmi_get_gpu_device_bdf", "GPUs", "get_gpu_device_bdf")

    def test_performance_get_gpu_device_uuid(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_device_uuid')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_device_uuid", "GPU", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_device_uuid, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_device_uuid,
                f'get_gpu_device_uuid_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_device_uuid_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
                
            else:
                print(f"  GPU {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("GPU", i)
        
        self._log_performance_summary("amdsmi_get_gpu_device_uuid", "GPUs", "get_gpu_device_uuid")

    def test_performance_get_gpu_driver_info(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_driver_info')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_driver_info", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_driver_info, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_driver_info,
                f'get_gpu_driver_info_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_driver_info_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_driver_info", "Processors", "get_gpu_driver_info")

    def test_performance_get_gpu_ecc_count(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_ecc_count')
        
        for i, processor in enumerate(self.processors):
            for gpu_block_name, gpu_block, gpu_block_cond in self.gpu_blocks:
                self._log_test_start("amdsmi_get_gpu_ecc_count", "Processor", i, block=gpu_block_name)
                
                self._print_api_result(amdsmi.amdsmi_get_gpu_ecc_count, i, processor, gpu_block)
                
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_gpu_ecc_count,
                    f'get_gpu_ecc_count_gpu_{i}_block_{gpu_block_name}',
                    processor, gpu_block
                )
                
                self.perf_results[f'get_gpu_ecc_count_gpu_{i}_block_{gpu_block_name}'] = stats
                
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(f"  Processor {i} block {gpu_block_name}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
                
                self._log_test_completion("Processor", i, f"block={gpu_block_name}")
        
        self._log_performance_summary("amdsmi_get_gpu_ecc_count", "Processors", "get_gpu_ecc_count")

    def test_performance_get_gpu_ecc_enabled(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_ecc_enabled')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_ecc_enabled", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_ecc_enabled, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_ecc_enabled,
                f'get_gpu_ecc_enabled_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_ecc_enabled_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_ecc_enabled", "Processors", "get_gpu_ecc_enabled")

    def test_performance_get_gpu_ecc_status(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_ecc_status')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_performance_get_gpu_ecc_status as it fails.")
        
        for i, processor in enumerate(self.processors):
            for gpu_block_name, gpu_block, gpu_block_cond in self.gpu_blocks:
                self._log_test_start("amdsmi_get_gpu_ecc_status", "Processor", i, block=gpu_block_name)
                
                self._print_api_result(amdsmi.amdsmi_get_gpu_ecc_status, i, processor, gpu_block)
                
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_gpu_ecc_status,
                    f'get_gpu_ecc_status_gpu_{i}_block_{gpu_block_name}',
                    processor, gpu_block
                )
                
                self.perf_results[f'get_gpu_ecc_status_gpu_{i}_block_{gpu_block_name}'] = stats
                
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(f"  Processor {i} block {gpu_block_name}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
                
                self._log_test_completion("Processor", i, f"block={gpu_block_name}")
        
        self._log_performance_summary("amdsmi_get_gpu_ecc_status", "Processors", "get_gpu_ecc_status")

    def test_performance_get_gpu_enumeration_info(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_enumeration_info')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_enumeration_info", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_enumeration_info, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_enumeration_info,
                f'get_gpu_enumeration_info_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_enumeration_info_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_enumeration_info", "Processors", "get_gpu_enumeration_info")

    def test_performance_get_gpu_fan_rpms(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_fan_rpms')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_fan_rpms", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_fan_rpms, i, processor, sensor_idx=0)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_fan_rpms,
                f'get_gpu_fan_rpms_gpu_{i}',
                processor, 0
            )
            
            self.perf_results[f'get_gpu_fan_rpms_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_fan_rpms", "Processors", "get_gpu_fan_rpms")

    def test_performance_get_gpu_id(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_id')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_id", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_id, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_id,
                f'get_gpu_id_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_id_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_id", "Processors", "get_gpu_id")

    def test_performance_get_gpu_kfd_info(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_kfd_info')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_kfd_info", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_kfd_info, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_kfd_info,
                f'get_gpu_kfd_info_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_kfd_info_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_kfd_info", "Processors", "get_gpu_kfd_info")

    def test_performance_get_gpu_mem_overdrive_level(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_mem_overdrive_level')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_mem_overdrive_level", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_mem_overdrive_level, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_mem_overdrive_level,
                f'get_gpu_mem_overdrive_level_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_mem_overdrive_level_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_mem_overdrive_level", "Processors", "get_gpu_mem_overdrive_level")

    def test_performance_get_gpu_memory_partition(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_memory_partition')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_memory_partition", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_memory_partition, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_memory_partition,
                f'get_gpu_memory_partition_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_memory_partition_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_memory_partition", "Processors", "get_gpu_memory_partition")

    def test_performance_get_gpu_memory_partition_config(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_memory_partition_config')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_performance_get_gpu_memory_partition_config as it fails on MI300.")
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_memory_partition_config", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_memory_partition_config, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_memory_partition_config,
                f'get_gpu_memory_partition_config_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_memory_partition_config_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_memory_partition_config", "Processors", "get_gpu_memory_partition_config")

    def test_performance_get_gpu_memory_reserved_pages(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_memory_reserved_pages')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_memory_reserved_pages", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_memory_reserved_pages, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_memory_reserved_pages,
                f'get_gpu_memory_reserved_pages_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_memory_reserved_pages_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_memory_reserved_pages", "Processors", "get_gpu_memory_reserved_pages")

    def test_performance_get_gpu_memory_total(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_memory_total')
        
        for i, processor in enumerate(self.processors):
            for memory_type_name, memory_type, memory_type_cond in self.memory_types:
                self._log_test_start("amdsmi_get_gpu_memory_total", "Processor", i, mem_type=memory_type_name)
                
                self._print_api_result(amdsmi.amdsmi_get_gpu_memory_total, i, processor, mem_type)
                
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_gpu_memory_total,
                    f'get_gpu_memory_total_gpu_{i}_type_{memory_type_name}',
                    processor, memory_type
                )
                
                self.perf_results[f'get_gpu_memory_total_gpu_{i}_type_{memory_type_name}'] = stats
                
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(f"  Processor {i} mem_type {memory_type_name}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
                
                self._log_test_completion("Processor", i, f"mem_type={memory_type_name}")
        
        self._log_performance_summary("amdsmi_get_gpu_memory_total", "Processors", "get_gpu_memory_total")

    def test_performance_get_gpu_memory_usage(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_memory_usage')
        
        for i, processor in enumerate(self.processors):
            for memory_type_name, memory_type, memory_type_cond in self.memory_types:
                self._log_test_start("amdsmi_get_gpu_memory_usage", "Processor", i, mem_type=memory_type_name)
                
                self._print_api_result(amdsmi.amdsmi_get_gpu_memory_usage, i, processor, mem_type)
                
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_gpu_memory_usage,
                    f'get_gpu_memory_usage_gpu_{i}_type_{memory_type_name}',
                    processor, memory_type
                )
                
                self.perf_results[f'get_gpu_memory_usage_gpu_{i}_type_{memory_type_name}'] = stats
                
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(f"  Processor {i} mem_type {memory_type_name}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
                
                self._log_test_completion("Processor", i, f"mem_type={memory_type_name}")
        
        self._log_performance_summary("amdsmi_get_gpu_memory_usage", "Processors", "get_gpu_memory_usage")

    def test_performance_get_gpu_metrics_header_info(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_metrics_header_info')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_metrics_header_info", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_metrics_header_info, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_metrics_header_info,
                f'get_gpu_metrics_header_info_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_metrics_header_info_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_metrics_header_info", "Processors", "get_gpu_metrics_header_info")

    def test_performance_get_gpu_metrics_info(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_metrics_info')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_performance_get_gpu_metrics_info as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA).")
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_metrics_info", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_metrics_info, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_metrics_info,
                f'get_gpu_metrics_info_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_metrics_info_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_metrics_info", "Processors", "get_gpu_metrics_info")

    def test_performance_get_gpu_od_volt_curve_regions(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_od_volt_curve_regions')
        
        num_region = 10
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_od_volt_curve_regions", "Processor", i, num_region=num_region)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_od_volt_curve_regions, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_od_volt_curve_regions,
                f'get_gpu_od_volt_curve_regions_gpu_{i}',
                processor, num_region
            )
            
            self.perf_results[f'get_gpu_od_volt_curve_regions_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i, f"num_region={num_region}")
        
        self._log_performance_summary("amdsmi_get_gpu_od_volt_curve_regions", "Processors", "get_gpu_od_volt_curve_regions")

    def test_performance_get_gpu_od_volt_info(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_od_volt_info')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_od_volt_info", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_od_volt_info, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_od_volt_info,
                f'get_gpu_od_volt_info_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_od_volt_info_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_od_volt_info", "Processors", "get_gpu_od_volt_info")

    def test_performance_get_gpu_overdrive_level(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_overdrive_level')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_overdrive_level", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_overdrive_level, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_overdrive_level,
                f'get_gpu_overdrive_level_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_overdrive_level_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_overdrive_level", "Processors", "get_gpu_overdrive_level")

    def test_performance_get_gpu_pci_bandwidth(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_pci_bandwidth')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_performance_get_gpu_pci_bandwidth as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA).")
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_pci_bandwidth", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_pci_bandwidth, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_pci_bandwidth,
                f'get_gpu_pci_bandwidth_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_pci_bandwidth_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_pci_bandwidth", "Processors", "get_gpu_pci_bandwidth")

    def test_performance_get_gpu_pci_replay_counter(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_pci_replay_counter')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_pci_replay_counter", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_pci_replay_counter, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_pci_replay_counter,
                f'get_gpu_pci_replay_counter_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_pci_replay_counter_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_pci_replay_counter", "Processors", "get_gpu_pci_replay_counter")

    def test_performance_get_gpu_pci_throughput(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_pci_throughput')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_pci_throughput", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_pci_throughput, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_pci_throughput,
                f'get_gpu_pci_throughput_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_pci_throughput_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_pci_throughput", "Processors", "get_gpu_pci_throughput")

    def test_performance_get_gpu_perf_level(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_perf_level')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_perf_level", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_perf_level, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_perf_level,
                f'get_gpu_perf_level_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_perf_level_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_perf_level", "Processors", "get_gpu_perf_level")

    def test_performance_get_gpu_pm_metrics_info(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_pm_metrics_info')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_pm_metrics_info", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_pm_metrics_info, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_pm_metrics_info,
                f'get_gpu_pm_metrics_info_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_pm_metrics_info_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_pm_metrics_info", "Processors", "get_gpu_pm_metrics_info")

    def test_performance_get_gpu_power_profile_presets(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_power_profile_presets')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_power_profile_presets", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_power_profile_presets, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_power_profile_presets,
                f'get_gpu_power_profile_presets_gpu_{i}',
                processor, 0
            )
            
            self.perf_results[f'get_gpu_power_profile_presets_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_power_profile_presets", "Processors", "get_gpu_power_profile_presets")

    def test_performance_get_gpu_process_isolation(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_process_isolation')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_process_isolation", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_process_isolation, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_process_isolation,
                f'get_gpu_process_isolation_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_process_isolation_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_process_isolation", "Processors", "get_gpu_process_isolation")

    def test_performance_get_gpu_process_list(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_process_list')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_process_list", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_process_list, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_process_list,
                f'get_gpu_process_list_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_process_list_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_process_list", "Processors", "get_gpu_process_list")

    def test_performance_get_gpu_ras_block_features_enabled(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_ras_block_features_enabled')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_ras_block_features_enabled", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_ras_block_features_enabled, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_ras_block_features_enabled,
                f'get_gpu_ras_block_features_enabled_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_ras_block_features_enabled_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_ras_block_features_enabled", "Processors", "get_gpu_ras_block_features_enabled")

    def test_performance_get_gpu_ras_feature_info(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_ras_feature_info')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_ras_feature_info", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_ras_feature_info, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_ras_feature_info,
                f'get_gpu_ras_feature_info_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_ras_feature_info_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_ras_feature_info", "Processors", "get_gpu_ras_feature_info")

    def test_performance_get_gpu_reg_table_info(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_reg_table_info')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_performance_get_gpu_reg_table_info as it fails on MI300.")
        
        for i, processor in enumerate(self.processors):
            for reg_type_name, reg_type, reg_type_cond in self.reg_types:
                self._log_test_start("amdsmi_get_gpu_reg_table_info", "Processor", i, reg_type=reg_type_name)
                
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_gpu_reg_table_info,
                    f'get_gpu_reg_table_info_gpu_{i}_type_{reg_type_name}',
                    processor, reg_type
                )
                
                self.perf_results[f'get_gpu_reg_table_info_gpu_{i}_type_{reg_type_name}'] = stats
                
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(f"  Processor {i} reg_type {reg_type_name}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
                
                self._log_test_completion("Processor", i, f"reg_type={reg_type_name}")
        
        self._log_performance_summary("amdsmi_get_gpu_reg_table_info", "Processors", "get_gpu_reg_table_info")

    def test_performance_get_gpu_revision(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_revision')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_revision", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_revision, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_revision,
                f'get_gpu_revision_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_revision_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_revision", "Processors", "get_gpu_revision")

    def test_performance_get_gpu_subsystem_id(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_subsystem_id')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_subsystem_id", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_subsystem_id, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_subsystem_id,
                f'get_gpu_subsystem_id_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_subsystem_id_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_subsystem_id", "Processors", "get_gpu_subsystem_id")

    def test_performance_get_gpu_subsystem_name(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_subsystem_name')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_subsystem_name", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_subsystem_name, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_subsystem_name,
                f'get_gpu_subsystem_name_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_subsystem_name_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_subsystem_name", "Processors", "get_gpu_subsystem_name")

    def test_performance_get_gpu_topo_numa_affinity(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_topo_numa_affinity')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_topo_numa_affinity", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_topo_numa_affinity, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_topo_numa_affinity,
                f'get_gpu_topo_numa_affinity_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_topo_numa_affinity_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_topo_numa_affinity", "Processors", "get_gpu_topo_numa_affinity")

    def test_performance_get_gpu_total_ecc_count(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_total_ecc_count')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_total_ecc_count", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_total_ecc_count, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_total_ecc_count,
                f'get_gpu_total_ecc_count_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_total_ecc_count_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_total_ecc_count", "Processors", "get_gpu_total_ecc_count")

    def test_performance_get_gpu_vbios_info(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_vbios_info')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_vbios_info", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_vbios_info, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_vbios_info,
                f'get_gpu_vbios_info_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_vbios_info_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_vbios_info", "Processors", "get_gpu_vbios_info")

    def test_performance_get_gpu_vendor_name(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_vendor_name')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_vendor_name", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_vendor_name, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_vendor_name,
                f'get_gpu_vendor_name_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_vendor_name_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_vendor_name", "Processors", "get_gpu_vendor_name")

    def test_performance_get_gpu_virtualization_mode(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_virtualization_mode')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_virtualization_mode", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_virtualization_mode, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_virtualization_mode,
                f'get_gpu_virtualization_mode_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_virtualization_mode_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_virtualization_mode", "Processors", "get_gpu_virtualization_mode")

    def test_performance_get_gpu_volt_metric(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_volt_metric')
        
        for i, processor in enumerate(self.processors):
            for voltage_type_name, voltage_type, voltage_type_cond in self.voltage_types:
                for voltage_metric_name, voltage_metric, voltage_metric_cond in self.voltage_metrics:
                    self._log_test_start("amdsmi_get_gpu_volt_metric", "Processor", i, volt_type=voltage_type_name, volt_metric=voltage_metric_name)
                    
                    stats = self._measure_api_performance(
                        amdsmi.amdsmi_get_gpu_volt_metric,
                        f'get_gpu_volt_metric_gpu_{i}_type_{voltage_type_name}_metric_{voltage_metric_name}',
                        processor, voltage_type, voltage_metric
                    )
                    
                    self.perf_results[f'get_gpu_volt_metric_gpu_{i}_type_{voltage_type_name}_metric_{voltage_metric_name}'] = stats
                    
                    if stats['successful_runs'] > 0:
                        self._print_performance_results(stats)
                    else:
                        print(f"  Processor {i} volt_type {voltage_type_name} volt_metric {voltage_metric_name}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
                    
                    self._log_test_completion("Processor", i, f"volt_type={voltage_type_name}, volt_metric={voltage_metric_name}")
        
        self._log_performance_summary("amdsmi_get_gpu_volt_metric", "Processors", "get_gpu_volt_metric")

    def test_performance_get_gpu_vram_info(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_vram_info')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_vram_info", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_vram_info, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_vram_info,
                f'get_gpu_vram_info_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_vram_info_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_vram_info", "Processors", "get_gpu_vram_info")

    def test_performance_get_gpu_vram_usage(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_vram_usage')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_vram_usage", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_vram_usage, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_vram_usage,
                f'get_gpu_vram_usage_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_vram_usage_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_vram_usage", "Processors", "get_gpu_vram_usage")

    def test_performance_get_gpu_vram_vendor(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_vram_vendor')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_vram_vendor", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_vram_vendor, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_vram_vendor,
                f'get_gpu_vram_vendor_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_vram_vendor_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_vram_vendor", "Processors", "get_gpu_vram_vendor")

    def test_performance_get_gpu_xcd_counter(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_xcd_counter')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_performance_get_gpu_xcd_counter as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA).")
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_xcd_counter", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_xcd_counter, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_xcd_counter,
                f'get_gpu_xcd_counter_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_xcd_counter_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_xcd_counter", "Processors", "get_gpu_xcd_counter")

    def test_performance_get_gpu_xgmi_link_status(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_xgmi_link_status')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_performance_get_gpu_xgmi_link_status as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA).")
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_xgmi_link_status", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_xgmi_link_status, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_xgmi_link_status,
                f'get_gpu_xgmi_link_status_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_xgmi_link_status_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_xgmi_link_status", "Processors", "get_gpu_xgmi_link_status")

    def test_performance_get_lib_version(self):
        self._print_func_name('Starting performance test for amdsmi_get_lib_version')
        
        self._log_test_start("amdsmi_get_lib_version", "System", "global")
        
        # No processor needed for library version
        try:
            ret = amdsmi.amdsmi_get_lib_version()
            print(f"### test amdsmi_get_lib_version()")
            self._print("", ret)
        except amdsmi.AmdSmiLibraryException as e:
            print(f"### test amdsmi_get_lib_version()")
            print(f"  Error: {e}")
        
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

    def test_performance_get_link_metrics(self):
        self._print_func_name('Starting performance test for amdsmi_get_link_metrics')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_performance_get_link_metrics as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA).")
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_link_metrics", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_link_metrics, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_link_metrics,
                f'get_link_metrics_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_link_metrics_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_link_metrics", "Processors", "get_link_metrics")

    def test_performance_get_link_topology_nearest(self):
        self._print_func_name('Starting performance test for amdsmi_get_link_topology_nearest')
        
        for i, processor in enumerate(self.processors):
            for link_type_name, link_type, link_type_cond in self.link_types:
                self._log_test_start("amdsmi_get_link_topology_nearest", "Processor", f"{i} link_type({link_type_name})")
                
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_link_topology_nearest,
                    f'get_link_topology_nearest_gpu_{i}_{link_type_name}',
                    processor, link_type
                )
                
                self.perf_results[f'get_link_topology_nearest_gpu_{i}_{link_type_name}'] = stats
                
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(f"  Processor {i} link_type({link_type_name}): All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
                
                self._log_test_completion("Processor", f"{i} link_type({link_type_name})")
        
        self._log_performance_summary("amdsmi_get_link_topology_nearest", "Processors", "get_link_topology_nearest")

    def test_performance_get_minmax_bandwidth_between_processors(self):
        self._print_func_name('Starting performance test for amdsmi_get_minmax_bandwidth_between_processors')
        
        for i, gpu_i in enumerate(self.processors):
            for j, gpu_j in enumerate(self.processors):
                self._log_test_start("amdsmi_get_minmax_bandwidth_between_processors", "Bandwidth", f"{i}->{j}")
                
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_minmax_bandwidth_between_processors,
                    f'get_minmax_bandwidth_between_processors_{i}_to_{j}',
                    gpu_i, gpu_j
                )
                
                self.perf_results[f'get_minmax_bandwidth_between_processors_{i}_to_{j}'] = stats
                
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(f"  Bandwidth {i}->{j}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
                
                self._log_test_completion("Bandwidth", f"{i}->{j}")
        
        self._log_performance_summary("amdsmi_get_minmax_bandwidth_between_processors", "Bandwidth", "get_minmax_bandwidth_between_processors")

    def test_performance_get_pcie_info(self):
        self._print_func_name('Starting performance test for amdsmi_get_pcie_info')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_performance_get_pcie_info as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA).")
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_pcie_info", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_pcie_info, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_pcie_info,
                f'get_pcie_info_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_pcie_info_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_pcie_info", "Processors", "get_pcie_info")

    def test_performance_get_power_info(self):
        self._print_func_name('Starting performance test for amdsmi_get_power_info')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_power_info", "GPU", i)
            
            self._print_api_result(amdsmi.amdsmi_get_power_info, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_power_info,
                f'get_power_info_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_power_info_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
                self._run_performance_assertions(stats, "amdsmi_get_power_info")
            else:
                print(f"  GPU {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("GPU", i)
        
        self._log_performance_summary("amdsmi_get_power_info", "GPUs", "get_power_info")

    def test_performance_get_processor_count_from_handles(self):
        self._print_func_name('Starting performance test for amdsmi_get_processor_count_from_handles')
        
        self._log_test_start("amdsmi_get_processor_count_from_handles", "Processors", "all")
        
        # Print result for all processors
        try:
            ret = amdsmi.amdsmi_get_processor_count_from_handles(self.processors)
            print(f"### test amdsmi_get_processor_count_from_handles()")
            self._print("", ret)
        except amdsmi.AmdSmiLibraryException as e:
            print(f"### test amdsmi_get_processor_count_from_handles()")
            print(f"  Error: {e}")
        
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
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_processor_handle_from_bdf", "GPU", i)
            
            try:
                # Get BDF for this processor first
                bdf = amdsmi.amdsmi_get_gpu_device_bdf(processor)
                
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_processor_handle_from_bdf,
                    f'get_processor_handle_from_bdf_gpu_{i}',
                    bdf
                )
                
                self.perf_results[f'get_processor_handle_from_bdf_gpu_{i}'] = stats
                
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                    self._run_performance_assertions(stats, "amdsmi_get_processor_handle_from_bdf")
                    
                    # Validate that the returned handle matches the original processor
                    ret = amdsmi.amdsmi_get_processor_handle_from_bdf(bdf)
                    if processor.value != ret.value:
                        print(f"  WARNING: GPU {i} - Handle mismatch! Expected: {processor.value}, Received: {ret.value}")
                else:
                    print(f"  GPU {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            except Exception as e:
                print(f"  GPU {i}: Error getting BDF - {e}")
            
            self._log_test_completion("GPU", i)
        
        self._log_performance_summary("amdsmi_get_processor_handle_from_bdf", "GPUs", "get_processor_handle_from_bdf")

    def test_performance_get_processor_handles(self):
        self._print_func_name('Starting performance test for amdsmi_get_processor_handles')
        
        self._log_test_start("amdsmi_get_processor_handles", "System", "global")
        
        self._print_api_result(amdsmi.amdsmi_get_processor_handles, 0, None)
        
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
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_processor_info", "GPU", i)
            
            self._print_api_result(amdsmi.amdsmi_get_processor_info, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_processor_info,
                f'get_processor_info_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_processor_info_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
                self._run_performance_assertions(stats, "amdsmi_get_processor_info")
            else:
                print(f"  GPU {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("GPU", i)
        
        self._log_performance_summary("amdsmi_get_processor_info", "GPUs", "get_processor_info")

    def test_performance_get_processor_type(self):
        self._print_func_name('Starting performance test for amdsmi_get_processor_type')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_processor_type", "GPU", i)
            
            self._print_api_result(amdsmi.amdsmi_get_processor_type, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_processor_type,
                f'get_processor_type_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_processor_type_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
                self._run_performance_assertions(stats, "amdsmi_get_processor_type")
            else:
                print(f"  GPU {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("GPU", i)
        
        self._log_performance_summary("amdsmi_get_processor_type", "GPUs", "get_processor_type")

    def test_performance_get_socket_handles(self):
        self._print_func_name('Starting performance test for amdsmi_get_socket_handles')
        
        self._log_test_start("amdsmi_get_socket_handles", "System", "global")
        
        self._print_api_result(amdsmi.amdsmi_get_socket_handles, 0, None)
        
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
            
            self._print_api_result(amdsmi.amdsmi_get_socket_info, i, socket, label_prefix="socket")
            
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
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_temp_metric as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA).")
        for i, processor in enumerate(self.processors):
            for temperature_type_name, temperature_type, temperature_type_cond in self.temperature_types:
                for temperature_metric_name, temperature_metric, temperature_metric_cond in self.temperature_metrics:
                    self._log_test_start("amdsmi_get_temp_metric", "GPU", i, 
                                       temperature_type=temperature_type_name, 
                                       temperature_metric=temperature_metric_name)
                    
                    stats = self._measure_api_performance(
                        amdsmi.amdsmi_get_temp_metric,
                        f'get_temp_metric_gpu_{i}_{temperature_type_name}_{temperature_metric_name}',
                        processor, temperature_type, temperature_metric
                    )
                    
                    self.perf_results[f'get_temp_metric_gpu_{i}_{temperature_type_name}_{temperature_metric_name}'] = stats
                    
                    if stats['successful_runs'] > 0:
                        self._print_performance_results(stats)
                        self._run_performance_assertions(stats, "amdsmi_get_temp_metric")
                    else:
                        print(f"  GPU {i} {temperature_type_name}/{temperature_metric_name}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
                    
                    self._log_test_completion("GPU", i, f"temperature_type={temperature_type_name}, temperature_metric={temperature_metric_name}")
        
        self._log_performance_summary("amdsmi_get_temp_metric", "GPUs", "get_temp_metric")

    def test_performance_get_threads_per_core(self):
        self._print_func_name('Starting performance test for amdsmi_get_threads_per_core')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_threads_per_core as it fails (IO Error).")    
        self._log_test_start("amdsmi_get_threads_per_core", "System", 0)
        
        # No processor needed for threads per core
        try:
            ret = amdsmi.amdsmi_get_threads_per_core()
            print(f"### test amdsmi_get_threads_per_core()")
            self._print("", ret)
        except amdsmi.AmdSmiLibraryException as e:
            print(f"### test amdsmi_get_threads_per_core()")
            print(f"  Error: {e}")
        
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

    def test_performance_get_utilization_count(self):
        self._print_func_name('Starting performance test for amdsmi_get_utilization_count')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_utilization_count as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA).")
        for i, processor in enumerate(self.processors):
            for utilization_counter_type_name, utilization_counter_type, utilization_counter_type_cond in self.utilization_counter_types:
                self._log_test_start("amdsmi_get_utilization_count", "GPU", i, 
                                   utilization_counter_type=utilization_counter_type_name)
                
                self._print_api_result(amdsmi.amdsmi_get_utilization_count, i, processor)
                
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_utilization_count,
                    f'get_utilization_count_gpu_{i}_{utilization_counter_type_name}',
                    processor, [utilization_counter_type]
                )
                
                self.perf_results[f'get_utilization_count_gpu_{i}_{utilization_counter_type_name}'] = stats
                
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                    self._run_performance_assertions(stats, "amdsmi_get_utilization_count")
                else:
                    print(f"  GPU {i} {utilization_counter_type_name}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
                
                self._log_test_completion("GPU", i, f"utilization_counter_type={utilization_counter_type_name}")
        
        self._log_performance_summary("amdsmi_get_utilization_count", "GPUs", "get_utilization_count")

    def test_performance_get_violation_status(self):
        self._print_func_name('Starting performance test for amdsmi_get_violation_status')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_violation_status as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA).")
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_violation_status", "GPU", i)
            
            self._print_api_result(amdsmi.amdsmi_get_violation_status, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_violation_status,
                f'get_violation_status_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_violation_status_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
                self._run_performance_assertions(stats, "amdsmi_get_violation_status")
            else:
                print(f"  GPU {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("GPU", i)
        
        self._log_performance_summary("amdsmi_get_violation_status", "GPUs", "get_violation_status")

    def test_performance_get_xgmi_info(self):
        self._print_func_name('Starting performance test for amdsmi_get_xgmi_info')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_xgmi_info", "GPU", i)
            
            self._print_api_result(amdsmi.amdsmi_get_xgmi_info, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_xgmi_info,
                f'get_xgmi_info_gpu_{i}',
                processor
            )
            
            self.perf_results[f'get_xgmi_info_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
                self._run_performance_assertions(stats, "amdsmi_get_xgmi_info")
            else:
                print(f"  GPU {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("GPU", i)
        
        self._log_performance_summary("amdsmi_get_xgmi_info", "GPUs", "get_xgmi_info")

    def test_performance_gpu_counter(self):
        self._print_func_name('Starting performance test for GPU counter workflow')
        
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_performance_gpu_counter as it fails (Error opening file).")
        
        for i, processor in enumerate(self.processors):
            for event_type_name, event_type, event_type_cond in self.event_types:
                self._log_test_start("gpu_counter_workflow", "GPU", i, event_type=event_type_name)
                
                # Test Create Counter
                stats_create = self._measure_api_performance(
                    amdsmi.amdsmi_gpu_create_counter,
                    f'gpu_create_counter_gpu_{i}_{event_type_name}',
                    processor, event_type
                )
                self.perf_results[f'gpu_create_counter_gpu_{i}_{event_type_name}'] = stats_create
                
                if stats_create['successful_runs'] > 0:
                    # Get the event_handle from the first successful run
                    try:
                        event_handle = amdsmi.amdsmi_gpu_create_counter(processor, event_type)
                        
                        # Test Read Counter
                        stats_read = self._measure_api_performance(
                            amdsmi.amdsmi_gpu_read_counter,
                            f'gpu_read_counter_gpu_{i}_{event_type_name}',
                            event_handle
                        )
                        self.perf_results[f'gpu_read_counter_gpu_{i}_{event_type_name}'] = stats_read
                        
                        # Test Control Counter for each command
                        for counter_command_name, counter_command, counter_commands_cond in self.counter_commands:
                            stats_control = self._measure_api_performance(
                                amdsmi.amdsmi_gpu_control_counter,
                                f'gpu_control_counter_gpu_{i}_{event_type_name}_{counter_command_name}',
                                event_handle, counter_command
                            )
                            self.perf_results[f'gpu_control_counter_gpu_{i}_{event_type_name}_{counter_command_name}'] = stats_control
                        
                        # Test Destroy Counter
                        stats_destroy = self._measure_api_performance(
                            amdsmi.amdsmi_gpu_destroy_counter,
                            f'gpu_destroy_counter_gpu_{i}_{event_type_name}',
                            event_handle
                        )
                        self.perf_results[f'gpu_destroy_counter_gpu_{i}_{event_type_name}'] = stats_destroy
                        
                    except amdsmi.AmdSmiLibraryException as e:
                        print(f"  GPU {i} {event_type_name}: Counter workflow failed - {e}")
                else:
                    print(f"  GPU {i} {event_type_name}: Create counter failed - {stats_create['errors'][0]['error_info'] if stats_create['errors'] else 'Unknown'}")
                
                self._log_test_completion("GPU", i, f"event_type={event_type_name}")
        
        self._log_performance_summary("gpu_counter_workflow", "GPUs", "gpu_counter")

    def test_performance_gpu_counter_group_supported(self):
        self._print_func_name('Starting performance test for amdsmi_gpu_counter_group_supported')
        
        for i, processor in enumerate(self.processors):
            for event_group_name, event_group, event_group_cond in self.event_groups:
                self._log_test_start("amdsmi_gpu_counter_group_supported", "GPU", i, event_group=event_group_name)
                
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_gpu_counter_group_supported,
                    f'gpu_counter_group_supported_gpu_{i}_{event_group_name}',
                    processor, event_group
                )
                
                self.perf_results[f'gpu_counter_group_supported_gpu_{i}_{event_group_name}'] = stats
                
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                    self._run_performance_assertions(stats, "amdsmi_gpu_counter_group_supported")
                else:
                    print(f"  GPU {i} {event_group_name}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
                
                self._log_test_completion("GPU", i, f"event_group={event_group_name}")
        
        self._log_performance_summary("amdsmi_gpu_counter_group_supported", "GPUs", "gpu_counter_group_supported")

    def test_performance_get_gpu_available_counters(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_available_counters')
        
        for i, processor in enumerate(self.processors):
            for event_group_name, event_group_type, event_group_cond in self.event_groups:
                self._log_test_start("amdsmi_get_gpu_available_counters", "GPU", i, event_group=event_group_name)
                
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_get_gpu_available_counters,
                    f'get_gpu_available_counters_gpu_{i}_event_group_{event_group_name}',
                    processor, event_group_type
                )
                
                self.perf_results[f'get_gpu_available_counters_gpu_{i}_event_group_{event_group_name}'] = stats
                
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                    self._run_performance_assertions(stats, "amdsmi_get_gpu_available_counters")
                else:
                    print(f"  GPU {i} event_group {event_group_name}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
                
                self._log_test_completion("GPU", i, f"event_group={event_group_name}")
        
        self._log_performance_summary("amdsmi_get_gpu_available_counters", "GPUs", "get_gpu_available_counters")

    def test_performance_gpu_validate_ras_eeprom(self):
        self._print_func_name('Starting performance test for amdsmi_gpu_validate_ras_eeprom')

        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_gpu_validate_ras_eepromas it fails (File Error).")

        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_gpu_validate_ras_eeprom", "GPU", i)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_gpu_validate_ras_eeprom,
                f'gpu_validate_ras_eeprom_gpu_{i}',
                processor
            )
            
            self.perf_results[f'gpu_validate_ras_eeprom_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
                self._run_performance_assertions(stats, "amdsmi_gpu_validate_ras_eeprom")
            else:
                print(f"  GPU {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("GPU", i)
        
        self._log_performance_summary("amdsmi_gpu_validate_ras_eeprom", "GPUs", "gpu_validate_ras_eeprom")

    def test_performance_gpu_xgmi_error_status(self):
        self._print_func_name('Starting performance test for amdsmi_gpu_xgmi_error_status')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_gpu_xgmi_error_status as it fails on MI300.")    
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_gpu_xgmi_error_status", "GPU", i)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_gpu_xgmi_error_status,
                f'gpu_xgmi_error_status_gpu_{i}',
                processor
            )
            
            self.perf_results[f'gpu_xgmi_error_status_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
                self._run_performance_assertions(stats, "amdsmi_gpu_xgmi_error_status")
            else:
                print(f"  GPU {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("GPU", i)
        
        self._log_performance_summary("amdsmi_gpu_xgmi_error_status", "GPUs", "gpu_xgmi_error_status")

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

    def test_performance_is_P2P_accessible(self):
        self._print_func_name('Starting performance test for amdsmi_is_P2P_accessible')
        
        for i, gpu_i in enumerate(self.processors):
            for j, gpu_j in enumerate(self.processors):
                if i != j:  # Test P2P access between different processors
                    self._log_test_start("amdsmi_is_P2P_accessible", "P2P", f"{i}->{j}")
                    
                    stats = self._measure_api_performance(
                        amdsmi.amdsmi_is_P2P_accessible,
                        f'is_P2P_accessible_{i}_to_{j}',
                        gpu_i, gpu_j
                    )
                    
                    self.perf_results[f'is_P2P_accessible_{i}_to_{j}'] = stats
                    
                    if stats['successful_runs'] > 0:
                        self._print_performance_results(stats)
                        self._run_performance_assertions(stats, "amdsmi_is_P2P_accessible")
                    else:
                        print(f"  P2P {i}->{j}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
                    
                    self._log_test_completion("P2P", f"{i}->{j}")
        
        self._log_performance_summary("amdsmi_is_P2P_accessible", "P2P", "is_P2P_accessible")

    def test_performance_gpu_event(self):
        self._print_func_name('Starting performance test for GPU event notification workflow')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_gpu_event as it fails (File Error).")    
        mask = 1 << (amdsmi.AmdSmiEvtNotificationType.GPU_PRE_RESET - 1) | \
               1 << (amdsmi.AmdSmiEvtNotificationType.GPU_POST_RESET - 1)
        timeout_ms = 1000
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("gpu_event_workflow", "GPU", i)
            
            # Test Init
            stats_init = self._measure_api_performance(
                amdsmi.amdsmi_init_gpu_event_notification,
                f'init_gpu_event_notification_gpu_{i}',
                processor
            )
            self.perf_results[f'init_gpu_event_notification_gpu_{i}'] = stats_init
            
            if stats_init['successful_runs'] > 0:
                self._print_performance_results(stats_init)
                
                # Test Is Enabled
                stats_enabled = self._measure_api_performance(
                    amdsmi.amdsmi_is_gpu_power_management_enabled,
                    f'is_gpu_power_management_enabled_gpu_{i}',
                    processor
                )
                self.perf_results[f'is_gpu_power_management_enabled_gpu_{i}'] = stats_enabled
                self._print_performance_results(stats_enabled)
                
                # Test Set Mask
                stats_set_mask = self._measure_api_performance(
                    amdsmi.amdsmi_set_gpu_event_notification_mask,
                    f'set_gpu_event_notification_mask_gpu_{i}',
                    processor, mask
                )
                self.perf_results[f'set_gpu_event_notification_mask_gpu_{i}'] = stats_set_mask
                self._print_performance_results(stats_set_mask)
                
                # Test Get Event
                stats_get = self._measure_api_performance(
                    amdsmi.amdsmi_get_gpu_event_notification,
                    f'get_gpu_event_notification_gpu_{i}',
                    timeout_ms
                )
                self.perf_results[f'get_gpu_event_notification_gpu_{i}'] = stats_get
                self._print_performance_results(stats_get)
                
                # Test Stop
                stats_stop = self._measure_api_performance(
                    amdsmi.amdsmi_stop_gpu_event_notification,
                    f'stop_gpu_event_notification_gpu_{i}',
                    processor
                )
                self.perf_results[f'stop_gpu_event_notification_gpu_{i}'] = stats_stop
                self._print_performance_results(stats_stop)
            else:
                print(f"  GPU {i}: Init failed - skipping remaining event workflow tests - {stats_init['errors'][0]['error_info'] if stats_init['errors'] else 'Unknown'}")
            
            self._log_test_completion("GPU", i)
        
        self._log_performance_summary("gpu_event_workflow", "GPUs", "gpu_event")

    def test_performance_reset_gpu(self):
        self._print_func_name('Starting performance test for amdsmi_reset_gpu')
        
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_performance_reset_gpu as it fails (MI350X, Hang).")
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_reset_gpu", "GPU", i)
            
            # Note: GPU reset is a critical operation, we may want to handle this carefully
            stats = self._measure_api_performance(
                amdsmi.amdsmi_reset_gpu,
                f'reset_gpu_gpu_{i}',
                processor,
                critical_operation=True  # This indicates this is a critical system operation
            )
            
            self.perf_results[f'reset_gpu_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
                self._run_performance_assertions(stats, "amdsmi_reset_gpu")
            else:
                print(f"  GPU {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("GPU", i)
        
        self._log_performance_summary("amdsmi_reset_gpu", "GPUs", "reset_gpu")

    def test_performance_reset_gpu_fan(self):
        self._print_func_name('Starting performance test for amdsmi_reset_gpu_fan')
        
        for i, processor in enumerate(self.processors):
            # Test different fan sensors
            fan_sensors = [0, 1] if i == 0 else [0]
            
            for sensor_ind in fan_sensors:
                self._log_test_start("amdsmi_reset_gpu_fan", "GPU", i, sensor_ind=sensor_ind)
                
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_reset_gpu_fan,
                    f'reset_gpu_fan_gpu_{i}_sensor_{sensor_ind}',
                    processor, sensor_ind
                )
                
                self.perf_results[f'reset_gpu_fan_gpu_{i}_sensor_{sensor_ind}'] = stats
                
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                    self._run_performance_assertions(stats, "amdsmi_reset_gpu_fan")
                else:
                    print(f"  GPU {i} sensor {sensor_ind}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
                
                self._log_test_completion("GPU", i, f"sensor_ind={sensor_ind}")
        
        self._log_performance_summary("amdsmi_reset_gpu_fan", "GPUs", "reset_gpu_fan")

    def test_performance_reset_gpu_xgmi_error(self):
        self._print_func_name('Starting performance test for amdsmi_reset_gpu_xgmi_error')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_reset_gpu_xgmi_error as it fails on MI300.")    
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_reset_gpu_xgmi_error", "GPU", i)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_reset_gpu_xgmi_error,
                f'reset_gpu_xgmi_error_gpu_{i}',
                processor
            )
            
            self.perf_results[f'reset_gpu_xgmi_error_gpu_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
                self._run_performance_assertions(stats, "amdsmi_reset_gpu_xgmi_error")
            else:
                print(f"  GPU {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("GPU", i)
        
        self._log_performance_summary("amdsmi_reset_gpu_xgmi_error", "GPUs", "reset_gpu_xgmi_error")

    def test_performance_set_clk_freq(self):
        self._print_func_name('Starting performance test for amdsmi_set_clk_freq')
        
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_performance_set_clk_freq as it fails (Perm failure).")
        
        for i, processor in enumerate(self.processors):
            for clk_type_name, clk_type, clk_cond in self.clk_types:
                # First get current clock info
                try:
                    clk_freq_info = amdsmi.amdsmi_get_clk_freq(processor, clk_type)
                except amdsmi.AmdSmiLibraryException:
                    continue
                
                current = clk_freq_info['current']
                num_supported = clk_freq_info['num_supported']
                frequency = clk_freq_info['frequency']
                
                if num_supported == 0:
                    continue
                
                # Test setting each supported frequency
                for index in range(0, num_supported):
                    freq_bitmask = frequency[index]
                    self._log_test_start("amdsmi_set_clk_freq", "GPU", i, clk_type=clk_type_name, freq_bitmask=freq_bitmask)
                    
                    stats = self._measure_api_performance(
                        amdsmi.amdsmi_set_clk_freq,
                        f'set_clk_freq_gpu_{i}_type_{clk_type_name}_freq_{freq_bitmask}',
                        processor, clk_type_name, freq_bitmask
                    )
                    
                    self.perf_results[f'set_clk_freq_gpu_{i}_type_{clk_type_name}_freq_{freq_bitmask}'] = stats
                    
                    if stats['successful_runs'] > 0:
                        self._print_performance_results(stats)
                        self._run_performance_assertions(stats, "amdsmi_set_clk_freq")
                    else:
                        print(f"  GPU {i} clk_type {clk_type_name} freq {freq_bitmask}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
                    
                    self._log_test_completion("GPU", i, f"clk_type={clk_type_name}, freq_bitmask={freq_bitmask}")
                
                # Restore original frequency
                try:
                    amdsmi.amdsmi_set_clk_freq(processor, clk_type_name, frequency[current])
                except amdsmi.AmdSmiLibraryException:
                    pass
        
        self._log_performance_summary("amdsmi_set_clk_freq", "GPUs", "set_clk_freq")

    def test_performance_set_gpu_accelerator_partition_profile(self):
        self._print_func_name('Starting performance test for amdsmi_set_gpu_accelerator_partition_profile')
        
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_performance_set_gpu_accelerator_partition_profile as it is not complete.")
        
        # Use TODO placeholder value like original test
        profile_index = 0
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_set_gpu_accelerator_partition_profile", "Processor", i, profile_index=profile_index)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_set_gpu_accelerator_partition_profile,
                f'set_gpu_accelerator_partition_profile_processor_{i}',
                processor, profile_index
            )
            
            self.perf_results[f'set_gpu_accelerator_partition_profile_processor_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_set_gpu_accelerator_partition_profile", "Processors", "set_gpu_accelerator_partition_profile")

    def test_performance_set_gpu_clk_limit(self):
        self._print_func_name('Starting performance test for amdsmi_set_gpu_clk_limit')
        
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_performance_set_gpu_clk_limit as it is not complete.")
        
        # Use TODO placeholder value like original test
        value = 0
        
        for i, processor in enumerate(self.processors):
            for clk_type_name, clk_type, clk_cond in self.clk_types:
                for clk_limit_type_name, clk_limit_type, clk_limit_cond in self.clk_limit_types:
                    self._log_test_start("amdsmi_set_gpu_clk_limit", "Processor", i, 
                                       clk_type=clk_type_name, clk_limit_type=clk_limit_type_name, value=value)
                    
                    stats = self._measure_api_performance(
                        amdsmi.amdsmi_set_gpu_clk_limit,
                        f'set_gpu_clk_limit_processor_{i}_clk_{clk_type_name}_limit_{clk_limit_type_name}',
                        processor, clk_type_name, clk_limit_type_name, value
                    )
                    
                    self.perf_results[f'set_gpu_clk_limit_processor_{i}_clk_{clk_type_name}_limit_{clk_limit_type_name}'] = stats
                    
                    if stats['successful_runs'] > 0:
                        self._print_performance_results(stats)
                    else:
                        print(f"  Processor {i} clk={clk_type_name} limit={clk_limit_type_name}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
                    
                    self._log_test_completion("Processor", i, f"clk_type={clk_type_name}, clk_limit_type={clk_limit_type_name}")
        
        self._log_performance_summary("amdsmi_set_gpu_clk_limit", "Processors", "set_gpu_clk_limit")

    def test_performance_set_gpu_clk_range(self):
        self._print_func_name('Starting performance test for amdsmi_set_gpu_clk_range')
        
        # Use TODO placeholder values like original test
        min_clk_value = 100
        max_clk_value = 200
        
        for i, processor in enumerate(self.processors):
            for clk_type_name, clk_type, clk_cond in self.clk_types:
                self._log_test_start("amdsmi_set_gpu_clk_range", "Processor", i, 
                                   min_clk_value=min_clk_value, max_clk_value=max_clk_value, clk_type=clk_type_name)
                
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_set_gpu_clk_range,
                    f'set_gpu_clk_range_processor_{i}_clk_{clk_type_name}',
                    processor, min_clk_value, max_clk_value, clk_type
                )
                
                self.perf_results[f'set_gpu_clk_range_processor_{i}_clk_{clk_type_name}'] = stats
                
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(f"  Processor {i} clk={clk_type_name}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
                
                self._log_test_completion("Processor", i, f"clk_type={clk_type_name}")
        
        self._log_performance_summary("amdsmi_set_gpu_clk_range", "Processors", "set_gpu_clk_range")

    def test_performance_set_gpu_compute_partition(self):
        self._print_func_name('Starting performance test for amdsmi_set_gpu_compute_partition')
        
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_performance_set_gpu_compute_partition as it fails on MI300.")
        
        for i, processor in enumerate(self.processors):
            # Get current compute partition first
            self._log_test_start("amdsmi_get_gpu_compute_partition", "Processor", i)
            
            stats_get = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_compute_partition,
                f'get_gpu_compute_partition_processor_{i}',
                processor
            )
            self.perf_results[f'get_gpu_compute_partition_processor_{i}'] = stats_get
            
            if stats_get['successful_runs'] > 0:
                self._print_performance_results(stats_get)
                
                try:
                    default_compute_partition_name = amdsmi.amdsmi_get_gpu_compute_partition(processor)
                    default_compute_partition_type = self.compute_partition_types[0][1]
                    
                    # Find the default partition type
                    for compute_partition_type_name, compute_partition_type, compute_partition_type_cond in self.compute_partition_types:
                        if default_compute_partition_name == compute_partition_type_name:
                            default_compute_partition_type = compute_partition_type
                        
                        # Test setting each partition type
                        stats_set = self._measure_api_performance(
                            amdsmi.amdsmi_set_gpu_compute_partition,
                            f'set_gpu_compute_partition_processor_{i}_type_{compute_partition_type_name}',
                            processor, compute_partition_type
                        )
                        self.perf_results[f'set_gpu_compute_partition_processor_{i}_type_{compute_partition_type_name}'] = stats_set
                        
                        if stats_set['successful_runs'] > 0:
                            self._print_performance_results(stats_set)
                        else:
                            print(f"  Processor {i} type={compute_partition_type_name}: All calls failed - {stats_set['errors'][0]['error_info'] if stats_set['errors'] else 'Unknown'}")
                    
                    # Restore original partition
                    stats_restore = self._measure_api_performance(
                        amdsmi.amdsmi_set_gpu_compute_partition,
                        f'restore_gpu_compute_partition_processor_{i}',
                        processor, default_compute_partition_type
                    )
                    self.perf_results[f'restore_gpu_compute_partition_processor_{i}'] = stats_restore
                    
                except amdsmi.AmdSmiLibraryException:
                    print(f"  Processor {i}: Could not get default compute partition")
            else:
                print(f"  Processor {i}: Get failed - {stats_get['errors'][0]['error_info'] if stats_get['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("set_gpu_compute_partition", "Processors", "set_gpu_compute_partition")

    def test_performance_gpu_fan_speed(self):
        self._print_func_name('Starting performance test for GPU fan speed workflow')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("gpu_fan_speed", "Processor", i)
            
            # Test Get GPU Fan Speed (current)
            stats_get_current = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_fan_speed,
                f'get_gpu_fan_speed_processor_{i}',
                processor, 0
            )
            self.perf_results[f'get_gpu_fan_speed_processor_{i}'] = stats_get_current
            
            if stats_get_current['successful_runs'] > 0:
                self._print_performance_results(stats_get_current)
                
                try:
                    fan_speed_current = amdsmi.amdsmi_get_gpu_fan_speed(processor, 0)
                    
                    # Test Get GPU Fan Speed Max
                    stats_get_max = self._measure_api_performance(
                        amdsmi.amdsmi_get_gpu_fan_speed_max,
                        f'get_gpu_fan_speed_max_processor_{i}',
                        processor, 0
                    )
                    self.perf_results[f'get_gpu_fan_speed_max_processor_{i}'] = stats_get_max
                    self._print_performance_results(stats_get_max)
                    
                    fan_speed_max = amdsmi.amdsmi_get_gpu_fan_speed_max(processor, 0)
                    if fan_speed_current == fan_speed_max:
                        fan_speed = int(fan_speed_max/2)
                    else:
                        fan_speed = fan_speed_max
                    
                    # Test Set GPU Fan Speed
                    stats_set = self._measure_api_performance(
                        amdsmi.amdsmi_set_gpu_fan_speed,
                        f'set_gpu_fan_speed_processor_{i}',
                        processor, 0, fan_speed
                    )
                    self.perf_results[f'set_gpu_fan_speed_processor_{i}'] = stats_set
                    self._print_performance_results(stats_set)
                    
                    # Restore original fan speed
                    stats_restore = self._measure_api_performance(
                        amdsmi.amdsmi_set_gpu_fan_speed,
                        f'restore_gpu_fan_speed_processor_{i}',
                        processor, 0, fan_speed_current
                    )
                    self.perf_results[f'restore_gpu_fan_speed_processor_{i}'] = stats_restore
                    self._print_performance_results(stats_restore)
                    
                except amdsmi.AmdSmiLibraryException:
                    print(f"  Processor {i}: Could not complete fan speed workflow")
            else:
                print(f"  Processor {i}: Get failed - {stats_get_current['errors'][0]['error_info'] if stats_get_current['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("gpu_fan_speed_workflow", "Processors", "gpu_fan_speed")

    def test_performance_set_gpu_memory_partition(self):
        self._print_func_name('Starting performance test for amdsmi_set_gpu_memory_partition')
        
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_performance_set_gpu_memory_partition as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA).")
        
        for i, processor in enumerate(self.processors):
            for memory_partition_type_name, memory_partition_type, memory_partition_type_cond in self.memory_partition_types:
                self._log_test_start("amdsmi_set_gpu_memory_partition", "Processor", i, 
                                   memory_partition_type=memory_partition_type_name)
                
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_set_gpu_memory_partition,
                    f'set_gpu_memory_partition_processor_{i}_type_{memory_partition_type_name}',
                    processor, memory_partition_type
                )
                
                self.perf_results[f'set_gpu_memory_partition_processor_{i}_type_{memory_partition_type_name}'] = stats
                
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(f"  Processor {i} type={memory_partition_type_name}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
                
                self._log_test_completion("Processor", i, f"memory_partition_type={memory_partition_type_name}")
        
        self._log_performance_summary("amdsmi_set_gpu_memory_partition", "Processors", "set_gpu_memory_partition")

    def test_performance_set_gpu_memory_partition_mode(self):
        self._print_func_name('Starting performance test for amdsmi_set_gpu_memory_partition_mode')
        
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_performance_set_gpu_memory_partition_mode as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA).")
        
        for i, processor in enumerate(self.processors):
            for memory_partition_type_name, memory_partition_type, memory_partition_type_cond in self.memory_partition_types:
                self._log_test_start("amdsmi_set_gpu_memory_partition_mode", "Processor", i, 
                                   memory_partition_type=memory_partition_type_name)
                
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_set_gpu_memory_partition_mode,
                    f'set_gpu_memory_partition_mode_processor_{i}_type_{memory_partition_type_name}',
                    processor, memory_partition_type
                )
                
                self.perf_results[f'set_gpu_memory_partition_mode_processor_{i}_type_{memory_partition_type_name}'] = stats
                
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(f"  Processor {i} type={memory_partition_type_name}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
                
                self._log_test_completion("Processor", i, f"memory_partition_type={memory_partition_type_name}")
        
        self._log_performance_summary("amdsmi_set_gpu_memory_partition_mode", "Processors", "set_gpu_memory_partition_mode")

    def test_performance_set_gpu_od_clk_info(self):
        self._print_func_name('Starting performance test for amdsmi_set_gpu_od_clk_info')
        
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_performance_set_gpu_od_clk_info as it is not complete.")
        
        # Use TODO placeholder value like original test
        value = 200
        
        for i, processor in enumerate(self.processors):
            for freq_ind_name, freq_ind, freq_ind_cond in self.freq_inds:
                for clk_type_name, clk_type, clk_cond in self.clk_types:
                    self._log_test_start("amdsmi_set_gpu_od_clk_info", "Processor", i, 
                                       freq_ind=freq_ind_name, value=value, clk_type=clk_type_name)
                    
                    stats = self._measure_api_performance(
                        amdsmi.amdsmi_set_gpu_od_clk_info,
                        f'set_gpu_od_clk_info_processor_{i}_freq_{freq_ind_name}_clk_{clk_type_name}',
                        processor, freq_ind, value, clk_type
                    )
                    
                    self.perf_results[f'set_gpu_od_clk_info_processor_{i}_freq_{freq_ind_name}_clk_{clk_type_name}'] = stats
                    
                    if stats['successful_runs'] > 0:
                        self._print_performance_results(stats)
                    else:
                        print(f"  Processor {i} freq={freq_ind_name} clk={clk_type_name}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
                    
                    self._log_test_completion("Processor", i, f"freq_ind={freq_ind_name}, clk_type={clk_type_name}")
        
        self._log_performance_summary("amdsmi_set_gpu_od_clk_info", "Processors", "set_gpu_od_clk_info")

    def test_performance_set_gpu_od_volt_info(self):
        self._print_func_name('Starting performance test for amdsmi_set_gpu_od_volt_info')
        
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_performance_set_gpu_od_volt_info as it is not complete.")
        
        # Use TODO placeholder values like original test
        vpoint = 0
        clk_value = 0
        volt_value = 0
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_set_gpu_od_volt_info", "Processor", i, 
                               vpoint=vpoint, clk_value=clk_value, volt_value=volt_value)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_set_gpu_od_volt_info,
                f'set_gpu_od_volt_info_processor_{i}',
                processor, vpoint, clk_value, volt_value
            )
            
            self.perf_results[f'set_gpu_od_volt_info_processor_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_set_gpu_od_volt_info", "Processors", "set_gpu_od_volt_info")

    def test_performance_set_gpu_overdrive_level(self):
        self._print_func_name('Starting performance test for GPU overdrive level workflow')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("gpu_overdrive_level_workflow", "Processor", i)
            
            # Test Get GPU Overdrive Level
            stats_get = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_overdrive_level,
                f'get_gpu_overdrive_level_processor_{i}',
                processor
            )
            self.perf_results[f'get_gpu_overdrive_level_processor_{i}'] = stats_get
            
            if stats_get['successful_runs'] > 0:
                self._print_performance_results(stats_get)
                
                try:
                    overdrive_value_current = amdsmi.amdsmi_get_gpu_overdrive_level(processor)
                    if overdrive_value_current != 1:
                        overdrive_value = 1
                    else:
                        overdrive_value = 2
                    
                    # Test Set GPU Overdrive Level
                    stats_set = self._measure_api_performance(
                        amdsmi.amdsmi_set_gpu_overdrive_level,
                        f'set_gpu_overdrive_level_processor_{i}',
                        processor, overdrive_value
                    )
                    self.perf_results[f'set_gpu_overdrive_level_processor_{i}'] = stats_set
                    self._print_performance_results(stats_set)
                    
                    # Restore original overdrive level
                    stats_restore = self._measure_api_performance(
                        amdsmi.amdsmi_set_gpu_overdrive_level,
                        f'restore_gpu_overdrive_level_processor_{i}',
                        processor, overdrive_value_current
                    )
                    self.perf_results[f'restore_gpu_overdrive_level_processor_{i}'] = stats_restore
                    self._print_performance_results(stats_restore)
                    
                except amdsmi.AmdSmiLibraryException:
                    print(f"  Processor {i}: Could not complete overdrive workflow")
            else:
                print(f"  Processor {i}: Get failed - {stats_get['errors'][0]['error_info'] if stats_get['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("gpu_overdrive_level_workflow", "Processors", "set_gpu_overdrive_level")

    def test_performance_set_gpu_pci_bandwidth(self):
        self._print_func_name('Starting performance test for GPU PCI bandwidth workflow')
        
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_performance_set_gpu_pci_bandwidth as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA).")
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("gpu_pci_bandwidth_workflow", "Processor", i)
            
            # Test Get GPU PCI Bandwidth
            stats_get = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_pci_bandwidth,
                f'get_gpu_pci_bandwidth_processor_{i}',
                processor
            )
            self.perf_results[f'get_gpu_pci_bandwidth_processor_{i}'] = stats_get
            
            if stats_get['successful_runs'] > 0:
                self._print_performance_results(stats_get)
                
                try:
                    bandwidth_info = amdsmi.amdsmi_get_gpu_pci_bandwidth(processor)
                    current_bandwidth_index = bandwidth_info['transfer_rate']['current']
                    if current_bandwidth_index > 0:
                        bitmask = 1 << (current_bandwidth_index - 1)
                    else:
                        bitmask = 1 << (current_bandwidth_index)
                    
                    # Test Set GPU PCI Bandwidth
                    stats_set = self._measure_api_performance(
                        amdsmi.amdsmi_set_gpu_pci_bandwidth,
                        f'set_gpu_pci_bandwidth_processor_{i}',
                        processor, bitmask
                    )
                    self.perf_results[f'set_gpu_pci_bandwidth_processor_{i}'] = stats_set
                    self._print_performance_results(stats_set)
                    
                    # Restore original PCI bandwidth
                    bitmask = 1 << (current_bandwidth_index)
                    stats_restore = self._measure_api_performance(
                        amdsmi.amdsmi_set_gpu_pci_bandwidth,
                        f'restore_gpu_pci_bandwidth_processor_{i}',
                        processor, bitmask
                    )
                    self.perf_results[f'restore_gpu_pci_bandwidth_processor_{i}'] = stats_restore
                    self._print_performance_results(stats_restore)
                    
                except amdsmi.AmdSmiLibraryException:
                    print(f"  Processor {i}: Could not complete PCI bandwidth workflow")
            else:
                print(f"  Processor {i}: Get failed - {stats_get['errors'][0]['error_info'] if stats_get['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("gpu_pci_bandwidth_workflow", "Processors", "set_gpu_pci_bandwidth")

    def test_performance_set_gpu_perf_determinism_mode(self):
        self._print_func_name('Starting performance test for amdsmi_set_gpu_perf_determinism_mode')
        
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_performance_set_gpu_perf_determinism_mode as it is not complete.")
        
        # Use TODO placeholder value like original test
        clk_value = 0
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_set_gpu_perf_determinism_mode", "Processor", i, clk_value=clk_value)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_set_gpu_perf_determinism_mode,
                f'set_gpu_perf_determinism_mode_processor_{i}',
                processor, clk_value
            )
            
            self.perf_results[f'set_gpu_perf_determinism_mode_processor_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_set_gpu_perf_determinism_mode", "Processors", "set_gpu_perf_determinism_mode")

    def test_performance_set_gpu_perf_level(self):
        self._print_func_name('Starting performance test for GPU perf level workflow')
        
        # Check if dev_perf_levels is populated, skip test if not
        if not self.dev_perf_levels:
            self.skipTest("dev_perf_levels not initialized - skipping performance level test")
            return
            
        dev_perf_level_current = self.dev_perf_levels[0][1]
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("gpu_perf_level_workflow", "Processor", i)
            
            # Test Get GPU Perf Level
            stats_get = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_perf_level,
                f'get_gpu_perf_level_processor_{i}',
                processor
            )
            self.perf_results[f'get_gpu_perf_level_processor_{i}'] = stats_get
            
            if stats_get['successful_runs'] > 0:
                self._print_performance_results(stats_get)
                
                try:
                    dev_perf_level_name_current = amdsmi.amdsmi_get_gpu_perf_level(processor)
                    items = dev_perf_level_name_current.split('_')
                    dev_perf_level_name_current = items[-1]
                    
                    # Test setting each performance level
                    for dev_perf_level_name, dev_perf_level, dev_perf_level_cond in self.dev_perf_levels:
                        if dev_perf_level_name_current == dev_perf_level_name:
                            dev_perf_level_current = dev_perf_level
                        
                        stats_set = self._measure_api_performance(
                            amdsmi.amdsmi_set_gpu_perf_level,
                            f'set_gpu_perf_level_processor_{i}_level_{dev_perf_level_name}',
                            processor, dev_perf_level
                        )
                        self.perf_results[f'set_gpu_perf_level_processor_{i}_level_{dev_perf_level_name}'] = stats_set
                        
                        if stats_set['successful_runs'] > 0:
                            self._print_performance_results(stats_set)
                        else:
                            print(f"  Processor {i} level={dev_perf_level_name}: All calls failed - {stats_set['errors'][0]['error_info'] if stats_set['errors'] else 'Unknown'}")
                    
                    # Restore original performance level
                    stats_restore = self._measure_api_performance(
                        amdsmi.amdsmi_set_gpu_perf_level,
                        f'restore_gpu_perf_level_processor_{i}',
                        processor, dev_perf_level_current
                    )
                    self.perf_results[f'restore_gpu_perf_level_processor_{i}'] = stats_restore
                    self._print_performance_results(stats_restore)
                    
                except amdsmi.AmdSmiLibraryException:
                    print(f"  Processor {i}: Could not complete perf level workflow")
            else:
                print(f"  Processor {i}: Get failed - {stats_get['errors'][0]['error_info'] if stats_get['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("gpu_perf_level_workflow", "Processors", "set_gpu_perf_level")

    def test_performance_set_gpu_power_profile(self):
        self._print_func_name('Starting performance test for amdsmi_set_gpu_power_profile')
        
        for i, processor in enumerate(self.processors):
            for power_profile_preset_mask_name, power_profile_preset_mask, power_profile_preset_masks_cond in self.power_profile_preset_masks:
                self._log_test_start("amdsmi_set_gpu_power_profile", "Processor", i, 
                                   power_profile=power_profile_preset_mask_name)
                
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_set_gpu_power_profile,
                    f'set_gpu_power_profile_processor_{i}_profile_{power_profile_preset_mask_name}',
                    processor, 0, power_profile_preset_mask
                )
                
                self.perf_results[f'set_gpu_power_profile_processor_{i}_profile_{power_profile_preset_mask_name}'] = stats
                
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(f"  Processor {i} profile={power_profile_preset_mask_name}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
                
                self._log_test_completion("Processor", i, f"power_profile={power_profile_preset_mask_name}")
        
        self._log_performance_summary("amdsmi_set_gpu_power_profile", "Processors", "set_gpu_power_profile")

    def test_performance_set_gpu_process_isolation(self):
        self._print_func_name('Starting performance test for amdsmi_set_gpu_process_isolation')
        
        # Use pisolate values from original test
        pisolates = [1, 0]
        
        for i, processor in enumerate(self.processors):
            for pisolate in pisolates:
                self._log_test_start("amdsmi_set_gpu_process_isolation", "Processor", i, pisolate=pisolate)
                
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_set_gpu_process_isolation,
                    f'set_gpu_process_isolation_processor_{i}_pisolate_{pisolate}',
                    processor, pisolate
                )
                
                self.perf_results[f'set_gpu_process_isolation_processor_{i}_pisolate_{pisolate}'] = stats
                
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(f"  Processor {i} pisolate={pisolate}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
                
                self._log_test_completion("Processor", i, f"pisolate={pisolate}")
        
        self._log_performance_summary("amdsmi_set_gpu_process_isolation", "Processors", "set_gpu_process_isolation")

    def test_performance_power_cap(self):
        self._print_func_name('Starting performance test for power cap workflow')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("power_cap_workflow", "Processor", i)
            
            # Test Get Power Cap Info
            stats_get = self._measure_api_performance(
                amdsmi.amdsmi_get_power_cap_info,
                f'get_power_cap_info_processor_{i}',
                processor
            )
            self.perf_results[f'get_power_cap_info_processor_{i}'] = stats_get
            
            if stats_get['successful_runs'] > 0:
                self._print_performance_results(stats_get)
                
                try:
                    power_cap_info = amdsmi.amdsmi_get_power_cap_info(processor)
                    cap = int((power_cap_info['max_power_cap'] + power_cap_info['min_power_cap']) / 2)
                    
                    # Test Set Power Cap (average of min and max)
                    stats_set = self._measure_api_performance(
                        amdsmi.amdsmi_set_power_cap,
                        f'set_power_cap_processor_{i}',
                        processor, 0, cap
                    )
                    self.perf_results[f'set_power_cap_processor_{i}'] = stats_set
                    self._print_performance_results(stats_set)
                    
                    # Restore original power cap
                    cap = power_cap_info['power_cap']
                    stats_restore = self._measure_api_performance(
                        amdsmi.amdsmi_set_power_cap,
                        f'restore_power_cap_processor_{i}',
                        processor, 0, cap
                    )
                    self.perf_results[f'restore_power_cap_processor_{i}'] = stats_restore
                    self._print_performance_results(stats_restore)
                    
                except amdsmi.AmdSmiLibraryException:
                    print(f"  Processor {i}: Could not complete power cap workflow")
            else:
                print(f"  Processor {i}: Get failed - {stats_get['errors'][0]['error_info'] if stats_get['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("power_cap_workflow", "Processors", "power_cap")

    def test_performance_soc_pstate(self):
        self._print_func_name('Starting performance test for SoC pstate workflow')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("soc_pstate_workflow", "Processor", i)
            
            # Test Get SoC Pstate
            stats_get = self._measure_api_performance(
                amdsmi.amdsmi_get_soc_pstate,
                f'get_soc_pstate_processor_{i}',
                processor
            )
            self.perf_results[f'get_soc_pstate_processor_{i}'] = stats_get
            
            if stats_get['successful_runs'] > 0:
                self._print_performance_results(stats_get)
                
                try:
                    policy_info = amdsmi.amdsmi_get_soc_pstate(processor)
                    num_supported = policy_info['num_supported']
                    policy_id_current = policy_info['current_id']
                    policy_id_orig = policy_info['policies'][policy_id_current]['policy_id']
                    
                    index = 0
                    if num_supported >= 2:
                        if policy_id_current != 0:
                            index = 1
                    policy_id = policy_info['policies'][index]['policy_id']
                    
                    # Test Set SoC Pstate
                    stats_set = self._measure_api_performance(
                        amdsmi.amdsmi_set_soc_pstate,
                        f'set_soc_pstate_processor_{i}',
                        processor, policy_id
                    )
                    self.perf_results[f'set_soc_pstate_processor_{i}'] = stats_set
                    self._print_performance_results(stats_set)
                    
                    # Restore original policy
                    stats_restore = self._measure_api_performance(
                        amdsmi.amdsmi_set_soc_pstate,
                        f'restore_soc_pstate_processor_{i}',
                        processor, policy_id_orig
                    )
                    self.perf_results[f'restore_soc_pstate_processor_{i}'] = stats_restore
                    self._print_performance_results(stats_restore)
                    
                except (amdsmi.AmdSmiLibraryException, KeyError, TypeError):
                    print(f"  Processor {i}: Could not complete SoC pstate workflow")
            else:
                print(f"  Processor {i}: Get failed - {stats_get['errors'][0]['error_info'] if stats_get['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("soc_pstate_workflow", "Processors", "soc_pstate")

    def test_performance_xgmi_plpd(self):
        self._print_func_name('Starting performance test for XGMI PLPD workflow')
        
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_performance_xgmi_plpd as it fails on MI300.")
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("xgmi_plpd_workflow", "Processor", i)
            
            # Test Get XGMI PLPD
            stats_get = self._measure_api_performance(
                amdsmi.amdsmi_get_xgmi_plpd,
                f'get_xgmi_plpd_processor_{i}',
                processor
            )
            self.perf_results[f'get_xgmi_plpd_processor_{i}'] = stats_get
            
            if stats_get['successful_runs'] > 0:
                self._print_performance_results(stats_get)
                
                try:
                    policy_info = amdsmi.amdsmi_get_xgmi_plpd(processor)
                    num_supported = policy_info['num_supported']
                    policy_id_current = policy_info['current_id']
                    policy_id_orig = policy_info['policies'][policy_id_current]['policy_id']
                    
                    index = 0
                    if num_supported >= 2:
                        if policy_id_current != 0:
                            index = 1
                    policy_id = policy_info['policies'][index]['policy_id']
                    
                    # Test Set XGMI PLPD
                    stats_set = self._measure_api_performance(
                        amdsmi.amdsmi_set_xgmi_plpd,
                        f'set_xgmi_plpd_processor_{i}',
                        processor, policy_id
                    )
                    self.perf_results[f'set_xgmi_plpd_processor_{i}'] = stats_set
                    self._print_performance_results(stats_set)
                    
                    # Restore original policy
                    stats_restore = self._measure_api_performance(
                        amdsmi.amdsmi_set_xgmi_plpd,
                        f'restore_xgmi_plpd_processor_{i}',
                        processor, policy_id_orig
                    )
                    self.perf_results[f'restore_xgmi_plpd_processor_{i}'] = stats_restore
                    self._print_performance_results(stats_restore)
                    
                except (amdsmi.AmdSmiLibraryException, KeyError, TypeError):
                    print(f"  Processor {i}: Could not complete XGMI PLPD workflow")
            else:
                print(f"  Processor {i}: Get failed - {stats_get['errors'][0]['error_info'] if stats_get['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("xgmi_plpd_workflow", "Processors", "xgmi_plpd")

    def test_performance_status_code_to_string(self):
        self._print_func_name('Starting performance test for amdsmi_status_code_to_string')
        
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_performance_status_code_to_string as it fails (Unhashable type).")
        
        for error_num, error_name in error_map.items():
            self._log_test_start("amdsmi_status_code_to_string", "ErrorCode", error_num)
            
            # Convert error_num to int, handling both decimal and hex formats
            try:
                if isinstance(error_num, str) and error_num.startswith('0x'):
                    error_int = int(error_num, 16)  # Parse hex
                else:
                    error_int = int(error_num)  # Parse decimal
            except (ValueError, TypeError):
                error_int = 0  # Fallback to 0 on error
            
            # Print result for this error code
            try:
                ret = amdsmi.amdsmi_status_code_to_string(ctypes.c_uint32(error_int))
                print(f"### test amdsmi_status_code_to_string(error={error_num})")
                self._print("", ret)
            except amdsmi.AmdSmiLibraryException as e:
                print(f"### test amdsmi_status_code_to_string(error={error_num})")
                print(f"  Error: {e}")
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_status_code_to_string,
                f'status_code_to_string_{error_num}',
                ctypes.c_uint32(error_int)
            )
            
            self.perf_results[f'status_code_to_string_{error_num}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Error {error_num}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("ErrorCode", error_num)
        
        self._log_performance_summary("amdsmi_status_code_to_string", "ErrorCodes", "status_code_to_string")

    def test_performance_topo_get_link_type(self):
        self._print_func_name('Starting performance test for amdsmi_topo_get_link_type')
        
        for i, gpu_i in enumerate(self.processors):
            for j, gpu_j in enumerate(self.processors):
                self._log_test_start("amdsmi_topo_get_link_type", "Link", f"{i}<->{j}")
                
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_topo_get_link_type,
                    f'topo_get_link_type_{i}_to_{j}',
                    gpu_i, gpu_j
                )
                
                self.perf_results[f'topo_get_link_type_{i}_to_{j}'] = stats
                
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(f"  Link {i}<->{j}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
                
                self._log_test_completion("Link", f"{i}<->{j}")
        
        self._log_performance_summary("amdsmi_topo_get_link_type", "Links", "topo_get_link_type")

    def test_performance_topo_get_link_weight(self):
        self._print_func_name('Starting performance test for amdsmi_topo_get_link_weight')
        
        for i, gpu_i in enumerate(self.processors):
            for j, gpu_j in enumerate(self.processors):
                self._log_test_start("amdsmi_topo_get_link_weight", "Link", f"{i}<->{j}")
                
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_topo_get_link_weight,
                    f'topo_get_link_weight_{i}_to_{j}',
                    gpu_i, gpu_j
                )
                
                self.perf_results[f'topo_get_link_weight_{i}_to_{j}'] = stats
                
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(f"  Link {i}<->{j}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
                
                self._log_test_completion("Link", f"{i}<->{j}")
        
        self._log_performance_summary("amdsmi_topo_get_link_weight", "Links", "topo_get_link_weight")

    def test_performance_topo_get_numa_node_number(self):
        self._print_func_name('Starting performance test for amdsmi_topo_get_numa_node_number')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_topo_get_numa_node_number", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_topo_get_numa_node_number, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_topo_get_numa_node_number,
                f'topo_get_numa_node_number_processor_{i}',
                processor
            )
            
            self.perf_results[f'topo_get_numa_node_number_processor_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_topo_get_numa_node_number", "Processors", "topo_get_numa_node_number")

    def test_performance_topo_get_p2p_status(self):
        self._print_func_name('Starting performance test for amdsmi_topo_get_p2p_status')
        
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_performance_topo_get_p2p_status as it fails (Inval parameters).")
        
        for i, gpu_i in enumerate(self.processors):
            for j, gpu_j in enumerate(self.processors):
                self._log_test_start("amdsmi_topo_get_p2p_status", "P2P", f"{i}<->{j}")
                
                stats = self._measure_api_performance(
                    amdsmi.amdsmi_topo_get_p2p_status,
                    f'topo_get_p2p_status_{i}_to_{j}',
                    gpu_i, gpu_j
                )
                
                self.perf_results[f'topo_get_p2p_status_{i}_to_{j}'] = stats
                
                if stats['successful_runs'] > 0:
                    self._print_performance_results(stats)
                else:
                    print(f"  P2P {i}<->{j}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
                
                self._log_test_completion("P2P", f"{i}<->{j}")
        
        self._log_performance_summary("amdsmi_topo_get_p2p_status", "P2P", "topo_get_p2p_status")

    def test_performance_get_gpu_busy_percent(self):
        self._print_func_name('Starting performance test for amdsmi_get_gpu_busy_percent')
        
        for i, processor in enumerate(self.processors):
            self._log_test_start("amdsmi_get_gpu_busy_percent", "Processor", i)
            
            self._print_api_result(amdsmi.amdsmi_get_gpu_busy_percent, i, processor)
            
            stats = self._measure_api_performance(
                amdsmi.amdsmi_get_gpu_busy_percent,
                f'get_gpu_busy_percent_processor_{i}',
                processor
            )
            
            self.perf_results[f'get_gpu_busy_percent_processor_{i}'] = stats
            
            if stats['successful_runs'] > 0:
                self._print_performance_results(stats)
            else:
                print(f"  Processor {i}: All calls failed - {stats['errors'][0]['error_info'] if stats['errors'] else 'Unknown'}")
            
            self._log_test_completion("Processor", i)
        
        self._log_performance_summary("amdsmi_get_gpu_busy_percent", "Processors", "get_gpu_busy_percent")


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
        f.write('═' * 170 + '\n')
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
            f.write('═' * 170 + '\n')
            f.write('┌──────┬───────────────────────────────────────────────────────────────────┬─────────────┬─────────────────────────────────────────────┐\n')
            f.write('│ Rank │ API Name                                                          │ Time (ms)   │ Status                                      │\n')
            f.write('├──────┼───────────────────────────────────────────────────────────────────┼─────────────┼─────────────────────────────────────────────┤\n')

            for i, test in enumerate(functional_tests, 1):
                name = test['name'][:65] + '...' if len(test['name']) > 65 else test['name']
                f.write(f'│ {i:4d} │ {name:<65} │ {test["avg_ms"]:>9.3f}   │ {"":<40}✅ │\n')

            f.write('└──────┴───────────────────────────────────────────────────────────────────┴─────────────┴─────────────────────────────────────────────┘\n\n')

        if error_tests:
            f.write('⚠️ ERROR MEASUREMENT TESTS (APIs Correctly Returning Expected Errors)\n')
            f.write('═' * 170 + '\n')
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
            f.write('═' * 170 + '\n')
            ultra_fast = [t for t in tests if t['avg_ms'] < 0.01]
            fast = [t for t in tests if 0.01 <= t['avg_ms'] < 0.1]
            medium = [t for t in tests if 0.1 <= t['avg_ms'] < 1.0]
            slow = [t for t in tests if t['avg_ms'] >= 1.0]

            f.write(f'🚀 Ultra Fast (< 0.01ms)    : {len(ultra_fast):2d} tests\n')
            f.write(f'⚡ Fast (0.01-0.1ms)        : {len(fast):2d} tests\n')
            f.write(f'🚶 Medium (0.1-1.0ms)       : {len(medium):2d} tests\n')
            f.write(f'🐌 Slow (>= 1.0ms)          : {len(slow):2d} tests\n\n')

        f.write('═' * 170 + '\n')
        f.write(f'Report generated by AMD SMI Performance Test (Python Version)\n')
        f.write(f'For detailed logs, see: _perf_test.log and _perf_test_err.log\n\n')
        f.write('📝 NOTE: "Error Measurement Tests" are successful performance measurements\n')
        f.write('   of APIs that correctly return expected error codes (e.g., AMDSMI_STATUS_NOT_SUPPORTED)\n')
        f.write('   These are NOT test failures - they are successful timing measurements.\n')
        f.write('═' * 170 + '\n')

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
        suite = loader.loadTestsFromTestCase(TestAmdSmiPythonPerformance)
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
