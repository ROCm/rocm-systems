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

"""
In Unit Testing, what is specifically tested within these units includes:
1. Correctness of logic: Ensuring that the code performs its intended operations
   accurately and produces the expected results.
2. Edge cases and boundary conditions: Testing how the unit behaves with extreme
   or unusual inputs, such as null values, empty strings, maximum/minimum values,
   or invalid data.
3. Error handling: Verifying that the unit correctly handles errors and exceptions,
   including throwing appropriate exceptions or returning specific error codes.
4. State changes: For classes or objects, testing that their internal state is modified
   as expected after method calls.
5. Interactions with dependencies (mocked): While unit tests aim for isolation, units
   often have dependencies on other parts of the system or external resources. In unit
   testing, these dependencies are typically "mocked" or "stubbed" to control their
   behavior and ensure the test focuses solely on the unit under examination.
"""

import ctypes
import os
import sys
import unittest

import common

amdsmi_path = os.environ.get("AMDSMI_PATH", "/opt/rocm/share/amd_smi")
if not os.path.exists(amdsmi_path):
    raise FileNotFoundError(
        f'AMDSMI_PATH "{amdsmi_path}" does not exist. Please set the correct path in your environment.'
    )
sys.path.append(amdsmi_path)
try:
    import amdsmi
except ImportError as exc:
    raise ImportError(f"Could not import {amdsmi_path}") from exc


verbose = common.VERBOSITY_NORMAL
if "-q" in sys.argv or "--quiet" in sys.argv:
    verbose = common.VERBOSITY_QUIET
elif any(a in ("-v", "-vv", "--verbose") for a in sys.argv):
    verbose = common.VERBOSITY_VERBOSE


class TestAmdSmiPythonBDF(unittest.TestCase):
    valid_bdfs = {
        "00:00.0": [0, 0, 0, 0],
        "01:01.1": [0, 1, 1, 1],
        "FF:1F.7": [0, 255, 31, 7],
        "FF:00.7": [0, 255, 0, 7],
        "11:01.2": [0, 17, 1, 2],
        "11:0a.2": [0, 17, 10, 2],
        "0000:FF:1F.7": [0, 255, 31, 7],
        "0001:ff:1F.7": [1, 255, 31, 7],
        "ffff:FF:1f.7": [65535, 255, 31, 7],
    }

    invalid_bdfs = {
        # invalid bdf strings, expect None
        None: None,
        "": None,
        "00:00:0": None,
        "00.00:0": None,
        "00:00.Z": None,
        "00:0Z.0": None,
        "0Z:00.0": None,
        "Z00:00.0": None,
        "A00:00.0": None,
        "0A00:00.0": None,
        "00:00.07": None,
        "00:00.8": None,
        "00:00.10": None,
        "00:00.11": None,
        "00:00.-1": None,
        "00:00.*-1": None,
        "00:00.123": None,
        "00:20.0": None,
        "00:45.0": None,
        "00:200.0": None,
        "00:002.0": None,
        "100:00.0": None,
        "0100:00.0": None,
        "00100:00.0": None,
        "0101:00.0": None,
        "00001:00.0": None,
        "10001:00.0": None,
        "45:0.0": None,
        ".00:00.0": None,
        "00.00.0": None,
        "00.0.0": None,
        "0.00.0": None,
        "000.00.0": None,
        "00 00 0": None,
        " 00:00.0": None,
        "00:00.0 ": None,
        "0000:00.00.0": None,
        "000:00:00.0": None,
        "00:00:00.1": None,
        "0:00:00.1": None,
        "0000 00 00 0": None,
        "-1-1:00:00.0": None,
        "AAAA:00:AA.0": None,
        "*1*1:00:00.0": None,
        "0000:00:00.07": None,
        "0000:00:00.8": None,
        "0000:00:00.10": None,
        "0000:00:00.11": None,
        "0000:00:00.-1": None,
        "0000:00:00.*-1": None,
        "0000:00:00.123": None,
        "0000:00:20.0": None,
        "0000:00:45.0": None,
        "0000:00:200.0": None,
        "0000:00:002.0": None,
        "0000:100:00.0": None,
        "0000:0100:00.0": None,
        "0000:00100:00.0": None,
        "0000:0101:00.0": None,
        "0000:00001:00.0": None,
        "0000:10001:00.0": None,
        "0000:45:0.0": None,
        ".0000.00:00.0": None,
        "0000.00.0.0": None,
        " 0000:00:00.0": None,
        "0000:00:00.0 ": None,
    }

    @classmethod
    def _convert_bdf_to_long(cls, bdf):
        if len(bdf) == 12:
            return bdf
        if len(bdf) == 7:
            return "0000:" + bdf
        return None

    def test_parse_bdf(self):
        # go through all bdfs
        expectations = self.valid_bdfs.copy()
        expectations.update(self.invalid_bdfs)
        for bdf in expectations:
            expected = expectations[bdf]
            result = amdsmi.amdsmi_interface._parse_bdf(bdf)
            self.assertEqual(
                result, expected, f"Expected {expected} for bdf {bdf}, but got {result}"
            )
        return

    def test_format_bdf(self):
        # go through valid bdfs
        expectations = self.valid_bdfs.copy()
        for bdf_string in expectations:
            # use key as result and value as input
            bdf_list = expectations[bdf_string]
            smi_bdf = amdsmi.amdsmi_interface._make_amdsmi_bdf_from_list(bdf_list)
            expected = TestAmdSmiPythonBDF._convert_bdf_to_long(bdf_string)
            if expected:
                expected = expected.lower()
            if smi_bdf:
                result = amdsmi.amdsmi_interface._format_bdf(smi_bdf)
            else:
                result = "None"
            self.assertEqual(
                result, expected, f"Expected {expected} for bdf {bdf_string}, but got {result}"
            )
        return

    def test_check_res(self):
        # expect retry error to raise SmiRetryException
        with self.assertRaises(amdsmi.AmdSmiRetryException) as retry_test:
            amdsmi.amdsmi_interface._check_res(
                (lambda: amdsmi.amdsmi_wrapper.AMDSMI_STATUS_RETRY)()
            )
        # except retry error to have AMDSMI_STATUS_RETRY error code
        self.assertEqual(
            retry_test.exception.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_RETRY
        )

        # expect timeout error to raise SmiTimeoutException
        with self.assertRaises(amdsmi.AmdSmiTimeoutException) as timeout_test:
            amdsmi.amdsmi_interface._check_res(
                (lambda: amdsmi.amdsmi_wrapper.AMDSMI_STATUS_TIMEOUT)()
            )
        # except timeout error to have AMDSMI_STATUS_RETRY error code
        self.assertEqual(
            timeout_test.exception.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_TIMEOUT
        )

        # expect invalid args error to raise AmdSmiLibraryException
        with self.assertRaises(amdsmi.AmdSmiLibraryException) as inval_test:
            amdsmi.amdsmi_interface._check_res(
                (lambda: amdsmi.amdsmi_wrapper.AMDSMI_STATUS_INVAL)()
            )
        # expect invalid args error to have AMDSMI_STATUS_INVAL error code
        self.assertEqual(
            inval_test.exception.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_INVAL
        )
        return


class TestAmdSmiPython(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    @classmethod
    def setUpClass(cls):
        cls.common = common.Common(verbose)

        if cls.common.verbose > common.VERBOSITY_QUIET:
            # Execute the following to print the asic and board info once per test run
            for i in range(len(cls.common.processors)):
                msg = f"gpu={i}"
                cls.common.print(msg)
                if i < len(cls.common.virt_mode):
                    msg = f"virtualization mode(gpu={i})"
                    cls.common.print(msg, cls.common.virt_mode[i])
                if i < len(cls.common.asic_info):
                    msg = f"asic info(gpu={i})"
                    cls.common.print(msg, cls.common.asic_info[i])
                if i < len(cls.common.board_info):
                    msg = f"board info(gpu={i})"
                    cls.common.print(msg, cls.common.board_info[i])
                cls.common.print("")
        return

    @classmethod
    def tearDownClass(cls):
        # Balance the amdsmi_smart_init() called in Common.__init__ during setUpClass.
        # Each test's tearDown shuts down its own init; this covers the class-level init.
        try:
            amdsmi.amdsmi_shut_down()
        except amdsmi.AmdSmiLibraryException:
            pass
        return

    def setUp(self):
        # Called before each test by unittest framework
        self.raise_exception = None
        self.common.amdsmi_smart_init()

        # Refresh processor handles after each init — old handles from Common.__init__
        # become stale after shutdown/reinit cycles and some API paths return
        # AMDSMI_STATUS_NOT_FOUND when using them.
        self.common.processors = amdsmi.amdsmi_get_processor_handles()
        return

    def tearDown(self):
        # Called after each test by unittest framework
        amdsmi.amdsmi_shut_down()
        return

    def test_clean_gpu_local_data(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_clean_gpu_local_data=amdsmi.amdsmi_clean_gpu_local_data)
        return

    def test_cpu_apb_disable(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_cpu_apb_disable=amdsmi.amdsmi_cpu_apb_disable, pstate=0)
        return

    def test_cpu_apb_enable(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_cpu_apb_enable=amdsmi.amdsmi_cpu_apb_enable)
        return

    def test_first_online_core_on_cpu_socket(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_first_online_core_on_cpu_socket as it fails (IO Error)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(
            amdsmi_first_online_core_on_cpu_socket=amdsmi.amdsmi_first_online_core_on_cpu_socket
        )
        return

    def test_get_clk_freq(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = (
                "\tSkipping test_get_clk_freq as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA)."
            )
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(amdsmi_get_clk_freq=amdsmi.amdsmi_get_clk_freq)
        return

    def test_get_clock_info(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_clock_info as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(amdsmi_get_clock_info=amdsmi.amdsmi_get_clock_info)
        return

    def test_get_cpu_cclk_limit(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_cpu_cclk_limit=amdsmi.amdsmi_get_cpu_cclk_limit)
        return

    def test_get_cpu_core_current_freq_limit(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_core_current_freq_limit=amdsmi.amdsmi_get_cpu_core_current_freq_limit
        )
        return

    def test_get_cpu_core_energy(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_cpu_core_energy as it fails (IO Error)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(amdsmi_get_cpu_core_energy=amdsmi.amdsmi_get_cpu_core_energy)
        return

    # no gpu but have list
    def test_get_cpu_current_io_bandwidth(self):
        self.common.print_func_name("")
        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            for encoding_name, encoding, encoding_cond in self.common.io_bw_encodings:
                msg = f"\t### amdsmi_get_cpu_current_io_bandwidth(gpu={i}, encoding={encoding} encoding_name={encoding_name}):"
                try:
                    ret = amdsmi.amdsmi_get_cpu_current_io_bandwidth(gpu, encoding, encoding_name)
                    self.common.print(msg, ret)
                    self.common.check_ret("", "", self.common.PASS)
                except amdsmi.AmdSmiLibraryException as e:
                    if self.common.check_ret(msg, e, encoding_cond):
                        self.raise_exception = e
                self.common.print("")
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_cpu_ddr_bw(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_cpu_ddr_bw=amdsmi.amdsmi_get_cpu_ddr_bw)
        return

    def test_get_cpu_dimm_power_consumption(self):
        self.common.print_func_name("")

        # TODO Find better way to get dimm_addr
        dimm_addr = 0

        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_dimm_power_consumption=amdsmi.amdsmi_get_cpu_dimm_power_consumption,
            dimm_addr=dimm_addr,
        )
        return

    def test_get_cpu_dimm_temp_range_and_refresh_rate(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_cpu_dimm_temp_range_and_refresh_rate as it fails."
            self.common.print(msg)
            self.skipTest(msg)

        # TODO Find better way to get dimm_addr
        dimm_addr = 0

        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_dimm_temp_range_and_refresh_rate=amdsmi.amdsmi_get_cpu_dimm_temp_range_and_refresh_rate,
            dimm_addr=dimm_addr,
        )
        return

    def test_get_cpu_dimm_thermal_sensor(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_cpu_dimm_thermal_sensor as it fails."
            self.common.print(msg)
            self.skipTest(msg)

        # TODO Find better way to get dimm_addr
        dimm_addr = 0

        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_dimm_thermal_sensor=amdsmi.amdsmi_get_cpu_dimm_thermal_sensor,
            dimm_addr=dimm_addr,
        )
        return

    def test_get_cpu_family(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_cpu_family as it fails (IO Error)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API(amdsmi_get_cpu_family=amdsmi.amdsmi_get_cpu_family)
        return

    def test_get_cpu_fclk_mclk(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_cpu_fclk_mclk=amdsmi.amdsmi_get_cpu_fclk_mclk)
        return

    def test_get_cpu_handles(self):
        self.common.print_func_name("")
        self.common.Test_API(amdsmi_get_cpu_handles=amdsmi.amdsmi_get_cpu_handles)
        return

    def test_get_cpu_hsmp_driver_version(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_cpu_hsmp_driver_version as it fails (IO Error)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_hsmp_driver_version=amdsmi.amdsmi_get_cpu_hsmp_driver_version
        )
        return

    def test_get_cpu_hsmp_proto_ver(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_cpu_hsmp_proto_ver as it fails (IO Error)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_hsmp_proto_ver=amdsmi.amdsmi_get_cpu_hsmp_proto_ver
        )
        return

    def test_get_cpu_model(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_cpu_model as it fails (IO Error)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API(amdsmi_get_cpu_model=amdsmi.amdsmi_get_cpu_model)
        return

    def test_get_cpu_prochot_status(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_prochot_status=amdsmi.amdsmi_get_cpu_prochot_status
        )
        return

    def test_get_cpu_pwr_svi_telemetry_all_rails(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_pwr_svi_telemetry_all_rails=amdsmi.amdsmi_get_cpu_pwr_svi_telemetry_all_rails
        )
        return

    def test_get_cpu_smu_fw_version(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_smu_fw_version=amdsmi.amdsmi_get_cpu_smu_fw_version
        )
        return

    def test_get_cpu_socket_c0_residency(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_socket_c0_residency=amdsmi.amdsmi_get_cpu_socket_c0_residency
        )
        return

    def test_get_cpu_socket_current_active_freq_limit(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_socket_current_active_freq_limit=amdsmi.amdsmi_get_cpu_socket_current_active_freq_limit
        )
        return

    def test_get_cpu_socket_energy(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_cpu_socket_energy as it fails (IO Error)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_socket_energy=amdsmi.amdsmi_get_cpu_socket_energy
        )
        return

    def test_get_cpu_socket_freq_range(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_socket_freq_range=amdsmi.amdsmi_get_cpu_socket_freq_range
        )
        return

    def test_get_cpu_socket_lclk_dpm_level(self):
        self.common.print_func_name("")

        # TODO nbio_id = 0
        nbio_id = 0

        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_socket_lclk_dpm_level=amdsmi.amdsmi_get_cpu_socket_lclk_dpm_level,
            nbio_id=nbio_id,
        )
        return

    def test_get_cpu_socket_power(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_cpu_socket_power=amdsmi.amdsmi_get_cpu_socket_power)
        return

    def test_get_cpu_socket_power_cap(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_socket_power_cap=amdsmi.amdsmi_get_cpu_socket_power_cap
        )
        return

    def test_get_cpu_socket_power_cap_max(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_socket_power_cap_max=amdsmi.amdsmi_get_cpu_socket_power_cap_max
        )
        return

    def test_get_cpu_socket_temperature(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_socket_temperature=amdsmi.amdsmi_get_cpu_socket_temperature
        )
        return

    def test_get_energy_count(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_energy_count as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(amdsmi_get_energy_count=amdsmi.amdsmi_get_energy_count)
        return

    # no gpu but have list
    def test_get_esmi_err_msg(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_esmi_err_msg as it fails (Unknown Error)."
            self.common.print(msg)
            self.skipTest(msg)

        for _, status_type, status_cond in self.common.status_types:
            msg = f"\t### amdsmi_get_esmi_err_msg(status_type={status_type}):"
            try:
                ret = amdsmi.amdsmi_get_esmi_err_msg(status_type)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except amdsmi.AmdSmiLibraryException as e:
                if self.common.check_ret(msg, e, status_cond):
                    self.raise_exception = e
            self.common.print("")
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_fw_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_fw_info=amdsmi.amdsmi_get_fw_info)
        return

    def test_get_gpu_accelerator_partition_profile(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_accelerator_partition_profile=amdsmi.amdsmi_get_gpu_accelerator_partition_profile
        )
        return

    def test_get_gpu_accelerator_partition_profile_config(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_accelerator_partition_profile_config=amdsmi.amdsmi_get_gpu_accelerator_partition_profile_config
        )
        return

    def test_get_gpu_activity(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_gpu_activity as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(amdsmi_get_gpu_activity=amdsmi.amdsmi_get_gpu_activity)
        return

    def test_get_gpu_asic_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_asic_info=amdsmi.amdsmi_get_gpu_asic_info)
        return

    def test_get_gpu_bad_page_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_bad_page_info=amdsmi.amdsmi_get_gpu_bad_page_info
        )
        return

    def test_get_gpu_bad_page_threshold(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_bad_page_threshold=amdsmi.amdsmi_get_gpu_bad_page_threshold
        )
        return

    def test_get_gpu_bdf_id(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_bdf_id=amdsmi.amdsmi_get_gpu_bdf_id)
        return

    def test_get_gpu_board_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_board_info=amdsmi.amdsmi_get_gpu_board_info)
        return

    def test_get_gpu_cache_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_cache_info=amdsmi.amdsmi_get_gpu_cache_info)
        return

    def test_get_gpu_compute_partition(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_compute_partition=amdsmi.amdsmi_get_gpu_compute_partition
        )
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

    def test_get_gpu_device_bdf(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_device_bdf=amdsmi.amdsmi_get_gpu_device_bdf)
        return

    def test_get_gpu_device_uuid(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_device_uuid=amdsmi.amdsmi_get_gpu_device_uuid)
        return

    def test_get_gpu_driver_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_driver_info=amdsmi.amdsmi_get_gpu_driver_info)
        return

    def test_get_gpu_ecc_count(self):
        self.common.print_func_name("")
        self.common.Test_Per_GPU_With_One_Enum(
            amdsmi_get_gpu_ecc_count=amdsmi.amdsmi_get_gpu_ecc_count,
            gpu_block=self.common.gpu_blocks,
        )
        return

    def test_get_gpu_ecc_enabled(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_ecc_enabled=amdsmi.amdsmi_get_gpu_ecc_enabled)
        return

    def test_get_gpu_ecc_status(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_gpu_ecc_status as it fails."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_Per_GPU_With_One_Enum(
            amdsmi_get_gpu_ecc_status=amdsmi.amdsmi_get_gpu_ecc_status,
            gpu_block=self.common.gpu_blocks,
        )
        return

    def test_get_gpu_enumeration_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_enumeration_info=amdsmi.amdsmi_get_gpu_enumeration_info
        )
        return

    def test_get_gpu_fan_rpms(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_fan_rpms=amdsmi.amdsmi_get_gpu_fan_rpms, index=0
        )
        return

    def test_get_gpu_id(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_id=amdsmi.amdsmi_get_gpu_id)
        return

    def test_get_gpu_kfd_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_kfd_info=amdsmi.amdsmi_get_gpu_kfd_info)
        return

    def test_get_gpu_mem_overdrive_level(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_mem_overdrive_level=amdsmi.amdsmi_get_gpu_mem_overdrive_level
        )
        return

    def test_get_gpu_memory_partition(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_memory_partition=amdsmi.amdsmi_get_gpu_memory_partition
        )
        return

    def test_get_gpu_memory_partition_config(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_gpu_memory_partition_config as it fails on MI300."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_memory_partition_config=amdsmi.amdsmi_get_gpu_memory_partition_config
        )
        return

    def test_get_gpu_memory_reserved_pages(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_memory_reserved_pages=amdsmi.amdsmi_get_gpu_memory_reserved_pages
        )
        return

    def test_get_gpu_memory_total(self):
        self.common.print_func_name("")
        self.common.Test_Per_GPU_With_One_Enum(
            amdsmi_get_gpu_memory_total=amdsmi.amdsmi_get_gpu_memory_total,
            memory_type=self.common.memory_types,
        )
        return

    def test_get_gpu_memory_usage(self):
        self.common.print_func_name("")
        self.common.Test_Per_GPU_With_One_Enum(
            amdsmi_get_gpu_memory_usage=amdsmi.amdsmi_get_gpu_memory_usage,
            memory_type=self.common.memory_types,
        )
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

    def test_get_gpu_od_volt_curve_regions(self):
        self.common.print_func_name("")

        # TODO num_region = 10
        num_region = 10

        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_od_volt_curve_regions=amdsmi.amdsmi_get_gpu_od_volt_curve_regions,
            num_region=num_region,
        )
        return

    def test_get_gpu_od_volt_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_od_volt_info=amdsmi.amdsmi_get_gpu_od_volt_info)
        return

    def test_get_gpu_overdrive_level(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_overdrive_level=amdsmi.amdsmi_get_gpu_overdrive_level
        )
        return

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

    def test_get_gpu_perf_level(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_perf_level=amdsmi.amdsmi_get_gpu_perf_level)
        return

    def test_get_gpu_pm_metrics_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_pm_metrics_info=amdsmi.amdsmi_get_gpu_pm_metrics_info
        )
        return

    def test_get_gpu_power_profile_presets(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_power_profile_presets=amdsmi.amdsmi_get_gpu_power_profile_presets,
            index=0,
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

    def test_get_gpu_ras_block_features_enabled(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_ras_block_features_enabled=amdsmi.amdsmi_get_gpu_ras_block_features_enabled
        )
        return

    def test_get_gpu_ras_feature_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_ras_feature_info=amdsmi.amdsmi_get_gpu_ras_feature_info
        )
        return

    def test_get_gpu_reg_table_info(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_gpu_reg_table_info as it fails on MI300."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_Per_GPU_With_One_Enum(
            amdsmi_get_gpu_reg_table_info=amdsmi.amdsmi_get_gpu_reg_table_info,
            reg_type=self.common.reg_types,
        )
        return

    def test_get_gpu_revision(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_revision=amdsmi.amdsmi_get_gpu_revision)
        return

    def test_get_gpu_subsystem_id(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_subsystem_id=amdsmi.amdsmi_get_gpu_subsystem_id)
        return

    def test_get_gpu_subsystem_name(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_subsystem_name=amdsmi.amdsmi_get_gpu_subsystem_name
        )
        return

    def test_get_gpu_topo_numa_affinity(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_topo_numa_affinity=amdsmi.amdsmi_get_gpu_topo_numa_affinity
        )
        return

    def test_get_gpu_total_ecc_count(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_total_ecc_count=amdsmi.amdsmi_get_gpu_total_ecc_count
        )
        return

    def test_get_gpu_vbios_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_vbios_info=amdsmi.amdsmi_get_gpu_vbios_info)
        return

    def test_get_gpu_vendor_name(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_vendor_name=amdsmi.amdsmi_get_gpu_vendor_name)
        return

    def test_get_gpu_virtualization_mode(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_virtualization_mode=amdsmi.amdsmi_get_gpu_virtualization_mode
        )
        return

    def test_get_gpu_volt_metric(self):
        self.common.print_func_name("")
        self.common.Test_Per_GPU_With_Two_Enums(
            amdsmi_get_gpu_volt_metric=amdsmi.amdsmi_get_gpu_volt_metric,
            voltage_type=self.common.voltage_types,
            voltage_metric=self.common.voltage_metrics,
        )
        return

    def test_get_gpu_vram_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_vram_info=amdsmi.amdsmi_get_gpu_vram_info)
        return

    def test_get_gpu_vram_usage(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_vram_usage=amdsmi.amdsmi_get_gpu_vram_usage)
        return

    def test_get_gpu_vram_vendor(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_vram_vendor=amdsmi.amdsmi_get_gpu_vram_vendor)
        return

    def test_get_gpu_xcd_counter(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_gpu_xcd_counter as it fails (MI350, AMDSMI_STATUS_UNEXPECTED_DATA)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(amdsmi_get_gpu_xcd_counter=amdsmi.amdsmi_get_gpu_xcd_counter)
        return

    def test_get_gpu_xgmi_link_status(self):
        self.common.print_func_name("")

        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_xgmi_link_status=amdsmi.amdsmi_get_gpu_xgmi_link_status
        )
        return

    def test_get_hsmp_metrics_table(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_hsmp_metrics_table=amdsmi.amdsmi_get_hsmp_metrics_table
        )
        return

    def test_get_hsmp_metrics_table_version(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_hsmp_metrics_table_version=amdsmi.amdsmi_get_hsmp_metrics_table_version
        )
        return

    def test_get_lib_version(self):
        self.common.print_func_name("")
        self.common.Test_API(amdsmi_get_lib_version=amdsmi.amdsmi_get_lib_version)
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
            link_type=self.common.link_types,
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

    def test_set_cpu_pcie_link_rate(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_NOT_COMPLETE:
            msg = "\tSkipping test_set_cpu_pcie_link_rate as it is not complete."
            self.common.print(msg)
            self.skipTest(msg)

        # TODO rate_ctrl = 0
        rate_ctrl = 0

        self.common.Test_API_Per_GPU(
            amdsmi_set_cpu_pcie_link_rate=amdsmi.amdsmi_set_cpu_pcie_link_rate, rate_ctrl=rate_ctrl
        )
        return

    def test_get_power_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_power_info=amdsmi.amdsmi_get_power_info)
        return

    def test_get_power_cap_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_power_cap_info=amdsmi.amdsmi_get_power_cap_info)
        return

    def test_get_processor_count_from_handles(self):
        self.common.print_func_name("")
        self.common.Test_API(
            amdsmi_get_processor_count_from_handles=amdsmi.amdsmi_get_processor_count_from_handles,
            processors=self.common.processors,
        )
        return

    # print data issues
    def test_get_processor_handles(self):
        self.common.print_func_name("")
        msg = "\t### amdsmi_get_processor_handles():"
        try:
            procs = amdsmi.amdsmi_get_processor_handles()
            self.common.print(msg, [id(addr) for addr in procs])
            self.assertGreaterEqual(len(self.common.processors), 1)
            self.assertLessEqual(len(self.common.processors), self.common.max_num_physical_devices)
            self.common.check_ret("", "", self.common.PASS)
        except amdsmi.AmdSmiLibraryException as e:
            if self.common.check_ret(msg, e, self.common.PASS):
                self.raise_exception = e
        self.common.print("")
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_processor_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_processor_info=amdsmi.amdsmi_get_processor_info)
        return

    def test_get_processor_type(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_processor_type=amdsmi.amdsmi_get_processor_type)
        return

    # data print issues
    def test_get_socket_handles(self):
        self.common.print_func_name("")
        msg = "\t### amdsmi_get_socket_handles():"
        try:
            ret = amdsmi.amdsmi_get_socket_handles()
            self.common.print(msg, [id(addr) for addr in ret])
            self.common.check_ret("", "", self.common.PASS)
        except amdsmi.AmdSmiLibraryException as e:
            if self.common.check_ret(msg, e, self.common.PASS):
                self.raise_exception = e
        self.common.print("")
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_temp_metric(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_temp_metric as it fails (Invalid param)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_Per_GPU_With_One_Enum(
            amdsmi_get_temp_metric=amdsmi.amdsmi_get_temp_metric,
            temperature_type=self.common.temperature_types,
            temperature_metric=self.common.temperature_metrics,
        )
        return

    def test_get_threads_per_core(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_threads_per_core as it fails (IO Error)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API(amdsmi_get_threads_per_core=amdsmi.amdsmi_get_threads_per_core)
        return

    def test_get_utilization_count(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_gpu_event as it fails (Data Read Error)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_Per_GPU_With_One_Enum(
            amdsmi_get_utilization_count=amdsmi.amdsmi_get_utilization_count,
            utilization_counter_type=self.common.utilization_counter_types,
        )
        return

    def test_get_violation_status(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_violation_status=amdsmi.amdsmi_get_violation_status)
        return

    def test_get_xgmi_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_xgmi_info=amdsmi.amdsmi_get_xgmi_info)
        return

    def test_gpu_counter_group_supported(self):
        self.common.print_func_name("")
        self.common.Test_Per_GPU_With_One_Enum(
            amdsmi_gpu_counter_group_supported=amdsmi.amdsmi_gpu_counter_group_supported,
            event_group=self.common.event_groups,
        )
        return

    def test_get_gpu_available_counters(self):
        self.common.print_func_name("")
        self.common.Test_Per_GPU_With_One_Enum(
            amdsmi_get_gpu_available_counters=amdsmi.amdsmi_get_gpu_available_counters,
            event_group=self.common.event_groups,
        )
        return

    def test_gpu_validate_ras_eeprom(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_gpu_validate_ras_eepromas it fails (File Error)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(
            amdsmi_gpu_validate_ras_eeprom=amdsmi.amdsmi_gpu_validate_ras_eeprom
        )
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

    def test_init(self):
        self.common.print_func_name("")
        self.common.Test_API(amdsmi_init=amdsmi.amdsmi_init)
        return

    def test_shut_down(self):
        self.common.print_func_name("")
        self.common.Test_API(amdsmi_shut_down=amdsmi.amdsmi_shut_down)
        return

    def test_is_P2P_accessible(self):
        self.common.print_func_name("")
        self.common.Test_Per_GPU_With_GPU(amdsmi_is_P2P_accessible=amdsmi.amdsmi_is_P2P_accessible)
        return

    def test_is_gpu_power_management_enabled(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_is_gpu_power_management_enabled=amdsmi.amdsmi_is_gpu_power_management_enabled
        )
        return

    def test_reset_gpu(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_reset_gpu as it fails (MI350X, Hang)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(amdsmi_reset_gpu=amdsmi.amdsmi_reset_gpu)
        return

    def test_reset_gpu_fan(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_reset_gpu_fan=amdsmi.amdsmi_reset_gpu_fan, index=0)
        return

    def test_reset_gpu_xgmi_error(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_reset_gpu_xgmi_error as it fails on MI300."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(amdsmi_reset_gpu_xgmi_error=amdsmi.amdsmi_reset_gpu_xgmi_error)
        return

    def test_set_cpu_df_pstate_range(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_NOT_COMPLETE:
            msg = "\tSkipping test_set_cpu_df_pstate_range as it is not complete."
            self.common.print(msg)
            self.skipTest(msg)

        # TODO max_pstate = 0, min_pstate = 0
        max_pstate = 0
        min_pstate = 0

        self.common.Test_API_Per_GPU(
            amdsmi_set_cpu_df_pstate_range=amdsmi.amdsmi_set_cpu_df_pstate_range,
            max_pstate=max_pstate,
            min_pstate=min_pstate,
        )
        return

    def test_set_cpu_gmi3_link_width_range(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_NOT_COMPLETE:
            msg = "\tSkipping test_set_cpu_gmi3_link_width_range as it is not complete."
            self.common.print(msg)
            self.skipTest(msg)

        # TODO min_link_width = 0, max_link_width = 0
        min_link_width = 0
        max_link_width = 0

        self.common.Test_API_Per_GPU(
            amdsmi_set_cpu_gmi3_link_width_range=amdsmi.amdsmi_set_cpu_gmi3_link_width_range,
            min_link_width=min_link_width,
            max_link_width=max_link_width,
        )
        return

    # param modes
    def test_set_cpu_pwr_efficiency_mode(self):
        self.common.print_func_name("")
        modes = [0, 1, 2]
        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            for mode in modes:
                msg = f"\t### amdsmi_set_cpu_pwr_efficiency_mode(gpu={i}, mode={mode}):"
                try:
                    amdsmi.amdsmi_set_cpu_pwr_efficiency_mode(gpu, mode)
                    self.common.print(msg, "")
                    self.common.check_ret("", "", self.common.PASS)
                except amdsmi.AmdSmiLibraryException as e:
                    if self.common.check_ret(msg, e, self.common.PASS):
                        self.raise_exception = e
                self.common.print("")
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_set_cpu_socket_lclk_dpm_level(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_NOT_COMPLETE:
            msg = "\tSkipping test_set_cpu_socket_lclk_dpm_level as it is not complete."
            self.common.print(msg)
            self.skipTest(msg)

        # TODO nbio_id = 0, min_val = 0, max_val = 0
        nbio_id = 0
        min_val = 0
        max_val = 0

        self.common.Test_API_Per_GPU(
            amdsmi_set_cpu_socket_lclk_dpm_level=amdsmi.amdsmi_set_cpu_socket_lclk_dpm_level,
            nbio_id=nbio_id,
            min_val=min_val,
            max_val=max_val,
        )
        return

    def test_set_cpu_xgmi_width(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_NOT_COMPLETE:
            msg = "\tSkipping test_set_cpu_xgmi_width as it is not complete."
            self.common.print(msg)
            self.skipTest(msg)

        # TODO min_width = 0, max_width = 0
        min_width = 0
        max_width = 0

        self.common.Test_API_Per_GPU(
            amdsmi_set_cpu_xgmi_width=amdsmi.amdsmi_set_cpu_xgmi_width,
            min_width=min_width,
            max_width=max_width,
        )
        return

    def test_set_gpu_accelerator_partition_profile(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_NOT_COMPLETE:
            msg = "\tSkipping test_set_gpu_accelerator_partition_profile as it is not complete."
            self.common.print(msg)
            self.skipTest(msg)

        # TODO profile_index = 0
        profile_index = 0

        self.common.Test_API_Per_GPU(
            amdsmi_set_gpu_accelerator_partition_profile=amdsmi.amdsmi_set_gpu_accelerator_partition_profile,
            profile_index=profile_index,
        )
        return

    # Uses clk_type_name instead of clk_type
    # Uses clk_limit_type_name instead of clk_limit_type
    def test_set_gpu_clk_limit(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_NOT_COMPLETE:
            msg = "\tSkipping test_set_gpu_clk_limit as it is not complete."
            self.common.print(msg)
            self.skipTest(msg)

        # TODO Find better way to set value
        value = 0

        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            for clk_type_name, _, clk_cond in self.common.clk_types:
                for (
                    clk_limit_type_name,
                    clk_limit_type,
                    clk_limit_cond,
                ) in self.common.clk_limit_types:
                    msg = f"\t### amdsmi_set_gpu_clk_limit(gpu={i}, clk_type={clk_type_name}, clk_limit_type={clk_limit_type_name}, value={value}):"
                    try:
                        amdsmi.amdsmi_set_gpu_clk_limit(
                            gpu, clk_type_name, clk_limit_type_name, value
                        )
                        self.common.print(msg, "")
                        self.common.check_ret("", "", self.common.PASS)
                    except amdsmi.AmdSmiLibraryException as e:
                        if not clk_cond == self.common.PASS:
                            self.common.check_ret(msg, e, clk_cond)
                            self.raise_exception = e
                        elif not clk_limit_type == self.common.PASS:
                            self.common.check_ret(msg, e, clk_limit_cond)
                            self.raise_exception = e
                        else:
                            self.common.check_ret(msg, e, self.common.PASS)
                            self.raise_exception = e
                    self.common.print("")
        if self.raise_exception:
            raise self.raise_exception
        return

    # out of order; min_clk_value, max_clk_value then clk_type
    def test_set_gpu_clk_range(self):
        self.common.print_func_name("")

        # TODO Find better way to set min_clk_value, max_clk_value
        min_clk_value = 100
        max_clk_value = 200

        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            for _, clk_type, clk_cond in self.common.clk_types:
                msg = f"\t### amdsmi_set_gpu_clk_range(gpu={i}, min_clk_value={min_clk_value}, max_clk_value={max_clk_value}, clk_type={clk_type}):"
                try:
                    amdsmi.amdsmi_set_gpu_clk_range(gpu, min_clk_value, max_clk_value, clk_type)
                    self.common.print(msg, "")
                    self.common.check_ret("", "", self.common.PASS)
                except amdsmi.AmdSmiLibraryException as e:
                    if self.common.check_ret(msg, e, clk_cond):
                        self.raise_exception = e
                self.common.print("")
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_set_gpu_memory_partition(self):
        self.common.print_func_name("")

        self.common.Test_Per_GPU_With_One_Enum(
            amdsmi_set_gpu_memory_partition=amdsmi.amdsmi_set_gpu_memory_partition,
            memory_partition_type=self.common.memory_partition_types,
        )
        return

    def test_set_gpu_memory_partition_mode(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_set_gpu_memory_partition_mode as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_Per_GPU_With_One_Enum(
            amdsmi_set_gpu_memory_partition_mode=amdsmi.amdsmi_set_gpu_memory_partition_mode,
            memory_partition_mode=self.common.memory_partition_types,
        )
        return

    # out of order freq_ind then value then clk_type
    def test_set_gpu_od_clk_info(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_NOT_COMPLETE:
            msg = "\tSkipping test_set_gpu_od_clk_info as it is not complete."
            self.common.print(msg)
            self.skipTest(msg)

        # TODO value = 0
        value = 200

        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            for freq_ind_name, freq_ind, freq_ind_cond in self.common.freq_inds:
                for clk_type_name, clk_type, clk_cond in self.common.clk_types:
                    msg = f"\t### amdsmi_set_gpu_od_clk_info(gpu={i}, freq_ind={freq_ind_name}, value={value}, clk_type={clk_type_name}):"
                    try:
                        amdsmi.amdsmi_set_gpu_od_clk_info(gpu, freq_ind, value, clk_type)
                        self.common.print(msg, "")
                        self.common.check_ret("", "", self.common.PASS)
                    except amdsmi.AmdSmiLibraryException as e:
                        if not freq_ind_cond == self.common.PASS:
                            self.common.check_ret(msg, e, freq_ind_cond)
                            self.raise_exception = e
                        elif not clk_cond == self.common.PASS:
                            self.common.check_ret(msg, e, clk_cond)
                            self.raise_exception = e
                        else:
                            self.common.check_ret(msg, e, self.common.PASS)
                            self.raise_exception = e
                    self.common.print("")
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_set_gpu_od_volt_info(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_NOT_COMPLETE:
            msg = "\tSkipping test_set_gpu_od_volt_info as it is not complete."
            self.common.print(msg)
            self.skipTest(msg)

        # TODO vpoint = 0 clk_value = 0 volt_value = 0
        vpoint = 0
        clk_value = 0
        volt_value = 0

        self.common.Test_API_Per_GPU(
            amdsmi_set_gpu_od_volt_info=amdsmi.amdsmi_set_gpu_od_volt_info,
            vpoint=vpoint,
            clk_value=clk_value,
            volt_value=volt_value,
        )
        return

    def test_set_gpu_perf_determinism_mode(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_NOT_COMPLETE:
            msg = "\tSkipping test_set_gpu_perf_determinism_mode as it is not complete."
            self.common.print(msg)
            self.skipTest(msg)

        # TODO clk_value = 0
        clk_value = 0

        self.common.Test_API_Per_GPU(
            amdsmi_set_gpu_perf_determinism_mode=amdsmi.amdsmi_set_gpu_perf_determinism_mode,
            clk_value=clk_value,
        )
        return

    # out of order: 0 then power_profile_preset_mask
    def test_set_gpu_power_profile(self):
        self.common.print_func_name("")
        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            for (
                power_profile_preset_mask_name,
                power_profile_preset_mask,
                power_profile_preset_masks_cond,
            ) in self.common.power_profile_preset_masks:
                msg = f"\t### amdsmi_set_gpu_power_profile(gpu={i}, power_profile_preset_mask={power_profile_preset_mask_name}):"
                try:
                    amdsmi.amdsmi_set_gpu_power_profile(gpu, 0, power_profile_preset_mask)
                    self.common.print(msg, "")
                    self.common.check_ret("", "", self.common.PASS)
                except amdsmi.AmdSmiLibraryException as e:
                    if self.common.check_ret(msg, e, power_profile_preset_masks_cond):
                        self.raise_exception = e
                self.common.print("")
        if self.raise_exception:
            raise self.raise_exception
        return

    # pisolates
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
    def test_status_code_to_string(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_status_code_to_string as it fails (Unhashable type)."
            self.common.print(msg)
            self.skipTest(msg)

        for error_num, _ in self.common.error_map.items():
            msg = f"\t### amdsmi_status_code_to_string(error_num={error_num}):"
            try:
                ret = amdsmi.amdsmi_status_code_to_string(ctypes.c_uint32(int(error_num, 0)))
                self.common.print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
            self.common.print("")
        if self.raise_exception:
            raise self.raise_exception
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

    def test_get_gpu_busy_percent(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_gpu_busy_percent=amdsmi.amdsmi_get_gpu_busy_percent)
        return

    def test_cpu_socket_boostlimit(self):
        self.common.print_func_name("")

        try:
            cpu_processors = amdsmi.amdsmi_get_cpusocket_handles()
        except amdsmi.AmdSmiLibraryException:
            cpu_processors = []
        if not cpu_processors:
            msg = "\tNo CPU processors found; skipping CPU-specific test"
            self.common.print(msg)
            self.skipTest(msg)

        # TODO boost_limit = 0
        boost_limit = 0
        for i, cpu in enumerate(cpu_processors):
            msg = f"cpu({i}):"
            msg1 = f"{msg} boost_limit({boost_limit}):"
            try:
                amdsmi.amdsmi_set_cpu_socket_boostlimit(cpu, boost_limit)
                self.common.print(msg1, "")
            except amdsmi.AmdSmiLibraryException as e:
                if self.common.check_ret(msg1, e, self.common.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return


if __name__ == "__main__":
    # Detect if ran without sudo or root privileges
    if os.geteuid() != 0:
        print(
            "Warning: Some tests may require elevated privileges (sudo/root) to run completely.\n",
            file=sys.stderr,
        )
        print("Please relaunch with elevated privileges.\n", file=sys.stderr)
        sys.exit(1)

    verbose = common.VERBOSITY_NORMAL
    # Parse verbosity from command line (updates the module-level default).
    # -v/-vv/--verbose all select VERBOSITY_VERBOSE; -q/--quiet selects QUIET.
    if "-q" in sys.argv or "--quiet" in sys.argv:
        verbose = common.VERBOSITY_QUIET
    elif any(a in ("-v", "-vv", "--verbose") for a in sys.argv):
        verbose = common.VERBOSITY_VERBOSE

    # If no -k or --keyword argument is given, print all available tests.
    # Do this before the -h check so the test list appears above unittest's help output.
    if not ("-k" in sys.argv or "--keyword" in sys.argv):
        if verbose > common.VERBOSITY_QUIET:
            common.print_tests(__name__)

    # Skip legend/title/"Running" preamble when the user just wants help text.
    if "-h" in sys.argv or "--help" in sys.argv:
        unittest.main()

    # Only show the dot-character legend when not in verbose mode; in verbose
    # mode each test prints its own result line so the dot legend is irrelevant.
    if verbose < common.VERBOSITY_VERBOSE:
        common.print_legend()

    if verbose > common.VERBOSITY_QUIET:
        print(f"AMD SMI Unit Tests\n")
        print("Running tests...\n")

    # WARNING: Future developers! Please read. :)
    # Avoid per-test ASIC skipping because:
    # 1) Masks API bugs — we should verify the API handles unsupported cases correctly, not skip past them.
    # 2) Unknown behavior — we don't know what the API actually does in unsupported configurations if we never run it.
    # 3) Tests may be wrong — skipped tests are never validated and can silently rot.
    # 4) Hides driver/firmware gaps — a missing implementation looks the same as "not supported"/etc...
    # 5) False coverage — a suite that skips isn't really passing, it's just not running.
    # 6) Skips become permanent — they rarely get revisited and turn into long-term technical debt.
    #
    # Preferred approach: Run the test. If the API returns an "unsupported" result, assert that response explicitly
    # rather than skipping.

    # ---------------------------------------------------------------------------
    # Skip approaches to AVOID in tests
    #
    # Approach                        | Example                                         | Problem
    # --------------------------------|-------------------------------------------------|------------------------------------------
    # Unconditional TODO skip         | if self.common.TODO_SKIP_FAIL: skipTest(...)    | Never runs; API behavior stays unknown
    # GFX filter / target version     | if gfx in GFX_FILTER: skipTest(...)             | Explicit but still hides API behavior
    # Feature flag skip               | if not gpu_supports_feature: skipTest(...)      | Doesn't verify API returns correct error
    # Exception swallow               | except Exception: pass                          | Hides failures silently; worse than skip
    # Broad except + skip             | except Exception: skipTest(...)                 | Skips on *any* error, including test bugs
    # Commented-out assertions        | # self.assertEqual(...)                         | Test always passes; nothing is verified
    #
    # Preferred approach:
    #   Run the test on all ASICs. If the feature is unsupported, assert the API
    #   returns the expected error code rather than skipping.
    #
    #   try:
    #       result = amdsmi.amdsmi_get_some_feature(processor)
    #       self.assertIsNotNone(result)
    #   except amdsmi.AmdSmiLibraryException as e:
    #       self.assertEqual(e.get_error_code(), amdsmi.AmdSmiStatus.AMDSMI_STATUS_NOT_SUPPORTED)
    # ---------------------------------------------------------------------------

    runner = unittest.TextTestRunner(
        stream=sys.stderr, verbosity=common.make_runner_verbosity(verbose)
    )
    common.expand_glob_k_arg(globals())
    unittest.main(testRunner=runner)
