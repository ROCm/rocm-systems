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

import ctypes
import inspect
import json
import os
import sys
import unittest


amdsmi_path = os.environ.get("AMDSMI_PATH", "/opt/rocm/share/amd_smi")
if not os.path.exists(amdsmi_path):
    raise FileNotFoundError(f"AMDSMI_PATH '{amdsmi_path}' does not exist. Please set the correct path in your environment.")
sys.path.append(amdsmi_path)

try:
    import amdsmi
except ImportError as exc:
    raise ImportError(f'Could not import {amdsmi_path}') from exc

not_supported_error_codes = \
[
    ( '2', 'AMDSMI_STATUS_NOT_SUPPORTED'),
    ( '3', 'AMDSMI_STATUS_NOT_YET_IMPLEMENTED'),
    ('49', 'AMDSMI_STATUS_NO_HSMP_MSG_SUP')
]

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

verbose=1
if '-q' in sys.argv or '--quiet' in sys.argv:
    verbose=0
elif '-v' in sys.argv or '--verbose' in sys.argv:
    verbose=2
has_info_printed = False


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

    def test_parse_bdf(self):
        # go through all bdfs
        expectations = self.valid_bdfs.copy()
        expectations.update(self.invalid_bdfs)
        for bdf in expectations:
            expected = expectations[bdf]
            result = amdsmi.amdsmi_interface._parse_bdf(bdf)
            self.assertEqual(result, expected,
                             "Expected {} for bdf {}, but got {}".format(
                                 expected, bdf, result))

    @classmethod
    def _convert_bdf_to_long(cls, bdf):
        if len(bdf) == 12:
            return bdf
        if len(bdf) == 7:
            return "0000:" + bdf
        return None

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
            self.assertEqual(result, expected,
                             "Expected {} for bdf {}, but got {}".format(
                                 expected, bdf_string, result))

    def test_check_res(self):
        # expect retry error to raise SmiRetryException
        with self.assertRaises(amdsmi.AmdSmiRetryException) as retry_test:
            amdsmi.amdsmi_interface._check_res(
                (lambda: amdsmi.amdsmi_wrapper.AMDSMI_STATUS_RETRY)())
        # except retry error to have AMDSMI_STATUS_RETRY error code
        self.assertEqual(retry_test.exception.get_error_code(),
                         amdsmi.amdsmi_wrapper.AMDSMI_STATUS_RETRY)

        # expect timeout error to raise SmiTimeoutException
        with self.assertRaises(amdsmi.AmdSmiTimeoutException) as timeout_test:
            amdsmi.amdsmi_interface._check_res(
                (lambda: amdsmi.amdsmi_wrapper.AMDSMI_STATUS_TIMEOUT)())
        # except timeout error to have AMDSMI_STATUS_RETRY error code
        self.assertEqual(timeout_test.exception.get_error_code(),
                         amdsmi.amdsmi_wrapper.AMDSMI_STATUS_TIMEOUT)

        # expect invalid args error to raise AmdSmiLibraryException
        with self.assertRaises(amdsmi.AmdSmiLibraryException) as inval_test:
            amdsmi.amdsmi_interface._check_res(
                (lambda: amdsmi.amdsmi_wrapper.AMDSMI_STATUS_INVAL)())
        # expect invalid args error to have AMDSMI_STATUS_INVAL error code
        self.assertEqual(inval_test.exception.get_error_code(),
                         amdsmi.amdsmi_wrapper.AMDSMI_STATUS_INVAL)

class TestAmdSmiPython(unittest.TestCase):

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

    PASS = 'AMDSMI_STATUS_SUCCESS'
    FAIL = 'AMDSMI_STATUS_INVAL'
    max_num_physical_devices = amdsmi.amdsmi_interface.AMDSMI_MAX_NUM_XCP * amdsmi.amdsmi_interface.AMDSMI_MAX_DEVICES

    # Tests marked wtih either of these flags will be skipped
    # and need to be implemented later.
    TODO_SKIP_FAIL = True
    TODO_SKIP_NOT_COMPLETE = True

    status_types = \
    [
        ('SUCCESS', amdsmi.AmdSmiStatus.SUCCESS, PASS),
        ('INVAL', amdsmi.AmdSmiStatus.INVAL, PASS),
        ('NOT_SUPPORTED', amdsmi.AmdSmiStatus.NOT_SUPPORTED, PASS),
        ('NOT_YET_IMPLEMENTED', amdsmi.AmdSmiStatus.NOT_YET_IMPLEMENTED, PASS),
        ('FAIL_LOAD_MODULE', amdsmi.AmdSmiStatus.FAIL_LOAD_MODULE, PASS),
        ('FAIL_LOAD_SYMBOL', amdsmi.AmdSmiStatus.FAIL_LOAD_SYMBOL, PASS),
        ('DRM_ERROR', amdsmi.AmdSmiStatus.DRM_ERROR, PASS),
        ('API_FAILED', amdsmi.AmdSmiStatus.API_FAILED, PASS),
        ('TIMEOUT', amdsmi.AmdSmiStatus.TIMEOUT, PASS),
        ('RETRY', amdsmi.AmdSmiStatus.RETRY, PASS),
        ('NO_PERM', amdsmi.AmdSmiStatus.NO_PERM, PASS),
        ('INTERRUPT', amdsmi.AmdSmiStatus.INTERRUPT, PASS),
        ('IO', amdsmi.AmdSmiStatus.IO, PASS),
        ('ADDRESS_FAULT', amdsmi.AmdSmiStatus.ADDRESS_FAULT, PASS),
        ('FILE_ERROR', amdsmi.AmdSmiStatus.FILE_ERROR, PASS),
        ('OUT_OF_RESOURCES', amdsmi.AmdSmiStatus.OUT_OF_RESOURCES, PASS),
        ('INTERNAL_EXCEPTION', amdsmi.AmdSmiStatus.INTERNAL_EXCEPTION, PASS),
        ('INPUT_OUT_OF_BOUNDS', amdsmi.AmdSmiStatus.INPUT_OUT_OF_BOUNDS, PASS),
        ('INIT_ERROR', amdsmi.AmdSmiStatus.INIT_ERROR, PASS),
        ('REFCOUNT_OVERFLOW', amdsmi.AmdSmiStatus.REFCOUNT_OVERFLOW, PASS),
        ('DIRECTORY_NOT_FOUND', amdsmi.AmdSmiStatus.DIRECTORY_NOT_FOUND, PASS),
        ('BUSY', amdsmi.AmdSmiStatus.BUSY, PASS),
        ('NOT_FOUND', amdsmi.AmdSmiStatus.NOT_FOUND, PASS),
        ('NOT_INIT', amdsmi.AmdSmiStatus.NOT_INIT, PASS),
        ('NO_SLOT', amdsmi.AmdSmiStatus.NO_SLOT, PASS),
        ('DRIVER_NOT_LOADED', amdsmi.AmdSmiStatus.DRIVER_NOT_LOADED, PASS),
        ('MORE_DATA', amdsmi.AmdSmiStatus.MORE_DATA, PASS),
        ('NO_DATA', amdsmi.AmdSmiStatus.NO_DATA, PASS),
        ('INSUFFICIENT_SIZE', amdsmi.AmdSmiStatus.INSUFFICIENT_SIZE, PASS),
        ('UNEXPECTED_SIZE', amdsmi.AmdSmiStatus.UNEXPECTED_SIZE, PASS),
        ('UNEXPECTED_DATA', amdsmi.AmdSmiStatus.UNEXPECTED_DATA, PASS),
        ('NON_AMD_CPU', amdsmi.AmdSmiStatus.NON_AMD_CPU, PASS),
        ('NO_ENERGY_DRV', amdsmi.AmdSmiStatus.NO_ENERGY_DRV, PASS),
        ('NO_MSR_DRV', amdsmi.AmdSmiStatus.NO_MSR_DRV, PASS),
        ('NO_HSMP_DRV', amdsmi.AmdSmiStatus.NO_HSMP_DRV, PASS),
        ('NO_HSMP_SUP', amdsmi.AmdSmiStatus.NO_HSMP_SUP, PASS),
        ('NO_HSMP_MSG_SUP', amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP, PASS),
        ('HSMP_TIMEOUT', amdsmi.AmdSmiStatus.HSMP_TIMEOUT, PASS),
        ('NO_DRV', amdsmi.AmdSmiStatus.NO_DRV, PASS),
        ('FILE_NOT_FOUND', amdsmi.AmdSmiStatus.FILE_NOT_FOUND, PASS),
        ('ARG_PTR_NULL', amdsmi.AmdSmiStatus.ARG_PTR_NULL, PASS),
        ('AMDGPU_RESTART_ERR', amdsmi.AmdSmiStatus.AMDGPU_RESTART_ERR, PASS),
        ('SETTING_UNAVAILABLE', amdsmi.AmdSmiStatus.SETTING_UNAVAILABLE, PASS),
        ('CORRUPTED_EEPROM', amdsmi.AmdSmiStatus.CORRUPTED_EEPROM, PASS),
        ('MAP_ERROR', amdsmi.AmdSmiStatus.MAP_ERROR, PASS),
        ('UNKNOWN_ERROR', amdsmi.AmdSmiStatus.UNKNOWN_ERROR, PASS)
    ]

    clk_types = \
    [
        ('SYS', amdsmi.AmdSmiClkType.SYS, PASS),
        ('GFX', amdsmi.AmdSmiClkType.GFX, PASS),
        ('DF', amdsmi.AmdSmiClkType.DF, PASS),
        ('DCEF', amdsmi.AmdSmiClkType.DCEF, [PASS, FAIL]),
        ('SOC', amdsmi.AmdSmiClkType.SOC, PASS),
        ('MEM', amdsmi.AmdSmiClkType.MEM, PASS),
        ('PCIE', amdsmi.AmdSmiClkType.PCIE, [PASS, FAIL]),
        ('VCLK0', amdsmi.AmdSmiClkType.VCLK0, PASS),
        ('VCLK1', amdsmi.AmdSmiClkType.VCLK1, PASS),
        ('DCLK0', amdsmi.AmdSmiClkType.DCLK0, PASS),
        ('DCLK1', amdsmi.AmdSmiClkType.DCLK1, PASS)
    ]

    clk_limit_types = \
    [
        ('MIN', amdsmi.AmdSmiClkLimitType.MIN, PASS),
        ('MAX', amdsmi.AmdSmiClkLimitType.MAX, PASS)
    ]

    io_bw_encodings = \
    [
        ('AGG_BW0', amdsmi.amdsmi_wrapper.AGG_BW0, PASS),
        ('RD_BW0', amdsmi.amdsmi_wrapper.RD_BW0, PASS),
        ('WR_BW0', amdsmi.amdsmi_wrapper.WR_BW0, PASS)
    ]

    event_groups = \
    [
        ('XGMI', amdsmi.AmdSmiEventGroup.XGMI, PASS),
        ('XGMI_DATA_OUT', amdsmi.AmdSmiEventGroup.XGMI_DATA_OUT, PASS),
        ('GRP_INVALID', amdsmi.AmdSmiEventGroup.GRP_INVALID, FAIL)
    ]

    gpu_blocks = \
    [
        ('INVALID', amdsmi.AmdSmiGpuBlock.INVALID, FAIL),
        ('UMC', amdsmi.AmdSmiGpuBlock.UMC, PASS),
        ('SDMA', amdsmi.AmdSmiGpuBlock.SDMA, PASS),
        ('GFX', amdsmi.AmdSmiGpuBlock.GFX, PASS),
        ('MMHUB', amdsmi.AmdSmiGpuBlock.MMHUB, PASS),
        ('ATHUB', amdsmi.AmdSmiGpuBlock.ATHUB, PASS),
        ('PCIE_BIF', amdsmi.AmdSmiGpuBlock.PCIE_BIF, PASS),
        ('HDP', amdsmi.AmdSmiGpuBlock.HDP, PASS),
        ('XGMI_WAFL', amdsmi.AmdSmiGpuBlock.XGMI_WAFL, PASS),
        ('DF', amdsmi.AmdSmiGpuBlock.DF, PASS),
        ('SMN', amdsmi.AmdSmiGpuBlock.SMN, PASS),
        ('SEM', amdsmi.AmdSmiGpuBlock.SEM, PASS),
        ('MP0', amdsmi.AmdSmiGpuBlock.MP0, PASS),
        ('MP1', amdsmi.AmdSmiGpuBlock.MP1, PASS),
        ('FUSE', amdsmi.AmdSmiGpuBlock.FUSE, PASS),
        ('MCA', amdsmi.AmdSmiGpuBlock.MCA, PASS),
        ('VCN', amdsmi.AmdSmiGpuBlock.VCN, PASS),
        ('JPEG', amdsmi.AmdSmiGpuBlock.JPEG, PASS),
        ('IH', amdsmi.AmdSmiGpuBlock.IH, PASS),
        ('MPIO', amdsmi.AmdSmiGpuBlock.MPIO, PASS),
        ('RESERVED', amdsmi.AmdSmiGpuBlock.RESERVED, FAIL)
    ]

    memory_types = \
    [
        ('VRAM', amdsmi.AmdSmiMemoryType.VRAM, PASS),
        ('VIS_VRAM', amdsmi.AmdSmiMemoryType.VIS_VRAM, PASS),
        ('GTT', amdsmi.AmdSmiMemoryType.GTT, PASS)
    ]

    processor_types = \
    [
        ('UNKNOWN', amdsmi.AmdSmiProcessorType.UNKNOWN, FAIL),
        ('AMD_GPU', amdsmi.AmdSmiProcessorType.AMD_GPU, PASS),
        ('AMD_CPU', amdsmi.AmdSmiProcessorType.AMD_CPU, PASS),
        ('NON_AMD_GPU', amdsmi.AmdSmiProcessorType.NON_AMD_GPU, PASS),
        ('NON_AMD_CPU', amdsmi.AmdSmiProcessorType.NON_AMD_CPU, PASS),
        ('AMD_CPU_CORE', amdsmi.AmdSmiProcessorType.AMD_CPU_CORE, PASS),
        ('AMD_APU', amdsmi.AmdSmiProcessorType.AMD_APU, PASS)
    ]

    reg_types = \
    [
        ('XGMI', amdsmi.AmdSmiRegType.XGMI, PASS),
        ('WAFL', amdsmi.AmdSmiRegType.WAFL, PASS),
        ('PCIE', amdsmi.AmdSmiRegType.PCIE, PASS),
        ('USR', amdsmi.AmdSmiRegType.USR, PASS),
        ('USR1', amdsmi.AmdSmiRegType.USR1, PASS)
    ]

    voltage_metrics = \
    [
        ('CURRENT', amdsmi.AmdSmiVoltageMetric.CURRENT, PASS),
        ('MAX', amdsmi.AmdSmiVoltageMetric.MAX, PASS),
        ('MIN_CRIT', amdsmi.AmdSmiVoltageMetric.MIN_CRIT, PASS),
        ('MIN', amdsmi.AmdSmiVoltageMetric.MIN, PASS),
        ('MAX_CRIT', amdsmi.AmdSmiVoltageMetric.MAX_CRIT, PASS),
        ('AVERAGE', amdsmi.AmdSmiVoltageMetric.AVERAGE, PASS),
        ('LOWEST', amdsmi.AmdSmiVoltageMetric.LOWEST, PASS),
        ('HIGHEST', amdsmi.AmdSmiVoltageMetric.HIGHEST, PASS)
    ]

    voltage_types = \
    [
        ('VDDGFX', amdsmi.AmdSmiVoltageType.VDDGFX, PASS),
        ('INVALID', amdsmi.AmdSmiVoltageType.INVALID, FAIL)
    ]

    link_types = \
    [
        ('AMDSMI_LINK_TYPE_INTERNAL', amdsmi.AmdSmiLinkType.AMDSMI_LINK_TYPE_INTERNAL, PASS),
        ('AMDSMI_LINK_TYPE_XGMI', amdsmi.AmdSmiLinkType.AMDSMI_LINK_TYPE_XGMI, PASS),
        ('AMDSMI_LINK_TYPE_PCIE', amdsmi.AmdSmiLinkType.AMDSMI_LINK_TYPE_PCIE, PASS),
        ('AMDSMI_LINK_TYPE_NOT_APPLICABLE', amdsmi.AmdSmiLinkType.AMDSMI_LINK_TYPE_NOT_APPLICABLE, FAIL),
        ('AMDSMI_LINK_TYPE_UNKNOWN', amdsmi.AmdSmiLinkType.AMDSMI_LINK_TYPE_UNKNOWN, FAIL)
    ]

    temperature_types = \
    [
        ('EDGE', amdsmi.AmdSmiTemperatureType.EDGE, PASS),
        ('HOTSPOT', amdsmi.AmdSmiTemperatureType.HOTSPOT, PASS),
        ('JUNCTION', amdsmi.AmdSmiTemperatureType.JUNCTION, PASS),
        ('VRAM', amdsmi.AmdSmiTemperatureType.VRAM, PASS),
        ('HBM_0', amdsmi.AmdSmiTemperatureType.HBM_0, PASS),
        ('HBM_1', amdsmi.AmdSmiTemperatureType.HBM_1, PASS),
        ('HBM_2', amdsmi.AmdSmiTemperatureType.HBM_2, PASS),
        ('HBM_3', amdsmi.AmdSmiTemperatureType.HBM_3, PASS),
        ('PLX', amdsmi.AmdSmiTemperatureType.PLX, PASS)
    ]

    temperature_metrics = \
    [
        ('CURRENT', amdsmi.AmdSmiTemperatureMetric.CURRENT, PASS),
        ('MAX', amdsmi.AmdSmiTemperatureMetric.MAX, PASS),
        ('MIN', amdsmi.AmdSmiTemperatureMetric.MIN, PASS),
        ('MAX_HYST', amdsmi.AmdSmiTemperatureMetric.MAX_HYST, PASS),
        ('MIN_HYST', amdsmi.AmdSmiTemperatureMetric.MIN_HYST, PASS),
        ('CRITICAL', amdsmi.AmdSmiTemperatureMetric.CRITICAL, PASS),
        ('CRITICAL_HYST', amdsmi.AmdSmiTemperatureMetric.CRITICAL_HYST, PASS),
        ('EMERGENCY', amdsmi.AmdSmiTemperatureMetric.EMERGENCY, PASS),
        ('EMERGENCY_HYST', amdsmi.AmdSmiTemperatureMetric.EMERGENCY_HYST, PASS),
        ('CRIT_MIN', amdsmi.AmdSmiTemperatureMetric.CRIT_MIN, PASS),
        ('CRIT_MIN_HYST', amdsmi.AmdSmiTemperatureMetric.CRIT_MIN_HYST, PASS),
        ('OFFSET', amdsmi.AmdSmiTemperatureMetric.OFFSET, PASS),
        ('LOWEST', amdsmi.AmdSmiTemperatureMetric.LOWEST, PASS),
        ('HIGHEST', amdsmi.AmdSmiTemperatureMetric.HIGHEST, PASS)
    ]

    utilization_counter_types = \
    [
        ('COARSE_GRAIN_GFX_ACTIVITY', amdsmi.AmdSmiUtilizationCounterType.COARSE_GRAIN_GFX_ACTIVITY, PASS),
        ('COARSE_GRAIN_MEM_ACTIVITY', amdsmi.AmdSmiUtilizationCounterType.COARSE_GRAIN_MEM_ACTIVITY, PASS),
        ('COARSE_DECODER_ACTIVITY', amdsmi.AmdSmiUtilizationCounterType.COARSE_DECODER_ACTIVITY, PASS),
        ('FINE_GRAIN_GFX_ACTIVITY', amdsmi.AmdSmiUtilizationCounterType.FINE_GRAIN_GFX_ACTIVITY, PASS),
        ('FINE_GRAIN_MEM_ACTIVITY', amdsmi.AmdSmiUtilizationCounterType.FINE_GRAIN_MEM_ACTIVITY, PASS),
        ('FINE_DECODER_ACTIVITY', amdsmi.AmdSmiUtilizationCounterType.FINE_DECODER_ACTIVITY, PASS),
        ('UTILIZATION_COUNTER_FIRST', amdsmi.AmdSmiUtilizationCounterType.UTILIZATION_COUNTER_FIRST, PASS),
        ('UTILIZATION_COUNTER_LAST', amdsmi.AmdSmiUtilizationCounterType.UTILIZATION_COUNTER_LAST, PASS)
    ]

    event_types = \
    [
        ('XGMI_0_NOP_TX', amdsmi.AmdSmiEventType.XGMI_0_NOP_TX, PASS),
        ('XGMI_0_REQUEST_TX', amdsmi.AmdSmiEventType.XGMI_0_REQUEST_TX, PASS),
        ('XGMI_0_RESPONSE_TX', amdsmi.AmdSmiEventType.XGMI_0_RESPONSE_TX, PASS),
        ('XGMI_0_BEATS_TX', amdsmi.AmdSmiEventType.XGMI_0_BEATS_TX, PASS),
        ('XGMI_1_NOP_TX', amdsmi.AmdSmiEventType.XGMI_1_NOP_TX, PASS),
        ('XGMI_1_REQUEST_TX', amdsmi.AmdSmiEventType.XGMI_1_REQUEST_TX, PASS),
        ('XGMI_1_RESPONSE_TX', amdsmi.AmdSmiEventType.XGMI_1_RESPONSE_TX, PASS),
        ('XGMI_1_BEATS_TX', amdsmi.AmdSmiEventType.XGMI_1_BEATS_TX, PASS),
        ('XGMI_DATA_OUT_0', amdsmi.AmdSmiEventType.XGMI_DATA_OUT_0, PASS),
        ('XGMI_DATA_OUT_1', amdsmi.AmdSmiEventType.XGMI_DATA_OUT_1, PASS),
        ('XGMI_DATA_OUT_2', amdsmi.AmdSmiEventType.XGMI_DATA_OUT_2, PASS),
        ('XGMI_DATA_OUT_3', amdsmi.AmdSmiEventType.XGMI_DATA_OUT_3, PASS),
        ('XGMI_DATA_OUT_4', amdsmi.AmdSmiEventType.XGMI_DATA_OUT_4, PASS),
        ('XGMI_DATA_OUT_5', amdsmi.AmdSmiEventType.XGMI_DATA_OUT_5, PASS)
    ]

    counter_commands = \
    [
        ('CMD_START', amdsmi.AmdSmiCounterCommand.CMD_START, PASS),
        ('CMD_STOP', amdsmi.AmdSmiCounterCommand.CMD_STOP, PASS)
    ]

    compute_partition_types = \
    [
        ('SPX', amdsmi.AmdSmiComputePartitionType.SPX, PASS),
        ('DPX', amdsmi.AmdSmiComputePartitionType.DPX, PASS),
        ('TPX', amdsmi.AmdSmiComputePartitionType.TPX, PASS),
        ('QPX', amdsmi.AmdSmiComputePartitionType.QPX, PASS),
        ('CPX', amdsmi.AmdSmiComputePartitionType.CPX, PASS),
        ('INVALID', amdsmi.AmdSmiComputePartitionType.INVALID, FAIL)
    ]

    memory_partition_types = \
    [
        ('NPS1', amdsmi.AmdSmiMemoryPartitionType.NPS1, PASS),
        ('NPS2', amdsmi.AmdSmiMemoryPartitionType.NPS2, PASS),
        ('NPS4', amdsmi.AmdSmiMemoryPartitionType.NPS4, PASS),
        ('NPS8', amdsmi.AmdSmiMemoryPartitionType.NPS8, PASS),
        ('UNKNOWN', amdsmi.AmdSmiMemoryPartitionType.UNKNOWN, FAIL)
    ]

    freq_inds = \
    [
        ('MIN', amdsmi.AmdSmiFreqInd.MIN, PASS),
        ('MAX', amdsmi.AmdSmiFreqInd.MAX, PASS),
        ('INVALID', amdsmi.AmdSmiFreqInd.INVALID, FAIL)
    ]

    dev_perf_levels = \
    [
        ('AUTO', amdsmi.AmdSmiDevPerfLevel.AUTO, PASS),
        ('LOW', amdsmi.AmdSmiDevPerfLevel.LOW, PASS),
        ('HIGH', amdsmi.AmdSmiDevPerfLevel.HIGH, PASS),
        ('MANUAL', amdsmi.AmdSmiDevPerfLevel.MANUAL, PASS),
        ('STABLE_STD', amdsmi.AmdSmiDevPerfLevel.STABLE_STD, PASS),
        ('STABLE_PEAK', amdsmi.AmdSmiDevPerfLevel.STABLE_PEAK, PASS),
        ('STABLE_MIN_MCLK', amdsmi.AmdSmiDevPerfLevel.STABLE_MIN_MCLK, PASS),
        ('STABLE_MIN_SCLK', amdsmi.AmdSmiDevPerfLevel.STABLE_MIN_SCLK, PASS),
        ('DETERMINISM', amdsmi.AmdSmiDevPerfLevel.DETERMINISM, PASS),
        ('UNKNOWN', amdsmi.AmdSmiDevPerfLevel.UNKNOWN, FAIL)
    ]

    power_profile_preset_masks = \
    [
        ('CUSTOM_MASK', amdsmi.AmdSmiPowerProfilePresetMasks.CUSTOM_MASK, PASS),
        ('VIDEO_MASK', amdsmi.AmdSmiPowerProfilePresetMasks.VIDEO_MASK, PASS),
        ('POWER_SAVING_MASK', amdsmi.AmdSmiPowerProfilePresetMasks.POWER_SAVING_MASK, PASS),
        ('COMPUTE_MASK', amdsmi.AmdSmiPowerProfilePresetMasks.COMPUTE_MASK, PASS),
        ('VR_MASK', amdsmi.AmdSmiPowerProfilePresetMasks.VR_MASK, PASS),
        ('THREE_D_FULL_SCR_MASK', amdsmi.AmdSmiPowerProfilePresetMasks.THREE_D_FULL_SCR_MASK, PASS),
        ('BOOTUP_DEFAULT', amdsmi.AmdSmiPowerProfilePresetMasks.BOOTUP_DEFAULT, PASS)
    ]

    def _print(self, msg, data=None):
        if verbose == 2:
            if data is None:
                print(msg, flush=True)
            elif any(data in value for value in not_supported_error_codes):
                print(f'{msg} {data}', flush=True)
            else:
                if isinstance(data, str) and data in error_map.values():
                    print(msg, end='')
                else:
                    print(msg)
                if isinstance(data, dict) or isinstance(data, list):
                    print(json.dumps(data, sort_keys=False, indent=4), flush=True)
                else:
                    print(data)
        return

    def _print_func_name(self, msg):
        if verbose == 2:
            stk = inspect.stack()
            if stk[1].function == '_callSetUp':
                return
            print(msg, flush=True)
            print(f'## {stk[1].function}()', flush=True)
        return

    def get_error_code(self, e):
        error_code = e.get_error_code()
        return error_map[error_code]

    def _check_ret(self, msg, _e, expected_code=None, printit=True):
        error_code_int = int(_e.get_error_code())
        error_code = str(error_code_int)
        if error_code in error_map:
            error_code_name = error_map[error_code]
        else:
            error_code_name = 'UNKNOWN_ERROR'

        # Check for when there are multiple passing conditions
        if isinstance(expected_code, list):
            for ec in expected_code:
                rc = self._check_ret(msg, _e, ec, False)  # Do not print msg, otherwise multiple msgs printed
                if not rc:
                    rc = self._check_ret(msg, _e, ec) # Call check again so msg is printed
                    return rc

            # No expected results found
            print(f'{msg}\nTest FAILED with expected results {expected_code} but received {error_code_name}', flush=True)
            return True

        # Check for single passing condition
        if any(error_code in value for value in not_supported_error_codes):
            if verbose == 2 and printit:
                print(f'{msg}\nTest SKIPPED with result {error_code_name}', flush=True)
        elif error_code_name == expected_code:
            if verbose == 2 and printit:
                print(f'{msg}\nTest PASSED with expected result {expected_code}', flush=True)
        else:
            if verbose == 2 and printit:
                print(f'{msg}\nTest FAILED with expected result {expected_code} but received {error_code_name}', flush=True)
            return True
        return False

    def setUp(self):
        # Called before each test by unittest framework
        self.raise_exception = None
        amdsmi.amdsmi_init()
        self.processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(self.processors), 1)
        self.assertLessEqual(len(self.processors), self.max_num_physical_devices)
        return

    def tearDown(self):
        # Called after each test by unittest framework
        amdsmi.amdsmi_shut_down()
        return

    def test_clean_gpu_local_data(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                amdsmi.amdsmi_clean_gpu_local_data(gpu)
                self._print(msg, '')
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_cpu_apb_disable(self):
        self._print_func_name('')
        pstate = 0
        for i, gpu in enumerate(self.processors):
            msg = f'### amdsmi_cpu_apb_disable(gpu={i}, pstate={pstate}):'
            try:
                amdsmi.amdsmi_cpu_apb_disable(gpu, pstate)
                self._print(msg, '')
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_cpu_apb_enable(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                amdsmi.amdsmi_cpu_apb_enable(gpu)
                self._print(msg, '')
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_first_online_core_on_cpu_socket(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_first_online_core_on_cpu_socket as it fails (IO Error).")
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_first_online_core_on_cpu_socket(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_clk_freq(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_clk_freq as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA).")
        for i, gpu in enumerate(self.processors):
            for clk_type_name, clk_type, clk_cond in self.clk_types:
                msg = f'gpu({i}): Clock Type({clk_type_name}):'
                try:
                    ret = amdsmi.amdsmi_get_clk_freq(gpu, clk_type)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, clk_cond):
                        self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_clock_info(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_clock_info as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA).")
        for i, gpu in enumerate(self.processors):
            for clk_type_name, clk_type, clk_cond in self.clk_types:
                msg = f'### test amdsmi_get_clock_info(gpu={i}, Clock Type={clk_type_name})'
                try:
                    ret = amdsmi.amdsmi_get_clock_info(gpu, clk_type)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, clk_cond):
                        self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_cpu_cclk_limit(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_cpu_cclk_limit(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return


    def test_get_cpu_core_current_freq_limit(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_cpu_core_current_freq_limit(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_cpu_core_energy(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_cpu_core_energy as it fails (IO Error).")
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_cpu_core_energy(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_cpu_current_io_bandwidth(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            for encoding_name, encoding, encoding_cond in self.io_bw_encodings:
                msg = f'gpu({i}): encodeing({encoding_name}):'
                try:
                    ret = amdsmi.amdsmi_get_cpu_current_io_bandwidth(gpu, encoding, encoding_name)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, encoding_cond):
                        self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_cpu_ddr_bw(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_cpu_ddr_bw(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_cpu_dimm_power_consumption(self):
        self._print_func_name('')
        # TODO Find better way to get dimm_addr
        dimm_addr = 0
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_cpu_dimm_power_consumption(gpu, dimm_addr)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_cpu_dimm_temp_range_and_refresh_rate(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_cpu_dimm_temp_range_and_refresh_rate as it fails.")
        # TODO Find better way to get dimm_addr
        dimm_addr = 0
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_cpu_dimm_temp_range_and_refresh_rate(gpu, dimm_addr)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_cpu_dimm_thermal_sensor(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_cpu_dimm_thermal_sensor as it fails.")
        # TODO Find better way to get dimm_addr
        dimm_addr = 0
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_cpu_dimm_thermal_sensor(gpu, dimm_addr)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_cpu_family(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_cpu_family as it fails (IO Error).")
        msg = ''
        try:
            ret = amdsmi.amdsmi_get_cpu_family()
            self._print(msg, ret)
        except amdsmi.AmdSmiLibraryException as e:
            if self._check_ret(msg, e, self.PASS):
                self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_cpu_fclk_mclk(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_cpu_fclk_mclk(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_cpu_handles(self):
        self._print_func_name('')
        msg = ''
        try:
            ret = amdsmi.amdsmi_get_cpu_handles()
            self._print(msg, ret)
        except amdsmi.AmdSmiLibraryException as e:
            if self._check_ret(msg, e, self.PASS):
                self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_cpu_hsmp_driver_version(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_cpu_hsmp_driver_version as it fails (IO Error).")
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_cpu_hsmp_driver_version(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_cpu_hsmp_proto_ver(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_cpu_hsmp_proto_ver as it fails (IO Error).")
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_cpu_hsmp_proto_ver(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_cpu_model(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_cpu_model as it fails (IO Error).")
        msg = ''
        try:
            ret = amdsmi.amdsmi_get_cpu_model()
            self._print(msg, ret)
        except amdsmi.AmdSmiLibraryException as e:
            if self._check_ret(msg, e, self.PASS):
                self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_cpu_prochot_status(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_cpu_prochot_status(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_cpu_pwr_svi_telemetry_all_rails(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_cpu_pwr_svi_telemetry_all_rails(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_cpu_smu_fw_version(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_cpu_smu_fw_version(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_cpu_socket_c0_residency(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_cpu_socket_c0_residency(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_cpu_socket_current_active_freq_limit(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_cpu_socket_current_active_freq_limit(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_cpu_socket_energy(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_cpu_socket_energy as it fails (IO Error).")
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_cpu_socket_energy(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_cpu_socket_freq_range(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_cpu_socket_freq_range(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_cpu_socket_lclk_dpm_level(self):
        self._print_func_name('')
        nbio_id = 0
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}): nbio_id({nbio_id}):'
            try:
                ret = amdsmi.amdsmi_get_cpu_socket_lclk_dpm_level(gpu, nbio_id)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_cpu_socket_power(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_cpu_socket_power(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_cpu_socket_power_cap_max(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_cpu_socket_power_cap_max(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_cpu_socket_temperature(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_cpu_socket_temperature(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_energy_count(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_energy_count as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA).")
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_energy_count(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_esmi_err_msg(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_esmi_err_msg as it fails (Unknown Error).")
        for status_type_name, status_type, status_cond in self.status_types:
            msg = f'status(AMDSMI_STATUS_{status_type_name}):'
            try:
                ret = amdsmi.amdsmi_get_esmi_err_msg(status_type)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, status_cond):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_fw_info(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_fw_info(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_accelerator_partition_profile(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_accelerator_partition_profile(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_accelerator_partition_profile_config(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_accelerator_partition_profile_config(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_activity(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_gpu_activity as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA).")
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_activity(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_asic_info(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'### test amdsmi_get_gpu_asic_info(gpu={i})'
            try:
                ret = amdsmi.amdsmi_get_gpu_asic_info(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_bad_page_info(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_bad_page_info(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_bad_page_threshold(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_bad_page_threshold(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_bdf_id(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_bdf_id(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_board_info(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'### test amdsmi_get_gpu_board_info(gpu={i})'
            try:
                ret = amdsmi.amdsmi_get_gpu_board_info(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_cache_info(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_cache_info(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_compute_partition(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_compute_partition(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_compute_process_gpus(self):
        self._print_func_name('')
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_get_gpu_compute_process_gpus as it is not complete (Inval Error).")
        # TODO Find better way to get pid
        pid = 0
        msg = f'pid({pid}):'
        try:
            ret = amdsmi.amdsmi_get_gpu_compute_process_gpus(pid)
            self._print(msg, ret)
        except amdsmi.AmdSmiLibraryException as e:
            if self._check_ret(msg, e, self.PASS):
                self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_compute_process_info(self):
        self._print_func_name('')
        msg = ''
        try:
            ret = amdsmi.amdsmi_get_gpu_compute_process_info()
            self._print(msg, ret)
        except amdsmi.AmdSmiLibraryException as e:
            if self._check_ret(msg, e, self.PASS):
                self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_compute_process_info_by_pid(self):
        self._print_func_name('')
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_get_gpu_compute_process_info_by_pid as it not complete (Device not found).")
        # TODO Find better way to get pid
        pid = 0
        msg = f'pid({pid}):'
        try:
            ret = amdsmi.amdsmi_get_gpu_compute_process_info_by_pid(pid)
            self._print(msg, ret)
        except amdsmi.AmdSmiLibraryException as e:
            if self._check_ret(msg, e, self.PASS):
                self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_device_bdf(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_device_bdf(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_device_uuid(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_device_uuid(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_driver_info(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_driver_info(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_ecc_count(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            for gpu_block_name, gpu_block, gpu_block_cond in self.gpu_blocks:
                msg = f'gpu({i}): gpu_block({gpu_block_name})'
                try:
                    ret = amdsmi.amdsmi_get_gpu_ecc_count(gpu, gpu_block)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, gpu_block_cond):
                        self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_ecc_enabled(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_ecc_enabled(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_ecc_status(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_gpu_ecc_status as it fails.")
        for i, gpu in enumerate(self.processors):
            for gpu_block_name, gpu_block, gpu_block_cond in self.gpu_blocks:
                msg = f'gpu({i}): gpu_block({gpu_block_name})'
                try:
                    ret = amdsmi.amdsmi_get_gpu_ecc_status(gpu, gpu_block)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, gpu_block_cond):
                        self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_enumeration_info(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_enumeration_info(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_fan_rpms(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_fan_rpms(gpu, 0)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_id(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_id(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_kfd_info(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_kfd_info(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_mem_overdrive_level(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_mem_overdrive_level(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_memory_partition(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_memory_partition(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_memory_partition_config(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_gpu_memory_partition_config as it fails on MI300.")
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_memory_partition_config(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_memory_reserved_pages(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_memory_reserved_pages(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_memory_total(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            for memory_type_name, memory_type, memory_type_cond in self.memory_types:
                msg = f'gpu({i}): memory_type({memory_type_name})'
                try:
                    ret = amdsmi.amdsmi_get_gpu_memory_total(gpu, memory_type)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, memory_type_cond):
                        self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_memory_usage(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            for memory_type_name, memory_type, memory_type_cond in self.memory_types:
                msg = f'gpu({i}): memory_type({memory_type_name})'
                try:
                    ret = amdsmi.amdsmi_get_gpu_memory_usage(gpu, memory_type)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, memory_type_cond):
                        self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_metrics_header_info(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_metrics_header_info(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_metrics_info(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_metrics_info(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_partition_metrics_info(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_partition_metrics_info(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception

    def test_get_gpu_od_volt_curve_regions(self):
        self._print_func_name('')
        num_region = 10
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}): num_region({num_region}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_od_volt_curve_regions(gpu, num_region)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_od_volt_info(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_od_volt_info(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_overdrive_level(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_overdrive_level(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_pci_bandwidth(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_gpu_pci_bandwidth as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA).")
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_pci_bandwidth(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_pci_replay_counter(self):
        self._print_func_name('')
        # TODO Check test_get_gpu_pci_replay_counter
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_pci_replay_counter(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_pci_throughput(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_pci_throughput(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_perf_level(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_perf_level(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_pm_metrics_info(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_pm_metrics_info(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_power_profile_presets(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_power_profile_presets(gpu, 0)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_process_isolation(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_process_isolation(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_process_list(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_process_list(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_ras_block_features_enabled(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_ras_block_features_enabled(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_ras_feature_info(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_ras_feature_info(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_reg_table_info(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            for reg_type_name, reg_type, reg_type_cond in self.reg_types:
                msg = f'gpu({i}): reg_type({reg_type_name}):'
                try:
                    ret = amdsmi.amdsmi_get_gpu_reg_table_info(gpu, reg_type)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, reg_type_cond):
                        self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_revision(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_revision(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_subsystem_id(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_subsystem_id(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_subsystem_name(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_subsystem_name(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_topo_numa_affinity(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_topo_numa_affinity(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_total_ecc_count(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_total_ecc_count(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_vbios_info(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_vbios_info(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_vendor_name(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_vendor_name(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_virtualization_mode(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_virtualization_mode(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_volt_metric(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            for voltage_type_name, voltage_type, voltage_type_cond in self.voltage_types:
                for voltage_metric_name, voltage_metric, voltage_metric_cond in self.voltage_metrics:
                    msg = f'gpu({i}): voltage_type({voltage_type_name}) voltage_metric({voltage_metric_name}):'
                    try:
                        ret = amdsmi.amdsmi_get_gpu_volt_metric(gpu, voltage_type, voltage_metric)
                        self._print(msg, ret)
                    except amdsmi.AmdSmiLibraryException as e:
                        if not voltage_type_cond == self.PASS:
                            if self._check_ret(msg, e, voltage_type_cond):
                                self.raise_exception = e
                        elif not voltage_metric_cond == self.PASS:
                            if self._check_ret(msg, e, voltage_metric_cond):
                                self.raise_exception = e
                        else:
                            if self._check_ret(msg, e, self.PASS):
                                self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_vram_info(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_vram_info(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_vram_usage(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_vram_usage(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_vram_vendor(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_vram_vendor(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_xcd_counter(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_gpu_xcd_counter as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA).")
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_xcd_counter(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_xgmi_link_status(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_gpu_xgmi_link_status as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA).")
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_xgmi_link_status(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_hsmp_metrics_table(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_hsmp_metrics_table(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_hsmp_metrics_table_version(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_hsmp_metrics_table_version(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_lib_version(self):
        self._print_func_name('')
        msg = ''
        try:
            ret = amdsmi.amdsmi_get_lib_version()
            self._print(msg, ret)
        except amdsmi.AmdSmiLibraryException as e:
            if self._check_ret(msg, e, self.PASS):
                self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_link_metrics(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_link_metrics as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA).")
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_link_metrics(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_link_topology_nearest(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            for link_type_name, link_type, link_type_cond in self.link_types:
                msg = f'gpu({i}): link_type({link_type_name})'
                try:
                    ret = amdsmi.amdsmi_get_link_topology_nearest(gpu, link_type)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, link_type_cond):
                        self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_minmax_bandwidth_between_processors(self):
        self._print_func_name('')
        for i, gpu_i in enumerate(self.processors):
            for j, gpu_j in enumerate(self.processors):
                msg = f'gpu({i},{j}):'
                try:
                    ret = amdsmi.amdsmi_get_minmax_bandwidth_between_processors(gpu_i, gpu_j)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if i == j:
                        if self._check_ret(msg, e, self.FAIL):
                            self.raise_exception = e
                    else:
                        if self._check_ret(msg, e, self.PASS):
                            self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_pcie_info(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_pcie_info as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA).")
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_pcie_info(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_set_cpu_pcie_link_rate(self):
        self._print_func_name('')
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_cpu_pcie_link_rate as it is not complete.")
        # TODO rate_ctrl = 0
        rate_ctrl = 0
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}): rate_ctrl({rate_ctrl}):'
            try:
                ret = amdsmi.amdsmi_set_cpu_pcie_link_rate(gpu, rate_ctrl)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_power_info(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_power_info(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_processor_count_from_handles(self):
        self._print_func_name('')
        msg = 'gpu():'
        try:
            ret = amdsmi.amdsmi_get_processor_count_from_handles(self.processors)
            self._print(msg, ret)
        except amdsmi.AmdSmiLibraryException as e:
            if self._check_ret(msg, e, self.PASS):
                self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_processor_handle_from_bdf(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                bdf = amdsmi.amdsmi_get_gpu_device_bdf(gpu)
                ret = amdsmi.amdsmi_get_processor_handle_from_bdf(bdf)
                if gpu.value != ret.value:
                    msg += f'{msg}Expected: {gpu.value}, Received: {ret.value}'
                    self.raise_exception = amdsmi.AmdSmiLibraryException(amdsmi.amdsmi_wrapper.AMDSMI_STATUS_INVAL)
                else:
                    self._print(msg)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_processor_handles(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            self._print(f'    {i:2d} processor_handles: {gpu}')
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_processor_handles_by_type(self):
        self._print_func_name('')
        socket_ids = amdsmi.amdsmi_get_socket_handles()
        for index, socket_id in enumerate(socket_ids):
            for processor_name, processor_type, processor_cond in self.processor_types:
                msg = f'socket({index}): processor_type({processor_name}):'
                try:
                    ret = amdsmi.amdsmi_get_processor_handles_by_type(socket_id, processor_type)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, processor_cond):
                        self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_processor_info(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_processor_info(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_processor_type(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_processor_type(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_socket_handles(self):
        self._print_func_name('')
        msg = ''
        try:
            ret = amdsmi.amdsmi_get_socket_handles()
            self._print(msg, [id(addr) for addr in ret])
        except amdsmi.AmdSmiLibraryException as e:
            if self._check_ret(msg, e, self.PASS):
                self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_socket_info(self):
        self._print_func_name('')
        sockets = amdsmi.amdsmi_get_socket_handles()
        self.assertGreaterEqual(len(sockets), 1)
        self.assertLessEqual(len(sockets), self.max_num_physical_devices)
        for i, socket in enumerate(sockets):
            msg = f'socket({i}):'
            try:
                ret = amdsmi.amdsmi_get_socket_info(socket)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_temp_metric(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_temp_metric as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA).")
        for i, gpu in enumerate(self.processors):
            for temperature_type_name, temperature_type, temperature_type_cond in self.temperature_types:
                for temperature_metric_name, temperature_metric, temperature_metric_cond in self.temperature_metrics:
                    msg = f'gpu({i}): temperature_type=({temperature_type_name}) temperature_metric({temperature_metric_name}):'
                    try:
                        ret = amdsmi.amdsmi_get_temp_metric(gpu, temperature_type, temperature_metric)
                        self._print(msg, ret)
                    except amdsmi.AmdSmiLibraryException as e:
                        if not temperature_type_cond == self.PASS:
                            if self._check_ret(msg, e, temperature_type_cond):
                                self.raise_exception = e
                        elif not temperature_metric_cond == self.PASS:
                            if self._check_ret(msg, e, temperature_metric_cond):
                                self.raise_exception = e
                        else:
                            if self._check_ret(msg, e, self.PASS):
                                self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_threads_per_core(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_threads_per_core as it fails (IO Error).")
        # TODO threads_per_core
        msg = 'threads_per_core:'
        try:
            ret = amdsmi.amdsmi_get_threads_per_core()
            self._print(msg, ret)
        except amdsmi.AmdSmiLibraryException as e:
            if self._check_ret(msg, e, self.PASS):
                self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_utilization_count(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_utilization_count as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA).")
        for i, gpu in enumerate(self.processors):
            for utilization_counter_type_name, utilization_counter_type, utilization_counter_type_cond in self.utilization_counter_types:
                msg = f'gpu({i}): utilization_counter_type({utilization_counter_type_name}):'
                try:
                    ret = amdsmi.amdsmi_get_utilization_count(gpu, [utilization_counter_type])
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, utilization_counter_type_cond):
                        self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_violation_status(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_violation_status as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA).")
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_violation_status(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_xgmi_info(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_xgmi_info(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_gpu_counter(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_gpu_counter as it fails (Error opening file).")
        for i, gpu in enumerate(self.processors):
            for event_type_name, event_type, event_type_cond in self.event_types:
                msg = f'gpu({i}): event_type({event_type_name}):'

                # Create
                msg1 = f'{msg} Create counter:'
                try:
                    event_handle = amdsmi.amdsmi_gpu_create_counter(gpu, event_type)
                    self._print(msg1, event_handle)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg1, e, event_type_cond):
                        self.raise_exception = e
                    # if any exception occurs, skip the rest of the loop
                    continue

                # Read
                msg1 = f'{msg} Read counter:'
                try:
                    amdsmi.amdsmi_gpu_read_counter(event_handle)
                    self._print(msg1)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg1, e, event_type_cond):
                        self.raise_exception = e

                # Control
                for counter_command_name, counter_command, counter_commands_cond in self.counter_commands:
                    msg1 = f'{msg} event_type({event_type_name}): counter_command({counter_command_name}):'
                    try:
                        amdsmi.amdsmi_gpu_control_counter(event_handle, counter_command)
                        self._print(msg1, '')
                    except amdsmi.AmdSmiLibraryException as e:
                        if self._check_ret(msg1, e, counter_commands_cond):
                            self.raise_exception = e

                # Destroy
                msg1 = f'{msg} Destroy counter:'
                try:
                    amdsmi.amdsmi_gpu_destroy_counter(event_handle)
                    self._print(msg1, '')
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg1, e, event_type_cond):
                        self.raise_exception = e

        if self.raise_exception:
            raise self.raise_exception
        return

    def test_gpu_counter_group_supported(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            for event_group_name, event_group, event_group_cond in self.event_groups:
                msg = f'gpu({i}): event_group({event_group_name}):'
                try:
                    amdsmi.amdsmi_gpu_counter_group_supported(gpu, event_group)
                    self._print(msg, '')
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, event_group_cond):
                        self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_available_counters(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            for event_group_name, event_group_type, event_group_cond in self.event_groups:
                msg = f'gpu({i}): event_group({event_group_name})'
                try:
                    ret = amdsmi.amdsmi_get_gpu_available_counters(gpu, event_group_type)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, event_group_cond):
                        self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_gpu_validate_ras_eeprom(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_gpu_validate_ras_eepromas it fails (File Error).")
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                amdsmi.amdsmi_gpu_validate_ras_eeprom(gpu)
                self._print(msg, '')
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_gpu_xgmi_error_status(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_gpu_xgmi_error_status as it fails on MI300.")
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_gpu_xgmi_error_status(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_init(self):
        self._print_func_name('')
        msg = ''
        try:
            amdsmi.amdsmi_init()
            self._print(msg, '')
        except amdsmi.AmdSmiLibraryException as e:
            if self._check_ret(msg, e, self.PASS):
                self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_shut_down(self):
        self._print_func_name('')
        msg = ''
        try:
            amdsmi.amdsmi_shut_down()
            self._print(msg, '')
        except amdsmi.AmdSmiLibraryException as e:
            if self._check_ret(msg, e, self.PASS):
                self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_is_P2P_accessible(self):
        self._print_func_name('')
        for i, gpu_i in enumerate(self.processors):
            for j, gpu_j in enumerate(self.processors):
                msg = f'gpu({i},{j}):'
                try:
                    ret = amdsmi.amdsmi_is_P2P_accessible(gpu_i, gpu_j)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, self.PASS):
                        self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_gpu_event(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_gpu_event as it fails (File Error).")
        mask = 1 << (amdsmi.AmdSmiEvtNotificationType.GPU_PRE_RESET -1) | \
               1 << (amdsmi.AmdSmiEvtNotificationType.GPU_POST_RESET -1)
        timeout_ms = 1000
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'

            # Init
            try:
                self._print(f'{msg} amdsmi_init_gpu_event_notification()')
                amdsmi.amdsmi_init_gpu_event_notification(gpu)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
                # Skip remaining tests on any exception when initializing
                continue

            # Is Enabled
            try:
                self._print(f'{msg} amdsmi_is_gpu_power_management_enabled()')
                ret = amdsmi.amdsmi_is_gpu_power_management_enabled(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e

            # Set Mask
            try:
                self._print(f'{msg} amdsmi_set_gpu_event_notification_mask()')
                amdsmi.amdsmi_set_gpu_event_notification_mask(gpu, mask)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e

            # Get
            try:
                self._print(f'{msg} amdsmi_get_gpu_event_notification()')
                ret = amdsmi.amdsmi_get_gpu_event_notification(timeout_ms)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e

            # Stop
            try:
                self._print(f'{msg} amdsmi_stop_gpu_event_notification()')
                amdsmi.amdsmi_stop_gpu_event_notification(gpu)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e

        if self.raise_exception:
            raise self.raise_exception
        return

    def test_reset_gpu(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_reset_gpu as it fails (MI350X, Hang).")
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                amdsmi.amdsmi_reset_gpu(gpu)
                self._print(msg, '')
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_reset_gpu_fan(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                amdsmi.amdsmi_reset_gpu_fan(gpu, 0)
                self._print(msg, '')
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_reset_gpu_xgmi_error(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_reset_gpu_xgmi_error as it fails on MI300.")
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                amdsmi.amdsmi_reset_gpu_xgmi_error(gpu)
                self._print(msg, '')
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_set_clk_freq(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_set_clk_freq as it fails (Perm failure).")
        for i, gpu in enumerate(self.processors):
            for clk_type_name, clk_type, clk_cond in self.clk_types:
                msg = f'gpu({i}): Get Clock Info({clk_type_name}):'
                try:
                    ret = amdsmi.amdsmi_get_clk_freq(gpu, clk_type)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, clk_cond):
                        self.raise_exception = e
                    continue
                clk_freq_info = ret
                current = clk_freq_info['current']
                num_supported = clk_freq_info['num_supported']
                frequency = clk_freq_info['frequency']
                if num_supported == 0:
                    self._print(f'No supported frequencies for clk_type={clk_type_name}')
                    continue
                found_error = False
                for index in range(0, num_supported):
                    msg = f'\tgpu({i}):'
                    try:
                        freq_bitmask = frequency[index]
                        msg = f'{msg} Set clk_type({clk_type_name}): freq_bitmask({freq_bitmask}):'
                        amdsmi.amdsmi_set_clk_freq(gpu, clk_type_name, freq_bitmask)
                        self._print(msg, '')
                    except amdsmi.AmdSmiLibraryException as e:
                        found_error = True
                        if self._check_ret(msg, e, clk_cond):
                            self.raise_exception = e
                if not found_error:
                    amdsmi.amdsmi_set_clk_freq(gpu, clk_type_name, frequency[current])
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_cpu_core_boostlimit(self):
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            msg1 = f'{msg} amdsmi_get_cpu_core_boostlimit():'
            try:
                boost_limit = amdsmi.amdsmi_get_cpu_core_boostlimit(gpu)
                msg1 = f'{msg1} boost_limit={boost_limit}'
                self._print(msg1, '')
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg1, e, self.PASS):
                    self.raise_exception = e
                continue

            msg1 = f'{msg} amdsmi_set_cpu_core_boostlimit():'
            try:
                amdsmi.amdsmi_set_cpu_core_boostlimit(gpu, boost_limit)
                self._print(msg1, '')
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg1, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_set_cpu_df_pstate_range(self):
        self._print_func_name('')
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_cpu_df_pstate_range as it is not complete.")
        # TODO max_pstate = 0, min_pstate = 0
        max_pstate = 0
        min_pstate = 0
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}): max_pstate({max_pstate}) min_pstate({min_pstate}):'
            try:
                amdsmi.amdsmi_set_cpu_df_pstate_range(gpu, max_pstate, min_pstate)
                self._print(msg, '')
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_set_cpu_gmi3_link_width_range(self):
        self._print_func_name('')
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_cpu_gmi3_link_width_range as it is not complete.")
        # TODO min_link_width = 0, max_link_width = 0
        min_link_width = 0
        max_link_width = 0
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}): min_link_width({min_link_width}) max_link_width({max_link_width}):'
            try:
                amdsmi.amdsmi_set_cpu_gmi3_link_width_range(gpu, min_link_width, max_link_width)
                self._print(msg, '')
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_set_cpu_pwr_efficiency_mode(self):
        self._print_func_name('')
        modes = [0, 1, 2]
        for i, gpu in enumerate(self.processors):
            for mode in modes:
                msg = f'gpu({i}): mode({mode}):'
                try:
                    amdsmi.amdsmi_set_cpu_pwr_efficiency_mode(gpu, mode)
                    self._print(msg, '')
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, self.PASS):
                        self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_cpu_socket_boostlimit(self):
        self._print_func_name('')
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_cpu_socket_boostlimit as it is not complete.")
        # TODO boost_limit = 0
        boost_limit = 0
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            msg1 = f'{msg} boost_limit({boost_limit}):'
            try:
                amdsmi.amdsmi_set_cpu_socket_boostlimit(gpu, boost_limit)
                self._print(msg1, '')
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg1, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_set_cpu_socket_lclk_dpm_level(self):
        self._print_func_name('')
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_cpu_socket_lclk_dpm_level as it is not complete.")
        # TODO nbio_id = 0, min_val = 0, max_val = 0
        nbio_id = 0
        min_val = 0
        max_val = 0
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}): nbio_id({nbio_id}) min_val({min_val}) max_val({max_val}):'
            try:
                amdsmi.amdsmi_set_cpu_socket_lclk_dpm_level(gpu, nbio_id, min_val, max_val)
                self._print(msg, '')
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_cpu_socket_power_cap(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            msg1 = f'{msg} amdsmi_get_cpu_socket_power_cap():'
            try:
                power_cap = amdsmi.amdsmi_get_cpu_socket_power_cap(gpu)
                self._print(msg1, '')
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg1, e, self.PASS):
                    self.raise_exception = e
                continue

            msg1 = f'{msg} power_cap={power_cap}'
            try:
                amdsmi.amdsmi_set_cpu_socket_power_cap(gpu, power_cap)
                self._print(msg1, '')
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg1, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_set_cpu_xgmi_width(self):
        self._print_func_name('')
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_cpu_xgmi_width as it is not complete.")
        # TODO min_width = 0, max_width = 0
        min_width = 0
        max_width = 0
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}): min_width({min_width} max_width({max_width}): )'
            try:
                amdsmi.amdsmi_set_cpu_xgmi_width(gpu, min_width , max_width)
                self._print(msg, '')
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_set_gpu_accelerator_partition_profile(self):
        self._print_func_name('')
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_gpu_accelerator_partition_profile as it is not complete.")
        # TODO profile_index = 0
        profile_index = 0
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}): profile_index({profile_index}):'
            try:
                amdsmi.amdsmi_set_gpu_accelerator_partition_profile(gpu, profile_index)
                self._print(msg, '')
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_set_gpu_clk_limit(self):
        self._print_func_name('')
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_gpu_clk_limit as it is not complete.")
        # TODO Find better way to set value
        value = 0
        for i, gpu in enumerate(self.processors):
            for clk_type_name, clk_type, clk_cond in self.clk_types:
                for clk_limit_type_name, clk_limit_type, clk_limit_cond in self.clk_limit_types:
                    msg = f'gpu({i}): value({value}) clock_type=({clk_type_name}) clock_limit_type({clk_limit_type_name}):'
                    try:
                        amdsmi.amdsmi_set_gpu_clk_limit(gpu, clk_type_name, clk_limit_type_name, value)
                        self._print(msg, '')
                    except amdsmi.AmdSmiLibraryException as e:
                        if not clk_cond == self.PASS:
                            self._check_ret(msg, e, clk_cond)
                            self.raise_exception = e
                        elif not clk_limit_type == self.PASS:
                            self._check_ret(msg, e, clk_limit_type)
                            self.raise_exception = e
                        else:
                            self._check_ret(msg, e, self.PASS)
                            self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_set_gpu_clk_range(self):
        self._print_func_name('')
        # TODO Find better way to set min_clk_value, max_clk_value
        min_clk_value = 100
        max_clk_value = 200
        for i, gpu in enumerate(self.processors):
            for clk_type_name, clk_type, clk_cond in self.clk_types:
                msg = f'gpu({i}): min_clk_value({min_clk_value}) max_clk_value({max_clk_value}) clk_type({clk_type_name}):'
                try:
                    amdsmi.amdsmi_set_gpu_clk_range(gpu, min_clk_value, max_clk_value, clk_type)
                    self._print(msg, '')
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, clk_cond):
                        self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_set_gpu_compute_partition(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_set_gpu_compute_partition as it fails on MI300.")
        for i, gpu in enumerate(self.processors):
            default_compute_partition_type = self.compute_partition_types[0][1]
            msg = f'gpu({i}): amdsmi_get_gpu_compute_partition()'
            try:
                default_compute_partition_name = amdsmi.amdsmi_get_gpu_compute_partition(gpu)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
                continue
            for compute_partition_type_name, compute_partition_type, compute_partition_type_cond in self.compute_partition_types:
                if default_compute_partition_name == compute_partition_type_name:
                    default_compute_partition_type = compute_partition_type
                msg = f'gpu({i}): compute_partition_type({compute_partition_type_name}):'
                try:
                    amdsmi.amdsmi_set_gpu_compute_partition(gpu, compute_partition_type)
                    self._print(msg, '')
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, compute_partition_type_cond):
                        self.raise_exception = e
            msg = f'gpu({i}): amdsmi_set_gpu_compute_partition({default_compute_partition_name})'
            try:
                amdsmi.amdsmi_set_gpu_compute_partition(gpu, default_compute_partition_type)
                self._print(msg, '')
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
                continue
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_gpu_fan_speed(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            msg1 = f'{msg} amdsmi_get_gpu_fan_speed()'
            try:
                # Determine current fan speed
                fan_speed_current = amdsmi.amdsmi_get_gpu_fan_speed(gpu, 0)
                msg1 = f'{msg1} fan_speed={fan_speed_current}'
                self._print(msg1, '')
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg1, e, self.PASS):
                    self.raise_exception = e
                continue

            try:
                # Determine max fan speed
                msg1 = f'{msg} amdsmi_get_gpu_fan_speed_max()'
                fan_speed_max = amdsmi.amdsmi_get_gpu_fan_speed_max(gpu, 0)
                msg1 = f'{msg1} fan_speed_max={fan_speed_max}'
                if fan_speed_current == fan_speed_max:
                    fan_speed = int(fan_speed_max/2)
                else:
                    fan_speed = fan_speed_max
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg1, e, self.PASS):
                    self.raise_exception = e
                continue

            try:
                # Set fan speed
                msg1 = f'{msg} fan_speed({fan_speed}):'
                amdsmi.amdsmi_set_gpu_fan_speed(gpu, 0, fan_speed)
                self._print(msg1, '')
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg1, e, self.PASS):
                    self.raise_exception = e

            try:
                # Set to original fan speed
                msg1 = f'{msg} fan_speed({fan_speed_current}):'
                amdsmi.amdsmi_set_gpu_fan_speed(gpu, 0, fan_speed_current)
                self._print(msg1, '')
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg1, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_set_gpu_memory_partition(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_set_gpu_memory_partition as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA).")
        for i, gpu in enumerate(self.processors):
            for memory_partition_type_name, memory_partition_type, memory_partition_type_cond in self.memory_partition_types:
                msg = f'gpu({i}): memory_partition_type({memory_partition_type_name}):'
                try:
                    amdsmi.amdsmi_set_gpu_memory_partition(gpu, memory_partition_type)
                    self._print(msg, '')
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, memory_partition_type_cond):
                        self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_set_gpu_memory_partition_mode(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_set_gpu_memory_partition_mode as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA).")
        for i, gpu in enumerate(self.processors):
            for memory_partition_type_name, memory_partition_type, memory_partition_type_cond in self.memory_partition_types:
                msg = f'gpu({i}): memory_partition_type({memory_partition_type_name}):'
                try:
                    amdsmi.amdsmi_set_gpu_memory_partition_mode(gpu, memory_partition_type)
                    self._print(msg, '')
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, memory_partition_type_cond):
                        self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_set_gpu_od_clk_info(self):
        self._print_func_name('')
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_gpu_od_clk_info as it is not complete.")
        # TODO value = 0
        value = 200
        for i, gpu in enumerate(self.processors):
            for freq_ind_name, freq_ind, freq_ind_cond in self.freq_inds:
                for clk_type_name, clk_type, clk_cond in self.clk_types:
                    msg = f'gpu({i}): freq_ind({freq_ind_name}) value({value}) clk_type({clk_type_name}):'
                    try:
                        amdsmi.amdsmi_set_gpu_od_clk_info(gpu, freq_ind, value, clk_type)
                        self._print(msg, '')
                    except amdsmi.AmdSmiLibraryException as e:
                        if not freq_ind_cond == self.PASS:
                            self._check_ret(msg, e, freq_ind_cond)
                            self.raise_exception = e
                        elif not clk_cond == self.PASS:
                            self._check_ret(msg, e, clk_cond)
                            self.raise_exception = e
                        else:
                            self._check_ret(msg, e, self.PASS)
                            self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_set_gpu_od_volt_info(self):
        self._print_func_name('')
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_gpu_od_volt_info as it is not complete.")
        # TODO vpoint = 0 clk_value = 0 volt_value = 0
        vpoint = 0
        clk_value = 0
        volt_value = 0
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}): vpoint({vpoint}) clk_value({clk_value}) volt_value({volt_value}):'
            try:
                amdsmi.amdsmi_set_gpu_od_volt_info(gpu, vpoint, clk_value, volt_value)
                self._print(msg, '')
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_set_gpu_overdrive_level(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                # Find current overdrive value
                overdrive_value_current = amdsmi.amdsmi_get_gpu_overdrive_level(gpu)
                if overdrive_value_current != 1:
                    overdrive_value = 1
                else:
                    overdrive_value = 2

                # Set overdrive value
                msg = f'gpu({i}): overdrive_value({overdrive_value}):'
                amdsmi.amdsmi_set_gpu_overdrive_level(gpu, overdrive_value)
                self._print(msg, '')

                # Set back to original overdrive value
                amdsmi.amdsmi_set_gpu_overdrive_level(gpu, overdrive_value_current)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_set_gpu_pci_bandwidth(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_set_gpu_pci_bandwidth as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA).")
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                # Get current PCI bandwidth info
                bandwidth_info = amdsmi.amdsmi_get_gpu_pci_bandwidth(gpu)
                current_bandwidth_index = bandwidth_info['transfer_rate']['current']
                if current_bandwidth_index > 0:
                    bitmask = 1 << (current_bandwidth_index - 1)
                else:
                    bitmask = 1 << (current_bandwidth_index)

                # Set PCI bandwidth
                msg = f'gpu({i}): bitmask({bitmask}):'
                amdsmi.amdsmi_set_gpu_pci_bandwidth(gpu, bitmask)
                self._print(msg, '')

                # Set back to original PCI bandwidth
                bitmask = 1 << (current_bandwidth_index)
                amdsmi.amdsmi_set_gpu_pci_bandwidth(gpu, bitmask)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_set_gpu_perf_determinism_mode(self):
        self._print_func_name('')
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_gpu_perf_determinism_mode as it is not complete.")
        # TODO clk_value = 0
        clk_value = 0
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}): clk_value({clk_value}):'
            try:
                amdsmi.amdsmi_set_gpu_perf_determinism_mode(gpu, clk_value)
                self._print(msg, '')
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_set_gpu_perf_level(self):
        self._print_func_name('')
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_gpu_perf_level as it is not complete.")
        dev_perf_level_current = self.dev_perf_levels[0][1]
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                dev_perf_level_name_current = amdsmi.amdsmi_get_gpu_perf_level(gpu)
                items = dev_perf_level_name_current.split('_')
                dev_perf_level_name_current = items[-1]
            except amdsmi.AmdSmiLibraryException as e:
                self._print(msg, e)
                continue
            for dev_perf_level_name, dev_perf_level, dev_perf_level_cond in self.dev_perf_levels:
                msg = f'gpu({i}):'
                try:
                    if dev_perf_level_name_current == dev_perf_level_name:
                        dev_perf_level_current = dev_perf_level

                    msg = f'{msg} dev_perf_level({dev_perf_level_name}):'
                    amdsmi.amdsmi_set_gpu_perf_level(gpu, dev_perf_level)
                    self._print(msg, '')
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, dev_perf_level_cond):
                        self.raise_exception = e
            try:
                amdsmi.amdsmi_set_gpu_perf_level(gpu, dev_perf_level_current)
            except amdsmi.AmdSmiLibraryException as e:
                self._print(msg, e)
                continue
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_set_gpu_power_profile(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            for power_profile_preset_mask_name, power_profile_preset_mask, power_profile_preset_masks_cond in self.power_profile_preset_masks:
                msg = f'gpu({i}): power_profile_preset_mask({power_profile_preset_mask_name}):'
                try:
                    amdsmi.amdsmi_set_gpu_power_profile(gpu, 0, power_profile_preset_mask)
                    self._print(msg, '')
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, power_profile_preset_masks_cond):
                        self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_set_gpu_process_isolation(self):
        self._print_func_name('')
        pisolates = [1, 0]
        for i, gpu in enumerate(self.processors):
            for pisolate in pisolates:
                msg = f'gpu({i}): pisolate({pisolate})'
                try:
                    amdsmi.amdsmi_set_gpu_process_isolation(gpu, pisolate)
                    self._print(msg)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, self.PASS):
                        self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_power_cap(self):
        '''test power cap'''
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            # Get Power Cap Info
            msg = f'### test amdsmi_get_power_cap_info(gpu={i})'
            try:
                power_cap_info = amdsmi.amdsmi_get_power_cap_info(gpu)
                self._print(msg, power_cap_info)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
                # Have to be able to get info before setting
                continue

            # Set to Average Power Cap
            try:
                cap =  int((power_cap_info['max_power_cap'] + power_cap_info['min_power_cap']) / 2)
                msg = f'### test amdsmi_set_power_cap(gpu={i}, 0, cap={cap})'
                amdsmi.amdsmi_set_power_cap(gpu, 0, cap)
                self._print(msg, '')
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e

            # Restore Power Cap
            try:
                cap = power_cap_info['power_cap']
                msg = f'### test amdsmi_set_power_cap(gpu={i}, 0, cap={cap})'
                amdsmi.amdsmi_set_power_cap(gpu, 0, cap)
                self._print(msg, '')
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e

        if self.raise_exception:
            raise self.raise_exception
        return

    def test_soc_pstate(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            # Get current policy info
            msg1 = f'{msg} amdsmi_get_soc_pstate'
            try:
                policy_info = amdsmi.amdsmi_get_soc_pstate(gpu)
                self._print(msg1, '')

                num_supported = policy_info['num_supported']
                if not isinstance(num_supported, int):
                    self._print('Cannot determine num_supported={num_supported}', '')
                    continue
                policy_id_current = policy_info['current_id']
                if not isinstance(policy_id_current, int):
                    self._print('Cannot determine policy_id_current={policy_id_current}', '')
                    continue
                policy_id_orig = policy_info['policies'][policy_id_current]['policy_id']
                if not isinstance(policy_id_orig, int):
                    self._print('Cannot determine orig policy_id={policy_id_orig}', '')
                    continue

                index = 0
                if num_supported >= 2:
                    if policy_id_current != 0:
                        index = 1
                policy_id = policy_info['policies'][index]['policy_id']
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg1, e, self.PASS):
                    self.raise_exception = e
                continue

            # Set SOC Pstate policy
            msg1 = f'{msg} policy_id({policy_id}):'
            try:
                amdsmi.amdsmi_set_soc_pstate(gpu, policy_id)
                self._print(msg1, '')
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg1, e, self.PASS):
                    self.raise_exception = e
                continue

            # Set back to original policy
            msg1 = f'{msg} policy_id({policy_id_orig}):'
            try:
                amdsmi.amdsmi_set_soc_pstate(gpu, policy_id_orig)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg1, e, self.PASS):
                    self.raise_exception = e

        if self.raise_exception:
            raise self.raise_exception
        return

    def test_xgmi_plpd(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_set_xgmi_plpd as it fails on MI300.")
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'

            # Get current policy info
            msg1 = f'{msg} amdsmi_get_xgmi_plpd()'
            try:
                policy_info = amdsmi.amdsmi_get_xgmi_plpd(gpu)
                self._print(msg1, '')

                num_supported = policy_info['num_supported']
                if not isinstance(num_supported, int):
                    self._print('Cannot determine num_supported={num_supported}', '')
                    continue
                policy_id_current = policy_info['current_id']
                if not isinstance(policy_id_current, int):
                    self._print('Cannot determine policy_id_current={policy_id_current}', '')
                    continue
                policy_id_orig = policy_info['policies'][policy_id_current]['policy_id']
                if not isinstance(policy_id_orig, int):
                    self._print('Cannot determine orig policy_id={policy_id_orig}', '')
                    continue
                index = 0
                if num_supported >= 2:
                    if policy_id_current != 0:
                        index = 1
                policy_id = policy_info['policies'][index]['policy_id']
                if not isinstance(policy_id, int):
                    self._print('Cannot determine policy_id={policy_id}', '')
                    continue
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg1, e, self.PASS):
                    self.raise_exception = e
                continue

            # Set policy
            msg1 = f'{msg} policy_id({policy_id}):'
            try:
                amdsmi.amdsmi_set_xgmi_plpd(gpu, policy_id)
                self._print(msg1, '')
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg1, e, self.PASS):
                    self.raise_exception = e

            # Set back to original policy
            try:
                msg1 = f'{msg} policy_id({policy_id_orig}):'
                amdsmi.amdsmi_set_xgmi_plpd(gpu, policy_id_orig)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg1, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_status_code_to_string(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_status_code_to_string as it fails (Unhashable type).")
        for error_num, error_name in error_map.items():
            msg = f'{error_name}({error_num}):'
            try:
                ret = amdsmi.amdsmi_status_code_to_string(ctypes.c_uint32(int(error_num)))
                self._print(f'{msg} {ret}')
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_topo_get_link_type(self):
        self._print_func_name('')
        for i, gpu_i in enumerate(self.processors):
            for j, gpu_j in enumerate(self.processors):
                msg = f'gpu({i},{j}):'
                try:
                    ret = amdsmi.amdsmi_topo_get_link_type(gpu_i, gpu_j)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, self.PASS):
                        self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_topo_get_link_weight(self):
        self._print_func_name('')
        for i, gpu_i in enumerate(self.processors):
            for j, gpu_j in enumerate(self.processors):
                msg = f'gpu({i},{j}):'
                try:
                    ret = amdsmi.amdsmi_topo_get_link_weight(gpu_i, gpu_j)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, self.PASS):
                        self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_topo_get_numa_node_number(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_topo_get_numa_node_number(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_topo_get_p2p_status(self):
        self._print_func_name('')
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_topo_get_p2p_status as it fails (Inval parameters).")
        for i, gpu_i in enumerate(self.processors):
            for j, gpu_j in enumerate(self.processors):
                msg = f'gpu({i},{j}):'
                try:
                    ret = amdsmi.amdsmi_topo_get_p2p_status(gpu_i, gpu_j)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, self.PASS):
                        self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_gpu_busy_percent(self):
        self._print_func_name('')
        for i, gpu in enumerate(self.processors):
            msg = f'gpu({i}):'
            try:
                ret = amdsmi.amdsmi_get_gpu_busy_percent(gpu)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

if __name__ == '__main__':
    unittest.main()
