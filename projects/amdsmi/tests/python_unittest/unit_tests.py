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

import inspect
import json
import sys
import unittest
sys.path.append("/opt/rocm/libexec/amdsmi_cli/")

try:
    import amdsmi
except ImportError:
    raise ImportError("Could not import /opt/rocm/libexec/amdsmi_cli/amdsmi_cli.py")

not_supported_error_codes = ['2', '3', '49']
not_supported_error_code_names = ['AMDSMI_STATUS_NOT_SUPPORTED', 'AMDSMI_STATUS_NOT_YET_IMPLEMENTED', 'AMDSMI_STATUS_NO_HSMP_MSG_SUP']

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

    PASS = 'AMDSMI_STATUS_SUCCESS'
    FAIL = 'AMDSMI_STATUS_INVAL'

    # Tests marked wtih either of these flags will be skipped
    # and need to be implemented later.
    TODO_SKIP_NYI = True
    TODO_SKIP_FAIL = True
    TODO_SKIP_NOT_COMPLETE = True

    clk_types = \
    [
        ('SYS', amdsmi.AmdSmiClkType.SYS, PASS),
        ('GFX', amdsmi.AmdSmiClkType.GFX, PASS),
        ('DF', amdsmi.AmdSmiClkType.DF, PASS),
        ('DCEF', amdsmi.AmdSmiClkType.DCEF, PASS),
        ('SOC', amdsmi.AmdSmiClkType.SOC, PASS),
        ('MEM', amdsmi.AmdSmiClkType.MEM, PASS),
        ('PCIE', amdsmi.AmdSmiClkType.PCIE, PASS),
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
        ('BOOTUP_DEFAULT', amdsmi.AmdSmiPowerProfilePresetMasks.BOOTUP_DEFAULT, PASS),
        ('INVALID', amdsmi.AmdSmiPowerProfilePresetMasks.INVALID, FAIL)
    ]

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
    def _convert_bdf_to_long(clz, bdf):
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
            expected = expected.lower()
            result = amdsmi.amdsmi_interface._format_bdf(smi_bdf)
            self.assertEqual(result, expected,
                             "Expected {} for bdf {}, but got {}".format(
                                 expected, bdf_string, result))

    def test_check_res(self):
        # expect retry error to raise SmiRetryException
        with self.assertRaises(amdsmi.AmdSmiRetryException) as retry_test:
            amdsmi.amdsmi_interface._check_res(
                (lambda: amdsmi.amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_RETRY)())
        # except retry error to have AMDSMI_STATUS_RETRY error code
        self.assertEqual(retry_test.exception.get_error_code(),
                         amdsmi.amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_RETRY)
        # expect invalid args error to raise AmdSmiLibraryException
        with self.assertRaises(amdsmi.AmdSmiLibraryException) as inval_test:
            amdsmi.amdsmi_interface._check_res(
                (lambda: amdsmi.amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_INVAL)())
        # expect invalid args error to have AMDSMI_STATUS_INVAL error code
        self.assertEqual(inval_test.exception.get_error_code(),
                         amdsmi.amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_INVAL)
        # for successfull call, expect no error is given
        result = amdsmi.amdsmi_interface._check_res(
            (lambda: amdsmi.amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_SUCCESS)())
        self.assertEqual(None, result)

    @classmethod
    def _print(self, msg, data=None, cond=None):
        if verbose == 2:
            if not data:
                print(msg, flush=True)
            elif data in not_supported_error_code_names:
                print(f'{msg}{data}', flush=True)
            else:
                if isinstance(data, str) and data in error_map.values():
                    print(msg, end='')
                else:
                    print(msg)
                print(json.dumps(data, sort_keys=False, indent=4), flush=True)
        return

    @classmethod
    def _print_func_name(self, msg):
        if verbose == 2:
            stk = inspect.stack()
            if stk[2].function == '_callSetUp':
                return
            print(msg, flush=True)
            print(f'{stk[2].function}', flush=True)
        return

    @classmethod
    def get_error_code(self, e):
        error_code = e.get_error_code()
        return error_map[error_code]

    @classmethod
    def _check_ret(self, msg, _e, expected_code=None):
        error_code = str(_e.get_error_code())
        error_code_name = error_map[error_code]
        if error_code in not_supported_error_codes:
            if verbose == 2:
                print(f'{msg}\nTest SKIPPED with result {error_code_name}', flush=True)
        elif error_code_name == expected_code:
            if verbose == 2:
                print(f'{msg}\nTest PASSED with expected result {expected_code}', flush=True)
        else:
            if verbose == 2:
                print(f'{msg}\nTest FAILED with expected result {expected_code} but received {error_code_name}', flush=True)
            return True
        return False

    def setUp(self):
        self._print_func_name('')
        amdsmi.amdsmi_init()

    def tearDown(self):
        amdsmi.amdsmi_shut_down()


    def test_clean_gpu_local_data(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f"gpu({i}): "
                ret = amdsmi.amdsmi_clean_gpu_local_data(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_cpu_apb_disable(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f"gpu({i}): "
                ret = amdsmi.amdsmi_cpu_apb_disable(processors[i], 0)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_cpu_apb_enable(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f"gpu({i}): "
                ret = amdsmi.amdsmi_cpu_apb_enable(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_first_online_core_on_cpu_socket(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_first_online_core_on_cpu_socket as it fails.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f"gpu({i}): "
                ret = amdsmi.amdsmi_first_online_core_on_cpu_socket(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception

    def test_get_clk_freq(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_clock_info as it fails.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            for clk_type_name, clk_type, clk_cond in self.clk_types:
                try:
                    msg = f"gpu({i}): Clock Type({clk_type_name}): "
                    ret = amdsmi.amdsmi_get_clk_freq(processors[i], clk_type)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, clk_cond):
                        raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_clock_info(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_clock_info as it fails.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            for clk_type_name, clk_type, clk_cond in self.clk_types:
                try:
                    msg = f'gpu({i}): Clock Type({clk_type_name}): '
                    ret = amdsmi.amdsmi_get_clock_info(processors[i], clk_type)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, clk_cond):
                        raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_cpu_cclk_limit(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_cpu_cclk_limit(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_cpu_core_boostlimit(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_cpu_core_boostlimit(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e

        if raise_exception:
            raise raise_exception


    def test_get_cpu_core_current_freq_limit(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f"gpu({i}): "
                ret = amdsmi.amdsmi_get_cpu_core_current_freq_limit(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_cpu_core_energy(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_cpu_core_energy as it fails.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f"gpu({i}): "
                ret = amdsmi.amdsmi_get_cpu_core_energy(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_cpu_current_io_bandwidth(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            for encoding_name, encoding, encoding_cond in self.io_bw_encodings:
                try:
                    msg = f'gpu({i}): encodeing({encoding_name}): '
                    ret = amdsmi.amdsmi_get_cpu_current_io_bandwidth(processors[i], encoding, encoding_name)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, encoding_cond):
                        raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_cpu_ddr_bw(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_cpu_ddr_bw(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_cpu_dimm_power_consumption(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_cpu_dimm_power_consumption as it fails.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        # TODO dimm_addr = 0
        dimm_addr = 0
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_cpu_dimm_power_consumption(processors[i], dimm_addr)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_cpu_dimm_temp_range_and_refresh_rate(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_cpu_dimm_temp_range_and_refresh_rate as it fails.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        # TODO dimm_addr = 0
        dimm_addr = 0
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_cpu_dimm_temp_range_and_refresh_rate(processors[i], dimm_addr)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_cpu_dimm_thermal_sensor(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_cpu_dimm_thermal_sensor as it fails.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        # TODO dimm_addr = 0
        dimm_addr = 0
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_cpu_dimm_thermal_sensor(processors[i], dimm_addr)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_cpu_family(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_cpu_family as it fails.")
        raise_exception = None

        try:
            msg = ''
            ret = amdsmi.amdsmi_get_cpu_family()
            self._print(msg, ret)
        except amdsmi.AmdSmiLibraryException as e:
            if self._check_ret(msg, e, self.PASS):
                raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_cpu_fclk_mclk(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_cpu_fclk_mclk(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_cpu_handles(self):
        if self.TODO_SKIP_NYI:
            self.skipTest("Skipping test_get_cpu_handles as it is not implemented yet.")
        raise_exception = None

        try:
            msg = ''
            ret = amdsmi.amdsmi_get_cpu_handles(amdsmi.amdsmi_interface.AMDSMI_MAX_DEVICES)
            self._print(msg, ret)
        except amdsmi.AmdSmiLibraryException as e:
            if self._check_ret(msg, e, self.PASS):
                raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_cpu_hsmp_driver_version(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_cpu_hsmp_driver_version as it fails.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_cpu_hsmp_driver_version(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_cpu_hsmp_proto_ver(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_cpu_hsmp_proto_ver as it fails.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_cpu_hsmp_proto_ver(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_cpu_model(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_cpu_model as it fails.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        try:
            msg = ''
            ret = amdsmi.amdsmi_get_cpu_model()
            self._print(msg, ret)
        except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_cpu_prochot_status(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_cpu_prochot_status(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_cpu_pwr_svi_telemetry_all_rails(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_cpu_pwr_svi_telemetry_all_rails(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_cpu_smu_fw_version(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_cpu_smu_fw_version(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_cpu_socket_c0_residency(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_cpu_socket_c0_residency(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_cpu_socket_current_active_freq_limit(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_cpu_socket_current_active_freq_limit(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_cpu_socket_energy(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_cpu_socket_energy as it fails.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_cpu_socket_energy(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_cpu_socket_freq_range(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_cpu_socket_freq_range(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_cpu_socket_lclk_dpm_level(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        nbio_id = 0
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): nbio_id({nbio_id}): '
                ret = amdsmi.amdsmi_get_cpu_socket_lclk_dpm_level(processors[i], nbio_id)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_cpu_socket_power(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_cpu_socket_power(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_cpu_socket_power_cap(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_cpu_socket_power_cap(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_cpu_socket_power_cap_max(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_cpu_socket_power_cap_max(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_cpu_socket_temperature(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_cpu_socket_temperature(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_energy_count(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_energy_count(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_esmi_err_msg(self):
        if self.TODO_SKIP_NYI:
            self.skipTest("Skipping test_get_esmi_err_msg as it is not implemented yet.")
        raise_exception = None

        for status_num in error_map:
            try:
                msg = f'status({error_map[status_num]}): '
                ret = amdsmi.amdsmi_get_esmi_err_msg(error_map[status_num])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_fw_info(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f"gpu({i}): "
                ret = amdsmi.amdsmi_get_fw_info(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_accelerator_partition_profile(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_accelerator_partition_profile(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_accelerator_partition_profile_config(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_accelerator_partition_profile_config(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_activity(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_activity(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_asic_info(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_asic_info(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_bad_page_info(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_bad_page_info(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_bad_page_threshold(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_bad_page_threshold(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_bdf_id(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_bdf_id(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_board_info(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_board_info(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_cache_info(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = gpu_cache_infos = amdsmi.amdsmi_get_gpu_cache_info(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_compute_partition(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_compute_partition(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_compute_process_gpus(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_gpu_compute_process_gpus as it fails.")
        raise_exception = None

        # TODO pid = 0
        pid = 0
        try:
            msg = f'pid({pid}): '
            ret = gpu_compute_process_gpuss = amdsmi.amdsmi_get_gpu_compute_process_gpus(pid)
            self._print(msg, ret)
        except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_compute_process_info(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_compute_process_info()
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_compute_process_info_by_pid(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_gpu_compute_process_info_by_pid as it fails.")
        raise_exception = None

        # TODO pid = 0
        pid = 0
        try:
            msg = f'pid({pid}): '
            ret = amdsmi.amdsmi_get_gpu_compute_process_info_by_pid(pid)
            self._print(msg, ret)
        except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_device_bdf(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_device_bdf(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_device_uuid(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_device_uuid(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_driver_info(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_driver_info(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_ecc_count(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            for gpu_block_name, gpu_block, gpu_block_cond in self.gpu_blocks:
                try:
                    msg = f'gpu({i}): gpu_block({gpu_block_name}) '
                    ret = amdsmi.amdsmi_get_gpu_ecc_count(processors[i], gpu_block)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, gpu_block_cond):
                        raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_ecc_enabled(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_ecc_enabled(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_ecc_status(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_gpu_ecc_status as it fails.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            for gpu_block_name, gpu_block, gpu_block_cond in self.gpu_blocks:
                try:
                    msg = f'gpu({i}): gpu_block({gpu_block_name}) '
                    ret = amdsmi.amdsmi_get_gpu_ecc_status(processors[i], gpu_block)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, gpu_block_cond):
                        raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_enumeration_info(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_enumeration_info(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_event_notification(self):
        if self.TODO_SKIP_NYI:
            self.skipTest("Skipping test_get_gpu_event_notification as it is not implemented yet.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_event_notification(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_fan_rpms(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_fan_rpms(processors[i], 0)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_fan_speed(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_fan_speed(processors[i], 0)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_fan_speed_max(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_fan_speed_max(processors[i], 0)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_id(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_id(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_kfd_info(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_kfd_info(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_mem_overdrive_level(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_mem_overdrive_level(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_memory_partition(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_memory_partition(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_memory_partition_config(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_memory_partition_config(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_memory_reserved_pages(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_memory_reserved_pages(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_memory_total(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            for memory_type_name, memory_type, memory_type_cond in self.memory_types:
                try:
                    msg = f'gpu({i}): memory_type({memory_type_name}) '
                    ret = amdsmi.amdsmi_get_gpu_memory_total(processors[i], memory_type)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, memory_type_cond):
                        raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_memory_usage(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            for memory_type_name, memory_type, memory_type_cond in self.memory_types:
                try:
                    msg = f'gpu({i}): memory_type({memory_type_name}) '
                    ret = amdsmi.amdsmi_get_gpu_memory_usage(processors[i], memory_type)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, memory_type_cond):
                        raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_metrics_header_info(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_metrics_header_info(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_metrics_info(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_metrics_info(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_od_volt_curve_regions(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        num_region = 10
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): num_region({num_region}): '
                ret = amdsmi.amdsmi_get_gpu_od_volt_curve_regions(processors[i], num_region)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_od_volt_info(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_gpu_od_volt_info as it fails.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_od_volt_info(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_overdrive_level(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_overdrive_level(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_pci_bandwidth(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_pci_bandwidth(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_pci_replay_counter(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_pci_replay_counter(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_pci_throughput(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_pci_throughput(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_perf_level(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_perf_level(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_pm_metrics_info(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_gpu_pm_metrics_info as it fails.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_pm_metrics_info(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_power_profile_presets(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_power_profile_presets(processors[i], 0)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_process_isolation(self):
        if self.TODO_SKIP_NYI:
            self.skipTest("Skipping test_get_gpu_process_isolation as it is not implemented yet.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_process_isolation(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_process_list(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_process_list(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_ras_block_features_enabled(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_ras_block_features_enabled(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_ras_feature_info(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_ras_feature_info(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_reg_table_info(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_gpu_reg_table_info as it fails.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            for reg_type_name, reg_type, reg_type_cond in self.reg_types:
                try:
                    msg = f'gpu({i}): reg_type({reg_type_name}): '
                    ret = amdsmi.amdsmi_get_gpu_reg_table_info(processors[i], reg_type)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, reg_type_cond):
                        raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_revision(self):
        if self.TODO_SKIP_NYI:
            self.skipTest("Skipping test_get_gpu_revision as it is not implemented yet.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_revision(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_subsystem_id(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_subsystem_id(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_subsystem_name(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_subsystem_name(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_topo_numa_affinity(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_topo_numa_affinity(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_total_ecc_count(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_total_ecc_count(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_vbios_info(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_vbios_info(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_vendor_name(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_vendor_name(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_virtualization_mode(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_virtualization_mode(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_volt_metric(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            for voltage_type_name, voltage_type, voltage_type_cond in self.voltage_types:
                for voltage_metric_name, voltage_metric, voltage_metric_cond in self.voltage_metrics:
                    try:
                        msg = f'gpu({i}): voltage_type({voltage_type_name}) voltage_metric({voltage_metric_name}): '
                        ret = amdsmi.amdsmi_get_gpu_volt_metric(processors[i], voltage_type, voltage_metric)
                        self._print(msg, ret)
                    except amdsmi.AmdSmiLibraryException as e:
                        if not voltage_type_cond == self.PASS:
                            if self._check_ret(msg, e, voltage_type_cond):
                                raise_exception = e
                        elif not voltage_metric_cond == self.PASS:
                            if self._check_ret(msg, e, voltage_metric_cond):
                                raise_exception = e
                        else:
                            if self._check_ret(msg, e, self.PASS):
                                raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_vram_info(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_vram_info(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_vram_usage(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_vram_usage(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_vram_vendor(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_vram_vendor(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_xcd_counter(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_xcd_counter(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_xgmi_link_status(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_gpu_xgmi_link_status(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_hsmp_metrics_table(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_hsmp_metrics_table(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_hsmp_metrics_table_version(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_hsmp_metrics_table_version(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_lib_version(self):
        raise_exception = None

        try:
            msg = f''
            ret = amdsmi.amdsmi_get_lib_version()
            self._print(msg, ret)
        except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_link_metrics(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_link_metrics as it fails.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_link_metrics(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_link_topology_nearest(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            for link_type_name, link_type, link_type_cond in self.link_types:
                try:
                    msg = f'gpu({i}): link_type({link_type_name}) '
                    ret = amdsmi.amdsmi_get_link_topology_nearest(processors[i], link_type)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, link_type_cond):
                        raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_minmax_bandwidth_between_processors(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_minmax_bandwidth_between_processors as it fails.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            for j in range(0, len(processors)):
                try:
                    msg = f'gpu({i},{j}): '
                    ret = amdsmi.amdsmi_get_minmax_bandwidth_between_processors(processors[i], processors[j])
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, self.PASS):
                        raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_pcie_info(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_pcie_info(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_power_cap_info(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_power_cap_info(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_power_info(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = power_info = amdsmi.amdsmi_get_power_info(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_processor_count_from_handles(self):
        if self.TODO_SKIP_NYI:
            self.skipTest("Skipping test_get_processor_count_from_handles as it is not implemented yet.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_processor_count_from_handles(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_processor_handle_from_bdf(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_processor_handle_from_bdf as it fails.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        #TODO: define bdf
        bdf = 0
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_processor_handle_from_bdf(bdf)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_processor_handles(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            self._print(f'    {i:2d} processor_handles: {processors[i]}')
        if raise_exception:
            raise raise_exception



    def test_get_processor_handles_by_type(self):
        if self.TODO_SKIP_NYI:
            self.skipTest("Skipping test_get_processor_handles_by_type as it is not implemented yet.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_processor_handles_by_type(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_processor_info(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_processor_info(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_processor_type(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_processor_type(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_soc_pstate(self):
        if self.TODO_SKIP_NYI:
            self.skipTest("Skipping test_get_soc_pstate as it is not implemented yet.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_soc_pstate(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_socket_handles(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_socket_handles as it fails.")
        raise_exception = None

        try:
            msg = f''
            ret = amdsmi.amdsmi_get_socket_handles()
            self._print(msg, ret)
        except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_socket_info(self):
        raise_exception = None

        sockets = amdsmi.amdsmi_get_socket_handles()
        self.assertGreaterEqual(len(sockets), 1)
        # TODO Find maximum number of sockets
        self.assertLessEqual(len(sockets), 32)
        for i in range(0, len(sockets)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_socket_info(sockets[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_temp_metric(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            for temperature_type_name, temperature_type, temperature_type_cond in self.temperature_types:
                for temperature_metric_name, temperature_metric, temperature_metric_cond in self.temperature_metrics:
                    try:
                        msg = f'gpu({i}): temperature_type=({temperature_type_name}) temperature_metric({temperature_metric_name}): '
                        ret = amdsmi.amdsmi_get_temp_metric(processors[i], temperature_type, temperature_metric)
                        self._print(msg, ret)
                    except amdsmi.AmdSmiLibraryException as e:
                        if not temperature_type_cond == self.PASS:
                            if self._check_ret(msg, e, temperature_type_cond):
                                raise_exception = e
                        elif not temperature_metric_cond == self.PASS:
                            if self._check_ret(msg, e, temperature_metric_cond):
                                raise_exception = e
                        else:
                            if self._check_ret(msg, e, self.PASS):
                                raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_threads_per_core(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_threads_per_core as it fails.")
        raise_exception = None

        # TODO threads_per_core 
        try:
            msg = f'threads_per_core: '
            ret = amdsmi.amdsmi_get_threads_per_core()
            self._print(msg, ret)
        except amdsmi.AmdSmiLibraryException as e:
            if self._check_ret(msg, e, self.PASS):
                raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_utilization_count(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_utilization_count as it fails.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            for utilization_counter_type_name, utilization_counter_type, utilization_counter_type_cond in self.utilization_counter_types:
                try:
                    msg = f'gpu({i}): utilization_counter_type({utilization_counter_type_name}): '
                    ret = amdsmi.amdsmi_get_utilization_count(processors[i], utilization_counter_type)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, utilization_counter_type_cond):
                        raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_violation_status(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_violation_status(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_xgmi_info(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_xgmi_info(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_xgmi_plpd(self):
        if self.TODO_SKIP_NYI:
            self.skipTest("Skipping test_get_xgmi_plpd as it is not implemented yet.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_get_xgmi_plpd(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_gpu_counter_group_supported(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_gpu_counter_group_supported as it fails.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            for event_group_name, event_group, event_group_cond in self.event_groups:
                try:
                    msg = f'gpu({i}): event_group({event_group_name}): '
                    ret = amdsmi.amdsmi_gpu_counter_group_supported(processors[i], event_group)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, event_group_cond):
                        raise_exception = e
        if raise_exception:
            raise raise_exception


    if False: # TODO: create_counter, destroy_counter, read_counter, get_gpu_available_counters, gpu_control_counter
        ''' Check these:
            test_get_gpu_pci_replay_counter
            test_get_gpu_xcd_counter
            test_gpu_counter_group_supported
        '''


    def test_gpu_create_counter(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_gpu_create_counter as it fails.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            for event_type_name, event_type, event_type_cond in self.event_types:
                try:
                    msg = f'gpu({i}): event_type({event_type_name}): '
                    ret = amdsmi.amdsmi_gpu_create_counter(processors[i], event_type)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, event_type_cond):
                        raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_gpu_destroy_counter(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_gpu_destroy_counter as it fails.")
        raise_exception = None

        # TODO event_handle = 0
        event_handle = 0
        try:
            msg = f'event_handle({event_handle}): '
            ret = event_handle
            amdsmi.amdsmi_gpu_destroy_counter(event_handle)
            self._print(msg, ret)
        except amdsmi.AmdSmiLibraryException as e:
            if self._check_ret(msg, e, self.PASS):
                raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_gpu_read_counter(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_gpu_read_counter as it fails.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        # TODO event_handle = 0
        event_handle = 0
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): event_handle({event_handle}): '
                ret = amdsmi.amdsmi_gpu_read_counter(event_handle)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_get_gpu_available_counters(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_get_gpu_available_counters as it fails.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            for event_group_name, event_group_type, event_group_cond in self.event_groups:
                try:
                    msg = f'gpu({i}): event_group({event_group_name}) '
                    ret = amdsmi.amdsmi_get_gpu_available_counters(processors[i], event_group_type)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, event_group_cond):
                        raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_gpu_control_counter(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_gpu_control_counter as it fails.")
        raise_exception = None

        # TODO event_handle = 0
        event_handle = 0
        for counter_command_name, counter_command, counter_commands_cond in self.counter_commands:
            try:
                msg = f'counter_command({counter_command_name}): '
                ret = amdsmi.amdsmi_gpu_control_counter(event_handle, counter_command)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_gpu_validate_ras_eeprom(self):
        if self.TODO_SKIP_NYI:
            self.skipTest("Skipping test_gpu_validate_ras_eeprom as it is not implemented yet.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_gpu_validate_ras_eeprom(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_gpu_xgmi_error_status(self):
        self.skipTest("Skipping test_gpu_xgmi_error_status currently not a valid test - temporarily disabled.")
        # See information in xgmi_read_write.cc file, it also skips this test for all ASICs.
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_gpu_xgmi_error_status(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_init(self):
        raise_exception = None

        try:
            msg = f''
            ret = amdsmi.amdsmi_init()
            self._print(msg, ret)
        except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_shut_down(self):
        raise_exception = None

        try:
            msg = f''
            ret = amdsmi.amdsmi_shut_down()
            self._print(msg, ret)
        except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_init_gpu_event_notification(self):
        if self.TODO_SKIP_NYI:
            self.skipTest("Skipping test_init_gpu_event_notification as it is not implemented yet.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_init_gpu_event_notification(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_is_P2P_accessible(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            for j in range(0, len(processors)):
                try:
                    msg = f'gpu({i},{j}): '
                    ret = amdsmi.amdsmi_is_P2P_accessible(processors[i], processors[j])
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, self.PASS):
                        raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_is_gpu_power_management_enabled(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_is_gpu_power_management_enabled(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_reset_gpu(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_reset_gpu(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_reset_gpu_fan(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_reset_gpu_fan(processors[i], 0)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_reset_gpu_xgmi_error(self):
        self.skipTest("Skipping test_reset_gpu_xgmi_error currently not a valid test - temporarily disabled.")
        # See information in xgmi_read_write.cc file, it also skips this test for all ASICs.
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_reset_gpu_xgmi_error(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_set_clk_freq(self):
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_clk_freq as it is not complete.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        # TODO freq_bitmask = 0
        freq_bitmask = 0
        for i in range(0, len(processors)):
            for clk_type_name, clk_type, clk_cond in self.clk_types:
                try:
                    msg = f'gpu({i}): clk_type({clk_type_name}): freq_bitmask({freq_bitmask}): '
                    ret = amdsmi.amdsmi_set_clk_freq(processors[i], clk_type, freq_bitmask)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, clk_cond):
                        raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_set_cpu_core_boostlimit(self):
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_cpu_core_boostlimit as it is not complete.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        # TODO boost_limit = 0
        boost_limit = 0
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): boost_limit({boost_limit}): '
                ret = amdsmi.amdsmi_set_cpu_core_boostlimit(processors[i], boost_limit)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_set_cpu_df_pstate_range(self):
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_cpu_df_pstate_range as it is not complete.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        # TODO max_pstate = 0, min_pstate = 0
        max_pstate = 0
        min_pstate = 0
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): max_pstate({max_pstate}) min_pstate({min_pstate}): '
                ret = amdsmi.amdsmi_set_cpu_df_pstate_range(processors[i], max_pstate, min_pstate)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_set_cpu_gmi3_link_width_range(self):
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_cpu_gmi3_link_width_range as it is not complete.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        # TODO min_link_width = 0, max_link_width = 0
        min_link_width = 0
        max_link_width = 0
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): min_link_width({min_link_width}) max_link_width({max_link_width}): '
                ret = amdsmi.amdsmi_set_cpu_gmi3_link_width_range(processors[i], min_link_width, max_link_width)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_set_cpu_pcie_link_rate(self):
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_cpu_pcie_link_rate as it is not complete.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        # TODO rate_ctrl = 0
        rate_ctrl = 0
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): rate_ctrl({rate_ctrl}): '
                ret = amdsmi.amdsmi_set_cpu_pcie_link_rate(processors[i], rate_ctrl)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_set_cpu_pwr_efficiency_mode(self):
        if self.TODO_SKIP_NYI:
            self.skipTest("Skipping test_set_cpu_pwr_efficiency_mode as it is not implemented is not yet implemented.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        # TODO mode = 0
        mode = 0
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): mode({mode}): '
                ret = amdsmi.set_cpu_pwr_efficiency_mode(processors[i], mode)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_set_cpu_socket_boostlimit(self):
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_cpu_socket_boostlimit as it is not complete.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        # TODO boost_limit = 0
        boost_limit = 0
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): boost_limit({boost_limit}): '
                ret = amdsmi.amdsmi_set_cpu_socket_boostlimit(processors[i], boost_limit)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_set_cpu_socket_lclk_dpm_level(self):
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_cpu_socket_lclk_dpm_level as it is not complete.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        # TODO nbio_id = 0, min_val = 0, max_val = 0
        nbio_id = 0
        min_val = 0
        max_val = 0
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): nbio_id({nbio_id}) min_val({min_val}) max_val({max_val}): '
                ret = amdsmi.amdsmi_set_cpu_socket_lclk_dpm_level(processors[i], nbio_id, min_val, max_val)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_set_cpu_socket_power_cap(self):
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_cpu_socket_power_cap as it is not complete.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        # TODO power_cap = 0
        power_cap = 0
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): power_cap({power_cap}): '
                ret = amdsmi.amdsmi_set_cpu_socket_power_cap(processors[i], power_cap)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_set_cpu_xgmi_width(self):
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_cpu_xgmi_width as it is not complete.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        # TODO min_width = 0, max_width = 0
        min_width = 0
        max_width = 0
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): min_width({min_width} max_width({max_width}): )'
                ret = amdsmi.amdsmi_set_cpu_xgmi_width(processors[i], min_width , max_width)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_set_gpu_accelerator_partition_profile(self):
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_gpu_accelerator_partition_profile as it is not complete.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        # TODO profile_index = 0
        profile_index = 0
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): profile_index({profile_index}): '
                ret = amdsmi.amdsmi_set_gpu_accelerator_partition_profile(processors[i], profile_index)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_set_gpu_clk_limit(self):
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_gpu_clk_limit as it is not complete.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        # TODO value = 0 is for min and max, need to separate out values
        value = 0
        for i in range(0, len(processors)):
            for clk_type_name, clk_type, clk_cond in self.clk_types:
                for clk_limit_type_name, clk_limit_type, clk_limit_cond in self.clk_limit_types:
                    try:
                        msg = f'gpu({i}): value({value}) clock_type=({clk_type_name}) clock_limit_type({clk_limit_type_name}): '
                        ret = amdsmi.amdsmi_set_gpu_clk_limit(processors[i], clk_type, clk_limit_type, value)
                        self._print(msg, ret)
                    except amdsmi.AmdSmiLibraryException as e:
                        if not clk_cond == self.PASS:
                            self._check_ret(msg, e, clk_cond)
                            raise_exception = e
                        elif not clk_limit_type == self.PASS:
                            self._check_ret(msg, e, clk_limit_type)
                            raise_exception = e
                        else:
                            self._check_ret(msg, e, self.PASS)
                            raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_set_gpu_clk_range(self):
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_gpu_clk_range as it is not complete.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        # TODO min_clk_value = 0 max_clk_value = 0
        min_clk_value = 0
        max_clk_value = 0
        for i in range(0, len(processors)):
            for clk_type_name, clk_type, clk_cond in self.clk_types:
                try:
                    msg = f'gpu({i}): min_clk_value({min_clk_value}) max_clk_value({max_clk_value}) clk_type({clk_type_name}): '
                    ret = amdsmi.amdsmi_set_gpu_clk_range(processors[i], min_clk_value, max_clk_value, clk_type)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, clk_cond):
                        raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_set_gpu_compute_partition(self):
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_gpu_compute_partition as it is not complete.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            for compute_partition_type_name, compute_partition_type, compute_partition_type_cond in self.compute_partition_types:
                try:
                    msg = f'gpu({i}): compute_partition_type({compute_partition_type_name}): '
                    ret = amdsmi.amdsmi_set_gpu_compute_partition(processors[i], compute_partition_type)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, compute_partition_type_cond):
                        raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_set_gpu_event_notification_mask(self):
        if self.TODO_SKIP_NYI:
            self.skipTest("Skipping test_set_gpu_event_notification_mask as it is not implemented yet.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_set_gpu_event_notification_mask(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_set_gpu_fan_speed(self):
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_gpu_fan_speed as it is not complete.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        # TODO fan_speed = 0
        fan_speed = 0
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): fan_speed({fan_speed}): '
                ret = amdsmi.amdsmi_set_gpu_fan_speed(processors[i], 0, fan_speed)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_set_gpu_memory_partition(self):
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_gpu_memory_partition as it is not complete.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            for memory_partition_type_name, memory_partition_type, memory_partition_type_cond in self.memory_partition_types:
                try:
                    msg = f'gpu({i}): memory_partition_type({memory_partition_type_name}): '
                    ret = amdsmi.amdsmi_set_gpu_memory_partition(processors[i], memory_partition_type)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, memory_partition_type_cond):
                        raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_set_gpu_memory_partition_mode(self):
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_gpu_memory_partition_mode as it is not complete.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            for memory_partition_type_name, memory_partition_type, memory_partition_type_cond in self.memory_partition_types:
                try:
                    msg = f'gpu({i}): memory_partition_type({memory_partition_type_name}): '
                    ret = amdsmi.amdsmi_set_gpu_memory_partition_mode(processors[i], memory_partition_type)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, memory_partition_type_cond):
                        raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_set_gpu_od_clk_info(self):
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_gpu_od_clk_info as it is not complete.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        # TODO value = 0
        value = 0
        for i in range(0, len(processors)):
            for freq_ind_name, freq_ind, freq_ind_cond in self.freq_inds:
                for clk_type_name, clk_type, clk_cond in self.clk_types:
                    try:
                        msg = f'gpu({i}): freq_ind({freq_ind_name}) value({value}) clk_type({clk_type_name}): '
                        ret = amdsmi.amdsmi_set_gpu_od_clk_info(processors[i], freq_ind, value, clk_type)
                        self._print(msg, ret)
                    except amdsmi.AmdSmiLibraryException as e:
                        if not freq_ind_cond == self.PASS:
                            self._check_ret(msg, e, freq_ind_cond)
                            raise_exception = e
                        elif not clk_cond == self.PASS:
                            self._check_ret(msg, e, clk_cond)
                            raise_exception = e
                        else:
                            self._check_ret(msg, e, self.PASS)
                            raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_set_gpu_od_volt_info(self):
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_gpu_od_volt_info as it is not complete.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        # TODO vpoint = 0 clk_value = 0 volt_value = 0
        vpoint = 0
        clk_value = 0
        volt_value = 0
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): vpoint({vpoint}) clk_value({clk_value}) volt_value({volt_value}): '
                ret = amdsmi.amdsmi_set_gpu_od_volt_info(processors[i], vpoint, clk_value, volt_value)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_set_gpu_overdrive_level(self):
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_gpu_overdrive_level as it is not complete.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        # TODO overdrive_value = 0
        overdrive_value = 0
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): overdrive_value({overdrive_value}): '
                ret = amdsmi.amdsmi_set_gpu_overdrive_level(processors[i], overdrive_value)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_set_gpu_pci_bandwidth(self):
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_gpu_pci_bandwidth as it is not complete.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        # TODO bitmask = 0
        bitmask = 0
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): bitmask({bitmask}): '
                ret = amdsmi.amdsmi_set_gpu_pci_bandwidth(processors[i], bitmask)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_set_gpu_perf_determinism_mode(self):
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_gpu_perf_determinism_mode as it is not complete.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        # TODO clk_value = 0
        clk_value = 0
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): clk_value({clk_value}): '
                ret = amdsmi.amdsmi_set_gpu_perf_determinism_mode(processors[i], clk_value)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_set_gpu_perf_level(self):
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_gpu_perf_level as it is not complete.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        # TODO Need to set back to default
        for i in range(0, len(processors)):
            for dev_perf_level_name, dev_perf_level, dev_perf_level_cond in self.dev_perf_levels:
                try:
                    msg = f'gpu({i}): dev_perf_level({dev_perf_level_name}): '
                    ret = amdsmi.amdsmi_set_gpu_perf_level(processors[i], dev_perf_level)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, dev_perf_level_cond):
                        raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_set_gpu_power_profile(self):
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_gpu_power_profile as it is not complete.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            for power_profile_preset_mask, power_profile_preset_masks_cond in self.power_profile_preset_masks:
                try:
                    msg = f'gpu({i}): '
                    ret = amdsmi.amdsmi_set_gpu_power_profile(processors[i], 0, power_profile_preset_mask)
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, power_profile_preset_masks_cond):
                        raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_set_gpu_process_isolation(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_set_gpu_process_isolation as it is not yet implemented.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        # TODO pisolate = 0
        pisolate = 0
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_set_gpu_process_isolation(processors[i], pisolate)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_set_power_cap(self):
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_power_cap as it is not complete.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        # TODO cap = 0
        cap = 0
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_set_power_cap(processors[i], 0, cap)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_set_soc_pstate(self):
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_soc_pstate as it is not complete.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        # TODO policy_id = 0
        policy_id = 0
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_set_soc_pstate(processors[i], policy_id)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_set_xgmi_plpd(self):
        if self.TODO_SKIP_NOT_COMPLETE:
            self.skipTest("Skipping test_set_xgmi_plpd as it is not complete.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        # TODO policy_id = 0
        policy_id = 0
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_set_xgmi_plpd(processors[i], policy_id)
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_status_code_to_string(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_status_code_to_string as it fails.")
        raise_exception = None

        # TODO status = 0
        status = 0
        status_code_to_string = amdsmi.amdsmi_status_code_to_string(status)
        self._print(f'    {status} status_code_to_string: {status_code_to_string}')
        if raise_exception:
            raise raise_exception



    def test_stop_gpu_event_notification(self):
        if self.TODO_SKIP_NYI:
            self.skipTest("Skipping test_stop_gpu_event_notification as it is not implemented yet.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_stop_gpu_event_notification(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_topo_get_link_type(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            for j in range(0, len(processors)):
                try:
                    msg = f'gpu({i},{j}): '
                    ret = amdsmi.amdsmi_topo_get_link_type(processors[i], processors[j])
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, self.PASS):
                        raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_topo_get_link_weight(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            for j in range(0, len(processors)):
                try:
                    msg = f'gpu({i},{j}): '
                    ret = amdsmi.amdsmi_topo_get_link_weight(processors[i], processors[j])
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, self.PASS):
                        raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_topo_get_numa_node_number(self):
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            try:
                msg = f'gpu({i}): '
                ret = amdsmi.amdsmi_topo_get_numa_node_number(processors[i])
                self._print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self._check_ret(msg, e, self.PASS):
                    raise_exception = e
        if raise_exception:
            raise raise_exception



    def test_topo_get_p2p_status(self):
        if self.TODO_SKIP_FAIL:
            self.skipTest("Skipping test_topo_get_p2p_status as it fails.")
        raise_exception = None

        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            for j in range(0, len(processors)):
                try:
                    msg = f'gpu({i},{j}): '
                    ret = amdsmi.amdsmi_topo_get_p2p_status(processors[i], processors[j])
                    self._print(msg, ret)
                except amdsmi.AmdSmiLibraryException as e:
                    if self._check_ret(msg, e, self.PASS):
                        raise_exception = e
        if raise_exception:
            raise raise_exception

def print_test_ids(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            print_test_ids(test)
        else:
            print(" -", test.id())

if __name__ == '__main__':
    import sys
    import unittest
    import os

    print("AMD SMI Unit Tests")
    verbose=1
    if '-q' in sys.argv or '--quiet' in sys.argv:
        verbose=0
    elif '-v' in sys.argv or '--verbose' in sys.argv:
        verbose=2

    # If no -k or --keyword argument is given, print all available tests
    if not ('-k' in sys.argv or '--keyword' in sys.argv):
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(sys.modules[__name__])
        print("==============================================================")
        print("Available tests:")
        print_test_ids(suite)

    # Provide Legend for test results, otherwise it is not clear what the output means
    print("==============================================================")
    print("Legend: . = pass, s = skipped, F = fail, E = error")
    print("==============================================================")
    print("Running tests...\n")

    # Detect if ran without sudo or root privileges
    if os.geteuid() != 0:
        print("Warning: Some tests may require elevated privileges (sudo/root) to run completely.\n")
        print("Please relaunch with elevated privileges.\n")
        sys.exit(1)

    runner = unittest.TextTestRunner(verbosity=verbose)
    unittest.main(testRunner=runner)
