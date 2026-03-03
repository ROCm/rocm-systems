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

import inspect
import json
import os
import sys
import unittest
import fcntl
import pathlib

amdsmi_path = os.environ.get("AMDSMI_PATH", "/opt/rocm/share/amd_smi")
if not os.path.exists(amdsmi_path):
    raise FileNotFoundError(
        f'AMDSMI_PATH "{amdsmi_path}" does not exist. Please set the correct path in your environment.'
    )
sys.path.append(amdsmi_path)
try:
    import amdsmi
except ImportError as e:
    raise ImportError(f'Could not import the "amdsmi" module from "{amdsmi_path}"') from e

#################################################
# Module level functions, not part of the class #
#################################################

VERBOSITY_QUIET = 0    # -q / --quiet
VERBOSITY_NORMAL = 1   # default (dot-per-test)
VERBOSITY_VERBOSE = 2  # -v / --verbose (per-test result lines)

def print_test_ids(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            print_test_ids(test)
        else:
            print(f' -{test.id()}', file=sys.stderr)
    return

def print_tests(module_name):
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[module_name])
    print('==============================================================', file=sys.stderr)
    print('Available tests:', file=sys.stderr)
    print_test_ids(suite)
    return

def print_legend():
    # Provide Legend for test results, otherwise it is not clear what the output means
    print('==============================================================', file=sys.stderr)
    print('Legend: . = pass, s = skipped, F = fail, E = error', file=sys.stderr)
    print('==============================================================', file=sys.stderr)
    print(file=sys.stderr)
    return


class Common(unittest.TestCase):
    DRIVER_INIT_FLAGS = \
        [ ('INIT_ALL_PROCESSORS', amdsmi.AmdSmiInitFlags.INIT_ALL_PROCESSORS),
          ('INIT_AMD_GPUS', amdsmi.AmdSmiInitFlags.INIT_AMD_GPUS),
          ('INIT_AMD_APUS', amdsmi.AmdSmiInitFlags.INIT_AMD_APUS),
          ('INIT_AMD_CPUS', amdsmi.AmdSmiInitFlags.INIT_AMD_CPUS)
        ]
    DRIVER_INIT_FLAGS_MAP = {flag_val: flag_name for flag_name, flag_val in DRIVER_INIT_FLAGS}
    VIRTUALIZATION_MODE_MAP = {
        amdsmi.AmdSmiVirtualizationMode.UNKNOWN: 'UNKNOWN',
        amdsmi.AmdSmiVirtualizationMode.BAREMETAL: 'BAREMETAL',
        amdsmi.AmdSmiVirtualizationMode.HOST: 'HOST',
        amdsmi.AmdSmiVirtualizationMode.GUEST: 'GUEST',
        amdsmi.AmdSmiVirtualizationMode.PASSTHROUGH: 'PASSTHROUGH'
    }

    def __init__(self, verbose, *args, **kwargs):
        self.verbose = verbose
        self.max_num_physical_devices = (
            amdsmi.amdsmi_interface.AMDSMI_MAX_NUM_XCP * amdsmi.amdsmi_interface.AMDSMI_MAX_DEVICES
        )
        self.PASS = "AMDSMI_STATUS_SUCCESS"
        self.FAIL = "AMDSMI_STATUS_INVAL"
        self.ANY_FAIL = "ANY_FAIL"

        # Tests marked wtih either of these flags will be skipped
        # and need to be implemented later.
        self.TODO_SKIP_FAIL = True
        self.TODO_SKIP_NOT_COMPLETE = True

        try:
            self.amdsmi_smart_init()

            # Get gpu
            self.processors = amdsmi.amdsmi_get_processor_handles()
            self.virt_mode = []
            self.asic_info = []
            self.board_info = []
            for gpu in self.processors:
                # Get virtualization mode info
                try:
                    ret = amdsmi.amdsmi_get_gpu_virtualization_mode(gpu)
                    mode_name = self.VIRTUALIZATION_MODE_MAP.get(ret['mode'], 'UNKNOWN')
                    self.virt_mode.append({'mode': mode_name})
                except amdsmi.AmdSmiLibraryException as e:
                    if self.verbose > VERBOSITY_QUIET:
                        print(f'In class Common, Cannot get virtualization mode information for gpu {gpu}, {e}')
                    self.virt_mode.append({'mode': 'UNKNOWN'})

                # Get asic info
                self.asic_info.append(amdsmi.amdsmi_get_gpu_asic_info(gpu))
                # Get board info
                self.board_info.append(amdsmi.amdsmi_get_gpu_board_info(gpu))

            amdsmi.amdsmi_shut_down()
        except amdsmi.AmdSmiLibraryException as e:
            print(f"In class Common, Cannot get processor information, {e}")

        self.not_supported_error_codes = [
            ("2", "AMDSMI_STATUS_NOT_SUPPORTED"),
            ("3", "AMDSMI_STATUS_NOT_YET_IMPLEMENTED"),
            ("49", "AMDSMI_STATUS_NO_HSMP_MSG_SUP"),
        ]

        self.error_map = {
            "0": "AMDSMI_STATUS_SUCCESS",
            "1": "AMDSMI_STATUS_INVAL",
            "2": "AMDSMI_STATUS_NOT_SUPPORTED",
            "3": "AMDSMI_STATUS_NOT_YET_IMPLEMENTED",
            "4": "AMDSMI_STATUS_FAIL_LOAD_MODULE",
            "5": "AMDSMI_STATUS_FAIL_LOAD_SYMBOL",
            "6": "AMDSMI_STATUS_DRM_ERROR",
            "7": "AMDSMI_STATUS_API_FAILED",
            "8": "AMDSMI_STATUS_TIMEOUT",
            "9": "AMDSMI_STATUS_RETRY",
            "10": "AMDSMI_STATUS_NO_PERM",
            "11": "AMDSMI_STATUS_INTERRUPT",
            "12": "AMDSMI_STATUS_IO",
            "13": "AMDSMI_STATUS_ADDRESS_FAULT",
            "14": "AMDSMI_STATUS_FILE_ERROR",
            "15": "AMDSMI_STATUS_OUT_OF_RESOURCES",
            "16": "AMDSMI_STATUS_INTERNAL_EXCEPTION",
            "17": "AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS",
            "18": "AMDSMI_STATUS_INIT_ERROR",
            "19": "AMDSMI_STATUS_REFCOUNT_OVERFLOW",
            "30": "AMDSMI_STATUS_BUSY",
            "31": "AMDSMI_STATUS_NOT_FOUND",
            "32": "AMDSMI_STATUS_NOT_INIT",
            "33": "AMDSMI_STATUS_NO_SLOT",
            "34": "AMDSMI_STATUS_DRIVER_NOT_LOADED",
            "39": "AMDSMI_STATUS_MORE_DATA",
            "40": "AMDSMI_STATUS_NO_DATA",
            "41": "AMDSMI_STATUS_INSUFFICIENT_SIZE",
            "42": "AMDSMI_STATUS_UNEXPECTED_SIZE",
            "43": "AMDSMI_STATUS_UNEXPECTED_DATA",
            "44": "AMDSMI_STATUS_NON_AMD_CPU",
            "45": "AMDSMI_STATUS_NO_ENERGY_DRV",
            "46": "AMDSMI_STATUS_NO_MSR_DRV",
            "47": "AMDSMI_STATUS_NO_HSMP_DRV",
            "48": "AMDSMI_STATUS_NO_HSMP_SUP",
            "49": "AMDSMI_STATUS_NO_HSMP_MSG_SUP",
            "50": "AMDSMI_STATUS_HSMP_TIMEOUT",
            "51": "AMDSMI_STATUS_NO_DRV",
            "52": "AMDSMI_STATUS_FILE_NOT_FOUND",
            "53": "AMDSMI_STATUS_ARG_PTR_NULL",
            "54": "AMDSMI_STATUS_AMDGPU_RESTART_ERR",
            "55": "AMDSMI_STATUS_SETTING_UNAVAILABLE",
            "56": "AMDSMI_STATUS_CORRUPTED_EEPROM",
            "0xFFFFFFFE": "AMDSMI_STATUS_MAP_ERROR",
            "0xFFFFFFFF": "AMDSMI_STATUS_UNKNOWN_ERROR",
        }

        self.status_types = [
            ("SUCCESS", amdsmi.AmdSmiStatus.SUCCESS, self.PASS),
            ("INVAL", amdsmi.AmdSmiStatus.INVAL, self.PASS),
            ("NOT_SUPPORTED", amdsmi.AmdSmiStatus.NOT_SUPPORTED, self.PASS),
            ("NOT_YET_IMPLEMENTED", amdsmi.AmdSmiStatus.NOT_YET_IMPLEMENTED, self.PASS),
            ("FAIL_LOAD_MODULE", amdsmi.AmdSmiStatus.FAIL_LOAD_MODULE, self.PASS),
            ("FAIL_LOAD_SYMBOL", amdsmi.AmdSmiStatus.FAIL_LOAD_SYMBOL, self.PASS),
            ("DRM_ERROR", amdsmi.AmdSmiStatus.DRM_ERROR, self.PASS),
            ("API_FAILED", amdsmi.AmdSmiStatus.API_FAILED, self.PASS),
            ("TIMEOUT", amdsmi.AmdSmiStatus.TIMEOUT, self.PASS),
            ("RETRY", amdsmi.AmdSmiStatus.RETRY, self.PASS),
            ("NO_PERM", amdsmi.AmdSmiStatus.NO_PERM, self.PASS),
            ("INTERRUPT", amdsmi.AmdSmiStatus.INTERRUPT, self.PASS),
            ("IO", amdsmi.AmdSmiStatus.IO, self.PASS),
            ("ADDRESS_FAULT", amdsmi.AmdSmiStatus.ADDRESS_FAULT, self.PASS),
            ("FILE_ERROR", amdsmi.AmdSmiStatus.FILE_ERROR, self.PASS),
            ("OUT_OF_RESOURCES", amdsmi.AmdSmiStatus.OUT_OF_RESOURCES, self.PASS),
            ("INTERNAL_EXCEPTION", amdsmi.AmdSmiStatus.INTERNAL_EXCEPTION, self.PASS),
            ("INPUT_OUT_OF_BOUNDS", amdsmi.AmdSmiStatus.INPUT_OUT_OF_BOUNDS, self.PASS),
            ("INIT_ERROR", amdsmi.AmdSmiStatus.INIT_ERROR, self.PASS),
            ("REFCOUNT_OVERFLOW", amdsmi.AmdSmiStatus.REFCOUNT_OVERFLOW, self.PASS),
            ("DIRECTORY_NOT_FOUND", amdsmi.AmdSmiStatus.DIRECTORY_NOT_FOUND, self.PASS),
            ("BUSY", amdsmi.AmdSmiStatus.BUSY, self.PASS),
            ("NOT_FOUND", amdsmi.AmdSmiStatus.NOT_FOUND, self.PASS),
            ("NOT_INIT", amdsmi.AmdSmiStatus.NOT_INIT, self.PASS),
            ("NO_SLOT", amdsmi.AmdSmiStatus.NO_SLOT, self.PASS),
            ("DRIVER_NOT_LOADED", amdsmi.AmdSmiStatus.DRIVER_NOT_LOADED, self.PASS),
            ("MORE_DATA", amdsmi.AmdSmiStatus.MORE_DATA, self.PASS),
            ("NO_DATA", amdsmi.AmdSmiStatus.NO_DATA, self.PASS),
            ("INSUFFICIENT_SIZE", amdsmi.AmdSmiStatus.INSUFFICIENT_SIZE, self.PASS),
            ("UNEXPECTED_SIZE", amdsmi.AmdSmiStatus.UNEXPECTED_SIZE, self.PASS),
            ("UNEXPECTED_DATA", amdsmi.AmdSmiStatus.UNEXPECTED_DATA, self.PASS),
            ("NON_AMD_CPU", amdsmi.AmdSmiStatus.NON_AMD_CPU, self.PASS),
            ("NO_ENERGY_DRV", amdsmi.AmdSmiStatus.NO_ENERGY_DRV, self.PASS),
            ("NO_MSR_DRV", amdsmi.AmdSmiStatus.NO_MSR_DRV, self.PASS),
            ("NO_HSMP_DRV", amdsmi.AmdSmiStatus.NO_HSMP_DRV, self.PASS),
            ("NO_HSMP_SUP", amdsmi.AmdSmiStatus.NO_HSMP_SUP, self.PASS),
            ("NO_HSMP_MSG_SUP", amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP, self.PASS),
            ("HSMP_TIMEOUT", amdsmi.AmdSmiStatus.HSMP_TIMEOUT, self.PASS),
            ("NO_DRV", amdsmi.AmdSmiStatus.NO_DRV, self.PASS),
            ("FILE_NOT_FOUND", amdsmi.AmdSmiStatus.FILE_NOT_FOUND, self.PASS),
            ("ARG_PTR_NULL", amdsmi.AmdSmiStatus.ARG_PTR_NULL, self.PASS),
            ("AMDGPU_RESTART_ERR", amdsmi.AmdSmiStatus.AMDGPU_RESTART_ERR, self.PASS),
            ("SETTING_UNAVAILABLE", amdsmi.AmdSmiStatus.SETTING_UNAVAILABLE, self.PASS),
            ("CORRUPTED_EEPROM", amdsmi.AmdSmiStatus.CORRUPTED_EEPROM, self.PASS),
            ("MAP_ERROR", amdsmi.AmdSmiStatus.MAP_ERROR, self.PASS),
            ("UNKNOWN_ERROR", amdsmi.AmdSmiStatus.UNKNOWN_ERROR, self.PASS),
        ]

        self.clk_types = [
            ("SYS", amdsmi.AmdSmiClkType.SYS, self.PASS),
            ("GFX", amdsmi.AmdSmiClkType.GFX, self.PASS),
            ("DF", amdsmi.AmdSmiClkType.DF, self.PASS),
            ("DCEF", amdsmi.AmdSmiClkType.DCEF, [self.PASS, self.FAIL]),
            ("SOC", amdsmi.AmdSmiClkType.SOC, self.PASS),
            ("MEM", amdsmi.AmdSmiClkType.MEM, self.PASS),
            ("PCIE", amdsmi.AmdSmiClkType.PCIE, [self.PASS, self.FAIL]),
            ("VCLK0", amdsmi.AmdSmiClkType.VCLK0, self.PASS),
            ("VCLK1", amdsmi.AmdSmiClkType.VCLK1, self.PASS),
            ("DCLK0", amdsmi.AmdSmiClkType.DCLK0, self.PASS),
            ("DCLK1", amdsmi.AmdSmiClkType.DCLK1, self.PASS),
        ]

        self.clk_limit_types = [
            ("MIN", amdsmi.AmdSmiClkLimitType.MIN, self.PASS),
            ("MAX", amdsmi.AmdSmiClkLimitType.MAX, self.PASS),
        ]

        self.io_bw_encodings = [
            ("AGG_BW0", amdsmi.amdsmi_interface.amdsmi_wrapper.AGG_BW0, self.PASS),
            ("RD_BW0", amdsmi.amdsmi_interface.amdsmi_wrapper.RD_BW0, self.PASS),
            ("WR_BW0", amdsmi.amdsmi_interface.amdsmi_wrapper.WR_BW0, self.PASS),
        ]

        self.gpu_blocks = [
            ("INVALID", amdsmi.AmdSmiGpuBlock.INVALID, self.FAIL),
            ("UMC", amdsmi.AmdSmiGpuBlock.UMC, self.PASS),
            ("SDMA", amdsmi.AmdSmiGpuBlock.SDMA, self.PASS),
            ("GFX", amdsmi.AmdSmiGpuBlock.GFX, self.PASS),
            ("MMHUB", amdsmi.AmdSmiGpuBlock.MMHUB, self.PASS),
            ("ATHUB", amdsmi.AmdSmiGpuBlock.ATHUB, self.PASS),
            ("PCIE_BIF", amdsmi.AmdSmiGpuBlock.PCIE_BIF, self.PASS),
            ("HDP", amdsmi.AmdSmiGpuBlock.HDP, self.PASS),
            ("XGMI_WAFL", amdsmi.AmdSmiGpuBlock.XGMI_WAFL, self.PASS),
            ("DF", amdsmi.AmdSmiGpuBlock.DF, self.PASS),
            ("SMN", amdsmi.AmdSmiGpuBlock.SMN, self.PASS),
            ("SEM", amdsmi.AmdSmiGpuBlock.SEM, self.PASS),
            ("MP0", amdsmi.AmdSmiGpuBlock.MP0, self.PASS),
            ("MP1", amdsmi.AmdSmiGpuBlock.MP1, self.PASS),
            ("FUSE", amdsmi.AmdSmiGpuBlock.FUSE, self.PASS),
            ("MCA", amdsmi.AmdSmiGpuBlock.MCA, self.PASS),
            ("VCN", amdsmi.AmdSmiGpuBlock.VCN, self.PASS),
            ("JPEG", amdsmi.AmdSmiGpuBlock.JPEG, self.PASS),
            ("IH", amdsmi.AmdSmiGpuBlock.IH, self.PASS),
            ("MPIO", amdsmi.AmdSmiGpuBlock.MPIO, self.PASS),
            ("RESERVED", amdsmi.AmdSmiGpuBlock.RESERVED, self.FAIL),
        ]

        self.memory_types = [
            ("VRAM", amdsmi.AmdSmiMemoryType.VRAM, self.PASS),
            ("VIS_VRAM", amdsmi.AmdSmiMemoryType.VIS_VRAM, self.PASS),
            ("GTT", amdsmi.AmdSmiMemoryType.GTT, self.PASS),
        ]

        self.reg_types = [
            ("XGMI", amdsmi.AmdSmiRegType.XGMI, self.PASS),
            ("WAFL", amdsmi.AmdSmiRegType.WAFL, self.PASS),
            ("PCIE", amdsmi.AmdSmiRegType.PCIE, self.PASS),
            ("USR", amdsmi.AmdSmiRegType.USR, self.PASS),
            ("USR1", amdsmi.AmdSmiRegType.USR1, self.PASS),
        ]

        self.voltage_metrics = [
            ("CURRENT", amdsmi.AmdSmiVoltageMetric.CURRENT, self.PASS),
            ("MAX", amdsmi.AmdSmiVoltageMetric.MAX, self.PASS),
            ("MIN_CRIT", amdsmi.AmdSmiVoltageMetric.MIN_CRIT, self.PASS),
            ("MIN", amdsmi.AmdSmiVoltageMetric.MIN, self.PASS),
            ("MAX_CRIT", amdsmi.AmdSmiVoltageMetric.MAX_CRIT, self.PASS),
            ("AVERAGE", amdsmi.AmdSmiVoltageMetric.AVERAGE, self.PASS),
            ("LOWEST", amdsmi.AmdSmiVoltageMetric.LOWEST, self.PASS),
            ("HIGHEST", amdsmi.AmdSmiVoltageMetric.HIGHEST, self.PASS),
        ]

        self.voltage_types = [
            ("VDDGFX", amdsmi.AmdSmiVoltageType.VDDGFX, self.PASS),
            ("VDDBOARD", amdsmi.AmdSmiVoltageType.VDDBOARD, self.PASS),
            ("INVALID", amdsmi.AmdSmiVoltageType.INVALID, self.FAIL),
        ]

        self.link_types = [
            (
                "AMDSMI_LINK_TYPE_INTERNAL",
                amdsmi.AmdSmiLinkType.AMDSMI_LINK_TYPE_INTERNAL,
                self.PASS,
            ),
            ("AMDSMI_LINK_TYPE_XGMI", amdsmi.AmdSmiLinkType.AMDSMI_LINK_TYPE_XGMI, self.PASS),
            ("AMDSMI_LINK_TYPE_PCIE", amdsmi.AmdSmiLinkType.AMDSMI_LINK_TYPE_PCIE, self.PASS),
            (
                "AMDSMI_LINK_TYPE_NOT_APPLICABLE",
                amdsmi.AmdSmiLinkType.AMDSMI_LINK_TYPE_NOT_APPLICABLE,
                self.FAIL,
            ),
            ("AMDSMI_LINK_TYPE_UNKNOWN", amdsmi.AmdSmiLinkType.AMDSMI_LINK_TYPE_UNKNOWN, self.FAIL),
        ]

        self.temperature_types = [
            ("EDGE", amdsmi.AmdSmiTemperatureType.EDGE, self.PASS),
            ("HOTSPOT", amdsmi.AmdSmiTemperatureType.HOTSPOT, self.PASS),
            ("JUNCTION", amdsmi.AmdSmiTemperatureType.JUNCTION, self.PASS),
            ("VRAM", amdsmi.AmdSmiTemperatureType.VRAM, self.PASS),
            ("HBM_0", amdsmi.AmdSmiTemperatureType.HBM_0, self.PASS),
            ("HBM_1", amdsmi.AmdSmiTemperatureType.HBM_1, self.PASS),
            ("HBM_2", amdsmi.AmdSmiTemperatureType.HBM_2, self.PASS),
            ("HBM_3", amdsmi.AmdSmiTemperatureType.HBM_3, self.PASS),
            ("PLX", amdsmi.AmdSmiTemperatureType.PLX, self.PASS),
        ]

        self.temperature_metrics = [
            ("CURRENT", amdsmi.AmdSmiTemperatureMetric.CURRENT, self.PASS),
            ("MAX", amdsmi.AmdSmiTemperatureMetric.MAX, self.PASS),
            ("MIN", amdsmi.AmdSmiTemperatureMetric.MIN, self.PASS),
            ("MAX_HYST", amdsmi.AmdSmiTemperatureMetric.MAX_HYST, self.PASS),
            ("MIN_HYST", amdsmi.AmdSmiTemperatureMetric.MIN_HYST, self.PASS),
            ("CRITICAL", amdsmi.AmdSmiTemperatureMetric.CRITICAL, self.PASS),
            ("CRITICAL_HYST", amdsmi.AmdSmiTemperatureMetric.CRITICAL_HYST, self.PASS),
            ("EMERGENCY", amdsmi.AmdSmiTemperatureMetric.EMERGENCY, self.PASS),
            ("EMERGENCY_HYST", amdsmi.AmdSmiTemperatureMetric.EMERGENCY_HYST, self.PASS),
            ("CRIT_MIN", amdsmi.AmdSmiTemperatureMetric.CRIT_MIN, self.PASS),
            ("CRIT_MIN_HYST", amdsmi.AmdSmiTemperatureMetric.CRIT_MIN_HYST, self.PASS),
            ("OFFSET", amdsmi.AmdSmiTemperatureMetric.OFFSET, self.PASS),
            ("LOWEST", amdsmi.AmdSmiTemperatureMetric.LOWEST, self.PASS),
            ("HIGHEST", amdsmi.AmdSmiTemperatureMetric.HIGHEST, self.PASS),
        ]

        self.utilization_counter_types = [
            (
                "COARSE_GRAIN_GFX_ACTIVITY",
                amdsmi.AmdSmiUtilizationCounterType.COARSE_GRAIN_GFX_ACTIVITY,
                self.PASS,
            ),
            (
                "COARSE_GRAIN_MEM_ACTIVITY",
                amdsmi.AmdSmiUtilizationCounterType.COARSE_GRAIN_MEM_ACTIVITY,
                self.PASS,
            ),
            (
                "COARSE_DECODER_ACTIVITY",
                amdsmi.AmdSmiUtilizationCounterType.COARSE_DECODER_ACTIVITY,
                self.PASS,
            ),
            (
                "FINE_GRAIN_GFX_ACTIVITY",
                amdsmi.AmdSmiUtilizationCounterType.FINE_GRAIN_GFX_ACTIVITY,
                self.PASS,
            ),
            (
                "FINE_GRAIN_MEM_ACTIVITY",
                amdsmi.AmdSmiUtilizationCounterType.FINE_GRAIN_MEM_ACTIVITY,
                self.PASS,
            ),
            (
                "FINE_DECODER_ACTIVITY",
                amdsmi.AmdSmiUtilizationCounterType.FINE_DECODER_ACTIVITY,
                self.PASS,
            ),
            (
                "UTILIZATION_COUNTER_FIRST",
                amdsmi.AmdSmiUtilizationCounterType.UTILIZATION_COUNTER_FIRST,
                self.PASS,
            ),
            (
                "UTILIZATION_COUNTER_LAST",
                amdsmi.AmdSmiUtilizationCounterType.UTILIZATION_COUNTER_LAST,
                self.PASS,
            ),
            ("UTILIZATION_COUNTER_BAD", 100, self.FAIL),
        ]

        self.event_groups = [
            ("XGMI", amdsmi.AmdSmiEventGroup.XGMI, self.PASS),
            ("XGMI_DATA_OUT", amdsmi.AmdSmiEventGroup.XGMI_DATA_OUT, self.PASS),
            ("GRP_INVALID", amdsmi.AmdSmiEventGroup.GRP_INVALID, self.FAIL),
        ]

        self.event_types = [
            ("XGMI_0_NOP_TX", amdsmi.AmdSmiEventType.XGMI_0_NOP_TX, self.PASS),
            ("XGMI_0_REQUEST_TX", amdsmi.AmdSmiEventType.XGMI_0_REQUEST_TX, self.PASS),
            ("XGMI_0_RESPONSE_TX", amdsmi.AmdSmiEventType.XGMI_0_RESPONSE_TX, self.PASS),
            ("XGMI_0_BEATS_TX", amdsmi.AmdSmiEventType.XGMI_0_BEATS_TX, self.PASS),
            ("XGMI_1_NOP_TX", amdsmi.AmdSmiEventType.XGMI_1_NOP_TX, self.PASS),
            ("XGMI_1_REQUEST_TX", amdsmi.AmdSmiEventType.XGMI_1_REQUEST_TX, self.PASS),
            ("XGMI_1_RESPONSE_TX", amdsmi.AmdSmiEventType.XGMI_1_RESPONSE_TX, self.PASS),
            ("XGMI_1_BEATS_TX", amdsmi.AmdSmiEventType.XGMI_1_BEATS_TX, self.PASS),
            ("XGMI_DATA_OUT_0", amdsmi.AmdSmiEventType.XGMI_DATA_OUT_0, self.PASS),
            ("XGMI_DATA_OUT_1", amdsmi.AmdSmiEventType.XGMI_DATA_OUT_1, self.PASS),
            ("XGMI_DATA_OUT_2", amdsmi.AmdSmiEventType.XGMI_DATA_OUT_2, self.PASS),
            ("XGMI_DATA_OUT_3", amdsmi.AmdSmiEventType.XGMI_DATA_OUT_3, self.PASS),
            ("XGMI_DATA_OUT_4", amdsmi.AmdSmiEventType.XGMI_DATA_OUT_4, self.PASS),
            ("XGMI_DATA_OUT_5", amdsmi.AmdSmiEventType.XGMI_DATA_OUT_5, self.PASS),
        ]

        self.counter_commands = [
            ("CMD_START", amdsmi.AmdSmiCounterCommand.CMD_START, self.PASS),
            ("CMD_STOP", amdsmi.AmdSmiCounterCommand.CMD_STOP, self.PASS),
        ]

        self.compute_partition_types = [
            ("SPX", amdsmi.AmdSmiComputePartitionType.SPX, self.PASS),
            ("DPX", amdsmi.AmdSmiComputePartitionType.DPX, self.PASS),
            ("TPX", amdsmi.AmdSmiComputePartitionType.TPX, self.PASS),
            ("QPX", amdsmi.AmdSmiComputePartitionType.QPX, self.PASS),
            ("CPX", amdsmi.AmdSmiComputePartitionType.CPX, self.PASS),
            ("INVALID", amdsmi.AmdSmiComputePartitionType.INVALID, self.FAIL),
        ]

        self.memory_partition_types = [
            ("NPS1", amdsmi.AmdSmiMemoryPartitionType.NPS1, self.PASS),
            ("NPS2", amdsmi.AmdSmiMemoryPartitionType.NPS2, self.PASS),
            ("NPS4", amdsmi.AmdSmiMemoryPartitionType.NPS4, self.PASS),
            ("NPS8", amdsmi.AmdSmiMemoryPartitionType.NPS8, self.PASS),
            ("UNKNOWN", amdsmi.AmdSmiMemoryPartitionType.UNKNOWN, self.FAIL),
        ]

        self.freq_inds = [
            ("MIN", amdsmi.AmdSmiFreqInd.MIN, self.PASS),
            ("MAX", amdsmi.AmdSmiFreqInd.MAX, self.PASS),
            ("INVALID", amdsmi.AmdSmiFreqInd.INVALID, self.FAIL),
        ]

        self.power_profile_preset_masks = [
            ("CUSTOM_MASK", amdsmi.AmdSmiPowerProfilePresetMasks.CUSTOM_MASK, self.PASS),
            ("VIDEO_MASK", amdsmi.AmdSmiPowerProfilePresetMasks.VIDEO_MASK, self.PASS),
            (
                "POWER_SAVING_MASK",
                amdsmi.AmdSmiPowerProfilePresetMasks.POWER_SAVING_MASK,
                self.PASS,
            ),
            ("COMPUTE_MASK", amdsmi.AmdSmiPowerProfilePresetMasks.COMPUTE_MASK, self.PASS),
            ("VR_MASK", amdsmi.AmdSmiPowerProfilePresetMasks.VR_MASK, self.PASS),
            (
                "THREE_D_FULL_SCR_MASK",
                amdsmi.AmdSmiPowerProfilePresetMasks.THREE_D_FULL_SCR_MASK,
                self.PASS,
            ),
            ("BOOTUP_DEFAULT", amdsmi.AmdSmiPowerProfilePresetMasks.BOOTUP_DEFAULT, self.PASS),
        ]

        self.processor_types = [
            ("UNKNOWN", amdsmi.AmdSmiProcessorType.UNKNOWN, self.FAIL),
            ("AMD_GPU", amdsmi.AmdSmiProcessorType.AMD_GPU, self.PASS),
            ("AMD_CPU", amdsmi.AmdSmiProcessorType.AMD_CPU, self.PASS),
            ("NON_AMD_GPU", amdsmi.AmdSmiProcessorType.NON_AMD_GPU, self.PASS),
            ("NON_AMD_CPU", amdsmi.AmdSmiProcessorType.NON_AMD_CPU, self.PASS),
            ("AMD_CPU_CORE", amdsmi.AmdSmiProcessorType.AMD_CPU_CORE, self.PASS),
            ("AMD_APU", amdsmi.AmdSmiProcessorType.AMD_APU, self.PASS),
        ]

        self.dev_perf_levels = [
            ("AUTO", amdsmi.AmdSmiDevPerfLevel.AUTO, self.PASS),
            ("LOW", amdsmi.AmdSmiDevPerfLevel.LOW, self.PASS),
            ("HIGH", amdsmi.AmdSmiDevPerfLevel.HIGH, self.PASS),
            ("MANUAL", amdsmi.AmdSmiDevPerfLevel.MANUAL, self.PASS),
            ("STABLE_STD", amdsmi.AmdSmiDevPerfLevel.STABLE_STD, self.PASS),
            ("STABLE_PEAK", amdsmi.AmdSmiDevPerfLevel.STABLE_PEAK, self.PASS),
            ("STABLE_MIN_MCLK", amdsmi.AmdSmiDevPerfLevel.STABLE_MIN_MCLK, self.PASS),
            ("STABLE_MIN_SCLK", amdsmi.AmdSmiDevPerfLevel.STABLE_MIN_SCLK, self.PASS),
            ("DETERMINISM", amdsmi.AmdSmiDevPerfLevel.DETERMINISM, self.PASS),
            ("UNKNOWN", amdsmi.AmdSmiDevPerfLevel.UNKNOWN, self.FAIL),
        ]

    def print(self, msg, data=None):
        if self.verbose > VERBOSITY_QUIET:
            if data is None:
                print(msg, flush=True)
            elif any(data in value for value in self.not_supported_error_codes):
                print(f"{msg} {data}", flush=True)
            else:
                if isinstance(data, str) and data in self.error_map.values():
                    print(msg, end="")
                else:
                    print(msg)
                if isinstance(data, dict) or isinstance(data, list):
                    print(json.dumps(data, sort_keys=False, indent=4), flush=True)
                else:
                    print(data)
        return

    def print_func_name(self, msg=None):
        if self.verbose == VERBOSITY_VERBOSE:
            stk = inspect.stack()
            if stk[1].function == "_callSetUp":
                return
            print(f"\n## {stk[1].function}()", flush=True)
            if msg:
                print(msg, flush=True)
        return

    def print_device_header(self, i, _):
        # Print virtualization mode info
        msg = f"virtualization mode(gpu={i})"
        self.print(f"\t{msg}")
        mode = self.virt_mode[i]["mode"]
        self.print(f"\t\tmode : {mode}")
        # Print asic info
        msg = f"asic info(gpu={i})"
        self.print(f"\t{msg}")
        for key, value in self.asic_info[i].items():
            self.print(f"\t\t{key} : {value}")
        # Print board info
        msg = f"board info(gpu={i})"
        self.print(f"\t{msg}")
        for key, value in self.board_info[i].items():
            self.print(f"\t\t{key} : {value}")
        return

    def get_error_code(self, exc):
        error_code = "-1"
        error_code_name = "UNKNOWN_ERROR"
        if hasattr(exc, "get_error_code"):
            error_code = str(exc.get_error_code())
            if error_code in self.error_map:
                error_code_name = self.error_map[error_code]
        return (error_code, error_code_name)

    def check_ret(self, msg, exc, expected_code_name=None, printIt=True):
        if isinstance(exc, str) and len(exc) == 0:
            error_code_name = expected_code_name
            if error_code_name in self.error_map.values():
                for key, value in self.error_map.items():
                    if value == error_code_name:
                        error_code = key
                        break
            else:
                error_code = "-1"
        elif hasattr(exc, "get_error_code"):
            error_code, error_code_name = self.get_error_code(exc)
        else:
            error_code = str(exc).split(':', maxsplit=1)[0]
            error_code_name = 'AMDSMI_STATUS_INVAL'

        # Check for when there are multiple passing conditions
        if isinstance(expected_code_name, list):
            for ec in expected_code_name:
                rc = self.check_ret(
                    msg, exc, ec, False
                )  # Do not print msg, otherwise multiple msgs printed
                if not rc:
                    rc = self.check_ret(msg, exc, ec)  # Call check again so msg is printed
                print(f"{msg}\n", end="")
            print(
                flush=True,
            )
            return True

        # Check for single passing condition
        status_msg = ""
        status_ret = False
        if any(error_code in value for value in self.not_supported_error_codes):
            status_msg = f'\tAMDSMI API Returned {error_code_name}'
        elif error_code_name == expected_code_name:
            status_msg = f"\tTest PASSED with expected result {expected_code_name}"
        elif error_code_name != self.PASS and expected_code_name == self.ANY_FAIL:
            status_msg = f"\tTest PASSED with expected result {expected_code_name} and received {error_code_name}"
        else:
            status_msg = f"\tTest FAILED with expected result {expected_code_name} but received {error_code_name}"
            status_ret = True
        if self.verbose > VERBOSITY_QUIET and printIt:
            if msg:
                print(f"{msg}\n", end="")
            print(f"{status_msg}", flush=True)
        return status_ret

    # Keeping just incase this will be needed for future tests
    # Have an example in integration power_cap test (commented out atm)
    def check_runtime_pm_status(self, render_minor: int) -> bool:
        """Check if device is in runtime suspend state."""
        try:
            # Read runtime_enabled
            device_path = f"/sys/class/drm/renderD{render_minor}/device"
            enabled_path = os.path.join(device_path, "power/runtime_enabled")
            with open(enabled_path, 'r') as f:
                enabled = f.read().strip()

            if "disabled" in enabled or "forbidden" in enabled:
                return False

            # Read runtime_status
            status_path = os.path.join(device_path, "power/runtime_status")
            with open(status_path, 'r') as f:
                status = f.read().strip()

            return "suspended" in status
        except (IOError, OSError):
            return False

    # Keeping just incase this will be needed for future tests
    # Have an example in integration power_cap test (commented out atm)
    def wake_device(self, render_minor: int) -> bool:
        """Wake device from runtime suspend using DRM ioctl."""
        render_path = f"/dev/dri/renderD{render_minor}"

        try:
            fd = os.open(render_path, os.O_RDWR | os.O_CLOEXEC)
            try:
                # DRM_IOCTL_AMDGPU_INFO = 0xc0206405 (from libdrm headers)
                # Just issuing any ioctl wakes the device
                DRM_IOCTL_AMDGPU_INFO = 0xc0206405
                request = bytes(32)  # Empty drm_amdgpu_info struct
                fcntl.ioctl(fd, DRM_IOCTL_AMDGPU_INFO, request)
            finally:
                os.close(fd)
            return True
        except (IOError, OSError) as e:
            self.print(f'Failed to wake device: {e}')
            return False

    # Keeping just incase this will be needed for future tests
    def get_gpu_id_from_device_handle(self, input_device_handle):
        """Get the gpu index from the device_handle.
        amdsmi_get_processor_handles() returns the list of device_handles in order of gpu_index
        """
        device_handles = amdsmi.amdsmi_get_processor_handles()
        for gpu_index, device_handle in enumerate(device_handles):
            if input_device_handle.value == device_handle.value:
                return gpu_index
        return None  # handle not found

    def _check_amdgpu_driver(self):
        """ Returns true if amdgpu is found in the list of initialized modules """
        amd_gpu_status_file = pathlib.Path("/sys/module/amdgpu/initstate")
        if amd_gpu_status_file.exists():
            try: 
                return amd_gpu_status_file.read_text(encoding="ascii").strip() == "live"
            except OSError:
                pass

        # If the driver is loaded either as a module OR built in, this dir will be populated
        drv = pathlib.Path("/sys/bus/pci/drivers/amdgpu")
        if not drv.exists():
            return False

        # Check if a symlink exists that loosely matches PCI BDF format
        # ex: 0000:03:00.0
        for p in drv.iterdir():
            if p.is_symlink() and ":" in p.name and "." in p.name:
                return True
        return False

    def _check_amd_hsmp_driver(self):
        """ Returns true if amd_hsmp or hsmp_acpi is found in the list of initialized modules """
        amd_cpu_status_file = pathlib.Path("/dev/hsmp")
        if amd_cpu_status_file.exists():
            return True
        return False

    def _init_with_flag(self, init_flag, driver_msg):
        ret = None
        try:
            ret = amdsmi.amdsmi_init(init_flag)
        except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
            if e.err_code in (
                amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_INIT,
                amdsmi.amdsmi_wrapper.AMDSMI_STATUS_DRIVER_NOT_LOADED,
            ):
                self.print(driver_msg)
                sys.exit(-1)
            raise
        return ret

    def amdsmi_smart_init(self):
        ''' Initializes AMDSMI Library based on live drivers found in the system.

        Checks for the presence of the amdgpu, amd_hsmp or hsmp_acpi drivers and initializes the
        AMD SMI library based on the live drivers found.

        Return:
            tuple: A tuple containing:
                - ret: The return value from amdsmi_init() (typically None on success)
                - init_flag: The flag used to initialize the AMD SMI library without error
                    (one of: INIT_AMD_APUS, INIT_AMD_GPUS, or INIT_AMD_CPUS)

        Raises:
            AmdSmiLibraryException: If initialization fails for reasons other than driver not loaded
            AmdSmiParameterException: If invalid parameters are passed to amdsmi_init
            SystemExit: If no compatible AMD drivers are detected on the system
        '''
        # Determine init flag from which drivers are live; msg is only emitted on init failure.
        if self._check_amdgpu_driver() and self._check_amd_hsmp_driver():
            init_flag = amdsmi.AmdSmiInitFlags.INIT_AMD_APUS
            msg = 'amdgpu and amd_hsmp/hsmp_acpi detected but INIT_AMD_APUS failed'
        elif self._check_amdgpu_driver():
            init_flag = amdsmi.AmdSmiInitFlags.INIT_AMD_GPUS
            msg = 'amdgpu detected but INIT_AMD_GPUS failed'
        elif self._check_amd_hsmp_driver():
            init_flag = amdsmi.AmdSmiInitFlags.INIT_AMD_CPUS
            msg = 'amd_hsmp/hsmp_acpi detected but INIT_AMD_CPUS failed'
        else:
            self.print('Drivers not loaded (amdgpu, amd_hsmp or hsmp_acpi drivers not found in modules)')
            sys.exit(-1)

        ret = self._init_with_flag(init_flag, msg)
        flag_name = self.DRIVER_INIT_FLAGS_MAP.get(init_flag, 'UNKNOWN')
        self.print(f'\tAMDSMI initialized with at least one driver | init flag: {flag_name} ({init_flag})')
        return (ret, init_flag)

    def Test_API(self, **kwargs):
        '''
            Tests API with zero or more arguments

            Arguments:
                func_name: API to be executed
            Optional:
                param1_name: Name of parameter 1
        '''
        iterator = iter(kwargs.items())
        func_name, func = next(iterator)
        param1_name, param1_value = next(iterator, (None, None))

        if not param1_name:
            msg = f'\t### {func_name}()'
        else:
            msg = f'\t### {func_name}({param1_name}={param1_value})'

        raise_exception = None
        try:
            if param1_name:
                data = func(param1_value)
            else:
                data = func()
            self.print(msg, data)
            self.check_ret('', '', self.PASS)
        except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
            if self.check_ret(msg, e, self.PASS):
                raise_exception = e
        if raise_exception:
            raise raise_exception
        return


    def Test_API_Per_GPU(self, **kwargs):
        '''
            Tests API per GPU with zero or more arguments

            Arguments:
                func_name: API to be executed
            Optional:
                param1_name: Name of parameter 1
                param2_name: Name of parameter 2
                param3_name: Name of parameter 3
        '''
        iterator = iter(kwargs.items())
        func_name, func = next(iterator)
        param1_name, param1_value = next(iterator, (None, None))
        param2_name = None
        param3_name = None
        if param1_name:
            param2_name, param2_value = next(iterator, (None, None))
            if param2_name:
                param3_name, param3_value = next(iterator, (None, None))

        raise_exception = None
        for i, gpu in enumerate(self.processors):
            self.print_device_header(i, gpu)
            if param3_name:
                msg = f'\t### {func_name}(gpu={i}, {param1_name}={param1_value}, {param2_name}={param2_value}, {param3_name}={param3_value})'
            elif param2_name:
                msg = f'\t### {func_name}(gpu={i}, {param1_name}={param1_value}, {param2_name}={param2_value})'
            elif param1_name:
                msg = f'\t### {func_name}(gpu={i}, {param1_name}={param1_value})'
            else:
                msg = f'\t### {func_name}(gpu={i})'
            try:
                if param3_name:
                    data = func(gpu, param1_value, param2_value, param3_value)
                elif param2_name:
                    data = func(gpu, param1_value, param2_value)
                elif param1_name:
                    data = func(gpu, param1_value)
                else:
                    data = func(gpu)
                self.print(msg, data)
                self.check_ret('', '', self.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.check_ret(msg, e, self.PASS):
                    raise_exception = e
            self.print('')
        if raise_exception:
            raise raise_exception
        return

    def Test_Per_GPU_With_One_Enum(self, **kwargs):
        '''
            Tests API per GPU per Enum with zero or more arguments

            Arguments:
                func_name: API to be executed
                value1_name=[(value_name, value, value_cond), ...]
            Optional:
                param1_name: Name of parameter 1
                param2_name: Name of parameter 2
        '''
        iterator = iter(kwargs.items())
        func_name, func = next(iterator)
        name1, values1 = next(iterator, (None, None))
        param1_name, param1_value = next(iterator, (None, None))
        if param1_name:
            param2_name, param2_value = next(iterator, (None, None))
        else:
            param2_name = None

        raise_exception = None
        for i, gpu in enumerate(self.processors):
            self.print_device_header(i, gpu)
            for value1_name, value1, value1_cond in values1:
                if param2_name:
                    msg = f'\t### {func_name}(gpu={i}, {name1}={value1_name}, {param1_name}={param1_value}, {param2_name}={param2_value})'
                elif param1_name:
                    msg = f'\t### {func_name}(gpu={i}, {name1}={value1_name}, {param1_name}={param1_value})'
                else:
                    msg = f'\t### {func_name}(gpu={i}, {name1}={value1_name})'
                try:
                    if param2_name:
                        data = func(gpu, value1, param1_value, param2_value)
                    elif param1_name:
                        data = func(gpu, value1, param1_value)
                    else:
                        data = func(gpu, value1)
                    self.print(msg, data)
                    self.check_ret('', '', self.PASS)
                except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                    if value1_cond != self.PASS:
                        if self.check_ret(msg, e, value1_cond):
                            raise_exception = e
                    else:
                        if self.check_ret(msg, e, self.PASS):
                            raise_exception = e
                self.print('')
        if raise_exception:
            raise raise_exception
        return

    def Test_Per_GPU_With_Two_Enums(self, **kwargs):
        '''
            Tests API per GPU per 2 Enums with zero or more arguments

            Arguments:
                func_name: API to be executed
                value1_name=[(value_name, value, value_cond), ...]
                value2_name=[(value_name, value, value_cond), ...]
            Optional:
                param1_name: Name of parameter 1
                param2_name: Name of parameter 2
        '''
        iterator = iter(kwargs.items())
        func_name, func = next(iterator)
        name1, values1 = next(iterator, (None, None))
        name2, values2 = next(iterator, (None, None))
        param1_name, param1_value = next(iterator, (None, None))
        if param1_name:
            param2_name, param2_value = next(iterator, (None, None))
        else:
            param2_name = None

        raise_exception = None
        for i, gpu in enumerate(self.processors):
            self.print_device_header(i, gpu)
            for value1_name, value1, value1_cond in values1:
                for value2_name, value2, value2_cond in values2:
                    if param2_name:
                        msg = f'\t### {func_name}(gpu={i}, {name1}={value1_name}, {name2}={value2_name}, {param1_name}={param1_value}, {param2_name}={param2_value})'
                    elif param1_name:
                        msg = f'\t### {func_name}(gpu={i}, {name1}={value1_name}, {name2}={value2_name}, {param1_name}={param1_value})'
                    else:
                        msg = f'\t### {func_name}(gpu={i}, {name1}={value1_name}, {name2}={value2_name})'
                    try:
                        if param2_name:
                            data = func(gpu, value1, value2, param1_value, param2_value)
                        elif param1_name:
                            data = func(gpu, value1, value2, param1_value)
                        else:
                            data = func(gpu, value1, value2)
                        self.print(msg, data)
                        self.check_ret('', '', self.PASS)
                    except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                        if value1_cond != self.PASS:
                            if self.check_ret(msg, e, value1_cond):
                                raise_exception = e
                        elif value2_cond != self.PASS:
                            if self.check_ret(msg, e, value2_cond):
                                raise_exception = e
                        else:
                            if self.check_ret(msg, e, self.PASS):
                                raise_exception = e
                    self.print('')
        if raise_exception:
            raise raise_exception
        return

    def Test_Per_GPU_With_GPU(self, **kwargs):
        '''
            Tests API per GPU per GPU with zero or more arguments

            Arguments:
                func_name: API to be executed
            Optional:
                param1_name: Name of parameter 1
                param2_name: Name of parameter 2
        '''
        iterator = iter(kwargs.items())
        func_name, func = next(iterator)
        param1_name, param1_value = next(iterator, (None, None))
        if param1_name:
            param2_name, param2_value = next(iterator, (None, None))
        else:
            param2_name = None

        raise_exception = None
        for i, gpu_i in enumerate(self.processors):
            self.print_device_header(i, gpu_i)
            for j, gpu_j in enumerate(self.processors):
                self.print_device_header(j, gpu_j)
                if param2_name:
                    msg = f'\t### {func_name}(gpu={i}, gpu={j}, {param1_name}={param1_value}, {param2_name}={param2_value})'
                elif param1_name:
                    msg = f'\t### {func_name}(gpu={i}, gpu={j}, {param1_name}={param1_value})'
                else:
                    msg = f'\t### {func_name}(gpu={i}, gpu={j})'
                try:
                    if param2_name:
                        data = func(gpu_i, gpu_j, param1_value, param2_value)
                    elif param1_name:
                        data = func(gpu_i, gpu_j, param1_value)
                    else:
                        data = func(gpu_i, gpu_j)
                    self.print(msg, data)
                    self.check_ret('', '', self.PASS)
                except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                    if i == j:
                        if self.check_ret(msg, e, self.FAIL):
                            raise_exception = e
                    else:
                        if self.check_ret(msg, e, self.PASS):
                            raise_exception = e
                self.print('')
        if raise_exception:
            raise raise_exception
        return

    def _skip_if_missing(self, names):
        def has_attr_recursive(obj, name):
            """Check if an attribute exists in obj or its submodules."""
            if hasattr(obj, name):
                return True
            # Try to find it in submodules
            for attr_name in dir(obj):
                try:
                    attr = getattr(obj, attr_name)
                    if hasattr(attr, "__dict__") and hasattr(attr, name):
                        return True
                except (AttributeError, ImportError):
                    pass
            return False

        missing = [name for name in names if not has_attr_recursive(amdsmi, name)]
        if missing:
            test_name = self.id().split('.')[-1]
            print_missing_msg = f"{test_name} | Missing amdsmi API(s) in amdsmi_interface.py: " + ", ".join(missing)
            print(file=sys.stderr)
            self.skipTest(f"{str(print_missing_msg)}")
        return
