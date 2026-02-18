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

amdsmi_path = os.environ.get('AMDSMI_PATH', '/opt/rocm/share/amd_smi')
if not os.path.exists(amdsmi_path):
    raise FileNotFoundError(f'AMDSMI_PATH "{amdsmi_path}" does not exist. Please set the correct path in your environment.')
sys.path.append(amdsmi_path)
try:
    import amdsmi
except ImportError:
    raise ImportError(f'Could not import the "amdsmi" module from "{amdsmi_path}"')


class Common(unittest.TestCase):
    def __init__(self, verbose, *args, **kwargs):
        self.verbose = verbose
        self.max_num_physical_devices = amdsmi.amdsmi_interface.AMDSMI_MAX_NUM_XCP * amdsmi.amdsmi_interface.AMDSMI_MAX_DEVICES
        self.PASS = 'AMDSMI_STATUS_SUCCESS'
        self.FAIL = 'AMDSMI_STATUS_INVAL'
        self.ANY_FAIL = 'ANY_FAIL'

        # Tests marked wtih either of these flags will be skipped
        # and need to be implemented later.
        self.TODO_SKIP_FAIL = True
        self.TODO_SKIP_NOT_COMPLETE = True

        self.virtualization_mode_map = \
        {
            '0': 'UNKNOWN',
            '1': 'BAREMETAL',
            '2': 'HOST',
            '3': 'GUEST',
            '4': 'PASSTHROUGH'
        }

        try:
            amdsmi.amdsmi_init()

            # Get gpu
            self.processors = amdsmi.amdsmi_get_processor_handles()
            self.virt_mode = []
            self.asic_info = []
            self.board_info = []
            #self.uuids = []
            #self.bdfs = []
            for i, gpu in enumerate(self.processors):
                #uuid = amdsmi.amdsmi_get_gpu_device_uuid(gpu)
                #self.uuids.append(uuid)
                #bdf = amdsmi.amdsmi_get_gpu_device_bdf(gpu)
                #self.bdfs.append(bdf)
                # Get virtualization mode info
                if False:
                    self.virt_mode.append(amdsmi.amdsmi_get_gpu_virtualization_mode(gpu))
                ret = amdsmi.amdsmi_get_gpu_virtualization_mode(gpu)
                mode_name = self.virtualization_mode_map[str(int(ret['mode']))]
                self.virt_mode.append({'mode': mode_name})

                # Get asic info
                self.asic_info.append(amdsmi.amdsmi_get_gpu_asic_info(gpu))
                # Get board info
                self.board_info.append(amdsmi.amdsmi_get_gpu_board_info(gpu))

            amdsmi.amdsmi_shut_down()
        except amdsmi.AmdSmiLibraryException as e:
            print(f'In class Common, Cannot get processor information, {e}')

        self.not_supported_error_codes = \
        [
            ( '2', 'AMDSMI_STATUS_NOT_SUPPORTED'),
            ( '3', 'AMDSMI_STATUS_NOT_YET_IMPLEMENTED'),
            ('49', 'AMDSMI_STATUS_NO_HSMP_MSG_SUP')
        ]

        self.error_map = \
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

        self.status_types = \
        [
            ('SUCCESS', amdsmi.AmdSmiStatus.SUCCESS, self.PASS),
            ('INVAL', amdsmi.AmdSmiStatus.INVAL, self.PASS),
            ('NOT_SUPPORTED', amdsmi.AmdSmiStatus.NOT_SUPPORTED, self.PASS),
            ('NOT_YET_IMPLEMENTED', amdsmi.AmdSmiStatus.NOT_YET_IMPLEMENTED, self.PASS),
            ('FAIL_LOAD_MODULE', amdsmi.AmdSmiStatus.FAIL_LOAD_MODULE, self.PASS),
            ('FAIL_LOAD_SYMBOL', amdsmi.AmdSmiStatus.FAIL_LOAD_SYMBOL, self.PASS),
            ('DRM_ERROR', amdsmi.AmdSmiStatus.DRM_ERROR, self.PASS),
            ('API_FAILED', amdsmi.AmdSmiStatus.API_FAILED, self.PASS),
            ('TIMEOUT', amdsmi.AmdSmiStatus.TIMEOUT, self.PASS),
            ('RETRY', amdsmi.AmdSmiStatus.RETRY, self.PASS),
            ('NO_PERM', amdsmi.AmdSmiStatus.NO_PERM, self.PASS),
            ('INTERRUPT', amdsmi.AmdSmiStatus.INTERRUPT, self.PASS),
            ('IO', amdsmi.AmdSmiStatus.IO, self.PASS),
            ('ADDRESS_FAULT', amdsmi.AmdSmiStatus.ADDRESS_FAULT, self.PASS),
            ('FILE_ERROR', amdsmi.AmdSmiStatus.FILE_ERROR, self.PASS),
            ('OUT_OF_RESOURCES', amdsmi.AmdSmiStatus.OUT_OF_RESOURCES, self.PASS),
            ('INTERNAL_EXCEPTION', amdsmi.AmdSmiStatus.INTERNAL_EXCEPTION, self.PASS),
            ('INPUT_OUT_OF_BOUNDS', amdsmi.AmdSmiStatus.INPUT_OUT_OF_BOUNDS, self.PASS),
            ('INIT_ERROR', amdsmi.AmdSmiStatus.INIT_ERROR, self.PASS),
            ('REFCOUNT_OVERFLOW', amdsmi.AmdSmiStatus.REFCOUNT_OVERFLOW, self.PASS),
            ('DIRECTORY_NOT_FOUND', amdsmi.AmdSmiStatus.DIRECTORY_NOT_FOUND, self.PASS),
            ('BUSY', amdsmi.AmdSmiStatus.BUSY, self.PASS),
            ('NOT_FOUND', amdsmi.AmdSmiStatus.NOT_FOUND, self.PASS),
            ('NOT_INIT', amdsmi.AmdSmiStatus.NOT_INIT, self.PASS),
            ('NO_SLOT', amdsmi.AmdSmiStatus.NO_SLOT, self.PASS),
            ('DRIVER_NOT_LOADED', amdsmi.AmdSmiStatus.DRIVER_NOT_LOADED, self.PASS),
            ('MORE_DATA', amdsmi.AmdSmiStatus.MORE_DATA, self.PASS),
            ('NO_DATA', amdsmi.AmdSmiStatus.NO_DATA, self.PASS),
            ('INSUFFICIENT_SIZE', amdsmi.AmdSmiStatus.INSUFFICIENT_SIZE, self.PASS),
            ('UNEXPECTED_SIZE', amdsmi.AmdSmiStatus.UNEXPECTED_SIZE, self.PASS),
            ('UNEXPECTED_DATA', amdsmi.AmdSmiStatus.UNEXPECTED_DATA, self.PASS),
            ('NON_AMD_CPU', amdsmi.AmdSmiStatus.NON_AMD_CPU, self.PASS),
            ('NO_ENERGY_DRV', amdsmi.AmdSmiStatus.NO_ENERGY_DRV, self.PASS),
            ('NO_MSR_DRV', amdsmi.AmdSmiStatus.NO_MSR_DRV, self.PASS),
            ('NO_HSMP_DRV', amdsmi.AmdSmiStatus.NO_HSMP_DRV, self.PASS),
            ('NO_HSMP_SUP', amdsmi.AmdSmiStatus.NO_HSMP_SUP, self.PASS),
            ('NO_HSMP_MSG_SUP', amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP, self.PASS),
            ('HSMP_TIMEOUT', amdsmi.AmdSmiStatus.HSMP_TIMEOUT, self.PASS),
            ('NO_DRV', amdsmi.AmdSmiStatus.NO_DRV, self.PASS),
            ('FILE_NOT_FOUND', amdsmi.AmdSmiStatus.FILE_NOT_FOUND, self.PASS),
            ('ARG_PTR_NULL', amdsmi.AmdSmiStatus.ARG_PTR_NULL, self.PASS),
            ('AMDGPU_RESTART_ERR', amdsmi.AmdSmiStatus.AMDGPU_RESTART_ERR, self.PASS),
            ('SETTING_UNAVAILABLE', amdsmi.AmdSmiStatus.SETTING_UNAVAILABLE, self.PASS),
            ('CORRUPTED_EEPROM', amdsmi.AmdSmiStatus.CORRUPTED_EEPROM, self.PASS),
            ('MAP_ERROR', amdsmi.AmdSmiStatus.MAP_ERROR, self.PASS),
            ('UNKNOWN_ERROR', amdsmi.AmdSmiStatus.UNKNOWN_ERROR, self.PASS)
        ]

        self.clk_types = \
        [
            ('SYS', amdsmi.AmdSmiClkType.SYS, self.PASS),
            ('GFX', amdsmi.AmdSmiClkType.GFX, self.PASS),
            ('DF', amdsmi.AmdSmiClkType.DF, self.PASS),
            ('DCEF', amdsmi.AmdSmiClkType.DCEF, [self.PASS, self.FAIL]),
            ('SOC', amdsmi.AmdSmiClkType.SOC, self.PASS),
            ('MEM', amdsmi.AmdSmiClkType.MEM, self.PASS),
            ('PCIE', amdsmi.AmdSmiClkType.PCIE, [self.PASS, self.FAIL]),
            ('VCLK0', amdsmi.AmdSmiClkType.VCLK0, self.PASS),
            ('VCLK1', amdsmi.AmdSmiClkType.VCLK1, self.PASS),
            ('DCLK0', amdsmi.AmdSmiClkType.DCLK0, self.PASS),
            ('DCLK1', amdsmi.AmdSmiClkType.DCLK1, self.PASS)
        ]

        self.clk_limit_types = \
        [
            ('MIN', amdsmi.AmdSmiClkLimitType.MIN, self.PASS),
            ('MAX', amdsmi.AmdSmiClkLimitType.MAX, self.PASS)
        ]

        self.io_bw_encodings = \
        [
            ('AGG_BW0', amdsmi.amdsmi_interface.amdsmi_wrapper.AGG_BW0, self.PASS),
            ('RD_BW0', amdsmi.amdsmi_interface.amdsmi_wrapper.RD_BW0, self.PASS),
            ('WR_BW0', amdsmi.amdsmi_interface.amdsmi_wrapper.WR_BW0, self.PASS)
        ]

        self.gpu_blocks = \
        [
            ('INVALID', amdsmi.AmdSmiGpuBlock.INVALID, self.FAIL),
            ('UMC', amdsmi.AmdSmiGpuBlock.UMC, self.PASS),
            ('SDMA', amdsmi.AmdSmiGpuBlock.SDMA, self.PASS),
            ('GFX', amdsmi.AmdSmiGpuBlock.GFX, self.PASS),
            ('MMHUB', amdsmi.AmdSmiGpuBlock.MMHUB, self.PASS),
            ('ATHUB', amdsmi.AmdSmiGpuBlock.ATHUB, self.PASS),
            ('PCIE_BIF', amdsmi.AmdSmiGpuBlock.PCIE_BIF, self.PASS),
            ('HDP', amdsmi.AmdSmiGpuBlock.HDP, self.PASS),
            ('XGMI_WAFL', amdsmi.AmdSmiGpuBlock.XGMI_WAFL, self.PASS),
            ('DF', amdsmi.AmdSmiGpuBlock.DF, self.PASS),
            ('SMN', amdsmi.AmdSmiGpuBlock.SMN, self.PASS),
            ('SEM', amdsmi.AmdSmiGpuBlock.SEM, self.PASS),
            ('MP0', amdsmi.AmdSmiGpuBlock.MP0, self.PASS),
            ('MP1', amdsmi.AmdSmiGpuBlock.MP1, self.PASS),
            ('FUSE', amdsmi.AmdSmiGpuBlock.FUSE, self.PASS),
            ('MCA', amdsmi.AmdSmiGpuBlock.MCA, self.PASS),
            ('VCN', amdsmi.AmdSmiGpuBlock.VCN, self.PASS),
            ('JPEG', amdsmi.AmdSmiGpuBlock.JPEG, self.PASS),
            ('IH', amdsmi.AmdSmiGpuBlock.IH, self.PASS),
            ('MPIO', amdsmi.AmdSmiGpuBlock.MPIO, self.PASS),
            ('RESERVED', amdsmi.AmdSmiGpuBlock.RESERVED, self.FAIL)
        ]

        self.memory_types = \
        [
            ('VRAM', amdsmi.AmdSmiMemoryType.VRAM, self.PASS),
            ('VIS_VRAM', amdsmi.AmdSmiMemoryType.VIS_VRAM, self.PASS),
            ('GTT', amdsmi.AmdSmiMemoryType.GTT, self.PASS)
        ]

        self.reg_types = \
        [
            ('XGMI', amdsmi.AmdSmiRegType.XGMI, self.PASS),
            ('WAFL', amdsmi.AmdSmiRegType.WAFL, self.PASS),
            ('PCIE', amdsmi.AmdSmiRegType.PCIE, self.PASS),
            ('USR', amdsmi.AmdSmiRegType.USR, self.PASS),
            ('USR1', amdsmi.AmdSmiRegType.USR1, self.PASS)
        ]

        self.voltage_metrics = \
        [
            ('CURRENT', amdsmi.AmdSmiVoltageMetric.CURRENT, self.PASS),
            ('MAX', amdsmi.AmdSmiVoltageMetric.MAX, self.PASS),
            ('MIN_CRIT', amdsmi.AmdSmiVoltageMetric.MIN_CRIT, self.PASS),
            ('MIN', amdsmi.AmdSmiVoltageMetric.MIN, self.PASS),
            ('MAX_CRIT', amdsmi.AmdSmiVoltageMetric.MAX_CRIT, self.PASS),
            ('AVERAGE', amdsmi.AmdSmiVoltageMetric.AVERAGE, self.PASS),
            ('LOWEST', amdsmi.AmdSmiVoltageMetric.LOWEST, self.PASS),
            ('HIGHEST', amdsmi.AmdSmiVoltageMetric.HIGHEST, self.PASS)
        ]

        self.voltage_types = \
        [
            ('VDDGFX', amdsmi.AmdSmiVoltageType.VDDGFX, self.PASS),
            ('VDDBOARD', amdsmi.AmdSmiVoltageType.VDDBOARD, self.PASS),
            ('INVALID', amdsmi.AmdSmiVoltageType.INVALID, self.FAIL)
        ]

        self.link_types = \
        [
            ('AMDSMI_LINK_TYPE_INTERNAL', amdsmi.AmdSmiLinkType.AMDSMI_LINK_TYPE_INTERNAL, self.PASS),
            ('AMDSMI_LINK_TYPE_XGMI', amdsmi.AmdSmiLinkType.AMDSMI_LINK_TYPE_XGMI, self.PASS),
            ('AMDSMI_LINK_TYPE_PCIE', amdsmi.AmdSmiLinkType.AMDSMI_LINK_TYPE_PCIE, self.PASS),
            ('AMDSMI_LINK_TYPE_NOT_APPLICABLE', amdsmi.AmdSmiLinkType.AMDSMI_LINK_TYPE_NOT_APPLICABLE, self.FAIL),
            ('AMDSMI_LINK_TYPE_UNKNOWN', amdsmi.AmdSmiLinkType.AMDSMI_LINK_TYPE_UNKNOWN, self.FAIL)
        ]

        self.temperature_types = \
        [
            ('EDGE', amdsmi.AmdSmiTemperatureType.EDGE, self.PASS),
            ('HOTSPOT', amdsmi.AmdSmiTemperatureType.HOTSPOT, self.PASS),
            ('JUNCTION', amdsmi.AmdSmiTemperatureType.JUNCTION, self.PASS),
            ('VRAM', amdsmi.AmdSmiTemperatureType.VRAM, self.PASS),
            ('HBM_0', amdsmi.AmdSmiTemperatureType.HBM_0, self.PASS),
            ('HBM_1', amdsmi.AmdSmiTemperatureType.HBM_1, self.PASS),
            ('HBM_2', amdsmi.AmdSmiTemperatureType.HBM_2, self.PASS),
            ('HBM_3', amdsmi.AmdSmiTemperatureType.HBM_3, self.PASS),
            ('PLX', amdsmi.AmdSmiTemperatureType.PLX, self.PASS)
        ]

        self.temperature_metrics = \
        [
            ('CURRENT', amdsmi.AmdSmiTemperatureMetric.CURRENT, self.PASS),
            ('MAX', amdsmi.AmdSmiTemperatureMetric.MAX, self.PASS),
            ('MIN', amdsmi.AmdSmiTemperatureMetric.MIN, self.PASS),
            ('MAX_HYST', amdsmi.AmdSmiTemperatureMetric.MAX_HYST, self.PASS),
            ('MIN_HYST', amdsmi.AmdSmiTemperatureMetric.MIN_HYST, self.PASS),
            ('CRITICAL', amdsmi.AmdSmiTemperatureMetric.CRITICAL, self.PASS),
            ('CRITICAL_HYST', amdsmi.AmdSmiTemperatureMetric.CRITICAL_HYST, self.PASS),
            ('EMERGENCY', amdsmi.AmdSmiTemperatureMetric.EMERGENCY, self.PASS),
            ('EMERGENCY_HYST', amdsmi.AmdSmiTemperatureMetric.EMERGENCY_HYST, self.PASS),
            ('CRIT_MIN', amdsmi.AmdSmiTemperatureMetric.CRIT_MIN, self.PASS),
            ('CRIT_MIN_HYST', amdsmi.AmdSmiTemperatureMetric.CRIT_MIN_HYST, self.PASS),
            ('OFFSET', amdsmi.AmdSmiTemperatureMetric.OFFSET, self.PASS),
            ('LOWEST', amdsmi.AmdSmiTemperatureMetric.LOWEST, self.PASS),
            ('HIGHEST', amdsmi.AmdSmiTemperatureMetric.HIGHEST, self.PASS)
        ]

        self.utilization_counter_types = \
        [
            ('COARSE_GRAIN_GFX_ACTIVITY', amdsmi.AmdSmiUtilizationCounterType.COARSE_GRAIN_GFX_ACTIVITY, self.PASS),
            ('COARSE_GRAIN_MEM_ACTIVITY', amdsmi.AmdSmiUtilizationCounterType.COARSE_GRAIN_MEM_ACTIVITY, self.PASS),
            ('COARSE_DECODER_ACTIVITY', amdsmi.AmdSmiUtilizationCounterType.COARSE_DECODER_ACTIVITY, self.PASS),
            ('FINE_GRAIN_GFX_ACTIVITY', amdsmi.AmdSmiUtilizationCounterType.FINE_GRAIN_GFX_ACTIVITY, self.PASS),
            ('FINE_GRAIN_MEM_ACTIVITY', amdsmi.AmdSmiUtilizationCounterType.FINE_GRAIN_MEM_ACTIVITY, self.PASS),
            ('FINE_DECODER_ACTIVITY', amdsmi.AmdSmiUtilizationCounterType.FINE_DECODER_ACTIVITY, self.PASS),
            ('UTILIZATION_COUNTER_FIRST', amdsmi.AmdSmiUtilizationCounterType.UTILIZATION_COUNTER_FIRST, self.PASS),
            ('UTILIZATION_COUNTER_LAST', amdsmi.AmdSmiUtilizationCounterType.UTILIZATION_COUNTER_LAST, self.PASS),
            ('UTILIZATION_COUNTER_BAD', 100, self.FAIL)
        ]

        self.event_groups = \
        [
            ('XGMI', amdsmi.AmdSmiEventGroup.XGMI, self.PASS),
            ('XGMI_DATA_OUT', amdsmi.AmdSmiEventGroup.XGMI_DATA_OUT, self.PASS),
            ('GRP_INVALID', amdsmi.AmdSmiEventGroup.GRP_INVALID, self.FAIL)
        ]

        self.event_types = \
        [
            ('XGMI_0_NOP_TX', amdsmi.AmdSmiEventType.XGMI_0_NOP_TX, self.PASS),
            ('XGMI_0_REQUEST_TX', amdsmi.AmdSmiEventType.XGMI_0_REQUEST_TX, self.PASS),
            ('XGMI_0_RESPONSE_TX', amdsmi.AmdSmiEventType.XGMI_0_RESPONSE_TX, self.PASS),
            ('XGMI_0_BEATS_TX', amdsmi.AmdSmiEventType.XGMI_0_BEATS_TX, self.PASS),
            ('XGMI_1_NOP_TX', amdsmi.AmdSmiEventType.XGMI_1_NOP_TX, self.PASS),
            ('XGMI_1_REQUEST_TX', amdsmi.AmdSmiEventType.XGMI_1_REQUEST_TX, self.PASS),
            ('XGMI_1_RESPONSE_TX', amdsmi.AmdSmiEventType.XGMI_1_RESPONSE_TX, self.PASS),
            ('XGMI_1_BEATS_TX', amdsmi.AmdSmiEventType.XGMI_1_BEATS_TX, self.PASS),
            ('XGMI_DATA_OUT_0', amdsmi.AmdSmiEventType.XGMI_DATA_OUT_0, self.PASS),
            ('XGMI_DATA_OUT_1', amdsmi.AmdSmiEventType.XGMI_DATA_OUT_1, self.PASS),
            ('XGMI_DATA_OUT_2', amdsmi.AmdSmiEventType.XGMI_DATA_OUT_2, self.PASS),
            ('XGMI_DATA_OUT_3', amdsmi.AmdSmiEventType.XGMI_DATA_OUT_3, self.PASS),
            ('XGMI_DATA_OUT_4', amdsmi.AmdSmiEventType.XGMI_DATA_OUT_4, self.PASS),
            ('XGMI_DATA_OUT_5', amdsmi.AmdSmiEventType.XGMI_DATA_OUT_5, self.PASS)
        ]

        self.counter_commands = \
        [
            ('CMD_START', amdsmi.AmdSmiCounterCommand.CMD_START, self.PASS),
            ('CMD_STOP', amdsmi.AmdSmiCounterCommand.CMD_STOP, self.PASS)
        ]

        self.compute_partition_types = \
        [
            ('SPX', amdsmi.AmdSmiComputePartitionType.SPX, self.PASS),
            ('DPX', amdsmi.AmdSmiComputePartitionType.DPX, self.PASS),
            ('TPX', amdsmi.AmdSmiComputePartitionType.TPX, self.PASS),
            ('QPX', amdsmi.AmdSmiComputePartitionType.QPX, self.PASS),
            ('CPX', amdsmi.AmdSmiComputePartitionType.CPX, self.PASS),
            ('INVALID', amdsmi.AmdSmiComputePartitionType.INVALID, self.FAIL)
        ]

        self.memory_partition_types = \
        [
            ('NPS1', amdsmi.AmdSmiMemoryPartitionType.NPS1, self.PASS),
            ('NPS2', amdsmi.AmdSmiMemoryPartitionType.NPS2, self.PASS),
            ('NPS4', amdsmi.AmdSmiMemoryPartitionType.NPS4, self.PASS),
            ('NPS8', amdsmi.AmdSmiMemoryPartitionType.NPS8, self.PASS),
            ('UNKNOWN', amdsmi.AmdSmiMemoryPartitionType.UNKNOWN, self.FAIL)
        ]

        self.freq_inds = \
        [
            ('MIN', amdsmi.AmdSmiFreqInd.MIN, self.PASS),
            ('MAX', amdsmi.AmdSmiFreqInd.MAX, self.PASS),
            ('INVALID', amdsmi.AmdSmiFreqInd.INVALID, self.FAIL)
        ]

        self.power_profile_preset_masks = \
        [
            ('CUSTOM_MASK', amdsmi.AmdSmiPowerProfilePresetMasks.CUSTOM_MASK, self.PASS),
            ('VIDEO_MASK', amdsmi.AmdSmiPowerProfilePresetMasks.VIDEO_MASK, self.PASS),
            ('POWER_SAVING_MASK', amdsmi.AmdSmiPowerProfilePresetMasks.POWER_SAVING_MASK, self.PASS),
            ('COMPUTE_MASK', amdsmi.AmdSmiPowerProfilePresetMasks.COMPUTE_MASK, self.PASS),
            ('VR_MASK', amdsmi.AmdSmiPowerProfilePresetMasks.VR_MASK, self.PASS),
            ('THREE_D_FULL_SCR_MASK', amdsmi.AmdSmiPowerProfilePresetMasks.THREE_D_FULL_SCR_MASK, self.PASS),
            ('BOOTUP_DEFAULT', amdsmi.AmdSmiPowerProfilePresetMasks.BOOTUP_DEFAULT, self.PASS)
        ]

        self.processor_types = \
        [
            ('UNKNOWN', amdsmi.AmdSmiProcessorType.UNKNOWN, self.FAIL),
            ('AMD_GPU', amdsmi.AmdSmiProcessorType.AMD_GPU, self.PASS),
            ('AMD_CPU', amdsmi.AmdSmiProcessorType.AMD_CPU, self.PASS),
            ('NON_AMD_GPU', amdsmi.AmdSmiProcessorType.NON_AMD_GPU, self.PASS),
            ('NON_AMD_CPU', amdsmi.AmdSmiProcessorType.NON_AMD_CPU, self.PASS),
            ('AMD_CPU_CORE', amdsmi.AmdSmiProcessorType.AMD_CPU_CORE, self.PASS),
            ('AMD_APU', amdsmi.AmdSmiProcessorType.AMD_APU, self.PASS)
        ]

        self.dev_perf_levels = \
        [
            ('AUTO', amdsmi.AmdSmiDevPerfLevel.AUTO, self.PASS),
            ('LOW', amdsmi.AmdSmiDevPerfLevel.LOW, self.PASS),
            ('HIGH', amdsmi.AmdSmiDevPerfLevel.HIGH, self.PASS),
            ('MANUAL', amdsmi.AmdSmiDevPerfLevel.MANUAL, self.PASS),
            ('STABLE_STD', amdsmi.AmdSmiDevPerfLevel.STABLE_STD, self.PASS),
            ('STABLE_PEAK', amdsmi.AmdSmiDevPerfLevel.STABLE_PEAK, self.PASS),
            ('STABLE_MIN_MCLK', amdsmi.AmdSmiDevPerfLevel.STABLE_MIN_MCLK, self.PASS),
            ('STABLE_MIN_SCLK', amdsmi.AmdSmiDevPerfLevel.STABLE_MIN_SCLK, self.PASS),
            ('DETERMINISM', amdsmi.AmdSmiDevPerfLevel.DETERMINISM, self.PASS),
            ('UNKNOWN', amdsmi.AmdSmiDevPerfLevel.UNKNOWN, self.FAIL)
        ]

    def print(self, msg, data=None):
        if self.verbose == 2:
            if data is None:
                print(msg, flush=True)
            elif any(data in value for value in self.not_supported_error_codes):
                print(f'{msg} {data}', flush=True)
            else:
                if isinstance(data, str) and data in self.error_map.values():
                    print(msg, end='')
                else:
                    print(msg)
                if isinstance(data, dict) or isinstance(data, list):
                    print(json.dumps(data, sort_keys=False, indent=4), flush=True)
                else:
                    print(data)
        return

    def print_func_name(self, msg=None):
        if self.verbose == 2:
            stk = inspect.stack()
            if stk[1].function == '_callSetUp':
                return
            print(f'\n## {stk[1].function}()', flush=True)
            if msg:
                print(msg, flush=True)
        return

    def print_device_header(self, i, gpu):
        # Print virtualization mode info
        msg = f'virtualization mode(gpu={i})'
        self.print(f'\t{msg}')
        mode = self.virt_mode[i]['mode']
        self.print(f'\t\tmode : {mode}')
        # Print asic info
        msg = f'asic info(gpu={i})'
        self.print(f'\t{msg}')
        for key, value in self.asic_info[i].items():
            self.print(f'\t\t{key} : {value}')
        # Print board info
        msg = f'board info(gpu={i})'
        self.print(f'\t{msg}')
        for key, value in self.board_info[i].items():
            self.print(f'\t\t{key} : {value}')
        return

    def get_error_code(self, exc):
        error_code = '-1'
        error_code_name = 'UNKNOWN_ERROR'
        if hasattr(exc, 'get_error_code'):
            error_code = str(exc.get_error_code())
            if error_code in self.error_map:
                error_code_name = self.error_map[error_code]
        return (error_code, error_code_name)

    def check_ret(self, msg, exc, expected_code_name=None, printIt=True):
        if isinstance(exc, str) and not len(exc):
            error_code_name = expected_code_name
            if error_code_name in self.error_map.values():
                for key, value in self.error_map.items():
                    if value == error_code_name:
                        error_code = key
                        break
            else:
                error_code = '-1'
        elif hasattr(exc, 'get_error_code'):
            error_code, error_code_name = self.get_error_code(exc)
        else:
            error_code = str(exc).split(':')[0]
            error_code_name = 'AMDSMI_STATUS_INVAL'

        # Check for when there are multiple passing conditions
        if isinstance(expected_code_name, list):
            for ec in expected_code_name:
                rc = self.check_ret(msg, exc, ec, False)  # Do not print msg, otherwise multiple msgs printed
                if not rc:
                    rc = self.check_ret(msg, exc, ec) # Call check again so msg is printed
                    return rc

            # No expected results found
            if msg:
                print(f'{msg}\n', end='')
            print(f'Test FAILED with expected results {expected_code_name} but received {error_code_name}', flush=True)
            return True

        # Check for single passing condition
        status_msg = ''
        status_ret = False
        if any(error_code in value for value in self.not_supported_error_codes):
            status_msg = f'\tAPI RETURNED {error_code_name}'
        elif error_code_name == expected_code_name:
            status_msg = f'\tTest PASSED with expected result {expected_code_name}'
        elif error_code_name != self.PASS and expected_code_name == self.ANY_FAIL:
            status_msg = f'\tTest PASSED with expected result {expected_code_name} and received {error_code_name}'
        else:
            status_msg = f'\tTest FAILED with expected result {expected_code_name} but received {error_code_name}'
            status_ret = True
        if self.verbose == 2 and printIt:
            if msg:
                print(f'{msg}\n', end='')
            print(f'{status_msg}', flush=True)
        return status_ret

    def _skip_if_missing(self, names):
        def has_attr_recursive(obj, name):
            """Check if an attribute exists in obj or its submodules."""
            if hasattr(obj, name):
                return True
            # Try to find it in submodules
            for attr_name in dir(obj):
                try:
                    attr = getattr(obj, attr_name)
                    if hasattr(attr, '__dict__') and hasattr(attr, name):
                        return True
                except (AttributeError, ImportError):
                    pass
            return False
        
        missing = [name for name in names if not has_attr_recursive(amdsmi, name)]
        if missing:
            test_name = self.id().split('.')[-1]
            print_missing_msg = f"{test_name} | Missing amdsmi API(s) in amdsmi_interface.py: " + ", ".join(missing)
            print(f"\n")
            self.skipTest(f"{str(print_missing_msg)}")
