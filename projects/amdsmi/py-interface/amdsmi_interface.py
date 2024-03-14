#
# Copyright (C) 2023 Advanced Micro Devices. All rights reserved.
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
#

import ctypes
import re
from typing import Union, Any, Dict, List
from enum import IntEnum
from collections.abc import Iterable

from . import amdsmi_wrapper
from .amdsmi_exception import *
import sys
import math
from time import localtime, asctime, time

MAX_NUM_PROCESSES = 1024

# gpu metrics macros defined in amdsmi.h
AMDSMI_NUM_HBM_INSTANCES = 4
AMDSMI_MAX_NUM_VCN = 4
AMDSMI_MAX_NUM_CLKS = 4
AMDSMI_MAX_NUM_XGMI_LINKS = 8
AMDSMI_MAX_NUM_GFX_CLKS = 8


class AmdSmiInitFlags(IntEnum):
    INIT_ALL_PROCESSORS = amdsmi_wrapper.AMDSMI_INIT_ALL_PROCESSORS
    INIT_AMD_CPUS = amdsmi_wrapper.AMDSMI_INIT_AMD_CPUS
    INIT_AMD_GPUS = amdsmi_wrapper.AMDSMI_INIT_AMD_GPUS
    INIT_AMD_APUS = amdsmi_wrapper.AMDSMI_INIT_AMD_APUS
    INIT_NON_AMD_CPUS = amdsmi_wrapper.AMDSMI_INIT_NON_AMD_CPUS
    INIT_NON_AMD_GPUS = amdsmi_wrapper.AMDSMI_INIT_NON_AMD_GPUS


class AmdSmiContainerTypes(IntEnum):
    LXC = amdsmi_wrapper.CONTAINER_LXC
    DOCKER = amdsmi_wrapper.CONTAINER_DOCKER


class AmdSmiDeviceType(IntEnum):
    UNKNOWN_DEVICE = amdsmi_wrapper.UNKNOWN
    AMD_GPU_DEVICE = amdsmi_wrapper.AMD_GPU
    AMD_CPU_DEVICE = amdsmi_wrapper.AMD_CPU
    NON_AMD_GPU_DEVICE = amdsmi_wrapper.NON_AMD_GPU
    NON_AMD_CPU_DEVICE = amdsmi_wrapper.NON_AMD_CPU


class AmdSmiMmIp(IntEnum):
    UVD = amdsmi_wrapper.AMDSMI_MM_UVD
    VCE = amdsmi_wrapper.AMDSMI_MM_VCE
    VCN = amdsmi_wrapper.AMDSMI_MM_VCN


class AmdSmiFwBlock(IntEnum):
    FW_ID_SMU = amdsmi_wrapper.FW_ID_SMU
    FW_ID_CP_CE = amdsmi_wrapper.FW_ID_CP_CE
    FW_ID_CP_PFP = amdsmi_wrapper.FW_ID_CP_PFP
    FW_ID_CP_ME = amdsmi_wrapper.FW_ID_CP_ME
    FW_ID_CP_MEC_JT1 = amdsmi_wrapper.FW_ID_CP_MEC_JT1
    FW_ID_CP_MEC_JT2 = amdsmi_wrapper.FW_ID_CP_MEC_JT2
    FW_ID_CP_MEC1 = amdsmi_wrapper.FW_ID_CP_MEC1
    FW_ID_CP_MEC2 = amdsmi_wrapper.FW_ID_CP_MEC2
    FW_ID_RLC = amdsmi_wrapper.FW_ID_RLC
    FW_ID_SDMA0 = amdsmi_wrapper.FW_ID_SDMA0
    FW_ID_SDMA1 = amdsmi_wrapper.FW_ID_SDMA1
    FW_ID_SDMA2 = amdsmi_wrapper.FW_ID_SDMA2
    FW_ID_SDMA3 = amdsmi_wrapper.FW_ID_SDMA3
    FW_ID_SDMA4 = amdsmi_wrapper.FW_ID_SDMA4
    FW_ID_SDMA5 = amdsmi_wrapper.FW_ID_SDMA5
    FW_ID_SDMA6 = amdsmi_wrapper.FW_ID_SDMA6
    FW_ID_SDMA7 = amdsmi_wrapper.FW_ID_SDMA7
    FW_ID_VCN = amdsmi_wrapper.FW_ID_VCN
    FW_ID_UVD = amdsmi_wrapper.FW_ID_UVD
    FW_ID_VCE = amdsmi_wrapper.FW_ID_VCE
    FW_ID_ISP = amdsmi_wrapper.FW_ID_ISP
    FW_ID_DMCU_ERAM = amdsmi_wrapper.FW_ID_DMCU_ERAM
    FW_ID_DMCU_ISR = amdsmi_wrapper.FW_ID_DMCU_ISR
    FW_ID_RLC_RESTORE_LIST_GPM_MEM = amdsmi_wrapper.FW_ID_RLC_RESTORE_LIST_GPM_MEM
    FW_ID_RLC_RESTORE_LIST_SRM_MEM = amdsmi_wrapper.FW_ID_RLC_RESTORE_LIST_SRM_MEM
    FW_ID_RLC_RESTORE_LIST_CNTL = amdsmi_wrapper.FW_ID_RLC_RESTORE_LIST_CNTL
    FW_ID_RLC_V = amdsmi_wrapper.FW_ID_RLC_V
    FW_ID_MMSCH = amdsmi_wrapper.FW_ID_MMSCH
    FW_ID_PSP_SYSDRV = amdsmi_wrapper.FW_ID_PSP_SYSDRV
    FW_ID_PSP_SOSDRV = amdsmi_wrapper.FW_ID_PSP_SOSDRV
    FW_ID_PSP_TOC = amdsmi_wrapper.FW_ID_PSP_TOC
    FW_ID_PSP_KEYDB = amdsmi_wrapper.FW_ID_PSP_KEYDB
    FW_ID_DFC = amdsmi_wrapper.FW_ID_DFC
    FW_ID_PSP_SPL = amdsmi_wrapper.FW_ID_PSP_SPL
    FW_ID_DRV_CAP = amdsmi_wrapper.FW_ID_DRV_CAP
    FW_ID_MC = amdsmi_wrapper.FW_ID_MC
    FW_ID_PSP_BL = amdsmi_wrapper.FW_ID_PSP_BL
    FW_ID_CP_PM4 = amdsmi_wrapper.FW_ID_CP_PM4
    FW_ID_RLC_P = amdsmi_wrapper.FW_ID_RLC_P
    FW_ID_SEC_POLICY_STAGE2 = amdsmi_wrapper.FW_ID_SEC_POLICY_STAGE2
    FW_ID_REG_ACCESS_WHITELIST = amdsmi_wrapper.FW_ID_REG_ACCESS_WHITELIST
    FW_ID_IMU_DRAM = amdsmi_wrapper.FW_ID_IMU_DRAM
    FW_ID_IMU_IRAM = amdsmi_wrapper.FW_ID_IMU_IRAM
    FW_ID_SDMA_TH0 = amdsmi_wrapper.FW_ID_SDMA_TH0
    FW_ID_SDMA_TH1 = amdsmi_wrapper.FW_ID_SDMA_TH1
    FW_ID_CP_MES = amdsmi_wrapper.FW_ID_CP_MES
    FW_ID_MES_STACK = amdsmi_wrapper.FW_ID_MES_STACK
    FW_ID_MES_THREAD1 = amdsmi_wrapper.FW_ID_MES_THREAD1
    FW_ID_MES_THREAD1_STACK = amdsmi_wrapper.FW_ID_MES_THREAD1_STACK
    FW_ID_RLX6 = amdsmi_wrapper.FW_ID_RLX6
    FW_ID_RLX6_DRAM_BOOT = amdsmi_wrapper.FW_ID_RLX6_DRAM_BOOT
    FW_ID_RS64_ME = amdsmi_wrapper.FW_ID_RS64_ME
    FW_ID_RS64_ME_P0_DATA = amdsmi_wrapper.FW_ID_RS64_ME_P0_DATA
    FW_ID_RS64_ME_P1_DATA = amdsmi_wrapper.FW_ID_RS64_ME_P1_DATA
    FW_ID_RS64_PFP = amdsmi_wrapper.FW_ID_RS64_PFP
    FW_ID_RS64_PFP_P0_DATA = amdsmi_wrapper.FW_ID_RS64_PFP_P0_DATA
    FW_ID_RS64_PFP_P1_DATA = amdsmi_wrapper.FW_ID_RS64_PFP_P1_DATA
    FW_ID_RS64_MEC = amdsmi_wrapper.FW_ID_RS64_MEC
    FW_ID_RS64_MEC_P0_DATA = amdsmi_wrapper.FW_ID_RS64_MEC_P0_DATA
    FW_ID_RS64_MEC_P1_DATA = amdsmi_wrapper.FW_ID_RS64_MEC_P1_DATA
    FW_ID_RS64_MEC_P2_DATA = amdsmi_wrapper.FW_ID_RS64_MEC_P2_DATA
    FW_ID_RS64_MEC_P3_DATA = amdsmi_wrapper.FW_ID_RS64_MEC_P3_DATA
    FW_ID_PPTABLE = amdsmi_wrapper.FW_ID_PPTABLE
    FW_ID_PSP_SOC = amdsmi_wrapper.FW_ID_PSP_SOC
    FW_ID_PSP_DBG = amdsmi_wrapper.FW_ID_PSP_DBG
    FW_ID_PSP_INTF = amdsmi_wrapper.FW_ID_PSP_INTF
    FW_ID_RLX6_CORE1 = amdsmi_wrapper.FW_ID_RLX6_CORE1
    FW_ID_RLX6_DRAM_BOOT_CORE1 = amdsmi_wrapper.FW_ID_RLX6_DRAM_BOOT_CORE1
    FW_ID_RLCV_LX7 = amdsmi_wrapper.FW_ID_RLCV_LX7
    FW_ID_RLC_SAVE_RESTORE_LIST = amdsmi_wrapper.FW_ID_RLC_SAVE_RESTORE_LIST
    FW_ID_ASD = amdsmi_wrapper.FW_ID_ASD
    FW_ID_TA_RAS = amdsmi_wrapper.FW_ID_TA_RAS
    FW_ID_TA_XGMI = amdsmi_wrapper.FW_ID_TA_XGMI
    FW_ID_RLC_SRLG = amdsmi_wrapper.FW_ID_RLC_SRLG
    FW_ID_RLC_SRLS = amdsmi_wrapper.FW_ID_RLC_SRLS
    FW_ID_PM = amdsmi_wrapper.FW_ID_PM
    FW_ID_DMCU = amdsmi_wrapper.FW_ID_DMCU


class AmdSmiClkType(IntEnum):
    SYS = amdsmi_wrapper.CLK_TYPE_SYS
    GFX = amdsmi_wrapper.CLK_TYPE_GFX
    DF = amdsmi_wrapper.CLK_TYPE_DF
    DCEF = amdsmi_wrapper.CLK_TYPE_DCEF
    SOC = amdsmi_wrapper.CLK_TYPE_SOC
    MEM = amdsmi_wrapper.CLK_TYPE_MEM
    PCIE = amdsmi_wrapper.CLK_TYPE_PCIE
    VCLK0 = amdsmi_wrapper.CLK_TYPE_VCLK0
    VCLK1 = amdsmi_wrapper.CLK_TYPE_VCLK1
    DCLK0 = amdsmi_wrapper.CLK_TYPE_DCLK0
    DCLK1 = amdsmi_wrapper.CLK_TYPE_DCLK1


class AmdSmiTemperatureType(IntEnum):
    EDGE = amdsmi_wrapper.TEMPERATURE_TYPE_EDGE
    HOTSPOT = amdsmi_wrapper.TEMPERATURE_TYPE_HOTSPOT
    JUNCTION = amdsmi_wrapper.TEMPERATURE_TYPE_JUNCTION
    VRAM = amdsmi_wrapper.TEMPERATURE_TYPE_VRAM
    HBM_0 = amdsmi_wrapper.TEMPERATURE_TYPE_HBM_0
    HBM_1 = amdsmi_wrapper.TEMPERATURE_TYPE_HBM_1
    HBM_2 = amdsmi_wrapper.TEMPERATURE_TYPE_HBM_2
    HBM_3 = amdsmi_wrapper.TEMPERATURE_TYPE_HBM_3
    PLX = amdsmi_wrapper.TEMPERATURE_TYPE_PLX


class AmdSmiDevPerfLevel(IntEnum):
    AUTO = amdsmi_wrapper.AMDSMI_DEV_PERF_LEVEL_AUTO
    LOW = amdsmi_wrapper.AMDSMI_DEV_PERF_LEVEL_LOW
    HIGH = amdsmi_wrapper.AMDSMI_DEV_PERF_LEVEL_HIGH
    MANUAL = amdsmi_wrapper.AMDSMI_DEV_PERF_LEVEL_MANUAL
    STABLE_STD = amdsmi_wrapper.AMDSMI_DEV_PERF_LEVEL_STABLE_STD
    STABLE_PEAK = amdsmi_wrapper.AMDSMI_DEV_PERF_LEVEL_STABLE_PEAK
    STABLE_MIN_MCLK = amdsmi_wrapper.AMDSMI_DEV_PERF_LEVEL_STABLE_MIN_MCLK
    STABLE_MIN_SCLK = amdsmi_wrapper.AMDSMI_DEV_PERF_LEVEL_STABLE_MIN_SCLK
    DETERMINISM = amdsmi_wrapper.AMDSMI_DEV_PERF_LEVEL_DETERMINISM
    UNKNOWN = amdsmi_wrapper.AMDSMI_DEV_PERF_LEVEL_UNKNOWN


class AmdSmiEventGroup(IntEnum):
    XGMI = amdsmi_wrapper.AMDSMI_EVNT_GRP_XGMI
    XGMI_DATA_OUT = amdsmi_wrapper.AMDSMI_EVNT_GRP_XGMI_DATA_OUT
    GRP_INVALID = amdsmi_wrapper.AMDSMI_EVNT_GRP_INVALID


class AmdSmiEventType(IntEnum):
    XGMI_0_NOP_TX = amdsmi_wrapper.AMDSMI_EVNT_XGMI_0_NOP_TX
    XGMI_0_REQUEST_TX = amdsmi_wrapper.AMDSMI_EVNT_XGMI_0_REQUEST_TX
    XGMI_0_RESPONSE_TX = amdsmi_wrapper.AMDSMI_EVNT_XGMI_0_RESPONSE_TX
    XGMI_0_BEATS_TX = amdsmi_wrapper.AMDSMI_EVNT_XGMI_0_BEATS_TX
    XGMI_1_NOP_TX = amdsmi_wrapper.AMDSMI_EVNT_XGMI_1_NOP_TX
    XGMI_1_REQUEST_TX = amdsmi_wrapper.AMDSMI_EVNT_XGMI_1_REQUEST_TX
    XGMI_1_RESPONSE_TX = amdsmi_wrapper.AMDSMI_EVNT_XGMI_1_RESPONSE_TX
    XGMI_1_BEATS_TX = amdsmi_wrapper.AMDSMI_EVNT_XGMI_1_BEATS_TX
    XGMI_DATA_OUT_0 = amdsmi_wrapper.AMDSMI_EVNT_XGMI_DATA_OUT_0
    XGMI_DATA_OUT_1 = amdsmi_wrapper.AMDSMI_EVNT_XGMI_DATA_OUT_1
    XGMI_DATA_OUT_2 = amdsmi_wrapper.AMDSMI_EVNT_XGMI_DATA_OUT_2
    XGMI_DATA_OUT_3 = amdsmi_wrapper.AMDSMI_EVNT_XGMI_DATA_OUT_3
    XGMI_DATA_OUT_4 = amdsmi_wrapper.AMDSMI_EVNT_XGMI_DATA_OUT_4
    XGMI_DATA_OUT_5 = amdsmi_wrapper.AMDSMI_EVNT_XGMI_DATA_OUT_5


class AmdSmiCounterCommand(IntEnum):
    CMD_START = amdsmi_wrapper.AMDSMI_CNTR_CMD_START
    CMD_STOP = amdsmi_wrapper.AMDSMI_CNTR_CMD_STOP


class AmdSmiEvtNotificationType(IntEnum):
    VMFAULT = amdsmi_wrapper.AMDSMI_EVT_NOTIF_VMFAULT
    THERMAL_THROTTLE = amdsmi_wrapper.AMDSMI_EVT_NOTIF_THERMAL_THROTTLE
    GPU_PRE_RESET = amdsmi_wrapper.AMDSMI_EVT_NOTIF_GPU_PRE_RESET
    GPU_POST_RESET = amdsmi_wrapper.AMDSMI_EVT_NOTIF_GPU_POST_RESET


class AmdSmiTemperatureMetric(IntEnum):
    CURRENT = amdsmi_wrapper.AMDSMI_TEMP_CURRENT
    MAX = amdsmi_wrapper.AMDSMI_TEMP_MAX
    MIN = amdsmi_wrapper.AMDSMI_TEMP_MIN
    MAX_HYST = amdsmi_wrapper.AMDSMI_TEMP_MAX_HYST
    MIN_HYST = amdsmi_wrapper.AMDSMI_TEMP_MIN_HYST
    CRITICAL = amdsmi_wrapper.AMDSMI_TEMP_CRITICAL
    CRITICAL_HYST = amdsmi_wrapper.AMDSMI_TEMP_CRITICAL_HYST
    EMERGENCY = amdsmi_wrapper.AMDSMI_TEMP_EMERGENCY
    EMERGENCY_HYST = amdsmi_wrapper.AMDSMI_TEMP_EMERGENCY_HYST
    CRIT_MIN = amdsmi_wrapper.AMDSMI_TEMP_CRIT_MIN
    CRIT_MIN_HYST = amdsmi_wrapper.AMDSMI_TEMP_CRIT_MIN_HYST
    OFFSET = amdsmi_wrapper.AMDSMI_TEMP_OFFSET
    LOWEST = amdsmi_wrapper.AMDSMI_TEMP_LOWEST
    HIGHEST = amdsmi_wrapper.AMDSMI_TEMP_HIGHEST


class AmdSmiVoltageMetric(IntEnum):
    CURRENT = amdsmi_wrapper.AMDSMI_VOLT_CURRENT
    MAX = amdsmi_wrapper.AMDSMI_VOLT_MAX
    MIN_CRIT = amdsmi_wrapper.AMDSMI_VOLT_MIN_CRIT
    MIN = amdsmi_wrapper.AMDSMI_VOLT_MIN
    MAX_CRIT = amdsmi_wrapper.AMDSMI_VOLT_MAX_CRIT
    AVERAGE = amdsmi_wrapper.AMDSMI_VOLT_AVERAGE
    LOWEST = amdsmi_wrapper.AMDSMI_VOLT_LOWEST
    HIGHEST = amdsmi_wrapper.AMDSMI_VOLT_HIGHEST


class AmdSmiVoltageType(IntEnum):
    VDDGFX = amdsmi_wrapper.AMDSMI_VOLT_TYPE_VDDGFX
    INVALID = amdsmi_wrapper.AMDSMI_VOLT_TYPE_INVALID


class AmdSmiComputePartitionType(IntEnum):
    CPX = amdsmi_wrapper.COMPUTE_PARTITION_CPX
    SPX = amdsmi_wrapper.COMPUTE_PARTITION_SPX
    DPX = amdsmi_wrapper.COMPUTE_PARTITION_DPX
    TPX = amdsmi_wrapper.COMPUTE_PARTITION_TPX
    QPX = amdsmi_wrapper.COMPUTE_PARTITION_QPX
    INVALID = amdsmi_wrapper.COMPUTE_PARTITION_INVALID


class AmdSmiMemoryPartitionType(IntEnum):
    NPS1 = amdsmi_wrapper.MEMORY_PARTITION_NPS1
    NPS2 = amdsmi_wrapper.MEMORY_PARTITION_NPS2
    NPS4 = amdsmi_wrapper.MEMORY_PARTITION_NPS4
    NPS8 = amdsmi_wrapper.MEMORY_PARTITION_NPS8
    UNKNOWN = amdsmi_wrapper.MEMORY_PARTITION_UNKNOWN


class AmdSmiPowerProfilePresetMasks(IntEnum):
    CUSTOM_MASK = amdsmi_wrapper.AMDSMI_PWR_PROF_PRST_CUSTOM_MASK
    VIDEO_MASK = amdsmi_wrapper.AMDSMI_PWR_PROF_PRST_VIDEO_MASK
    POWER_SAVING_MASK = amdsmi_wrapper.AMDSMI_PWR_PROF_PRST_POWER_SAVING_MASK
    COMPUTE_MASK = amdsmi_wrapper.AMDSMI_PWR_PROF_PRST_COMPUTE_MASK
    VR_MASK = amdsmi_wrapper.AMDSMI_PWR_PROF_PRST_VR_MASK
    THREE_D_FULL_SCR_MASK = amdsmi_wrapper.AMDSMI_PWR_PROF_PRST_3D_FULL_SCR_MASK
    BOOTUP_DEFAULT = amdsmi_wrapper.AMDSMI_PWR_PROF_PRST_BOOTUP_DEFAULT
    INVALID = amdsmi_wrapper.AMDSMI_PWR_PROF_PRST_INVALID


class AmdSmiGpuBlock(IntEnum):
    INVALID = amdsmi_wrapper.AMDSMI_GPU_BLOCK_INVALID
    UMC = amdsmi_wrapper.AMDSMI_GPU_BLOCK_UMC
    SDMA = amdsmi_wrapper.AMDSMI_GPU_BLOCK_SDMA
    GFX = amdsmi_wrapper.AMDSMI_GPU_BLOCK_GFX
    MMHUB = amdsmi_wrapper.AMDSMI_GPU_BLOCK_MMHUB
    ATHUB = amdsmi_wrapper.AMDSMI_GPU_BLOCK_ATHUB
    PCIE_BIF = amdsmi_wrapper.AMDSMI_GPU_BLOCK_PCIE_BIF
    HDP = amdsmi_wrapper.AMDSMI_GPU_BLOCK_HDP
    XGMI_WAFL = amdsmi_wrapper.AMDSMI_GPU_BLOCK_XGMI_WAFL
    DF = amdsmi_wrapper.AMDSMI_GPU_BLOCK_DF
    SMN = amdsmi_wrapper.AMDSMI_GPU_BLOCK_SMN
    SEM = amdsmi_wrapper.AMDSMI_GPU_BLOCK_SEM
    MP0 = amdsmi_wrapper.AMDSMI_GPU_BLOCK_MP0
    MP1 = amdsmi_wrapper.AMDSMI_GPU_BLOCK_MP1
    FUSE = amdsmi_wrapper.AMDSMI_GPU_BLOCK_FUSE
    RESERVED = amdsmi_wrapper.AMDSMI_GPU_BLOCK_RESERVED


class AmdSmiRasErrState(IntEnum):
    NONE = amdsmi_wrapper.AMDSMI_RAS_ERR_STATE_NONE
    DISABLED = amdsmi_wrapper.AMDSMI_RAS_ERR_STATE_DISABLED
    PARITY = amdsmi_wrapper.AMDSMI_RAS_ERR_STATE_PARITY
    SING_C = amdsmi_wrapper.AMDSMI_RAS_ERR_STATE_SING_C
    MULT_UC = amdsmi_wrapper.AMDSMI_RAS_ERR_STATE_MULT_UC
    POISON = amdsmi_wrapper.AMDSMI_RAS_ERR_STATE_POISON
    ENABLED = amdsmi_wrapper.AMDSMI_RAS_ERR_STATE_ENABLED
    INVALID = amdsmi_wrapper.AMDSMI_RAS_ERR_STATE_INVALID


class AmdSmiMemoryType(IntEnum):
    VRAM = amdsmi_wrapper.AMDSMI_MEM_TYPE_VRAM
    VIS_VRAM = amdsmi_wrapper.AMDSMI_MEM_TYPE_VIS_VRAM
    GTT = amdsmi_wrapper.AMDSMI_MEM_TYPE_GTT


class AmdSmiFreqInd(IntEnum):
    MIN = amdsmi_wrapper.AMDSMI_FREQ_IND_MIN
    MAX = amdsmi_wrapper.AMDSMI_FREQ_IND_MAX
    INVALID = amdsmi_wrapper.AMDSMI_FREQ_IND_INVALID


class AmdSmiXgmiStatus(IntEnum):
    NO_ERRORS = amdsmi_wrapper.AMDSMI_XGMI_STATUS_NO_ERRORS
    ERROR = amdsmi_wrapper.AMDSMI_XGMI_STATUS_ERROR
    MULTIPLE_ERRORS = amdsmi_wrapper.AMDSMI_XGMI_STATUS_MULTIPLE_ERRORS


class AmdSmiMemoryPageStatus(IntEnum):
    RESERVED = amdsmi_wrapper.AMDSMI_MEM_PAGE_STATUS_RESERVED
    PENDING = amdsmi_wrapper.AMDSMI_MEM_PAGE_STATUS_PENDING
    UNRESERVABLE = amdsmi_wrapper.AMDSMI_MEM_PAGE_STATUS_UNRESERVABLE


class AmdSmiIoLinkType(IntEnum):
    UNDEFINED = amdsmi_wrapper.AMDSMI_IOLINK_TYPE_UNDEFINED
    PCIEXPRESS = amdsmi_wrapper.AMDSMI_IOLINK_TYPE_PCIEXPRESS
    XGMI = amdsmi_wrapper.AMDSMI_IOLINK_TYPE_XGMI
    NUMIOLINKTYPES = amdsmi_wrapper.AMDSMI_IOLINK_TYPE_NUMIOLINKTYPES
    SIZE = amdsmi_wrapper.AMDSMI_IOLINK_TYPE_SIZE


class AmdSmiUtilizationCounterType(IntEnum):
    COARSE_GRAIN_GFX_ACTIVITY = amdsmi_wrapper.AMDSMI_COARSE_GRAIN_GFX_ACTIVITY
    COARSE_GRAIN_MEM_ACTIVITY = amdsmi_wrapper.AMDSMI_COARSE_GRAIN_MEM_ACTIVITY
    UTILIZATION_COUNTER_FIRST = amdsmi_wrapper.AMDSMI_UTILIZATION_COUNTER_FIRST
    UTILIZATION_COUNTER_LAST = amdsmi_wrapper.AMDSMI_UTILIZATION_COUNTER_LAST


class AmdSmiProcessorType(IntEnum):
    UNKNOWN = amdsmi_wrapper.UNKNOWN
    AMD_GPU = amdsmi_wrapper.AMD_GPU
    AMD_CPU = amdsmi_wrapper.AMD_CPU
    NON_AMD_GPU = amdsmi_wrapper.NON_AMD_GPU
    NON_AMD_CPU = amdsmi_wrapper.NON_AMD_CPU

class AmdSmiEventReader:
    def __init__(
        self, processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
        event_types: List[AmdSmiEvtNotificationType]
    ):
        if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
            raise AmdSmiParameterException(
                processor_handle, amdsmi_wrapper.amdsmi_processor_handle
            )
        if not isinstance(event_types, Iterable):
            raise AmdSmiParameterException(
                event_types, Iterable
            )

        for event_type in event_types:
            if not isinstance(event_type, AmdSmiEvtNotificationType):
                raise AmdSmiParameterException(
                    event_type, AmdSmiEvtNotificationType
                )

        self.processor_handle = processor_handle
        mask = 0
        for event_type in event_types:
            mask |= (1 << (int(event_type) - 1))

        _check_res(amdsmi_wrapper.amdsmi_init_gpu_event_notification(processor_handle))
        _check_res(amdsmi_wrapper.amdsmi_set_gpu_event_notification_mask(
            processor_handle, ctypes.c_uint64(mask)))

    def read(self, timestamp, num_elem=10):
        self.event_info = (amdsmi_wrapper.amdsmi_evt_notification_data_t * num_elem)()
        _check_res(
            amdsmi_wrapper.amdsmi_get_gpu_event_notification(
                ctypes.c_int(timestamp),
                ctypes.byref(ctypes.c_uint32(num_elem)),
                self.event_info,
            )
        )

        ret = []
        for i in range(0, num_elem):
            unique_event_values = set(event.value for event in AmdSmiEvtNotificationType)
            if self.event_info[i].event in unique_event_values:
                ret.append(
                    {
                        "processor_handle": self.event_info[i].processor_handle,
                        "event": AmdSmiEvtNotificationType(self.event_info[i].event).name,
                        "message": self.event_info[i].message.decode("utf-8"),
                    }
                )

        return ret

    def stop(self):
        _check_res(amdsmi_wrapper.amdsmi_stop_gpu_event_notification(
            self.processor_handle))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


_AMDSMI_MAX_DRIVER_VERSION_LENGTH = 80
_AMDSMI_GPU_UUID_SIZE = 38
_AMDSMI_STRING_LENGTH = 80


def _format_bad_page_info(bad_page_info, bad_page_count: ctypes.c_uint32) -> List[Dict]:
    """
    Format bad page info data retrieved.

    Parameters:
        bad_page_info(`POINTER(amdsmi_retired_page_record_t)`): Pointer to bad page info
        retrieved.
        bad_page_count(`c_uint32`): Bad page count.

    Returns:
        `list`: List containing formatted bad pages.
    """
    if not isinstance(
        bad_page_info, ctypes.POINTER(
            amdsmi_wrapper.amdsmi_retired_page_record_t)
    ):
        raise AmdSmiParameterException(
            bad_page_info, ctypes.POINTER(
                amdsmi_wrapper.amdsmi_retired_page_record_t)
        )

    table_records = []
    for i in range(bad_page_count.value):
        table_records.append(
            {
                "value": i,
                "page_address": bad_page_info[i].page_address,
                "page_size": bad_page_info[i].page_size,
                "status": bad_page_info[i].status,
            }
        )
    return table_records


def _format_bdf(amdsmi_bdf: amdsmi_wrapper.amdsmi_bdf_t) -> str:
    """
    Format BDF struct to readable data.

    Parameters:
        amdsmi_bdf(`amdsmi_bdf_t`): Struct containing BDF data that
        will be formatted.

    Returns:
        `str`: String containing BDF data in a readable format.
    """
    domain = hex(amdsmi_bdf.fields.domain_number)[2:].zfill(4)
    bus = hex(amdsmi_bdf.fields.bus_number)[2:].zfill(2)
    device = hex(amdsmi_bdf.fields.device_number)[2:].zfill(2)
    function = hex(amdsmi_bdf.fields.function_number)[2:]

    return domain + ":" + bus + ":" + device + "." + function


def _check_res(ret_code) -> None:
    """
    Wrapper for amdsmi function calls. Checks the status returned
    by the call. Raises exceptions if the status was inappropriate.

    Parameters:
        ret_code(`amdsmi_status_t`): Status code returned by function
        call.

    Returns:
        `None`.
    """
    if ret_code == amdsmi_wrapper.AMDSMI_STATUS_RETRY:
        raise AmdSmiRetryException()

    if ret_code == amdsmi_wrapper.AMDSMI_STATUS_TIMEOUT:
        raise AmdSmiTimeoutException()

    if ret_code != amdsmi_wrapper.AMDSMI_STATUS_SUCCESS:
        raise AmdSmiLibraryException(ret_code)


def _parse_bdf(bdf):
    if bdf is None:
        return None
    extended_regex = re.compile(
        r'^([0-9a-fA-F]{4}):([0-9a-fA-F]{2}):([0-1][0-9a-fA-F])\.([0-7])$')
    if extended_regex.match(bdf) is None:
        simple_regex = re.compile(
            r'^([0-9a-fA-F]{2}):([0-1][0-9a-fA-F])\.([0-7])$')
        if simple_regex.match(bdf) is None:
            return None
        else:
            return [0] + [int(x, 16) for x in simple_regex.match(bdf).groups()]
    else:
        return [int(x, 16) for x in extended_regex.match(bdf).groups()]


def _make_amdsmi_bdf_from_list(bdf):
    if len(bdf) != 4:
        return None
    amdsmi_bdf = amdsmi_wrapper.amdsmi_bdf_t()
    amdsmi_bdf.fields.function_number = bdf[3]
    amdsmi_bdf.fields.device_number = bdf[2]
    amdsmi_bdf.fields.bus_number = bdf[1]
    amdsmi_bdf.fields.domain_number = bdf[0]
    return amdsmi_bdf


def amdsmi_get_socket_handles() -> List[amdsmi_wrapper.amdsmi_socket_handle]:
    """
    Function that gets socket handles. Wraps the same named function call.

    Parameters:
        `None`.

    Returns:
        `List`: List containing all of the found socket handles.
    """
    socket_count = ctypes.c_uint32(0)
    null_ptr = ctypes.POINTER(amdsmi_wrapper.amdsmi_socket_handle)()
    _check_res(
        amdsmi_wrapper.amdsmi_get_socket_handles(
            ctypes.byref(socket_count), null_ptr)
    )
    socket_handles = (amdsmi_wrapper.amdsmi_socket_handle *
                      socket_count.value)()
    _check_res(
        amdsmi_wrapper.amdsmi_get_socket_handles(
            ctypes.byref(socket_count), socket_handles)
    )
    sockets = [
        amdsmi_wrapper.amdsmi_socket_handle(socket_handles[sock_idx])
        for sock_idx in range(socket_count.value)
    ]

    return sockets

def amdsmi_get_cpusocket_handles() -> List[amdsmi_wrapper.amdsmi_socket_handle]:
    """
    Function that gets cpu socket handles. Wraps the same named function call.

    Parameters:
        `None`.

    Returns:
        `List`: List containing all of the found cpu socket handles.
    """
    socket_handles = amdsmi_get_socket_handles()
    cpu_handles = []
    type = amdsmi_wrapper.AMD_CPU
    for socket in socket_handles:
        cpu_count = ctypes.c_uint32()
        null_ptr = ctypes.POINTER(amdsmi_wrapper.amdsmi_processor_handle)()
        _check_res(
            amdsmi_wrapper.amdsmi_get_processor_handles_by_type(
                socket,
                type,
                null_ptr,
                ctypes.byref(cpu_count),
            )
        )
        processor_handles = (
            amdsmi_wrapper.amdsmi_processor_handle * cpu_count.value)()
        _check_res(
            amdsmi_wrapper.amdsmi_get_processor_handles_by_type(
                socket,
                type,
                processor_handles,
                ctypes.byref(cpu_count)
            )
        )
        cpu_handles.extend(
            [
                amdsmi_wrapper.amdsmi_processor_handle(processor_handles[dev_idx])
                for dev_idx in range(cpu_count.value)
            ]
        )
    return cpu_handles


def amdsmi_get_socket_info(socket_handle):
    if not isinstance(socket_handle, amdsmi_wrapper.amdsmi_socket_handle):
        raise AmdSmiParameterException(
            socket_handle, amdsmi_wrapper.amdsmi_socket_handle)
    socket_info = ctypes.create_string_buffer(128)

    _check_res(
        amdsmi_wrapper.amdsmi_get_socket_info(
            socket_handle, ctypes.c_size_t(128), socket_info)
    )

    return socket_info.value.decode()

def amdsmi_get_processor_info(processor_handle):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle)
    processor_info = ctypes.create_string_buffer(128)

    core_id = ctypes.c_uint32()
    _check_res(
        amdsmi_wrapper.amdsmi_get_processor_info(
            processor_handle, ctypes.c_size_t(128), processor_info)
    )

    return processor_info.value.decode()


def amdsmi_get_processor_handles() -> List[amdsmi_wrapper.amdsmi_processor_handle]:
    socket_handles = amdsmi_get_socket_handles()
    devices = []
    for socket in socket_handles:
        device_count = ctypes.c_uint32()
        null_ptr = ctypes.POINTER(amdsmi_wrapper.amdsmi_processor_handle)()
        _check_res(
            amdsmi_wrapper.amdsmi_get_processor_handles(
                socket,
                ctypes.byref(device_count),
                null_ptr,
            )
        )
        processor_handles = (
            amdsmi_wrapper.amdsmi_processor_handle * device_count.value)()
        _check_res(
            amdsmi_wrapper.amdsmi_get_processor_handles(
                socket,
                ctypes.byref(device_count),
                processor_handles,
            )
        )
        devices.extend(
            [
                amdsmi_wrapper.amdsmi_processor_handle(processor_handles[dev_idx])
                for dev_idx in range(device_count.value)
            ]
        )

    return devices

def amdsmi_get_cpucore_handles() -> List[amdsmi_wrapper.amdsmi_processor_handle]:
    socket_handles = amdsmi_get_socket_handles()
    core_handles = []
    type = amdsmi_wrapper.AMD_CPU_CORE

    for socket in socket_handles:
        core_count = ctypes.c_uint32()
        null_ptr = ctypes.POINTER(amdsmi_wrapper.amdsmi_processor_handle)()
        _check_res(
            amdsmi_wrapper.amdsmi_get_processor_handles_by_type(
                socket,
                type,
                null_ptr,
                ctypes.byref(core_count),
            )
        )
        c_handles = (
            amdsmi_wrapper.amdsmi_processor_handle * core_count.value)()
        _check_res(
            amdsmi_wrapper.amdsmi_get_processor_handles_by_type(
                socket,
                type,
                c_handles,
                ctypes.byref(core_count)
            )
        )
        core_handles.extend(
            [
                amdsmi_wrapper.amdsmi_processor_handle(c_handles[dev_idx])
                for dev_idx in range(core_count.value)
            ]
        )
    return core_handles


def amdsmi_get_cpu_hsmp_proto_ver(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> int:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    proto_ver = ctypes.c_uint32()
    _check_res(
        amdsmi_wrapper.amdsmi_get_cpu_hsmp_proto_ver(
            processor_handle, ctypes.byref(proto_ver)
        )
    )

    return proto_ver.value

def amdsmi_get_cpu_smu_fw_version(processor_handle: amdsmi_wrapper.amdsmi_processor_handle):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    smu_fw = amdsmi_wrapper.amdsmi_smu_fw_version_t()

    _check_res(amdsmi_wrapper.amdsmi_get_cpu_smu_fw_version(processor_handle, smu_fw))

    return {
        "smu_fw_debug_ver_num": smu_fw.debug,
        "smu_fw_minor_ver_num": smu_fw.minor,
        "smu_fw_major_ver_num": smu_fw.major
    }

def amdsmi_get_cpu_core_energy(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle
) -> int:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    penergy = ctypes.c_uint64()
    _check_res(
        amdsmi_wrapper.amdsmi_get_cpu_core_energy(
            processor_handle, ctypes.byref(penergy)
        )
    )

    return f"{float(penergy.value * pow(10, -6))} J"

def amdsmi_get_cpu_socket_energy(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle
) -> int:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    penergy = ctypes.c_uint64()
    _check_res(
        amdsmi_wrapper.amdsmi_get_cpu_socket_energy(
            processor_handle, ctypes.byref(penergy)
        )
    )

    return f"{float(penergy.value * pow(10, -6))} J"

def amdsmi_get_cpu_prochot_status(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle
) -> int:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    prochot = ctypes.c_uint32()
    _check_res(
        amdsmi_wrapper.amdsmi_get_cpu_prochot_status(
            processor_handle, ctypes.byref(prochot)
        )
    )

    return prochot.value

def amdsmi_get_cpu_fclk_mclk(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    fclk = ctypes.c_uint32()
    mclk = ctypes.c_uint32()
    _check_res(
        amdsmi_wrapper.amdsmi_get_cpu_fclk_mclk(
            processor_handle, ctypes.byref(fclk), ctypes.byref(mclk)
        )
    )

    return {
       "fclk": f"{fclk.value} MHz",
       "mclk": f"{mclk.value} MHz"
    }

def amdsmi_get_cpu_cclk_limit(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle
) -> int:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    cclk = ctypes.c_uint32()
    _check_res(
        amdsmi_wrapper.amdsmi_get_cpu_cclk_limit(
            processor_handle, ctypes.byref(cclk)
        )
    )

    return f"{cclk.value} MHz"

def amdsmi_get_cpu_socket_current_active_freq_limit(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    amdsmi_wrapper.amdsmi_get_cpu_socket_current_active_freq_limit.argtypes = [amdsmi_wrapper.amdsmi_processor_handle, ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_char_p * len(amdsmi_wrapper.amdsmi_hsmp_freqlimit_src_names))]
    freq = ctypes.c_uint16()
    src_type = (ctypes.c_char_p * len(amdsmi_wrapper.amdsmi_hsmp_freqlimit_src_names))()

    _check_res(
        amdsmi_wrapper.amdsmi_get_cpu_socket_current_active_freq_limit(
            processor_handle, ctypes.byref(freq), src_type
        )
    )

    freq_src = []
    for names in src_type:
        if names is not None:
            freq_src.append(names.decode('utf-8'))

    return {
            "freq": f"{freq.value} MHz",
            "freq_src": f"{freq_src}"
    }

def amdsmi_get_cpu_socket_freq_range(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    freq_max = ctypes.c_uint16()
    freq_min = ctypes.c_uint16()
    _check_res(
        amdsmi_wrapper.amdsmi_get_cpu_socket_freq_range(
            processor_handle, ctypes.byref(freq_max), ctypes.byref(freq_min)
        )
    )

    return {
       "max_socket_freq": f"{freq_max.value} MHz",
       "min_socket_freq": f"{freq_min.value} MHz"
    }

def amdsmi_get_cpu_core_current_freq_limit(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle
) -> int:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    freq = ctypes.c_uint32()
    _check_res(
        amdsmi_wrapper.amdsmi_get_cpu_core_current_freq_limit(
            processor_handle, ctypes.byref(freq)
        )
    )

    return f"{freq.value} MHz"

def amdsmi_get_cpu_socket_power(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle
) -> int:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    ppower = ctypes.c_uint32()
    _check_res(
        amdsmi_wrapper.amdsmi_get_cpu_socket_power(
            processor_handle, ctypes.byref(ppower)
        )
    )

    return f"{ppower.value} mW"

def amdsmi_get_cpu_socket_power_cap(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle
) -> int:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    pcap = ctypes.c_uint32()
    _check_res(
        amdsmi_wrapper.amdsmi_get_cpu_socket_power_cap(
            processor_handle, ctypes.byref(pcap)
        )
    )

    return f"{pcap.value} mW"

def amdsmi_get_cpu_socket_power_cap_max(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle
) -> int:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    pmax = ctypes.c_uint32()
    _check_res(
        amdsmi_wrapper.amdsmi_get_cpu_socket_power_cap_max(
            processor_handle, ctypes.byref(pmax)
        )
    )

    return f"{pmax.value} mW"

def amdsmi_get_cpu_pwr_svi_telemetry_all_rails(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle
) -> int:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    power = ctypes.c_uint32()
    _check_res(
        amdsmi_wrapper.amdsmi_get_cpu_pwr_svi_telemetry_all_rails(
            processor_handle, ctypes.byref(power)
        )
    )

    return f"{power.value} mW"

def amdsmi_set_cpu_socket_power_cap(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle, power_cap: int
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(power_cap, int):
        raise AmdSmiParameterException(power_cap, int)

    power_cap = ctypes.c_uint32(power_cap)

    _check_res(
        amdsmi_wrapper.amdsmi_set_cpu_socket_power_cap(
            processor_handle, power_cap)
    )

def amdsmi_set_cpu_pwr_efficiency_mode(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle, mode: int
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(mode, int):
        raise AmdSmiParameterException(mode, int)
    mode = ctypes.c_uint8(mode)

    _check_res(
        amdsmi_wrapper.amdsmi_set_cpu_pwr_efficiency_mode(
            processor_handle, mode)
    )

def amdsmi_get_cpu_core_boostlimit(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle
) -> int:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    boostlimit = ctypes.c_uint32()
    _check_res(
        amdsmi_wrapper.amdsmi_get_cpu_core_boostlimit(
            processor_handle, ctypes.byref(boostlimit)
        )
    )

    return f"{boostlimit.value} MHz"

def amdsmi_get_cpu_socket_c0_residency(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle
) -> int:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    c0_residency = ctypes.c_uint32()
    _check_res(
        amdsmi_wrapper.amdsmi_get_cpu_socket_c0_residency(
            processor_handle, ctypes.byref(c0_residency)
        )
    )

    return f"{c0_residency.value} %"

def amdsmi_set_cpu_core_boostlimit(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle, boostlimit: int
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(boostlimit, int):
        raise AmdSmiParameterException(boostlimit, int)
    boostlimit = ctypes.c_uint32(boostlimit)

    _check_res(
        amdsmi_wrapper.amdsmi_set_cpu_core_boostlimit(
            processor_handle, boostlimit)
    )

def amdsmi_set_cpu_socket_boostlimit(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle, boostlimit: int
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(boostlimit, int):
        raise AmdSmiParameterException(boostlimit, int)
    boostlimit = ctypes.c_uint32(boostlimit)

    _check_res(
        amdsmi_wrapper.amdsmi_set_cpu_socket_boostlimit(
            processor_handle, boostlimit)
    )

def amdsmi_get_cpu_ddr_bw(processor_handle: amdsmi_wrapper.amdsmi_processor_handle):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    ddr_bw = amdsmi_wrapper.amdsmi_ddr_bw_metrics_t()

    _check_res(amdsmi_wrapper.amdsmi_get_cpu_ddr_bw(processor_handle, ddr_bw))

    return {
        "ddr_bw_max_bw": f"{ddr_bw.max_bw} Gbps",
        "ddr_bw_utilized_bw": f"{ddr_bw.utilized_bw} Gbps",
        "ddr_bw_utilized_pct": f"{ddr_bw.utilized_pct} %"
    }

def amdsmi_get_cpu_socket_temperature(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle
) -> int:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    ptmon = ctypes.c_uint32()

    _check_res(
        amdsmi_wrapper.amdsmi_get_cpu_socket_temperature(
            processor_handle, ctypes.byref(ptmon)
        )
    )

    return f"{ptmon.value} Degrees C"

def amdsmi_get_cpu_dimm_temp_range_and_refresh_rate(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    dimm_addr: int):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(dimm_addr, int):
        raise AmdSmiParameterException(dimm_addr, int)

    dimm_addr = ctypes.c_uint8(dimm_addr)
    dimm = amdsmi_wrapper.amdsmi_temp_range_refresh_rate_t()

    _check_res(amdsmi_wrapper.amdsmi_get_cpu_dimm_temp_range_and_refresh_rate(processor_handle,
                                                                                dimm_addr,
                                                                                ctypes.byref(dimm)))

    return {
        "dimm_temperature_range": dimm.range,
        "dimm_refresh_rate": dimm.ref_rate
    }

def amdsmi_get_cpu_dimm_power_consumption(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    dimm_addr: int):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(dimm_addr, int):
        raise AmdSmiParameterException(dimm_addr, int)

    dimm_addr = ctypes.c_uint8(dimm_addr)
    dimm = amdsmi_wrapper.amdsmi_dimm_power_t()

    _check_res(amdsmi_wrapper.amdsmi_get_cpu_dimm_power_consumption(processor_handle,
                                                                    dimm_addr,
                                                                    ctypes.byref(dimm)))

    return {
        "dimm_power_consumed": f"{dimm.power} mW",
        "dimm_power_update_rate": f"{dimm.update_rate} ms",
        "dimm_dimm_addr": dimm.dimm_addr
    }

def amdsmi_get_cpu_dimm_thermal_sensor(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    dimm_addr: int):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(dimm_addr, int):
        raise AmdSmiParameterException(dimm_addr, int)

    dimm_addr = ctypes.c_uint8(dimm_addr)
    dimm_thermal = amdsmi_wrapper.amdsmi_dimm_thermal_t()

    _check_res(amdsmi_wrapper.amdsmi_get_cpu_dimm_thermal_sensor(processor_handle,
                                                                    dimm_addr,
                                                                    ctypes.byref(dimm_thermal)))

    return {
        "dimm_thermal_sensor_value": dimm_thermal.sensor,
        "dimm_thermal_update_rate": f"{dimm_thermal.update_rate} ms",
        "dimm_thermal_dimm_addr": dimm_thermal.dimm_addr,
        "dimm_thermal_temperature": f"{dimm_thermal.temp} Degrees C"
    }

def amdsmi_set_cpu_xgmi_width(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle, min_width: int, max_width: int
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(min_width, int):
        raise AmdSmiParameterException(min_width, int)
    if not isinstance(max_width, int):
        raise AmdSmiParameterException(max_width, int)

    min_width = ctypes.c_uint8(min_width)
    max_width = ctypes.c_uint8(max_width)

    _check_res(
        amdsmi_wrapper.amdsmi_set_cpu_xgmi_width(
            processor_handle, min_width, max_width)
    )

def amdsmi_set_cpu_gmi3_link_width_range(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    min_link_width: int, max_link_width: int
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(min_link_width, int):
        raise AmdSmiParameterException(min_link_width, int)
    if not isinstance(max_link_width, int):
        raise AmdSmiParameterException(max_link_width, int)

    min_link_width = ctypes.c_uint8(min_link_width)
    max_link_width = ctypes.c_uint8(max_link_width)

    _check_res(
        amdsmi_wrapper.amdsmi_set_cpu_gmi3_link_width_range(
            processor_handle, min_link_width, max_link_width)
    )

def amdsmi_cpu_apb_enable(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    _check_res(
        amdsmi_wrapper.amdsmi_cpu_apb_enable(processor_handle)
    )

def amdsmi_cpu_apb_disable(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    pstate: int
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(pstate, int):
        raise AmdSmiParameterException(pstate, int)

    pstate = ctypes.c_uint8(pstate)

    _check_res(
        amdsmi_wrapper.amdsmi_cpu_apb_disable(
            processor_handle, pstate)
    )

def amdsmi_set_cpu_socket_lclk_dpm_level(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    nbio_id: int, min_val: int, max_val: int
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(nbio_id, int):
        raise AmdSmiParameterException(nbio_id, int)
    if not isinstance(min_val, int):
        raise AmdSmiParameterException(min_val, int)
    if not isinstance(max_val, int):
        raise AmdSmiParameterException(max_val, int)

    nbio_id = ctypes.c_uint8(nbio_id)
    min_val = ctypes.c_uint8(min_val)
    max_val = ctypes.c_uint8(max_val)

    _check_res(
        amdsmi_wrapper.amdsmi_set_cpu_socket_lclk_dpm_level(
            processor_handle, nbio_id, min_val, max_val)
    )

def amdsmi_get_cpu_socket_lclk_dpm_level(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    nbio_id: int
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(nbio_id, int):
        raise AmdSmiParameterException(nbio_id, int)

    nbio_id = ctypes.c_uint8(nbio_id)
    dpm_level = amdsmi_wrapper.amdsmi_dpm_level_t()

    _check_res(amdsmi_wrapper.amdsmi_get_cpu_socket_lclk_dpm_level(processor_handle, nbio_id, dpm_level))

    return {
        "nbio_max_dpm_level": dpm_level.max_dpm_level,
        "nbio_min_dpm_level": dpm_level.min_dpm_level
    }

def amdsmi_set_cpu_pcie_link_rate(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    rate_ctrl: int
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(rate_ctrl, int):
        raise AmdSmiParameterException(rate_ctrl, int)

    rate_ctrl = ctypes.c_uint8(rate_ctrl)
    prev_mode = ctypes.c_uint8()

    _check_res(
        amdsmi_wrapper.amdsmi_set_cpu_pcie_link_rate(
            processor_handle, rate_ctrl, ctypes.byref(prev_mode))
    )

    return f"{prev_mode.value}"

def amdsmi_set_cpu_df_pstate_range(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    max_pstate: int, min_pstate: int
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(max_pstate, int):
        raise AmdSmiParameterException(max_pstate, int)
    if not isinstance(min_pstate, int):
        raise AmdSmiParameterException(min_pstate, int)

    max_pstate = ctypes.c_uint8(max_pstate)
    min_pstate = ctypes.c_uint8(min_pstate)

    _check_res(
        amdsmi_wrapper.amdsmi_set_cpu_df_pstate_range(
            processor_handle, max_pstate, min_pstate))

def amdsmi_get_cpu_current_io_bandwidth(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    encoding: int,
    link_name: str
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    link = amdsmi_wrapper.amdsmi_link_id_bw_type_t()
    link.bw_type = ctypes.c_uint32(encoding)
    link.link_name = ctypes.create_string_buffer(link_name.encode('utf-8'))
    io_bw = ctypes.c_uint32()

    _check_res(
        amdsmi_wrapper.amdsmi_get_cpu_current_io_bandwidth(
            processor_handle, link, ctypes.byref(io_bw))
    )

    return f"{io_bw.value} Mbps"

def amdsmi_get_cpu_current_xgmi_bw(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    encoding: int,
    link_name: str
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    link = amdsmi_wrapper.amdsmi_link_id_bw_type_t()
    link.bw_type = ctypes.c_uint32(encoding)
    link.link_name = ctypes.create_string_buffer(link_name.encode('utf-8'))
    xgmi_bw = ctypes.c_uint32()

    _check_res(
        amdsmi_wrapper.amdsmi_get_cpu_current_xgmi_bw(
            processor_handle, link, ctypes.byref(xgmi_bw))
    )

    return f"{xgmi_bw.value} Mbps"

def amdsmi_get_hsmp_metrics_table_version(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    metric_tbl_version = ctypes.c_uint32()

    _check_res(
        amdsmi_wrapper.amdsmi_get_hsmp_metrics_table_version(
            processor_handle, ctypes.byref(metric_tbl_version))
    )

    return metric_tbl_version.value


NO_OF_32BITS = (sys.getsizeof(ctypes.c_uint32) * 8)
NO_OF_64BITS = (sys.getsizeof(ctypes.c_uint64) * 8)
KILO = math.pow(10, 3)

# Get 2's complement of 32 bit unsigned integer
def check_msb_32(num):
    msb = 1 << (NO_OF_32BITS - 1)

    '''If msb = 1 , then take 2's complement of the number'''
    if num & msb:
        num = ~num + 1
        return num
    else:
       return num

# Get 2's complement of 64 bit unsigned integer
def check_msb_64(num):
    msb = 1 << (NO_OF_64BITS - 1)

    '''If msb = 1 , then take 2's complement of the number'''
    if num & msb:
        num = ~num + 1
        return num
    else:
        return num

def amdsmi_get_hsmp_metrics_table(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    mtbl = amdsmi_wrapper.amdsmi_hsmp_metrics_table_t()

    '''Encodings for the metric table defined for hsmp'''
    fraction_q10 = 1 / math.pow(2, 10)
    fraction_uq10 = fraction_q10
    fraction_uq16 = 1 / math.pow(2, 16)

    _check_res(
            amdsmi_wrapper.amdsmi_get_hsmp_metrics_table(
                   processor_handle, mtbl
            )
    )

    rawtime = int(mtbl.timestamp)
    rawtime = time()
    timeinfo = localtime(rawtime)

    return {
        "mtbl_accumulation_counter": mtbl.accumulation_counter,
        "mtbl_max_socket_temperature": f"{round(check_msb_32(mtbl.max_socket_temperature) * fraction_q10 ,3)} °C",
        "mtbl_max_vr_temperature": f"{round(check_msb_32(mtbl.max_vr_temperature) * fraction_q10 ,3)} °C",
        "mtbl_max_hbm_temperature": f"{round(check_msb_32(mtbl.max_hbm_temperature) * fraction_q10 ,3)} °C",
        "mtbl_max_socket_temperature_acc": f"{round(check_msb_64(mtbl.max_socket_temperature_acc) * fraction_q10 ,3)} °C",
        "mtbl_max_vr_temperature_acc": f"{round(check_msb_64(mtbl.max_vr_temperature_acc) * fraction_q10 ,3)} °C",
        "mtbl_max_hbm_temperature_acc": f"{round(check_msb_64(mtbl.max_hbm_temperature_acc) * fraction_q10 ,3)} °C",
        "mtbl_socket_power_limit": f"{round(mtbl.socket_power_limit * fraction_uq10 ,3)} W",
        "mtbl_max_socket_power_limit": f"{round(mtbl.max_socket_power_limit * fraction_uq10 ,3)} W",
        "mtbl_socket_power": f"{round(mtbl.socket_power * fraction_uq10 ,3)} W",
        "mtbl_timestamp_raw": mtbl.timestamp,
        "mtbl_timestamp_readable": f"{asctime(timeinfo)}",
        "mtbl_socket_energy_acc": f"{round((mtbl.socket_energy_acc * fraction_uq16)/KILO ,3)} kJ",
        "mtbl_ccd_energy_acc": f"{round((mtbl.ccd_energy_acc * fraction_uq16)/KILO ,3)} kJ",
        "mtbl_xcd_energy_acc": f"{round((mtbl.xcd_energy_acc * fraction_uq16)/KILO ,3)} kJ",
        "mtbl_aid_energy_acc": f"{round((mtbl.aid_energy_acc * fraction_uq16)/KILO ,3)} kJ",
        "mtbl_hbm_energy_acc": f"{round((mtbl.hbm_energy_acc * fraction_uq16)/KILO ,3)} kJ",
        "mtbl_cclk_frequency_limit": f"{round(mtbl.cclk_frequency_limit * fraction_uq10 ,3)} GHz",
        "mtbl_gfxclk_frequency_limit": f"{round(mtbl.gfxclk_frequency_limit * fraction_uq10 ,3)} MHz",
        "mtbl_fclk_frequency": f"{round(mtbl.fclk_frequency * fraction_uq10 ,3)} MHz",
        "mtbl_uclk_frequency": f"{round(mtbl.uclk_frequency * fraction_uq10 ,3)} MHz",
        "mtbl_socclk_frequency": f"{[round(x*fraction_uq10 ,3) for x in list(mtbl.socclk_frequency)]} MHz",
        "mtbl_vclk_frequency": f"{[round(x*fraction_uq10 ,3) for x in list(mtbl.vclk_frequency)]} MHz",
        "mtbl_dclk_frequency": f"{[round(x*fraction_uq10 ,3) for x in list(mtbl.dclk_frequency)]} MHz",
        "mtbl_lclk_frequency": f"{[round(x*fraction_uq10 ,3) for x in list(mtbl.lclk_frequency)]} MHz",
        "mtbl_fclk_frequency_table": f"{[round(x*fraction_uq10 ,3) for x in list(mtbl.fclk_frequency_table)]} MHz",
        "mtbl_uclk_frequency_table": f"{[round(x*fraction_uq10 ,3) for x in list(mtbl.uclk_frequency_table)]} MHz",
        "mtbl_socclk_frequency_table": f"{[round(x*fraction_uq10 ,3) for x in list(mtbl.socclk_frequency_table)]} MHz",
        "mtbl_vclk_frequency_table": f"{[round(x*fraction_uq10 ,3) for x in list(mtbl.vclk_frequency_table)]} MHz",
        "mtbl_dclk_frequency_table": f"{[round(x*fraction_uq10 ,3) for x in list(mtbl.dclk_frequency_table)]} MHz",
        "mtbl_lclk_frequency_table": f"{[round(x*fraction_uq10 ,3) for x in list(mtbl.lclk_frequency_table)]} MHz",
        "mtbl_cclk_frequency_acc": f"{[round(x*fraction_uq10 ,3) for x in list(mtbl.cclk_frequency_acc)]} GHz",
        "mtbl_gfxclk_frequency_acc": f"{[round(x*fraction_uq10 ,3) for x in list(mtbl.gfxclk_frequency_acc)]} MHz",
        "mtbl_gfxclk_frequency": f"{[round(x*fraction_uq10 ,3) for x in list(mtbl.gfxclk_frequency)]} MHz",
        "mtbl_max_cclk_frequency": f"{round(mtbl.max_cclk_frequency * fraction_uq10 ,3)} GHz",
        "mtbl_min_cclk_frequency": f"{round(mtbl.min_cclk_frequency * fraction_uq10 ,3)} GHz",
        "mtbl_max_gfxclk_frequency": f"{round(mtbl.max_gfxclk_frequency * fraction_uq10 ,3)} MHz",
        "mtbl_min_gfxclk_frequency": f"{round(mtbl.min_gfxclk_frequency * fraction_uq10 ,3)} MHz",
        "mtbl_max_lclk_dpm_range": mtbl.max_lclk_dpm_range,
        "mtbl_min_lclk_dpm_range": mtbl.min_lclk_dpm_range,
        "mtbl_xgmi_width": round(mtbl.xgmi_width * fraction_uq10 ,3),
        "mtbl_xgmi_bitrate": f"{round(mtbl.xgmi_bitrate * fraction_uq10 ,3)} Gbps",
        "mtbl_xgmi_read_bandwidth_acc": f"{[round(x*fraction_uq10 ,3) for x in list(mtbl.xgmi_read_bandwidth_acc)]} Gbps",
        "mtbl_xgmi_write_bandwidth_acc": f"{[round(x*fraction_uq10 ,3) for x in list(mtbl.xgmi_write_bandwidth_acc)]} Gbps",
        "mtbl_socket_c0_residency": f"{round(mtbl.socket_c0_residency * fraction_uq10 ,3)} %",
        "mtbl_socket_gfx_busy": f"{round(mtbl.socket_gfx_busy * fraction_uq10 ,3)} %",
        "mtbl_hbm_bandwidth_utilization": f"{round(mtbl.dram_bandwidth_utilization * fraction_uq10 ,3)} %",
        "mtbl_socket_c0_residency_acc": round(mtbl.socket_c0_residency_acc * fraction_uq10 ,3),
        "mtbl_socket_gfx_busy_acc": round(mtbl.socket_gfx_busy_acc * fraction_uq10 ,3),
        "mtbl_hbm_bandwidth_acc": f"{round(mtbl.dram_bandwidth_acc * fraction_uq10 ,3)} Gbps",
        "mtbl_max_hbm_bandwidth": f"{round(mtbl.max_dram_bandwidth * fraction_uq10 ,3)} Gbps",
        "mtbl_dram_bandwidth_utilization_acc": round(mtbl.dram_bandwidth_utilization_acc * fraction_uq10 ,3),
        "mtbl_pcie_bandwidth_acc": f"{[round(x*fraction_uq10 ,3) for x in list(mtbl.pcie_bandwidth_acc)]} Gbps",
        "mtbl_prochot_residency_acc": mtbl.prochot_residency_acc,
        "mtbl_ppt_residency_acc": mtbl.ppt_residency_acc,
        "mtbl_socket_thm_residency_acc": mtbl.socket_thm_residency_acc,
        "mtbl_vr_thm_residency_acc": mtbl.vr_thm_residency_acc,
        "mtbl_hbm_thm_residency_acc": mtbl.hbm_thm_residency_acc
    }

def amdsmi_first_online_core_on_cpu_socket(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    pcore_ind = ctypes.c_uint32()

    _check_res(
        amdsmi_wrapper.amdsmi_first_online_core_on_cpu_socket(
            processor_handle, ctypes.byref(pcore_ind))
    )

    return pcore_ind.value

def amdsmi_get_cpu_family():
    family = ctypes.c_uint32()
    _check_res(
        amdsmi_wrapper.amdsmi_get_cpu_family(ctypes.byref(family))
    )
    return family.value

def amdsmi_get_cpu_model():
    model = ctypes.c_uint32()
    _check_res(
        amdsmi_wrapper.amdsmi_get_cpu_model(ctypes.byref(model))
    )
    return model.value

def amdsmi_init(flag=AmdSmiInitFlags.INIT_AMD_GPUS):
    if not isinstance(flag, AmdSmiInitFlags):
        raise AmdSmiParameterException(flag, AmdSmiInitFlags)
    _check_res(amdsmi_wrapper.amdsmi_init(flag))


def amdsmi_shut_down():
    _check_res(amdsmi_wrapper.amdsmi_shut_down())


def amdsmi_get_processor_type(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> ctypes.c_uint32:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    dev_type = amdsmi_wrapper.processor_type_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_processor_type(
            processor_handle, ctypes.byref(dev_type))
    )

    return {
        "processor_type": AmdSmiProcessorType(dev_type.value).name
    }


def amdsmi_get_gpu_device_bdf(processor_handle: amdsmi_wrapper.amdsmi_processor_handle) -> str:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    bdf_info = amdsmi_wrapper.amdsmi_bdf_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_device_bdf(
            processor_handle, ctypes.byref(bdf_info))
    )

    return _format_bdf(bdf_info)


def amdsmi_get_gpu_asic_info(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> Dict[str, Any]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    asic_info_struct = amdsmi_wrapper.amdsmi_asic_info_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_asic_info(
            processor_handle, ctypes.byref(asic_info_struct))
    )

    asic_info = {
        "market_name": asic_info_struct.market_name.decode("utf-8"),
        "vendor_id": asic_info_struct.vendor_id,
        "vendor_name": asic_info_struct.vendor_name.decode("utf-8"),
        "subvendor_id": asic_info_struct.subvendor_id,
        "device_id": asic_info_struct.device_id,
        "rev_id": hex(asic_info_struct.rev_id),
        "asic_serial": asic_info_struct.asic_serial.decode("utf-8"),
        "oam_id": asic_info_struct.oam_id
    }

    string_values = ["market_name", "vendor_name"]
    for value in string_values:
        if not asic_info[value]:
            asic_info[value] = "N/A"

    hex_values = ["vendor_id", "subvendor_id", "device_id"]
    for value in hex_values:
        if asic_info[value]:
            asic_info[value] = hex(asic_info[value])
        else:
            asic_info[value] = "N/A"

    # Convert asic serial (hex string) to hex output format
    if asic_info["asic_serial"]:
        asic_serial_string = asic_info["asic_serial"]
        asic_serial_hex = int(asic_serial_string, base=16)
        asic_info["asic_serial"] = str.format("0x{:016X}", asic_serial_hex)
    else:
        asic_info["asic_serial"] = "N/A"

    # Check for max value as a sign for not applicable
    if asic_info["oam_id"] == 0xFFFF: # uint 16 max
        asic_info["oam_id"] = "N/A"

    # Remove commas from vendor name for clean output
    asic_info["vendor_name"] = asic_info["vendor_name"].replace(',', '')

    return asic_info


def amdsmi_get_power_cap_info(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> Dict[str, Any]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    power_info = amdsmi_wrapper.amdsmi_power_cap_info_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_power_cap_info(
            processor_handle, ctypes.c_uint32(0), ctypes.byref(power_info)
        )
    )

    return {"power_cap": power_info.power_cap,
            "default_power_cap": power_info.default_power_cap,
            "dpm_cap": power_info.dpm_cap,
            "min_power_cap": power_info.min_power_cap,
            "max_power_cap": power_info.max_power_cap}


def amdsmi_get_gpu_vram_info(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> Dict[str, Any]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    vram_info = amdsmi_wrapper.amdsmi_vram_info_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_vram_info(
            processor_handle, ctypes.byref(vram_info))
    )

    return {
        "vram_type": vram_info.vram_type,
        "vram_vendor": vram_info.vram_vendor,
        "vram_size_mb": vram_info.vram_size_mb,
    }


def amdsmi_get_gpu_cache_info(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> List[Dict[str, Any]]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    cache_info_struct = amdsmi_wrapper.amdsmi_gpu_cache_info_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_cache_info(
            processor_handle, ctypes.byref(cache_info_struct))
    )

    cache_info_list = []
    for cache_index in range(cache_info_struct.num_cache_types):
        # Put cache_properties at the start of the dictionary for readability
        cache_dict = {
            "cache_properties": [], # This will be a list of strings
            "cache_size": cache_info_struct.cache[cache_index].cache_size,
            "cache_level": cache_info_struct.cache[cache_index].cache_level,
            "max_num_cu_shared": cache_info_struct.cache[cache_index].max_num_cu_shared,
            "num_cache_instance": cache_info_struct.cache[cache_index].num_cache_instance
        }

        # Check against cache properties bitmask
        cache_properties = cache_info_struct.cache[cache_index].cache_properties
        data_cache = cache_properties & amdsmi_wrapper.AMDSMI_CACHE_PROPERTY_DATA_CACHE
        inst_cache = cache_properties & amdsmi_wrapper.AMDSMI_CACHE_PROPERTY_INST_CACHE
        cpu_cache = cache_properties & amdsmi_wrapper.AMDSMI_CACHE_PROPERTY_CPU_CACHE
        simd_cache = cache_properties & amdsmi_wrapper.AMDSMI_CACHE_PROPERTY_SIMD_CACHE

        cache_properties_status = [data_cache, inst_cache, cpu_cache, simd_cache]
        cache_property_list = []
        for cache_property in cache_properties_status:
            if cache_property:
                property_name = amdsmi_wrapper.amdsmi_cache_property_type_t__enumvalues[cache_property]
                property_name = property_name.replace("AMDSMI_CACHE_PROPERTY_", "")
                cache_property_list.append(property_name)

        cache_dict["cache_properties"] = cache_property_list
        cache_info_list.append(cache_dict)

    if not cache_info_list:
        raise AmdSmiLibraryException(amdsmi_wrapper.AMDSMI_STATUS_NO_DATA)

    return {
        "cache": cache_info_list
    }


def amdsmi_get_gpu_vbios_info(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> Dict[str, Any]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    vbios_info = amdsmi_wrapper.amdsmi_vbios_info_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_vbios_info(
            processor_handle, ctypes.byref(vbios_info))
    )

    return {
        "name": vbios_info.name.decode("utf-8"),
        "build_date": vbios_info.build_date.decode("utf-8"),
        "part_number": vbios_info.part_number.decode("utf-8"),
        "version": vbios_info.version.decode("utf-8"),
    }


def amdsmi_get_gpu_activity(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> Dict[str, Any]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    engine_usage = amdsmi_wrapper.amdsmi_engine_usage_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_activity(
            processor_handle, ctypes.byref(engine_usage)
        )
    )

    activity_dict = {
        "gfx_activity": engine_usage.gfx_activity,
        "umc_activity": engine_usage.umc_activity,
        "mm_activity": engine_usage.mm_activity,
    }

    for key, value in activity_dict.items():
        if value == 0xFFFF:
            activity_dict[key] = "N/A"

    return activity_dict


def amdsmi_get_clock_info(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    clock_type: AmdSmiClkType,
) -> Dict[str, int]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(clock_type, AmdSmiClkType):
        raise AmdSmiParameterException(clock_type, AmdSmiClkType)

    clock_measure = amdsmi_wrapper.amdsmi_clk_info_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_clock_info(
            processor_handle,
            clock_type,
            ctypes.byref(clock_measure),
        )
    )

    return {
        "cur_clk": clock_measure.cur_clk,
        "max_clk": clock_measure.max_clk,
        "min_clk": clock_measure.min_clk,
        "sleep_clk" : clock_measure.sleep_clk,
    }


def amdsmi_get_gpu_bad_page_info(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> Union[list, str]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    num_pages = ctypes.c_uint32()
    retired_page_record = ctypes.POINTER(
        amdsmi_wrapper.amdsmi_retired_page_record_t)()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_bad_page_info(
            processor_handle, ctypes.byref(num_pages), retired_page_record
        )
    )
    table_records = _format_bad_page_info(retired_page_record, num_pages)
    if num_pages.value == 0:
        return "No bad pages found."
    else:
        table_records = _format_bad_page_info(retired_page_record, num_pages)

    return table_records


def amdsmi_get_gpu_total_ecc_count(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> Dict[str, Any]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    error_count = amdsmi_wrapper.amdsmi_error_count_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_total_ecc_count(
            processor_handle, ctypes.byref(error_count)
        )
    )

    return {
        "correctable_count": error_count.correctable_count,
        "uncorrectable_count": error_count.uncorrectable_count,
    }


def amdsmi_get_gpu_board_info(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> Dict[str, Any]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    board_info = amdsmi_wrapper.amdsmi_board_info_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_board_info(
            processor_handle, ctypes.byref(board_info))
    )

    return {
        "model_number": board_info.model_number.decode("utf-8").strip(),
        "product_serial": board_info.product_serial.decode("utf-8").strip(),
        "fru_id": board_info.fru_id.decode("utf-8").strip(),
        "product_name": board_info.product_name.decode("utf-8").strip(),
        "manufacturer_name": board_info.manufacturer_name.decode("utf-8").strip()
    }


def amdsmi_get_gpu_ras_feature_info(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> Dict[str, Any]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    ras_feature = amdsmi_wrapper.amdsmi_ras_feature_t()

    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_ras_feature_info(
            processor_handle, ctypes.byref(ras_feature)
        )
    )

    return {
        "eeprom_version": hex(ras_feature.ras_eeprom_version),
        "parity_schema" : bool(ras_feature.ecc_correction_schema_flag & 1),
        "single_bit_schema" : bool(ras_feature.ecc_correction_schema_flag & 2),
        "double_bit_schema" : bool(ras_feature.ecc_correction_schema_flag & 4),
        "poison_schema" : bool(ras_feature.ecc_correction_schema_flag & 8)
    }


def amdsmi_get_gpu_ras_block_features_enabled(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> List[Dict[str, Any]]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    ras_state = amdsmi_wrapper.amdsmi_ras_err_state_t()
    ras_states = []
    for gpu_block in AmdSmiGpuBlock:
        if gpu_block.name == "RESERVED" or gpu_block.name == "INVALID":
            continue
        if gpu_block.name == "LAST":
            gpu_block.name = "FUSE"
        _check_res(
            amdsmi_wrapper.amdsmi_get_gpu_ras_block_features_enabled(
                processor_handle,
                amdsmi_wrapper.amdsmi_gpu_block_t(gpu_block.value),
                ctypes.byref(ras_state),
            )
        )
        ras_states.append(
            {
                "block": gpu_block.name,
                "status": AmdSmiRasErrState(ras_state.value).name,
            }
        )

    return ras_states


def amdsmi_get_gpu_process_list(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> List[amdsmi_wrapper.amdsmi_process_handle_t]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    max_processes = ctypes.c_uint32(MAX_NUM_PROCESSES)

    process_list = (amdsmi_wrapper.amdsmi_process_handle_t *
                    max_processes.value)()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_process_list(
            processor_handle, ctypes.byref(max_processes), process_list
        )
    )

    return [amdsmi_wrapper.amdsmi_process_handle_t(process_list[x])\
    for x in range(0, max_processes.value)]


def amdsmi_get_gpu_process_info(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    process: amdsmi_wrapper.amdsmi_process_handle_t,
) -> Dict[str, Any]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    if not isinstance(process, amdsmi_wrapper.amdsmi_process_handle_t):
        raise AmdSmiParameterException(
            process, amdsmi_wrapper.amdsmi_process_handle_t)

    info = amdsmi_wrapper.amdsmi_proc_info_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_process_info(
            processor_handle, process, ctypes.byref(info)
        )
    )

    return {
        "name": info.name.decode("utf-8"),
        "pid": info.pid,
        "mem": info.mem,
        "engine_usage": {
            "gfx": info.engine_usage.gfx,
            "enc": info.engine_usage.enc
        },
        "memory_usage": {
            "gtt_mem": info.memory_usage.gtt_mem,
            "cpu_mem": info.memory_usage.cpu_mem,
            "vram_mem": info.memory_usage.vram_mem,
        },
    }


def amdsmi_get_gpu_device_uuid(processor_handle: amdsmi_wrapper.amdsmi_processor_handle) -> str:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    uuid = ctypes.create_string_buffer(_AMDSMI_GPU_UUID_SIZE)

    uuid_length = ctypes.c_uint32()
    uuid_length.value = _AMDSMI_GPU_UUID_SIZE

    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_device_uuid(
            processor_handle, ctypes.byref(uuid_length), uuid
        )
    )

    return uuid.value.decode("utf-8")


def amdsmi_get_gpu_driver_info(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> Dict[str, Any]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    length = ctypes.c_int()
    length.value = _AMDSMI_MAX_DRIVER_VERSION_LENGTH

    info = amdsmi_wrapper.amdsmi_driver_info_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_driver_info(
            processor_handle, ctypes.byref(info)
        )
    )

    return {
        "driver_name": info.driver_name.decode("utf-8"),
        "driver_version": info.driver_version.decode("utf-8"),
        "driver_date": info.driver_date.decode("utf-8")
    }


def amdsmi_get_power_info(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> Dict[str, ctypes.c_uint32]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    power_measure = amdsmi_wrapper.amdsmi_power_info_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_power_info(
            processor_handle, ctypes.byref(power_measure)
        )
    )

    return {
        "current_socket_power": power_measure.current_socket_power,
        "average_socket_power": power_measure.average_socket_power,
        "gfx_voltage": power_measure.gfx_voltage,
        "soc_voltage": power_measure.soc_voltage,
        "mem_voltage": power_measure.mem_voltage,
        "power_limit" : power_measure.power_limit,
    }


def amdsmi_is_gpu_power_management_enabled(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle
    ) -> bool:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(processor_handle, amdsmi_wrapper.amdsmi_processor_handle)

    is_power_management_enabled = ctypes.c_bool()
    _check_res(
        amdsmi_wrapper.amdsmi_is_gpu_power_management_enabled(
            processor_handle, ctypes.byref(is_power_management_enabled)
        )
    )

    return is_power_management_enabled.value


def amdsmi_get_fw_info(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle
) -> List[Dict[str, Any]]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle)
    fw_info = amdsmi_wrapper.amdsmi_fw_info_t()
    _check_res(amdsmi_wrapper.amdsmi_get_fw_info(
        processor_handle, ctypes.byref(fw_info)))

    hex_format_fw = [AmdSmiFwBlock.FW_ID_PSP_SOSDRV,
                     AmdSmiFwBlock.FW_ID_TA_RAS,
                     AmdSmiFwBlock.FW_ID_TA_XGMI,
                     AmdSmiFwBlock.FW_ID_UVD,
                     AmdSmiFwBlock.FW_ID_VCE,
                     AmdSmiFwBlock.FW_ID_VCN]

    dec_format_fw = [AmdSmiFwBlock.FW_ID_PM]

    firmwares = []
    for i in range(0, fw_info.num_fw_info):
        fw_name = AmdSmiFwBlock(fw_info.fw_info_list[i].fw_id)
        fw_version = fw_info.fw_info_list[i].fw_version

        if fw_name in hex_format_fw:
            fw_version_string = ".".join(re.findall('..?', hex(fw_version)[2:]))
        elif fw_name in dec_format_fw:
            # Convert every two hex digits to decimal and join them with a dot
            dec_version_string = ''
            for ver1,ver2 in zip(hex(fw_version)[2::2], hex(fw_version)[3::2]):
                dec_version_string += str(int(f"0x{ver1}{ver2}", 0)) + "."
            fw_version_string = dec_version_string.strip('.')
        else:
            fw_version_string = str(fw_version)

        firmwares.append({
            'fw_name': fw_name,
            'fw_version': fw_version_string.upper(),
        })
    return {
        'fw_list': firmwares
    }


def amdsmi_get_gpu_vram_usage(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> Dict[str, Any]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    vram_usage = amdsmi_wrapper.amdsmi_vram_usage_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_vram_usage(
            processor_handle, ctypes.byref(vram_usage))
    )

    return {"vram_total": vram_usage.vram_total, "vram_used": vram_usage.vram_used}


def amdsmi_get_pcie_info(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> Dict[str, Any]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    pcie_info = amdsmi_wrapper.amdsmi_pcie_info_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_pcie_info(
            processor_handle, ctypes.byref(pcie_info)
        )
    )

    return {
        "pcie_static": {
            "max_pcie_width": pcie_info.pcie_static.max_pcie_width,
            "max_pcie_speed": pcie_info.pcie_static.max_pcie_speed,
            "pcie_interface_version": pcie_info.pcie_static.pcie_interface_version,
            "slot_type": pcie_info.pcie_static.slot_type,
            },
        "pcie_metric": {
            "pcie_width": pcie_info.pcie_metric.pcie_width,
            "pcie_speed": pcie_info.pcie_metric.pcie_speed,
            "pcie_bandwidth": pcie_info.pcie_metric.pcie_bandwidth,
            "pcie_replay_count": pcie_info.pcie_metric.pcie_replay_count,
            "pcie_l0_to_recovery_count": pcie_info.pcie_metric.pcie_l0_to_recovery_count,
            "pcie_replay_roll_over_count": pcie_info.pcie_metric.pcie_replay_roll_over_count,
            "pcie_nak_sent_count": pcie_info.pcie_metric.pcie_nak_sent_count,
            "pcie_nak_received_count": pcie_info.pcie_metric.pcie_nak_received_count,
        }
    }


def amdsmi_get_processor_handle_from_bdf(bdf):
    bdf = _parse_bdf(bdf)
    if bdf is None:
        raise AmdSmiBdfFormatException(bdf)
    amdsmi_bdf = _make_amdsmi_bdf_from_list(bdf)
    processor_handle = amdsmi_wrapper.amdsmi_processor_handle()
    _check_res(amdsmi_wrapper.amdsmi_get_processor_handle_from_bdf(
        amdsmi_bdf, ctypes.byref(processor_handle)))
    return processor_handle


def amdsmi_get_gpu_vendor_name(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> str:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    length = ctypes.c_uint64()
    length.value = _AMDSMI_STRING_LENGTH

    vendor_name = ctypes.create_string_buffer(_AMDSMI_STRING_LENGTH)

    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_vendor_name(
            processor_handle, vendor_name, length)
    )

    return vendor_name.value.decode("utf-8")


def amdsmi_get_gpu_id(processor_handle: amdsmi_wrapper.amdsmi_processor_handle):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    id = ctypes.c_uint16()

    _check_res(amdsmi_wrapper.amdsmi_get_gpu_id(
        processor_handle, ctypes.byref(id)))

    return id.value


def amdsmi_get_gpu_vram_vendor(processor_handle: amdsmi_wrapper.amdsmi_processor_handle):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    length = ctypes.c_uint32()
    length.value = _AMDSMI_STRING_LENGTH

    vram_vendor = ctypes.create_string_buffer(_AMDSMI_STRING_LENGTH)

    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_vram_vendor(
            processor_handle, vram_vendor, length)
    )

    return vram_vendor.value.decode("utf-8")


def amdsmi_get_gpu_subsystem_id(processor_handle: amdsmi_wrapper.amdsmi_processor_handle):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    id = ctypes.c_uint16()

    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_subsystem_id(
            processor_handle, ctypes.byref(id))
    )

    return id.value


def amdsmi_get_gpu_subsystem_name(processor_handle: amdsmi_wrapper.amdsmi_processor_handle):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    length = ctypes.c_uint64()
    length.value = _AMDSMI_STRING_LENGTH

    name = ctypes.create_string_buffer(_AMDSMI_STRING_LENGTH)

    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_subsystem_name(
            processor_handle, name, length)
    )

    return name.value.decode("utf-8")


def amdsmi_get_lib_version():
    version = amdsmi_wrapper.amdsmi_version_t()

    _check_res(amdsmi_wrapper.amdsmi_get_lib_version(ctypes.byref(version)))

    return {
        "year": version.year,
        "major": version.major,
        "minor": version.minor,
        "release": version.release,
        "build": version.build.contents.value.decode("utf-8")
    }


def amdsmi_topo_get_numa_node_number(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    numa_node_number = ctypes.c_uint32()

    _check_res(
        amdsmi_wrapper.amdsmi_topo_get_numa_node_number(
            processor_handle, ctypes.byref(numa_node_number)
        )
    )

    return numa_node_number.value


def amdsmi_topo_get_link_weight(
    processor_handle_src: amdsmi_wrapper.amdsmi_processor_handle,
    processor_handle_dst: amdsmi_wrapper.amdsmi_processor_handle,
):
    if not isinstance(processor_handle_src, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle_src, amdsmi_wrapper.amdsmi_processor_handle
        )

    if not isinstance(processor_handle_dst, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle_dst, amdsmi_wrapper.amdsmi_processor_handle
        )

    weight = ctypes.c_uint64()

    _check_res(
        amdsmi_wrapper.amdsmi_topo_get_link_weight(
            processor_handle_src, processor_handle_dst, ctypes.byref(weight)
        )
    )

    return weight.value


def amdsmi_get_minmax_bandwidth_between_processors(
    processor_handle_src: amdsmi_wrapper.amdsmi_processor_handle,
    processor_handle_dst: amdsmi_wrapper.amdsmi_processor_handle,
):
    if not isinstance(processor_handle_src, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle_src, amdsmi_wrapper.amdsmi_processor_handle
        )

    if not isinstance(processor_handle_dst, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle_dst, amdsmi_wrapper.amdsmi_processor_handle
        )

    min_bandwidth = ctypes.c_uint64()
    max_bandwidth = ctypes.c_uint64()

    _check_res(
        amdsmi_wrapper.amdsmi_get_minmax_bandwidth_between_processors(
            processor_handle_src,
            processor_handle_dst,
            ctypes.byref(min_bandwidth),
            ctypes.byref(max_bandwidth),
        )
    )

    return {"min_bandwidth": min_bandwidth.value, "max_bandwidth": max_bandwidth.value}


def amdsmi_topo_get_link_type(
    processor_handle_src: amdsmi_wrapper.amdsmi_processor_handle,
    processor_handle_dst: amdsmi_wrapper.amdsmi_processor_handle,
):
    if not isinstance(processor_handle_src, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle_src, amdsmi_wrapper.amdsmi_processor_handle
        )

    if not isinstance(processor_handle_dst, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle_dst, amdsmi_wrapper.amdsmi_processor_handle
        )

    hops = ctypes.c_uint64()
    #type = AmdSmiIoLinkType()
    type = ctypes.c_uint32()

    _check_res(
        amdsmi_wrapper.amdsmi_topo_get_link_type(
            #processor_handle_src, processor_handle_dst, ctypes.byref(hops), type
            processor_handle_src, processor_handle_dst, ctypes.byref(
                hops), ctypes.byref(type)
        )
    )

    return {"hops": hops.value, "type": type.value}


def amdsmi_is_P2P_accessible(
    processor_handle_src: amdsmi_wrapper.amdsmi_processor_handle,
    processor_handle_dst: amdsmi_wrapper.amdsmi_processor_handle,
):
    if not isinstance(processor_handle_src, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle_src, amdsmi_wrapper.amdsmi_processor_handle
        )

    if not isinstance(processor_handle_dst, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle_dst, amdsmi_wrapper.amdsmi_processor_handle
        )

    accessible = ctypes.c_bool()

    _check_res(
        amdsmi_wrapper.amdsmi_is_P2P_accessible(
            processor_handle_src, processor_handle_dst, ctypes.byref(accessible)
        )
    )

    return accessible.value


def amdsmi_get_gpu_compute_partition(processor_handle: amdsmi_wrapper.amdsmi_processor_handle):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    length = ctypes.c_uint32()
    length.value = _AMDSMI_STRING_LENGTH

    compute_partition = ctypes.create_string_buffer(_AMDSMI_STRING_LENGTH)

    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_compute_partition(
            processor_handle, compute_partition, length
        )
    )

    return compute_partition.value.decode("utf-8")


def amdsmi_set_gpu_compute_partition(processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
                                     compute_partition: AmdSmiComputePartitionType):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    if not isinstance(compute_partition, AmdSmiComputePartitionType):
        raise AmdSmiParameterException(compute_partition, AmdSmiComputePartitionType)

    _check_res(
        amdsmi_wrapper.amdsmi_set_gpu_compute_partition(
            processor_handle, compute_partition
        )
    )


def amdsmi_reset_gpu_compute_partition(processor_handle: amdsmi_wrapper.amdsmi_processor_handle):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    _check_res(amdsmi_wrapper.amdsmi_reset_gpu_compute_partition(processor_handle))


def amdsmi_get_gpu_memory_partition(processor_handle: amdsmi_wrapper.amdsmi_processor_handle):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    length = ctypes.c_uint32()
    length.value = _AMDSMI_STRING_LENGTH

    memory_partition = ctypes.create_string_buffer(_AMDSMI_STRING_LENGTH)

    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_memory_partition(
            processor_handle, memory_partition, length
        )
    )

    return memory_partition.value.decode("utf-8")


def amdsmi_set_gpu_memory_partition(processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
                                    memory_partition: AmdSmiMemoryPartitionType):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    if not isinstance(memory_partition, AmdSmiMemoryPartitionType):
        raise AmdSmiParameterException(memory_partition, AmdSmiMemoryPartitionType)

    _check_res(
        amdsmi_wrapper.amdsmi_set_gpu_memory_partition(
            processor_handle, memory_partition
        )
    )


def amdsmi_reset_gpu_memory_partition(processor_handle: amdsmi_wrapper.amdsmi_processor_handle):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    _check_res(amdsmi_wrapper.amdsmi_reset_gpu_memory_partition(processor_handle))


def amdsmi_get_xgmi_info(processor_handle: amdsmi_wrapper.amdsmi_processor_handle):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    xgmi_info = amdsmi_wrapper.amdsmi_xgmi_info_t()

    _check_res(amdsmi_wrapper.amdsmi_get_xgmi_info(processor_handle, xgmi_info))

    return {
        "xgmi_lanes": xgmi_info.xgmi_lanes,
        "xgmi_hive_id": xgmi_info.xgmi_hive_id,
        "xgmi_node_id": xgmi_info.xgmi_node_id,
        "index": xgmi_info.index,
    }


def amdsmi_gpu_counter_group_supported(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    event_group: AmdSmiEventGroup,
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(event_group, AmdSmiEventGroup):
        raise AmdSmiParameterException(event_group, AmdSmiEventGroup)

    _check_res(
        amdsmi_wrapper.amdsmi_gpu_counter_group_supported(
            processor_handle, event_group)
    )


def amdsmi_gpu_create_counter(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    event_type: AmdSmiEventType,
) -> amdsmi_wrapper.amdsmi_event_handle_t:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(event_type, AmdSmiEventType):
        raise AmdSmiParameterException(event_type, AmdSmiEventType)

    event_handle = amdsmi_wrapper.amdsmi_event_handle_t()
    _check_res(
        amdsmi_wrapper.amdsmi_gpu_create_counter(
            processor_handle, event_type, ctypes.byref(event_handle)
        )
    )

    return event_handle


def amdsmi_gpu_destroy_counter(event_handle: amdsmi_wrapper.amdsmi_event_handle_t):
    if not isinstance(event_handle, amdsmi_wrapper.amdsmi_event_handle_t):
        raise AmdSmiParameterException(
            event_handle, amdsmi_wrapper.amdsmi_event_handle_t
        )
    _check_res(amdsmi_wrapper.amdsmi_gpu_destroy_counter(event_handle))


def amdsmi_gpu_control_counter(
    event_handle: amdsmi_wrapper.amdsmi_event_handle_t,
    counter_command: AmdSmiCounterCommand,
):
    if not isinstance(event_handle, amdsmi_wrapper.amdsmi_event_handle_t):
        raise AmdSmiParameterException(
            event_handle, amdsmi_wrapper.amdsmi_event_handle_t
        )
    if not isinstance(counter_command, AmdSmiCounterCommand):
        raise AmdSmiParameterException(counter_command, AmdSmiCounterCommand)
    command_args = ctypes.c_void_p()

    _check_res(
        amdsmi_wrapper.amdsmi_gpu_control_counter(
            event_handle, counter_command, command_args
        )
    )


def amdsmi_gpu_read_counter(
    event_handle: amdsmi_wrapper.amdsmi_event_handle_t,
) -> Dict[str, Any]:
    if not isinstance(event_handle, amdsmi_wrapper.amdsmi_event_handle_t):
        raise AmdSmiParameterException(
            event_handle, amdsmi_wrapper.amdsmi_event_handle_t
        )

    counter_value = amdsmi_wrapper.amdsmi_counter_value_t()

    _check_res(
        amdsmi_wrapper.amdsmi_gpu_read_counter(
            event_handle, ctypes.byref(counter_value))
    )

    return {
        "value": counter_value.value,
        "time_enabled": counter_value.time_enabled,
        "time_running": counter_value.time_running,
    }


def amdsmi_get_gpu_available_counters(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    event_group: AmdSmiEventGroup,
) -> int:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(event_group, AmdSmiEventGroup):
        raise AmdSmiParameterException(event_group, AmdSmiEventGroup)
    available = ctypes.c_uint32()

    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_available_counters(
            processor_handle, event_group, ctypes.byref(available)
        )
    )

    return available.value


def amdsmi_set_gpu_perf_level(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    perf_level: AmdSmiDevPerfLevel,
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(perf_level, AmdSmiDevPerfLevel):
        raise AmdSmiParameterException(perf_level, AmdSmiDevPerfLevel)

    _check_res(amdsmi_wrapper.amdsmi_set_gpu_perf_level(
        processor_handle, perf_level))


def amdsmi_reset_gpu(processor_handle: amdsmi_wrapper.amdsmi_processor_handle):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    _check_res(amdsmi_wrapper.amdsmi_reset_gpu(processor_handle))


def amdsmi_set_gpu_fan_speed(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle, sensor_idx: int, fan_speed: int
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(sensor_idx, int):
        raise AmdSmiParameterException(sensor_idx, int)
    if not isinstance(fan_speed, int):
        raise AmdSmiParameterException(fan_speed, int)
    sensor_idx = ctypes.c_uint32(sensor_idx)
    fan_speed = ctypes.c_uint64(fan_speed)

    _check_res(
        amdsmi_wrapper.amdsmi_set_gpu_fan_speed(
            processor_handle, sensor_idx, fan_speed)
    )


def amdsmi_reset_gpu_fan(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle, sensor_idx: int
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(sensor_idx, int):
        raise AmdSmiParameterException(sensor_idx, int)
    sensor_idx = ctypes.c_uint32(sensor_idx)

    _check_res(amdsmi_wrapper.amdsmi_reset_gpu_fan(processor_handle, sensor_idx))


def amdsmi_set_clk_freq(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    clk_type: AmdSmiClkType,
    freq_bitmask: int,
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(clk_type, AmdSmiClkType):
        raise AmdSmiParameterException(clk_type, AmdSmiParameterException)
    if not isinstance(freq_bitmask, int):
        raise AmdSmiParameterException(freq_bitmask, int)
    freq_bitmask = ctypes.c_uint64(freq_bitmask)
    _check_res(
        amdsmi_wrapper.amdsmi_set_clk_freq(
            processor_handle, clk_type, freq_bitmask
        )
    )

def amdsmi_set_dpm_policy(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    policy_id: int,
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    _check_res(
        amdsmi_wrapper.amdsmi_set_dpm_policy(
            processor_handle, policy_id
        )
    )

def amdsmi_set_gpu_overdrive_level(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle, overdrive_value: int
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(overdrive_value, int):
        raise AmdSmiParameterException(overdrive_value, int)
    overdrive_value = ctypes.c_uint32(overdrive_value)

    _check_res(
        amdsmi_wrapper.amdsmi_set_gpu_overdrive_level(
            processor_handle, overdrive_value)
    )


def amdsmi_get_gpu_bdf_id(processor_handle: amdsmi_wrapper.amdsmi_processor_handle):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    bdfid = ctypes.c_uint64()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_bdf_id(
            processor_handle, ctypes.byref(bdfid))
    )

    return bdfid.value

def amdsmi_set_gpu_pci_bandwidth(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle, bitmask: int
) -> None:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    if not isinstance(bitmask, int):
        raise AmdSmiParameterException(bitmask, int)

    _check_res(
        amdsmi_wrapper.amdsmi_set_gpu_pci_bandwidth(
            processor_handle, ctypes.c_uint64(bitmask)
        )
    )


def _format_transfer_rate(transfer_rate):
    return {
        'num_supported': transfer_rate.num_supported,
        'current': transfer_rate.current,
        'frequency': list(transfer_rate.frequency)
    }


def amdsmi_get_gpu_pci_bandwidth(processor_handle: amdsmi_wrapper.amdsmi_processor_handle):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    bandwidth = amdsmi_wrapper.amdsmi_pcie_bandwidth_t()

    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_pci_bandwidth(
            processor_handle, ctypes.byref(bandwidth))
    )

    transfer_rate = _format_transfer_rate(bandwidth.transfer_rate)

    return {
        'transfer_rate': transfer_rate,
        'lanes': list(bandwidth.lanes)
    }


def amdsmi_get_gpu_pci_throughput(processor_handle: amdsmi_wrapper.amdsmi_processor_handle):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    sent = ctypes.c_uint64()
    received = ctypes.c_uint64()
    max_pkt_sz = ctypes.c_uint64()

    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_pci_throughput(processor_handle, ctypes.byref(
            sent), ctypes.byref(received), ctypes.byref(max_pkt_sz))
    )

    return {
        'sent': sent.value,
        'received': received.value,
        'max_pkt_sz': max_pkt_sz.value
    }


def amdsmi_get_gpu_pci_replay_counter(processor_handle: amdsmi_wrapper.amdsmi_processor_handle):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    counter = ctypes.c_uint64()

    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_pci_replay_counter(
            processor_handle, ctypes.byref(counter))
    )

    return counter.value


def amdsmi_get_gpu_topo_numa_affinity(processor_handle: amdsmi_wrapper.amdsmi_processor_handle):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    numa_node = ctypes.c_int32()

    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_topo_numa_affinity(
            processor_handle, ctypes.byref(numa_node))
    )

    return numa_node.value


def amdsmi_set_power_cap(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle, sensor_ind: int, cap: int
) -> None:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    if not isinstance(sensor_ind, int):
        raise AmdSmiParameterException(sensor_ind, int)

    if not isinstance(cap, int):
        raise AmdSmiParameterException(cap, int)

    _check_res(
        amdsmi_wrapper.amdsmi_set_power_cap(
            processor_handle, ctypes.c_uint32(sensor_ind), ctypes.c_uint64(cap)
        )
    )


def amdsmi_set_gpu_power_profile(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    reserved: int,
    profile: AmdSmiPowerProfilePresetMasks,
) -> None:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    if not isinstance(reserved, int):
        raise AmdSmiParameterException(reserved, int)

    if not isinstance(profile, AmdSmiPowerProfilePresetMasks):
        raise AmdSmiParameterException(profile, AmdSmiPowerProfilePresetMasks)

    _check_res(
        amdsmi_wrapper.amdsmi_set_gpu_power_profile(
            processor_handle, ctypes.c_uint32(reserved), profile
        )
    )


def amdsmi_get_energy_count(processor_handle: amdsmi_wrapper.amdsmi_processor_handle):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    power = ctypes.c_uint64()
    counter_resolution = ctypes.c_float()
    timestamp = ctypes.c_uint64()

    _check_res(
        amdsmi_wrapper.amdsmi_get_energy_count(processor_handle, ctypes.byref(
            power), ctypes.byref(counter_resolution), ctypes.byref(timestamp))
    )

    return {
        'power': power.value,
        'counter_resolution': counter_resolution.value,
        'timestamp': timestamp.value,
    }


def amdsmi_set_gpu_clk_range(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    min_clk_value: int,
    max_clk_value: int,
    clk_type: AmdSmiClkType,
) -> None:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    if not isinstance(min_clk_value, int):
        raise AmdSmiParameterException(min_clk_value, int)

    if not isinstance(max_clk_value, int):
        raise AmdSmiParameterException(min_clk_value, int)

    if not isinstance(clk_type, AmdSmiClkType):
        raise AmdSmiParameterException(clk_type, AmdSmiClkType)

    _check_res(
        amdsmi_wrapper.amdsmi_set_gpu_clk_range(
            processor_handle,
            ctypes.c_uint64(min_clk_value),
            ctypes.c_uint64(max_clk_value),
            clk_type,
        )
    )


def amdsmi_get_gpu_memory_total(processor_handle: amdsmi_wrapper.amdsmi_processor_handle, mem_type: AmdSmiMemoryType):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    if not isinstance(mem_type, AmdSmiMemoryType):
        raise AmdSmiParameterException(
            mem_type, AmdSmiMemoryType
        )

    total = ctypes.c_uint64()

    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_memory_total(
            processor_handle, mem_type, ctypes.byref(total))
    )

    return total.value


def amdsmi_set_gpu_od_clk_info(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    level: AmdSmiFreqInd,
    value: int,
    clk_type: AmdSmiClkType,
) -> None:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    if not isinstance(level, AmdSmiFreqInd):
        raise AmdSmiParameterException(level, AmdSmiFreqInd)

    if not isinstance(value, int):
        raise AmdSmiParameterException(value, int)

    if not isinstance(clk_type, AmdSmiClkType):
        raise AmdSmiParameterException(clk_type, AmdSmiClkType)

    _check_res(
        amdsmi_wrapper.amdsmi_set_gpu_od_clk_info(
            processor_handle, level, ctypes.c_uint64(value), clk_type
        )
    )


def amdsmi_get_gpu_memory_usage(processor_handle: amdsmi_wrapper.amdsmi_processor_handle, mem_type: AmdSmiMemoryType):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    if not isinstance(mem_type, AmdSmiMemoryType):
        raise AmdSmiParameterException(
            mem_type, AmdSmiMemoryType
        )

    used = ctypes.c_uint64()

    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_memory_usage(
            processor_handle, mem_type, ctypes.byref(used))
    )

    return used.value


def amdsmi_set_gpu_od_volt_info(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    vpoint: int,
    clk_value: int,
    volt_value: int,
) -> None:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    if not isinstance(vpoint, int):
        raise AmdSmiParameterException(vpoint, int)

    if not isinstance(clk_value, int):
        raise AmdSmiParameterException(clk_value, int)

    if not isinstance(volt_value, int):
        raise AmdSmiParameterException(volt_value, int)

    _check_res(
        amdsmi_wrapper.amdsmi_set_gpu_od_volt_info(
            processor_handle,
            ctypes.c_uint32(vpoint),
            ctypes.c_uint64(clk_value),
            ctypes.c_uint64(volt_value),
        )
    )



def amdsmi_get_gpu_fan_rpms(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle, sensor_idx: int
) -> int:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(sensor_idx, int):
        raise AmdSmiParameterException(sensor_idx, int)
    fan_speed = ctypes.c_int64()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_fan_rpms(
            processor_handle, sensor_idx, ctypes.byref(fan_speed)
        )
    )

    return fan_speed.value


def amdsmi_get_gpu_fan_speed(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle, sensor_idx: int
) -> int:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(sensor_idx, int):
        raise AmdSmiParameterException(sensor_idx, int)
    fan_speed = ctypes.c_int64()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_fan_speed(
            processor_handle, sensor_idx, ctypes.byref(fan_speed)
        )
    )

    return fan_speed.value


def amdsmi_get_gpu_fan_speed_max(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle, sensor_idx: int
) -> int:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(sensor_idx, int):
        raise AmdSmiParameterException(sensor_idx, int)
    fan_speed = ctypes.c_uint64()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_fan_speed_max(
            processor_handle, sensor_idx, ctypes.byref(fan_speed)
        )
    )

    return fan_speed.value


def amdsmi_get_temp_metric(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    sensor_type: AmdSmiTemperatureType,
    metric: AmdSmiTemperatureMetric,
) -> int:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(sensor_type, AmdSmiTemperatureType):
        raise AmdSmiParameterException(sensor_type, AmdSmiTemperatureType)
    if not isinstance(metric, AmdSmiTemperatureMetric):
        raise AmdSmiParameterException(metric, AmdSmiTemperatureMetric)

    temp_value = ctypes.c_int64()
    _check_res(
        amdsmi_wrapper.amdsmi_get_temp_metric(
            processor_handle, sensor_type, metric, ctypes.byref(temp_value)
        )
    )

    return temp_value.value


def amdsmi_get_gpu_volt_metric(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    sensor_type: AmdSmiVoltageType,
    metric: AmdSmiVoltageMetric,
) -> int:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(sensor_type, AmdSmiVoltageType):
        raise AmdSmiParameterException(sensor_type, AmdSmiVoltageType)
    if not isinstance(metric, AmdSmiVoltageMetric):
        raise AmdSmiParameterException(metric, AmdSmiVoltageMetric)

    voltage = ctypes.c_int64()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_volt_metric(
            processor_handle, sensor_type, metric, ctypes.byref(voltage)
        )
    )

    return voltage.value


def amdsmi_get_utilization_count(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    counter_types: List[AmdSmiUtilizationCounterType]
) -> List[Dict[str, Any]]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if len(counter_types) == 0:
        raise AmdSmiLibraryException(amdsmi_wrapper.AMDSMI_STATUS_INVAL)
    counters = []
    for counter_type in counter_types:
        if not isinstance(counter_type, AmdSmiUtilizationCounterType):
            raise AmdSmiParameterException(
                counter_type, AmdSmiUtilizationCounterType)
        counter = amdsmi_wrapper.amdsmi_utilization_counter_t()
        counter.type = counter_type
        counters.append(counter)

    count = ctypes.c_uint32(len(counters))
    timestamp = ctypes.c_uint64()
    util_counter_list = (amdsmi_wrapper.amdsmi_utilization_counter_t * len(counters))(*counters)

    _check_res(
        amdsmi_wrapper.amdsmi_get_utilization_count(
            processor_handle, util_counter_list, count, ctypes.byref(timestamp)
        )
    )
    if count.value != len(counters):
        raise AmdSmiLibraryException(amdsmi_wrapper.AMDSMI_STATUS_API_FAILED)

    result = [{"timestamp": timestamp.value}]
    for index in range(count.value):
        counter_type = amdsmi_wrapper.amdsmi_utilization_counter_type_t__enumvalues[
            util_counter_list[index].type
        ]
        if counter_type == "AMDSMI_UTILIZATION_COUNTER_FIRST":
            counter_type = "AMDSMI_COARSE_GRAIN_GPU_ACTIVITY"
        if counter_type == "AMDSMI_UTILIZATION_COUNTER_LAST":
            counter_type = "AMDSMI_COARSE_GRAIN_MEM_ACTIVITY"
        result.append(
            {"type": counter_type, "value": util_counter_list[index].value})

    return result


def amdsmi_get_gpu_perf_level(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> str:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    perf = amdsmi_wrapper.amdsmi_dev_perf_level_t()

    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_perf_level(
            processor_handle, ctypes.byref(perf))
    )

    result = amdsmi_wrapper.amdsmi_dev_perf_level_t__enumvalues[perf.value]
    if result == "AMDSMI_DEV_PERF_LEVEL_FIRST":
        result = "AMDSMI_DEV_PERF_LEVEL_AUTO"
    if result == "AMDSMI_DEV_PERF_LEVEL_LAST":
        result = "AMDSMI_DEV_PERF_LEVEL_DETERMINISM"

    return result


def amdsmi_set_gpu_perf_determinism_mode(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle, clkvalue: int
) -> None:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(clkvalue, int):
        raise AmdSmiParameterException(clkvalue, int)

    _check_res(amdsmi_wrapper.amdsmi_set_gpu_perf_determinism_mode(
        processor_handle, clkvalue))


def amdsmi_get_gpu_overdrive_level(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> int:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    od_level = ctypes.c_uint32()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_overdrive_level(
            processor_handle, ctypes.byref(od_level)
        )
    )

    return od_level.value


def amdsmi_get_clk_freq(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle, clk_type: AmdSmiClkType
) -> Dict[str, Any]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(clk_type, AmdSmiClkType):
        raise AmdSmiParameterException(clk_type, AmdSmiClkType)

    freq = amdsmi_wrapper.amdsmi_frequencies_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_clk_freq(
            processor_handle, clk_type, ctypes.byref(freq)
        )
    )

    return {
        "num_supported": freq.num_supported,
        "current": freq.current,
        "frequency": list(freq.frequency)[: freq.num_supported - 1],
    }

def amdsmi_get_dpm_policy(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> Dict[str, Any]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    policy = amdsmi_wrapper.amdsmi_dpm_policy_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_dpm_policy(
            processor_handle, ctypes.byref(policy)
        )
    )
        
    polices = []
    for i in range(0, policy.num_supported):
        id = policy.policies[i].policy_id
        desc = policy.policies[i].policy_description
        polices.append({
            'policy_id' : id,
            'policy_description': desc.decode()
        })
    current_id = policy.policies[policy.current].policy_id

    return  {
        "num_supported": policy.num_supported,
        "current_id": current_id,
        "policies": polices,
    }

def amdsmi_get_gpu_od_volt_info(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> Dict[str, Any]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    freq_data = amdsmi_wrapper.amdsmi_od_volt_freq_data_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_od_volt_info(
            processor_handle, ctypes.byref(freq_data)
        )
    )

    return {
        "curr_sclk_range": {
            "lower_bound": freq_data.curr_sclk_range.lower_bound,
            "upper_bound": freq_data.curr_sclk_range.upper_bound,
        },
        "curr_mclk_range": {
            "lower_bound": freq_data.curr_mclk_range.lower_bound,
            "upper_bound": freq_data.curr_mclk_range.upper_bound,
        },
        "sclk_freq_limits": {
            "lower_bound": freq_data.sclk_freq_limits.lower_bound,
            "upper_bound": freq_data.sclk_freq_limits.upper_bound,
        },
        "mclk_freq_limits": {
            "lower_bound": freq_data.mclk_freq_limits.lower_bound,
            "upper_bound": freq_data.mclk_freq_limits.upper_bound,
        },
        "curve.vc_points": list(freq_data.curve.vc_points),
        "num_regions": freq_data.num_regions,
    }


def amdsmi_get_gpu_metrics_info(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> Dict[str, Any]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    gpu_metrics = amdsmi_wrapper.amdsmi_gpu_metrics_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_metrics_info(
            processor_handle, ctypes.byref(gpu_metrics)
        )
    )

    gpu_metrics_output = {
        "temperature_edge": gpu_metrics.temperature_edge,
        "temperature_hotspot": gpu_metrics.temperature_hotspot,
        "temperature_mem": gpu_metrics.temperature_mem,
        "temperature_vrgfx": gpu_metrics.temperature_vrgfx,
        "temperature_vrsoc": gpu_metrics.temperature_vrsoc,
        "temperature_vrmem": gpu_metrics.temperature_vrmem,
        "average_gfx_activity": gpu_metrics.average_gfx_activity,
        "average_umc_activity": gpu_metrics.average_umc_activity,
        "average_mm_activity": gpu_metrics.average_mm_activity,
        "average_socket_power": gpu_metrics.average_socket_power,
        "energy_accumulator": gpu_metrics.energy_accumulator,
        "system_clock_counter": gpu_metrics.system_clock_counter,
        "average_gfxclk_frequency": gpu_metrics.average_gfxclk_frequency,
        "average_socclk_frequency": gpu_metrics.average_socclk_frequency,
        "average_uclk_frequency": gpu_metrics.average_uclk_frequency,
        "average_vclk0_frequency": gpu_metrics.average_vclk0_frequency,
        "average_dclk0_frequency": gpu_metrics.average_dclk0_frequency,
        "average_vclk1_frequency": gpu_metrics.average_vclk1_frequency,
        "average_dclk1_frequency": gpu_metrics.average_dclk1_frequency,
        "current_gfxclk": gpu_metrics.current_gfxclk,
        "current_socclk": gpu_metrics.current_socclk,
        "current_uclk": gpu_metrics.current_uclk,
        "current_vclk0": gpu_metrics.current_vclk0,
        "current_dclk0": gpu_metrics.current_dclk0,
        "current_vclk1": gpu_metrics.current_vclk1,
        "current_dclk1": gpu_metrics.current_dclk1,
        "throttle_status": gpu_metrics.throttle_status,
        "current_fan_speed": gpu_metrics.current_fan_speed,
        "pcie_link_width": gpu_metrics.pcie_link_width,
        "pcie_link_speed": gpu_metrics.pcie_link_speed,
        "gfx_activity_acc": gpu_metrics.gfx_activity_acc,
        "mem_activity_acc": gpu_metrics.mem_activity_acc,
        "temperature_hbm": list(gpu_metrics.temperature_hbm),
        "firmware_timestamp": gpu_metrics.firmware_timestamp,
        "voltage_soc": gpu_metrics.voltage_soc,
        "voltage_gfx": gpu_metrics.voltage_gfx,
        "voltage_mem": gpu_metrics.voltage_mem,
        "indep_throttle_status": gpu_metrics.indep_throttle_status,
        "current_socket_power": gpu_metrics.current_socket_power,
        "vcn_activity": list(gpu_metrics.vcn_activity),
        "gfxclk_lock_status": gpu_metrics.gfxclk_lock_status,
        "xgmi_link_width": gpu_metrics.xgmi_link_width,
        "xgmi_link_speed": gpu_metrics.xgmi_link_speed,
        "pcie_bandwidth_acc": gpu_metrics.pcie_bandwidth_acc,
        "pcie_bandwidth_inst": gpu_metrics.pcie_bandwidth_inst,
        "pcie_l0_to_recov_count_acc": gpu_metrics.pcie_l0_to_recov_count_acc,
        "pcie_replay_count_acc": gpu_metrics.pcie_replay_count_acc,
        "pcie_replay_rover_count_acc": gpu_metrics.pcie_replay_rover_count_acc,
        "xgmi_read_data_acc": list(gpu_metrics.xgmi_read_data_acc),
        "xgmi_write_data_acc": list(gpu_metrics.xgmi_write_data_acc),
        "current_gfxclks": list(gpu_metrics.current_gfxclks),
        "current_socclks": list(gpu_metrics.current_socclks),
        "current_vclk0s": list(gpu_metrics.current_vclk0s),
        "current_dclk0s": list(gpu_metrics.current_dclk0s),
        "pcie_nak_sent_count_acc": gpu_metrics.pcie_nak_sent_count_acc,
        "pcie_nak_rcvd_count_acc": gpu_metrics.pcie_nak_rcvd_count_acc,
        "jpeg_activity": list(gpu_metrics.jpeg_activity),
    }

    # Validate support for each gpu_metric
    uint_16_metrics = ['temperature_edge', 'temperature_hotspot', 'temperature_mem',
                     'temperature_vrgfx', 'temperature_vrsoc', 'temperature_vrmem',
                     'average_gfx_activity', 'average_umc_activity', 'average_mm_activity',
                     'average_socket_power', 'average_gfxclk_frequency', 'average_socclk_frequency',
                     'average_uclk_frequency', 'average_vclk0_frequency', 'average_dclk0_frequency',
                     'average_vclk1_frequency', 'average_dclk1_frequency', 'current_gfxclk',
                     'current_socclk', 'current_uclk', 'current_vclk0', 'current_dclk0',
                     'current_vclk1', 'current_dclk1', 'current_fan_speed', 'pcie_link_width',
                     'pcie_link_speed', 'voltage_soc', 'voltage_gfx', 'voltage_mem',
                     'current_socket_power', 'xgmi_link_width', 'xgmi_link_speed']
    for metric in uint_16_metrics:
        if gpu_metrics_output[metric] == 0xFFFF:
            gpu_metrics_output[metric] = "N/A"

    uint_32_metrics = ['gfx_activity_acc','mem_activity_acc', 'pcie_nak_sent_count_acc', 'pcie_nak_rcvd_count_acc']
    for metric in uint_32_metrics:
        if gpu_metrics_output[metric] == 0xFFFFFFFF:
            gpu_metrics_output[metric] = "N/A"

    uint_64_metrics = ['energy_accumulator', 'system_clock_counter', 'firmware_timestamp',
                      'pcie_bandwidth_acc', 'pcie_bandwidth_inst',
                      'pcie_l0_to_recov_count_acc', 'pcie_replay_count_acc',
                      'pcie_replay_rover_count_acc']
    for metric in uint_64_metrics:
        if gpu_metrics_output[metric] == 0xFFFFFFFFFFFFFFFF:
            gpu_metrics_output[metric] = "N/A"

    # Custom validation for metrics in a bool format
    uint_32_bool_metrics = ['throttle_status', 'gfxclk_lock_status']
    for metric in uint_32_bool_metrics:
        if gpu_metrics_output[metric] == 0xFFFFFFFF:
            gpu_metrics_output[metric] = "N/A"
        else:
            gpu_metrics_output[metric] = bool(gpu_metrics_output[metric])

    # Custom validation for metrics in a list format
    uint_16_clock_list_metrics = ['current_gfxclks', 'current_socclks', 'current_vclk0s', 'current_dclk0s']
    for clock in uint_16_clock_list_metrics:
        for index, clk in enumerate(gpu_metrics_output[clock]):
            if clk == 0xFFFF:
                gpu_metrics_output[clock][index] = "N/A"

    uint_16_activity_list_metrics = ['vcn_activity', 'jpeg_activity']
    for activity_metric in uint_16_activity_list_metrics:
        for index, activity in enumerate(gpu_metrics_output[activity_metric]):
            if activity == 0xFFFF or activity > 110:
                gpu_metrics_output[activity_metric][index] = "N/A"

    uint_64_xgmi_metrics = ['xgmi_read_data_acc', 'xgmi_write_data_acc']
    for metric in uint_64_xgmi_metrics:
        for index, data in enumerate(gpu_metrics_output[metric]):
            if data == 0xFFFFFFFFFFFFFFFF:
                gpu_metrics_output[metric][index] = "N/A"

    # Custom validation for specific gpu_metrics
    for index, temp in enumerate(gpu_metrics_output['temperature_hbm']):
        if temp == 0xFFFF:
            gpu_metrics_output['temperature_hbm'][index] = "N/A"

    if gpu_metrics_output['indep_throttle_status'] == 0xFFFFFFFFFFFFFFFF:
        gpu_metrics_output['indep_throttle_status'] = "N/A"
    else:
        gpu_metrics_output['indep_throttle_status'] = bool(gpu_metrics_output['indep_throttle_status'])

    return gpu_metrics_output


def amdsmi_get_gpu_od_volt_curve_regions(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle, num_regions: int
) -> List[Dict[str, Any]]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(num_regions, int):
        raise AmdSmiParameterException(num_regions, int)

    region_count = ctypes.c_uint32(num_regions)
    buffer = (amdsmi_wrapper.amdsmi_freq_volt_region_t * num_regions)()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_od_volt_curve_regions(
            processor_handle, ctypes.byref(region_count), buffer
        )
    )

    result = []

    for index in range(region_count.value):
        result.extend(
            [
                {
                    "freq_range": {
                        "lower_bound": buffer[index].freq_range.lower_bound,
                        "upper_bound": buffer[index].freq_range.upper_bound,
                    },
                    "volt_range": {
                        "lower_bound": buffer[index].volt_range.lower_bound,
                        "upper_bound": buffer[index].volt_range.upper_bound,
                    },
                }
            ]
        )

    return result


def amdsmi_get_gpu_power_profile_presets(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle, sensor_idx: int
) -> Dict[str, Any]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(sensor_idx, int):
        raise AmdSmiParameterException(sensor_idx, int)

    status = amdsmi_wrapper.amdsmi_power_profile_status_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_power_profile_presets(
            processor_handle, sensor_idx, ctypes.byref(status)
        )
    )

    return {
        "available_profiles": status.available_profiles,
        "current": status.current,
        "num_profiles": status.num_profiles,
    }


def amdsmi_get_gpu_ecc_count(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle, block: AmdSmiGpuBlock
) -> Dict[str, int]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    if not isinstance(block, AmdSmiGpuBlock):
        raise AmdSmiParameterException(block, AmdSmiGpuBlock)

    ec = amdsmi_wrapper.amdsmi_error_count_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_ecc_count(
            processor_handle, block, ctypes.byref(ec))
    )

    return {
        "correctable_count": ec.correctable_count,
        "uncorrectable_count": ec.uncorrectable_count,
    }


def amdsmi_get_gpu_ecc_enabled(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> int:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    blocks = ctypes.c_uint64(0)
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_ecc_enabled(
            processor_handle, ctypes.byref(blocks))
    )

    return blocks.value


def amdsmi_get_gpu_ecc_status(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle, block: AmdSmiGpuBlock
) -> AmdSmiRasErrState:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    if not isinstance(block, AmdSmiGpuBlock):
        raise AmdSmiParameterException(block, AmdSmiGpuBlock)

    state = amdsmi_wrapper.amdsmi_ras_err_state_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_ecc_status(
            processor_handle, block, ctypes.byref(state)
        )
    )

    return AmdSmiRasErrState(state.value)


def amdsmi_status_code_to_string(status: amdsmi_wrapper.amdsmi_status_t) -> str:
    if not isinstance(status, amdsmi_wrapper.amdsmi_status_t):
        raise AmdSmiParameterException(status, amdsmi_wrapper.amdsmi_status_t)

    status_string_p_p = ctypes.pointer(ctypes.pointer(ctypes.c_char()))

    _check_res(amdsmi_wrapper.amdsmi_status_code_to_string(
        status, status_string_p_p))

    return amdsmi_wrapper.string_cast(status_string_p_p.contents)


def amdsmi_get_gpu_compute_process_info() -> List[Dict[str, int]]:
    num_items = ctypes.c_uint32(0)
    nullptr = ctypes.POINTER(amdsmi_wrapper.amdsmi_process_info_t)()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_compute_process_info(
            nullptr, ctypes.byref(num_items))
    )

    procs = (amdsmi_wrapper.amdsmi_process_info_t * num_items.value)()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_compute_process_info(
            procs, ctypes.byref(num_items))
    )

    return [
        {
            "process_id": proc.process_id,
            "pasid": proc.pasid,
            "vram_usage": proc.vram_usage,
            "sdma_usage": proc.sdma_usage,
            "cu_occupancy": proc.cu_occupancy,
        }
        for proc in procs
    ]


def amdsmi_get_gpu_compute_process_info_by_pid(pid: int) -> Dict[str, int]:
    if not isinstance(pid, int):
        raise AmdSmiParameterException(pid, int)

    proc = amdsmi_wrapper.amdsmi_process_info_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_compute_process_info_by_pid(
            ctypes.c_uint32(pid), ctypes.byref(proc)
        )
    )

    return {
        "process_id": proc.process_id,
        "pasid": proc.pasid,
        "vram_usage": proc.vram_usage,
        "sdma_usage": proc.sdma_usage,
        "cu_occupancy": proc.cu_occupancy,
    }


def amdsmi_get_gpu_compute_process_gpus(pid: int) -> List[int]:
    if not isinstance(pid, int):
        raise AmdSmiParameterException(pid, int)

    num_devices = ctypes.c_uint32(0)
    nullptr = ctypes.POINTER(ctypes.c_uint32)()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_compute_process_gpus(
            pid, nullptr, ctypes.byref(num_devices)
        )
    )

    dv_indices = (ctypes.c_uint32 * num_devices.value)()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_compute_process_gpus(
            pid, dv_indices, ctypes.byref(num_devices)
        )
    )

    return [dv_index.value for dv_index in dv_indices]


def amdsmi_gpu_xgmi_error_status(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> AmdSmiXgmiStatus:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    status = amdsmi_wrapper.amdsmi_xgmi_status_t()
    _check_res(
        amdsmi_wrapper.amdsmi_gpu_xgmi_error_status(
            processor_handle, ctypes.byref(status))
    )

    return AmdSmiXgmiStatus(status.value).value


def amdsmi_reset_gpu_xgmi_error(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> None:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    _check_res(amdsmi_wrapper.amdsmi_reset_gpu_xgmi_error(processor_handle))


def amdsmi_get_gpu_memory_reserved_pages(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> Union[list, str]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    num_pages = ctypes.c_uint32()
    retired_page_record = ctypes.POINTER(
        amdsmi_wrapper.amdsmi_retired_page_record_t)()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_memory_reserved_pages(
            processor_handle, ctypes.byref(num_pages), retired_page_record
        )
    )

    table_records = _format_bad_page_info(retired_page_record, num_pages)
    if num_pages.value == 0:
        return "No bad pages found."
    else:
        table_records = _format_bad_page_info(retired_page_record, num_pages)

    return table_records


def amdsmi_get_gpu_metrics_header_info(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> Dict[str, int]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    header_info = amdsmi_wrapper.amd_metrics_table_header_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_metrics_header_info(
            ctypes.byref(header_info)
        )
    )

    return {
        "structure_size": header_info.structure_size.value,
        "format_revision": header_info.format_revision.value,
        "content_revision": header_info.content_revision.value
    }
