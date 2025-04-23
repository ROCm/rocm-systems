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
import json
import logging
import math
import os
import re
import sys
from collections.abc import Iterable
from enum import IntEnum, Enum
from pathlib import Path
from time import asctime, localtime, time
from typing import Any, Dict, List, Tuple, Union

from . import amdsmi_wrapper
from .amdsmi_exception import *


### Non Library Specific Constants ###
class MaxUIntegerTypes(IntEnum):
    UINT8_T  = 0xFF
    UINT16_T = 0xFFFF
    UINT32_T = 0xFFFFFFFF
    UINT64_T = 0xFFFFFFFFFFFFFFFF

NO_OF_32BITS = (sys.getsizeof(ctypes.c_uint32) * 8)
NO_OF_64BITS = (sys.getsizeof(ctypes.c_uint64) * 8)
KILO = math.pow(10, 3)
###############################

MAX_NUM_PROCESSES = 1024

# gpu metrics macros defined in amdsmi.h
AMDSMI_NUM_HBM_INSTANCES = 4
AMDSMI_MAX_NUM_VCN = 4
AMDSMI_MAX_NUM_CLKS = 4
AMDSMI_MAX_NUM_XGMI_LINKS = 8
AMDSMI_MAX_NUM_GFX_CLKS = 8
AMDSMI_MAX_AID = 4
AMDSMI_MAX_ENGINES = 8
AMDSMI_MAX_NUM_JPEG = 32
AMDSMI_MAX_NUM_XCC = 8
AMDSMI_MAX_NUM_XCP = 8

# Max number of DPM policies
AMDSMI_MAX_NUM_PM_POLICIES = 32

# Max supported frequencies
AMDSMI_MAX_NUM_FREQUENCIES = 33

# Max Fan speed
AMDSMI_MAX_FAN_SPEED = 255

# Max Votlage Curve Points
AMDSMI_NUM_VOLTAGE_CURVE_POINTS = 3

# Max size definitions
AMDSMI_MAX_MM_IP_COUNT = 8
AMDSMI_MAX_DATE_LENGTH = 32
AMDSMI_MAX_STRING_LENGTH = 64
AMDSMI_MAX_DEVICES = 32
AMDSMI_MAX_NAME = 32
AMDSMI_MAX_DRIVER_VERSION_LENGTH = 80
AMDSMI_256_LENGTH = 256
AMDSMI_MAX_CONTAINER_TYPE = 2
AMDSMI_MAX_CACHE_TYPES = 10
AMDSMI_MAX_NUM_XGMI_PHYSICAL_LINK = 64
AMDSMI_GPU_UUID_SIZE = 38
MAX_AMDSMI_NAME_LENGTH = 64
MAX_EVENT_NOTIFICATION_MSG_SIZE = 96
_AMDSMI_STRING_LENGTH = 80


class AmdSmiInitFlags(IntEnum):
    INIT_ALL_PROCESSORS = amdsmi_wrapper.AMDSMI_INIT_ALL_PROCESSORS
    INIT_AMD_CPUS = amdsmi_wrapper.AMDSMI_INIT_AMD_CPUS
    INIT_AMD_GPUS = amdsmi_wrapper.AMDSMI_INIT_AMD_GPUS
    INIT_AMD_APUS = amdsmi_wrapper.AMDSMI_INIT_AMD_APUS
    INIT_NON_AMD_CPUS = amdsmi_wrapper.AMDSMI_INIT_NON_AMD_CPUS
    INIT_NON_AMD_GPUS = amdsmi_wrapper.AMDSMI_INIT_NON_AMD_GPUS


class AmdSmiContainerTypes(IntEnum):
    LXC = amdsmi_wrapper.AMDSMI_CONTAINER_LXC
    DOCKER = amdsmi_wrapper.AMDSMI_CONTAINER_DOCKER


class AmdSmiDeviceType(IntEnum):
    UNKNOWN_DEVICE = amdsmi_wrapper.AMDSMI_PROCESSOR_TYPE_UNKNOWN
    AMD_GPU_DEVICE = amdsmi_wrapper.AMDSMI_PROCESSOR_TYPE_AMD_GPU
    AMD_CPU_DEVICE = amdsmi_wrapper.AMDSMI_PROCESSOR_TYPE_AMD_CPU
    NON_AMD_GPU_DEVICE = amdsmi_wrapper.AMDSMI_PROCESSOR_TYPE_NON_AMD_GPU
    NON_AMD_CPU_DEVICE = amdsmi_wrapper.AMDSMI_PROCESSOR_TYPE_NON_AMD_CPU


class AmdSmiMmIp(IntEnum):
    UVD = amdsmi_wrapper.AMDSMI_MM_UVD
    VCE = amdsmi_wrapper.AMDSMI_MM_VCE
    VCN = amdsmi_wrapper.AMDSMI_MM_VCN


class AmdSmiFwBlock(IntEnum):
    AMDSMI_FW_ID_SMU = amdsmi_wrapper.AMDSMI_FW_ID_SMU
    AMDSMI_FW_ID_CP_CE = amdsmi_wrapper.AMDSMI_FW_ID_CP_CE
    AMDSMI_FW_ID_CP_PFP = amdsmi_wrapper.AMDSMI_FW_ID_CP_PFP
    AMDSMI_FW_ID_CP_ME = amdsmi_wrapper.AMDSMI_FW_ID_CP_ME
    AMDSMI_FW_ID_CP_MEC_JT1 = amdsmi_wrapper.AMDSMI_FW_ID_CP_MEC_JT1
    AMDSMI_FW_ID_CP_MEC_JT2 = amdsmi_wrapper.AMDSMI_FW_ID_CP_MEC_JT2
    AMDSMI_FW_ID_CP_MEC1 = amdsmi_wrapper.AMDSMI_FW_ID_CP_MEC1
    AMDSMI_FW_ID_CP_MEC2 = amdsmi_wrapper.AMDSMI_FW_ID_CP_MEC2
    AMDSMI_FW_ID_RLC = amdsmi_wrapper.AMDSMI_FW_ID_RLC
    AMDSMI_FW_ID_SDMA0 = amdsmi_wrapper.AMDSMI_FW_ID_SDMA0
    AMDSMI_FW_ID_SDMA1 = amdsmi_wrapper.AMDSMI_FW_ID_SDMA1
    AMDSMI_FW_ID_SDMA2 = amdsmi_wrapper.AMDSMI_FW_ID_SDMA2
    AMDSMI_FW_ID_SDMA3 = amdsmi_wrapper.AMDSMI_FW_ID_SDMA3
    AMDSMI_FW_ID_SDMA4 = amdsmi_wrapper.AMDSMI_FW_ID_SDMA4
    AMDSMI_FW_ID_SDMA5 = amdsmi_wrapper.AMDSMI_FW_ID_SDMA5
    AMDSMI_FW_ID_SDMA6 = amdsmi_wrapper.AMDSMI_FW_ID_SDMA6
    AMDSMI_FW_ID_SDMA7 = amdsmi_wrapper.AMDSMI_FW_ID_SDMA7
    AMDSMI_FW_ID_VCN = amdsmi_wrapper.AMDSMI_FW_ID_VCN
    AMDSMI_FW_ID_UVD = amdsmi_wrapper.AMDSMI_FW_ID_UVD
    AMDSMI_FW_ID_VCE = amdsmi_wrapper.AMDSMI_FW_ID_VCE
    AMDSMI_FW_ID_ISP = amdsmi_wrapper.AMDSMI_FW_ID_ISP
    AMDSMI_FW_ID_DMCU_ERAM = amdsmi_wrapper.AMDSMI_FW_ID_DMCU_ERAM
    AMDSMI_FW_ID_DMCU_ISR = amdsmi_wrapper.AMDSMI_FW_ID_DMCU_ISR
    AMDSMI_FW_ID_RLC_RESTORE_LIST_GPM_MEM = amdsmi_wrapper.AMDSMI_FW_ID_RLC_RESTORE_LIST_GPM_MEM
    AMDSMI_FW_ID_RLC_RESTORE_LIST_SRM_MEM = amdsmi_wrapper.AMDSMI_FW_ID_RLC_RESTORE_LIST_SRM_MEM
    AMDSMI_FW_ID_RLC_RESTORE_LIST_CNTL = amdsmi_wrapper.AMDSMI_FW_ID_RLC_RESTORE_LIST_CNTL
    AMDSMI_FW_ID_RLC_V = amdsmi_wrapper.AMDSMI_FW_ID_RLC_V
    AMDSMI_FW_ID_MMSCH = amdsmi_wrapper.AMDSMI_FW_ID_MMSCH
    AMDSMI_FW_ID_PSP_SYSDRV = amdsmi_wrapper.AMDSMI_FW_ID_PSP_SYSDRV
    AMDSMI_FW_ID_PSP_SOSDRV = amdsmi_wrapper.AMDSMI_FW_ID_PSP_SOSDRV
    AMDSMI_FW_ID_PSP_TOC = amdsmi_wrapper.AMDSMI_FW_ID_PSP_TOC
    AMDSMI_FW_ID_PSP_KEYDB = amdsmi_wrapper.AMDSMI_FW_ID_PSP_KEYDB
    AMDSMI_FW_ID_DFC = amdsmi_wrapper.AMDSMI_FW_ID_DFC
    AMDSMI_FW_ID_PSP_SPL = amdsmi_wrapper.AMDSMI_FW_ID_PSP_SPL
    AMDSMI_FW_ID_DRV_CAP = amdsmi_wrapper.AMDSMI_FW_ID_DRV_CAP
    AMDSMI_FW_ID_MC = amdsmi_wrapper.AMDSMI_FW_ID_MC
    AMDSMI_FW_ID_PSP_BL = amdsmi_wrapper.AMDSMI_FW_ID_PSP_BL
    AMDSMI_FW_ID_CP_PM4 = amdsmi_wrapper.AMDSMI_FW_ID_CP_PM4
    AMDSMI_FW_ID_RLC_P = amdsmi_wrapper.AMDSMI_FW_ID_RLC_P
    AMDSMI_FW_ID_SEC_POLICY_STAGE2 = amdsmi_wrapper.AMDSMI_FW_ID_SEC_POLICY_STAGE2
    AMDSMI_FW_ID_REG_ACCESS_WHITELIST = amdsmi_wrapper.AMDSMI_FW_ID_REG_ACCESS_WHITELIST
    AMDSMI_FW_ID_IMU_DRAM = amdsmi_wrapper.AMDSMI_FW_ID_IMU_DRAM
    AMDSMI_FW_ID_IMU_IRAM = amdsmi_wrapper.AMDSMI_FW_ID_IMU_IRAM
    AMDSMI_FW_ID_SDMA_TH0 = amdsmi_wrapper.AMDSMI_FW_ID_SDMA_TH0
    AMDSMI_FW_ID_SDMA_TH1 = amdsmi_wrapper.AMDSMI_FW_ID_SDMA_TH1
    AMDSMI_FW_ID_CP_MES = amdsmi_wrapper.AMDSMI_FW_ID_CP_MES
    AMDSMI_FW_ID_MES_STACK = amdsmi_wrapper.AMDSMI_FW_ID_MES_STACK
    AMDSMI_FW_ID_MES_THREAD1 = amdsmi_wrapper.AMDSMI_FW_ID_MES_THREAD1
    AMDSMI_FW_ID_MES_THREAD1_STACK = amdsmi_wrapper.AMDSMI_FW_ID_MES_THREAD1_STACK
    AMDSMI_FW_ID_RLX6 = amdsmi_wrapper.AMDSMI_FW_ID_RLX6
    AMDSMI_FW_ID_RLX6_DRAM_BOOT = amdsmi_wrapper.AMDSMI_FW_ID_RLX6_DRAM_BOOT
    AMDSMI_FW_ID_RS64_ME = amdsmi_wrapper.AMDSMI_FW_ID_RS64_ME
    AMDSMI_FW_ID_RS64_ME_P0_DATA = amdsmi_wrapper.AMDSMI_FW_ID_RS64_ME_P0_DATA
    AMDSMI_FW_ID_RS64_ME_P1_DATA = amdsmi_wrapper.AMDSMI_FW_ID_RS64_ME_P1_DATA
    AMDSMI_FW_ID_RS64_PFP = amdsmi_wrapper.AMDSMI_FW_ID_RS64_PFP
    AMDSMI_FW_ID_RS64_PFP_P0_DATA = amdsmi_wrapper.AMDSMI_FW_ID_RS64_PFP_P0_DATA
    AMDSMI_FW_ID_RS64_PFP_P1_DATA = amdsmi_wrapper.AMDSMI_FW_ID_RS64_PFP_P1_DATA
    AMDSMI_FW_ID_RS64_MEC = amdsmi_wrapper.AMDSMI_FW_ID_RS64_MEC
    AMDSMI_FW_ID_RS64_MEC_P0_DATA = amdsmi_wrapper.AMDSMI_FW_ID_RS64_MEC_P0_DATA
    AMDSMI_FW_ID_RS64_MEC_P1_DATA = amdsmi_wrapper.AMDSMI_FW_ID_RS64_MEC_P1_DATA
    AMDSMI_FW_ID_RS64_MEC_P2_DATA = amdsmi_wrapper.AMDSMI_FW_ID_RS64_MEC_P2_DATA
    AMDSMI_FW_ID_RS64_MEC_P3_DATA = amdsmi_wrapper.AMDSMI_FW_ID_RS64_MEC_P3_DATA
    AMDSMI_FW_ID_PPTABLE = amdsmi_wrapper.AMDSMI_FW_ID_PPTABLE
    AMDSMI_FW_ID_PSP_SOC = amdsmi_wrapper.AMDSMI_FW_ID_PSP_SOC
    AMDSMI_FW_ID_PSP_DBG = amdsmi_wrapper.AMDSMI_FW_ID_PSP_DBG
    AMDSMI_FW_ID_PSP_INTF = amdsmi_wrapper.AMDSMI_FW_ID_PSP_INTF
    AMDSMI_FW_ID_RLX6_CORE1 = amdsmi_wrapper.AMDSMI_FW_ID_RLX6_CORE1
    AMDSMI_FW_ID_RLX6_DRAM_BOOT_CORE1 = amdsmi_wrapper.AMDSMI_FW_ID_RLX6_DRAM_BOOT_CORE1
    AMDSMI_FW_ID_RLCV_LX7 = amdsmi_wrapper.AMDSMI_FW_ID_RLCV_LX7
    AMDSMI_FW_ID_RLC_SAVE_RESTORE_LIST = amdsmi_wrapper.AMDSMI_FW_ID_RLC_SAVE_RESTORE_LIST
    AMDSMI_FW_ID_ASD = amdsmi_wrapper.AMDSMI_FW_ID_ASD
    AMDSMI_FW_ID_TA_RAS = amdsmi_wrapper.AMDSMI_FW_ID_TA_RAS
    AMDSMI_FW_ID_TA_XGMI = amdsmi_wrapper.AMDSMI_FW_ID_TA_XGMI
    AMDSMI_FW_ID_RLC_SRLG = amdsmi_wrapper.AMDSMI_FW_ID_RLC_SRLG
    AMDSMI_FW_ID_RLC_SRLS = amdsmi_wrapper.AMDSMI_FW_ID_RLC_SRLS
    AMDSMI_FW_ID_PM = amdsmi_wrapper.AMDSMI_FW_ID_PM
    AMDSMI_FW_ID_DMCU = amdsmi_wrapper.AMDSMI_FW_ID_DMCU


class AmdSmiClkType(IntEnum):
    SYS = amdsmi_wrapper.AMDSMI_CLK_TYPE_SYS
    GFX = amdsmi_wrapper.AMDSMI_CLK_TYPE_GFX
    DF = amdsmi_wrapper.AMDSMI_CLK_TYPE_DF
    DCEF = amdsmi_wrapper.AMDSMI_CLK_TYPE_DCEF
    SOC = amdsmi_wrapper.AMDSMI_CLK_TYPE_SOC
    MEM = amdsmi_wrapper.AMDSMI_CLK_TYPE_MEM
    PCIE = amdsmi_wrapper.AMDSMI_CLK_TYPE_PCIE
    VCLK0 = amdsmi_wrapper.AMDSMI_CLK_TYPE_VCLK0
    VCLK1 = amdsmi_wrapper.AMDSMI_CLK_TYPE_VCLK1
    DCLK0 = amdsmi_wrapper.AMDSMI_CLK_TYPE_DCLK0
    DCLK1 = amdsmi_wrapper.AMDSMI_CLK_TYPE_DCLK1


class AmdSmiTemperatureType(IntEnum):
    EDGE = amdsmi_wrapper.AMDSMI_TEMPERATURE_TYPE_EDGE
    HOTSPOT = amdsmi_wrapper.AMDSMI_TEMPERATURE_TYPE_HOTSPOT
    JUNCTION = amdsmi_wrapper.AMDSMI_TEMPERATURE_TYPE_JUNCTION
    VRAM = amdsmi_wrapper.AMDSMI_TEMPERATURE_TYPE_VRAM
    HBM_0 = amdsmi_wrapper.AMDSMI_TEMPERATURE_TYPE_HBM_0
    HBM_1 = amdsmi_wrapper.AMDSMI_TEMPERATURE_TYPE_HBM_1
    HBM_2 = amdsmi_wrapper.AMDSMI_TEMPERATURE_TYPE_HBM_2
    HBM_3 = amdsmi_wrapper.AMDSMI_TEMPERATURE_TYPE_HBM_3
    PLX = amdsmi_wrapper.AMDSMI_TEMPERATURE_TYPE_PLX


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
    NONE = amdsmi_wrapper.AMDSMI_EVT_NOTIF_NONE
    VMFAULT = amdsmi_wrapper.AMDSMI_EVT_NOTIF_VMFAULT
    THERMAL_THROTTLE = amdsmi_wrapper.AMDSMI_EVT_NOTIF_THERMAL_THROTTLE
    GPU_PRE_RESET = amdsmi_wrapper.AMDSMI_EVT_NOTIF_GPU_PRE_RESET
    GPU_POST_RESET = amdsmi_wrapper.AMDSMI_EVT_NOTIF_GPU_POST_RESET
    RING_HANG = amdsmi_wrapper.AMDSMI_EVT_NOTIF_RING_HANG


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

class AmdSmiAcceleratorPartitionResourceType(IntEnum):
    XCC = amdsmi_wrapper.AMDSMI_ACCELERATOR_XCC
    ENCODER = amdsmi_wrapper.AMDSMI_ACCELERATOR_ENCODER
    DECODER = amdsmi_wrapper.AMDSMI_ACCELERATOR_DECODER
    DMA = amdsmi_wrapper.AMDSMI_ACCELERATOR_DMA
    JPEG = amdsmi_wrapper.AMDSMI_ACCELERATOR_JPEG
    MAX = amdsmi_wrapper.AMDSMI_ACCELERATOR_MAX


class AmdSmiAcceleratorPartitionType(IntEnum):
    SPX = amdsmi_wrapper.AMDSMI_ACCELERATOR_PARTITION_SPX
    DPX = amdsmi_wrapper.AMDSMI_ACCELERATOR_PARTITION_DPX
    TPX = amdsmi_wrapper.AMDSMI_ACCELERATOR_PARTITION_TPX
    QPX = amdsmi_wrapper.AMDSMI_ACCELERATOR_PARTITION_QPX
    CPX = amdsmi_wrapper.AMDSMI_ACCELERATOR_PARTITION_CPX
    INVALID = amdsmi_wrapper.AMDSMI_ACCELERATOR_PARTITION_INVALID


class AmdSmiComputePartitionType(IntEnum):
    SPX = amdsmi_wrapper.AMDSMI_COMPUTE_PARTITION_SPX
    DPX = amdsmi_wrapper.AMDSMI_COMPUTE_PARTITION_DPX
    TPX = amdsmi_wrapper.AMDSMI_COMPUTE_PARTITION_TPX
    QPX = amdsmi_wrapper.AMDSMI_COMPUTE_PARTITION_QPX
    CPX = amdsmi_wrapper.AMDSMI_COMPUTE_PARTITION_CPX
    INVALID = amdsmi_wrapper.AMDSMI_COMPUTE_PARTITION_INVALID


class AmdSmiMemoryPartitionType(IntEnum):
    NPS1 = amdsmi_wrapper.AMDSMI_MEMORY_PARTITION_NPS1
    NPS2 = amdsmi_wrapper.AMDSMI_MEMORY_PARTITION_NPS2
    NPS4 = amdsmi_wrapper.AMDSMI_MEMORY_PARTITION_NPS4
    NPS8 = amdsmi_wrapper.AMDSMI_MEMORY_PARTITION_NPS8
    UNKNOWN = amdsmi_wrapper.AMDSMI_MEMORY_PARTITION_UNKNOWN


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
    MCA = amdsmi_wrapper.AMDSMI_GPU_BLOCK_MCA
    VCN = amdsmi_wrapper.AMDSMI_GPU_BLOCK_VCN
    JPEG = amdsmi_wrapper.AMDSMI_GPU_BLOCK_JPEG
    IH = amdsmi_wrapper.AMDSMI_GPU_BLOCK_IH
    MPIO = amdsmi_wrapper.AMDSMI_GPU_BLOCK_MPIO
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


class AmdSmiCperNotifyType(Enum):
    CMC = amdsmi_wrapper.AMDSMI_CPER_NOTIFY_TYPE_CMC
    CPE = amdsmi_wrapper.AMDSMI_CPER_NOTIFY_TYPE_CPE
    MCE = amdsmi_wrapper.AMDSMI_CPER_NOTIFY_TYPE_MCE
    PCIE = amdsmi_wrapper.AMDSMI_CPER_NOTIFY_TYPE_PCIE
    INIT = amdsmi_wrapper.AMDSMI_CPER_NOTIFY_TYPE_INIT
    NMI = amdsmi_wrapper.AMDSMI_CPER_NOTIFY_TYPE_NMI
    BOOT = amdsmi_wrapper.AMDSMI_CPER_NOTIFY_TYPE_BOOT
    DMAr = amdsmi_wrapper.AMDSMI_CPER_NOTIFY_TYPE_DMAR
    SEA =  amdsmi_wrapper.AMDSMI_CPER_NOTIFY_TYPE_SEA
    SEI = amdsmi_wrapper.AMDSMI_CPER_NOTIFY_TYPE_SEI
    PEI = amdsmi_wrapper.AMDSMI_CPER_NOTIFY_TYPE_PEI
    CXL_COMPONENT = amdsmi_wrapper.AMDSMI_CPER_NOTIFY_TYPE_CXL_COMPONENT


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


class AmdSmiLinkType(IntEnum):
    AMDSMI_LINK_TYPE_INTERNAL = amdsmi_wrapper.AMDSMI_LINK_TYPE_INTERNAL
    AMDSMI_LINK_TYPE_XGMI = amdsmi_wrapper.AMDSMI_LINK_TYPE_XGMI
    AMDSMI_LINK_TYPE_PCIE = amdsmi_wrapper.AMDSMI_LINK_TYPE_PCIE
    AMDSMI_LINK_TYPE_NOT_APPLICABLE = amdsmi_wrapper.AMDSMI_LINK_TYPE_NOT_APPLICABLE
    AMDSMI_LINK_TYPE_UNKNOWN = amdsmi_wrapper.AMDSMI_LINK_TYPE_UNKNOWN


class AmdSmiUtilizationCounterType(IntEnum):
    COARSE_GRAIN_GFX_ACTIVITY = amdsmi_wrapper.AMDSMI_COARSE_GRAIN_GFX_ACTIVITY
    COARSE_GRAIN_MEM_ACTIVITY = amdsmi_wrapper.AMDSMI_COARSE_GRAIN_MEM_ACTIVITY
    COARSE_DECODER_ACTIVITY = amdsmi_wrapper.AMDSMI_COARSE_DECODER_ACTIVITY
    FINE_GRAIN_GFX_ACTIVITY = amdsmi_wrapper.AMDSMI_FINE_GRAIN_GFX_ACTIVITY
    FINE_GRAIN_MEM_ACTIVITY = amdsmi_wrapper.AMDSMI_FINE_GRAIN_MEM_ACTIVITY
    FINE_DECODER_ACTIVITY = amdsmi_wrapper.AMDSMI_FINE_DECODER_ACTIVITY
    UTILIZATION_COUNTER_FIRST = amdsmi_wrapper.AMDSMI_UTILIZATION_COUNTER_FIRST
    UTILIZATION_COUNTER_LAST = amdsmi_wrapper.AMDSMI_UTILIZATION_COUNTER_LAST


class AmdSmiProcessorType(IntEnum):
    UNKNOWN = amdsmi_wrapper.AMDSMI_PROCESSOR_TYPE_UNKNOWN
    AMDSMI_PROCESSOR_TYPE_AMD_GPU = amdsmi_wrapper.AMDSMI_PROCESSOR_TYPE_AMD_GPU
    AMDSMI_PROCESSOR_TYPE_AMD_CPU = amdsmi_wrapper.AMDSMI_PROCESSOR_TYPE_AMD_CPU
    AMDSMI_PROCESSOR_TYPE_NON_AMD_GPU = amdsmi_wrapper.AMDSMI_PROCESSOR_TYPE_NON_AMD_GPU
    AMDSMI_PROCESSOR_TYPE_NON_AMD_CPU = amdsmi_wrapper.AMDSMI_PROCESSOR_TYPE_NON_AMD_CPU


class AmdSmiRegType(IntEnum):
    XGMI = amdsmi_wrapper.AMDSMI_REG_XGMI
    WAFL = amdsmi_wrapper.AMDSMI_REG_WAFL
    PCIE = amdsmi_wrapper.AMDSMI_REG_PCIE
    USR = amdsmi_wrapper.AMDSMI_REG_USR
    USR1 = amdsmi_wrapper.AMDSMI_REG_USR1


class AmdSmiVirtualizationMode(IntEnum):
    UNKNOWN = amdsmi_wrapper.AMDSMI_VIRTUALIZATION_MODE_UNKNOWN
    BAREMETAL = amdsmi_wrapper.AMDSMI_VIRTUALIZATION_MODE_BAREMETAL
    HOST = amdsmi_wrapper.AMDSMI_VIRTUALIZATION_MODE_HOST
    GUEST = amdsmi_wrapper.AMDSMI_VIRTUALIZATION_MODE_GUEST
    PASSTHROUGH = amdsmi_wrapper.AMDSMI_VIRTUALIZATION_MODE_PASSTHROUGH


class AmdSmiVramType(IntEnum):
    UNKNOWN = amdsmi_wrapper.AMDSMI_VRAM_TYPE_UNKNOWN
    HBM = amdsmi_wrapper.AMDSMI_VRAM_TYPE_HBM
    HBM2 = amdsmi_wrapper.AMDSMI_VRAM_TYPE_HBM2
    HBM2E = amdsmi_wrapper.AMDSMI_VRAM_TYPE_HBM2E
    HBM3 = amdsmi_wrapper.AMDSMI_VRAM_TYPE_HBM3
    DDR2 = amdsmi_wrapper.AMDSMI_VRAM_TYPE_DDR2
    DDR3 = amdsmi_wrapper.AMDSMI_VRAM_TYPE_DDR3
    DDR4 = amdsmi_wrapper.AMDSMI_VRAM_TYPE_DDR4
    GDDR1 = amdsmi_wrapper.AMDSMI_VRAM_TYPE_GDDR1
    GDDR2 = amdsmi_wrapper.AMDSMI_VRAM_TYPE_GDDR2
    GDDR3 = amdsmi_wrapper.AMDSMI_VRAM_TYPE_GDDR3
    GDDR4 = amdsmi_wrapper.AMDSMI_VRAM_TYPE_GDDR4
    GDDR5 = amdsmi_wrapper.AMDSMI_VRAM_TYPE_GDDR5
    GDDR6 = amdsmi_wrapper.AMDSMI_VRAM_TYPE_GDDR6
    GDDR7 = amdsmi_wrapper.AMDSMI_VRAM_TYPE_GDDR7
    MAX = amdsmi_wrapper.AMDSMI_VRAM_TYPE__MAX


class AmdSmiVramVendor(IntEnum):
    SAMSUNG = amdsmi_wrapper.AMDSMI_VRAM_VENDOR_SAMSUNG
    INFINEON = amdsmi_wrapper.AMDSMI_VRAM_VENDOR_INFINEON
    ELPIDA = amdsmi_wrapper.AMDSMI_VRAM_VENDOR_ELPIDA
    ETRON = amdsmi_wrapper.AMDSMI_VRAM_VENDOR_ETRON
    NANYA = amdsmi_wrapper.AMDSMI_VRAM_VENDOR_NANYA
    HYNIX = amdsmi_wrapper.AMDSMI_VRAM_VENDOR_HYNIX
    MOSEL = amdsmi_wrapper.AMDSMI_VRAM_VENDOR_MOSEL
    WINBOND = amdsmi_wrapper.AMDSMI_VRAM_VENDOR_WINBOND
    ESMT = amdsmi_wrapper.AMDSMI_VRAM_VENDOR_ESMT
    MICRON = amdsmi_wrapper.AMDSMI_VRAM_VENDOR_MICRON
    UNKNOWN = amdsmi_wrapper.AMDSMI_VRAM_VENDOR_UNKNOWN


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
            if event_type != AmdSmiEvtNotificationType.NONE:
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
                if AmdSmiEvtNotificationType(self.event_info[i].event).name != "NONE":
                    processor_handle = amdsmi_wrapper.amdsmi_processor_handle(self.event_info[i].processor_handle)
                    ret.append(
                        {
                            "processor_handle": processor_handle,
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


def _format_bad_page_info(bad_page_info, bad_page_count: ctypes.c_uint32) -> List[Dict]:
    """
    Format bad page info data retrieved.

    Parameters:
        bad_page_info(`amdsmi_retired_page_record_t`): A populated list of amdsmi_retired_page_record_t(s)
        retrieved. Ex: (amdsmi_wrapper.amdsmi_retired_page_record_t * #)()
        bad_page_count(`c_uint32`): Bad page count.

    Returns:
        `list`: List containing formatted bad pages. Can be empty
    """
    if bad_page_count == 0:
        return []

    # Check if each struct within bad_page_info is valid
    for bad_page in bad_page_info:
        if not isinstance(bad_page, amdsmi_wrapper.amdsmi_retired_page_record_t):
            raise AmdSmiParameterException(
                bad_page, amdsmi_wrapper.amdsmi_retired_page_record_t
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
    domain = hex(amdsmi_bdf.struct_amdsmi_bdf_t.domain_number)[2:].zfill(4)
    bus = hex(amdsmi_bdf.struct_amdsmi_bdf_t.bus_number)[2:].zfill(2)
    device = hex(amdsmi_bdf.struct_amdsmi_bdf_t.device_number)[2:].zfill(2)
    function = hex(amdsmi_bdf.struct_amdsmi_bdf_t.function_number)[2:]

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
    amdsmi_bdf.struct_amdsmi_bdf_t.function_number = bdf[3]
    amdsmi_bdf.struct_amdsmi_bdf_t.device_number = bdf[2]
    amdsmi_bdf.struct_amdsmi_bdf_t.bus_number = bdf[1]
    amdsmi_bdf.struct_amdsmi_bdf_t.domain_number = bdf[0]
    return amdsmi_bdf


def _pad_hex_value(value, length):
    """ Pad a hexadecimal value with a given length of zeros

    :param value: A hexadecimal value to be padded with zeros
    :param length: Number of zeros to pad the hexadecimal value
    :param return original string string or
        padded hex of confirmed hex output (using length provided)
    """
    # Ensure value entered meets the minimum length and is hexadecimal
    if len(value) > 2 and length > 1 and value[:2].lower() == '0x' \
        and all(c in '0123456789abcdefABCDEF' for c in value[2:]):
        # Pad with zeros after '0x' prefix
        return '0x' + value[2:].zfill(length)
    return value


def _validate_if_max_uint(value, uint_type: MaxUIntegerTypes, isActivity=False, isBool=False):
    return_val = "N/A"
    if not isinstance(value, list):
        if (value == uint_type) or (isActivity and value > 100):
            return return_val
        else:
            if isBool:
                return bool(value)
            else:
               return value
    else:
        return_val = []
        for _, v in enumerate(value):
            if (v == uint_type) or (isActivity and v > 100):
                return_val.append("N/A")
            else:
                return_val.append(v)
    if isBool:
        return bool(return_val)
    else:
        return return_val


def _notifyTypeToString(notify_type_b):
    guid = []
    # Iterate over only the first 8 bytes, but backwards
    for i in notify_type_b[7::-1]:
        guid.append(format(i, '02x'))
    hex_string = "".join(guid)
    hex_value = int(hex_string, 16)
    if hex_value in AmdSmiCperNotifyType._value2member_map_:
        # Convert to the corresponding enum name
        return AmdSmiCperNotifyType(hex_value).name
    else:
        return "Unknown"


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
    cpu_count = ctypes.c_uint32(0)
    null_ptr = ctypes.POINTER(amdsmi_wrapper.amdsmi_processor_handle)()
    _check_res(
        amdsmi_wrapper.amdsmi_get_cpu_handles(
            ctypes.byref(cpu_count), null_ptr)
    )
    proc_handles = (amdsmi_wrapper.amdsmi_processor_handle *
                      cpu_count.value)()
    _check_res(
        amdsmi_wrapper.amdsmi_get_cpu_handles(
            ctypes.byref(cpu_count), proc_handles)
    )
    cpu_handles = [
        amdsmi_wrapper.amdsmi_processor_handle(proc_handles[sock_idx])
        for sock_idx in range(cpu_count.value)
    ]
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
    cores_count = ctypes.c_uint32(0)
    null_ptr = ctypes.POINTER(amdsmi_wrapper.amdsmi_processor_handle)()
    _check_res(
        amdsmi_wrapper.amdsmi_get_cpucore_handles(
            ctypes.byref(cores_count), null_ptr)
    )
    proc_handles = (amdsmi_wrapper.amdsmi_processor_handle *
                      cores_count.value)()
    _check_res(
        amdsmi_wrapper.amdsmi_get_cpucore_handles(
            ctypes.byref(cores_count), proc_handles)
    )
    core_handles = [
        amdsmi_wrapper.amdsmi_processor_handle(proc_handles[sock_idx])
        for sock_idx in range(cores_count.value)
    ]

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

def amdsmi_get_cpu_hsmp_driver_version(processor_handle: amdsmi_wrapper.amdsmi_processor_handle):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    hsmp_driver_version = amdsmi_wrapper.amdsmi_hsmp_driver_version_t()

    _check_res(amdsmi_wrapper.amdsmi_get_cpu_hsmp_driver_version(processor_handle, hsmp_driver_version))

    return {
        "hsmp_driver_major_ver_num": hsmp_driver_version.major,
        "hsmp_driver_minor_ver_num": hsmp_driver_version.minor,
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

def amdsmi_get_threads_per_core():
    threads_per_core = ctypes.c_uint32()
    _check_res(
        amdsmi_wrapper.amdsmi_get_threads_per_core(
            ctypes.byref(threads_per_core)
        )
    )

    return threads_per_core.value

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
        "mtbl_hbm_thm_residency_acc": mtbl.hbm_thm_residency_acc,
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

def amdsmi_get_cpu_model_name(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    cpu_info = amdsmi_wrapper.amdsmi_cpu_info_t()

    _check_res(
            amdsmi_wrapper.amdsmi_get_cpu_model_name(
                   processor_handle, cpu_info
            )
    )
    return f"{cpu_info.model_name}"

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


def amdsmi_get_gpu_device_uuid(processor_handle: amdsmi_wrapper.amdsmi_processor_handle) -> str:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    uuid = ctypes.create_string_buffer(AMDSMI_GPU_UUID_SIZE)

    uuid_length = ctypes.c_uint32()
    uuid_length.value = AMDSMI_GPU_UUID_SIZE

    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_device_uuid(
            processor_handle, ctypes.byref(uuid_length), uuid
        )
    )

    return uuid.value.decode("utf-8")


def amdsmi_get_gpu_enumeration_info(processor_handle: amdsmi_wrapper.amdsmi_processor_handle) -> Dict[str, Any]:
    """
    Retrieves GPU enumeration information including DRM card ID, DRM render ID, HIP ID, and HIP UUID.

    Parameters:
        processor_handle (amdsmi_processor_handle): The processor handle.

    Returns:
        Dict[str, Any]: A dictionary containing the retrieved enumeration information.

    Raises:
        AmdSmiParameterException: If the input parameters are invalid.
    """
    # Validate the processor handle
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    # Create an instance of the enumeration info struct
    enumeration_info = amdsmi_wrapper.amdsmi_enumeration_info_t()

    # Call the C function to populate the struct
    status = amdsmi_wrapper.amdsmi_get_gpu_enumeration_info(processor_handle, ctypes.byref(enumeration_info))

    # Validate the status result
    _check_res(status)

    # Convert the struct fields into a dictionary and return
    enumeration_info = {
        "drm_render": _validate_if_max_uint(enumeration_info.drm_render, MaxUIntegerTypes.UINT32_T),
        "drm_card": _validate_if_max_uint(enumeration_info.drm_card, MaxUIntegerTypes.UINT32_T),
        "hsa_id": _validate_if_max_uint(enumeration_info.hsa_id, MaxUIntegerTypes.UINT32_T),
        "hip_id": _validate_if_max_uint(enumeration_info.hip_id, MaxUIntegerTypes.UINT32_T),
        "hip_uuid": enumeration_info.hip_uuid.decode('utf-8')
    }

    return enumeration_info

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

    market_name = _pad_hex_value(asic_info_struct.market_name.decode("utf-8"), 4)
    target_graphics_version = hex(asic_info_struct.target_graphics_version)[2:]
    asic_info = {
        "market_name": market_name,
        "vendor_id": asic_info_struct.vendor_id,
        "vendor_name": asic_info_struct.vendor_name.decode("utf-8"),
        "subvendor_id": asic_info_struct.subvendor_id,
        "device_id": asic_info_struct.device_id,
        "rev_id": _pad_hex_value(hex(asic_info_struct.rev_id), 2),
        "asic_serial": asic_info_struct.asic_serial.decode("utf-8"),
        "oam_id": asic_info_struct.oam_id,
        "num_compute_units": asic_info_struct.num_of_compute_units,
        "target_graphics_version": "gfx" + target_graphics_version
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

    # Check for max value as a sign for not applicable
    if asic_info["num_compute_units"] == 0xFFFFFFFF: # uint 32 max
        asic_info["num_compute_units"] = "N/A"

    # Remove commas from vendor name for clean output
    asic_info["vendor_name"] = asic_info["vendor_name"].replace(',', '')

    return asic_info


def amdsmi_get_gpu_kfd_info(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> Dict[str, Any]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    kfd_info_struct = amdsmi_wrapper.amdsmi_kfd_info_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_kfd_info(
            processor_handle, ctypes.byref(kfd_info_struct))
    )

    kfd_info = {
        "kfd_id": _validate_if_max_uint(kfd_info_struct.kfd_id, MaxUIntegerTypes.UINT64_T),
        "node_id": _validate_if_max_uint(kfd_info_struct.node_id, MaxUIntegerTypes.UINT32_T),
        "current_partition_id": _validate_if_max_uint(kfd_info_struct.current_partition_id, MaxUIntegerTypes.UINT32_T)
    }

    return kfd_info


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


def amdsmi_get_gpu_pm_metrics_info(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> Dict[str, Any]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    pm_metrics = ctypes.POINTER(amdsmi_wrapper.amdsmi_name_value_t)()
    num_mets = ctypes.c_uint32()

    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_pm_metrics_info(
            processor_handle, ctypes.pointer(pm_metrics), ctypes.byref(num_mets)
        )
    )

    results = []
    for i in range(num_mets.value):
        item = {
            'name': pm_metrics[i].name,
            'value': pm_metrics[i].value
        }
        results.append(item)
    amdsmi_wrapper.amdsmi_free_name_value_pairs(pm_metrics)
    return results


def amdsmi_get_gpu_reg_table_info(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    reg_type: AmdSmiRegType,
) -> Dict[str, Any]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    reg_metrics = ctypes.POINTER(amdsmi_wrapper.amdsmi_name_value_t)()
    num_regs = ctypes.c_uint32()

    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_reg_table_info(
            processor_handle, reg_type, ctypes.pointer(reg_metrics), ctypes.byref(num_regs)
        )
    )

    results = []
    for i in range(num_regs.value):
        item = {
            'name': reg_metrics[i].name,
            'value': reg_metrics[i].value
        }
        results.append(item)
    amdsmi_wrapper.amdsmi_free_name_value_pairs(reg_metrics)
    return results


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
        "vram_size": vram_info.vram_size,
        "vram_bit_width": _validate_if_max_uint(vram_info.vram_bit_width, MaxUIntegerTypes.UINT32_T),
        "vram_max_bandwidth": _validate_if_max_uint(vram_info.vram_max_bandwidth, MaxUIntegerTypes.UINT64_T),
    }


def amdsmi_get_gpu_xgmi_link_status(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    ) -> Dict[str, Any]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    status_info = amdsmi_wrapper.amdsmi_xgmi_link_status_t()
    _check_res(
    amdsmi_wrapper.amdsmi_get_gpu_xgmi_link_status(
        processor_handle, ctypes.byref(status_info))
    )

    link_status = []
    count = 0
    for link in status_info.status:
        if count == status_info.total_links:
            break
        if amdsmi_wrapper.amdsmi_xgmi_link_status_type_t__enumvalues[link] == 'AMDSMI_XGMI_LINK_DISABLE': # XGMI link is disabled
            link_status.append("X")
        elif amdsmi_wrapper.amdsmi_xgmi_link_status_type_t__enumvalues[link] == 'AMDSMI_XGMI_LINK_UP': # XGMI Link is up
            link_status.append("U")
        elif amdsmi_wrapper.amdsmi_xgmi_link_status_type_t__enumvalues[link] == 'AMDSMI_XGMI_LINK_DOWN': # XGMI Link is down
            link_status.append("D")
        else:
            link_status.append("N/A")
        count += 1

    return_dict = {
        "status"     : link_status,
        "total_links": status_info.total_links,
    }
    return return_dict


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

    clk_type_str = AmdSmiClkType(clock_type).name

    dict_ret = {
        "clk": _validate_if_max_uint(clock_measure.clk, MaxUIntegerTypes.UINT32_T),
        "min_clk": _validate_if_max_uint(clock_measure.min_clk, MaxUIntegerTypes.UINT32_T),
        "max_clk": _validate_if_max_uint(clock_measure.max_clk, MaxUIntegerTypes.UINT32_T),
        "clk_locked": _validate_if_max_uint(clock_measure.clk_locked, MaxUIntegerTypes.UINT8_T, isBool=True),
        "clk_deep_sleep" : _validate_if_max_uint(clock_measure.clk_deep_sleep, MaxUIntegerTypes.UINT8_T, isBool=True),
    }
    logging.debug("amdsmi_interface.py | amdsmi_get_clock_info | clk_type = " + clk_type_str + " | return_dictionary = \n" + str(json.dumps(dict_ret, indent=4)))
    return dict_ret


def amdsmi_get_gpu_bad_page_info(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> Union[list, str]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    num_pages = ctypes.c_uint32()
    nullptr = ctypes.POINTER(amdsmi_wrapper.amdsmi_retired_page_record_t)()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_bad_page_info(
            processor_handle, ctypes.byref(num_pages), nullptr
        )
    )

    if num_pages.value == 0:
        return []

    bad_pages = (amdsmi_wrapper.amdsmi_retired_page_record_t * num_pages.value)()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_bad_page_info(
            processor_handle, ctypes.byref(num_pages), bad_pages
        )
    )

    return _format_bad_page_info(bad_pages, num_pages)

def amdsmi_get_gpu_bad_page_threshold(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> Dict[str, Any]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    threshold = ctypes.c_uint32()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_bad_page_threshold(
            processor_handle, ctypes.byref(threshold)
        )
    )

    return threshold.value

def amdsmi_get_violation_status(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> Dict[str, Any]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    violation_status = amdsmi_wrapper.amdsmi_violation_status_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_violation_status(
            processor_handle, ctypes.byref(violation_status))
    )

    dict_return = {
        "reference_timestamp": _validate_if_max_uint(violation_status.reference_timestamp, MaxUIntegerTypes.UINT64_T),
        "violation_timestamp": _validate_if_max_uint(violation_status.violation_timestamp, MaxUIntegerTypes.UINT64_T),
        "acc_counter": _validate_if_max_uint(violation_status.acc_counter, MaxUIntegerTypes.UINT64_T),
        "acc_prochot_thrm": _validate_if_max_uint(violation_status.acc_prochot_thrm, MaxUIntegerTypes.UINT64_T),
        "acc_ppt_pwr": _validate_if_max_uint(violation_status.acc_ppt_pwr, MaxUIntegerTypes.UINT64_T),                           #PVIOL
        "acc_socket_thrm": _validate_if_max_uint(violation_status.acc_socket_thrm, MaxUIntegerTypes.UINT64_T),                   #TVIOL
        "acc_vr_thrm": _validate_if_max_uint(violation_status.acc_vr_thrm, MaxUIntegerTypes.UINT64_T),
        "acc_hbm_thrm": _validate_if_max_uint(violation_status.acc_hbm_thrm, MaxUIntegerTypes.UINT64_T),
        "acc_gfx_clk_below_host_limit": _validate_if_max_uint(violation_status.acc_gfx_clk_below_host_limit, MaxUIntegerTypes.UINT64_T),
        "per_prochot_thrm": _validate_if_max_uint(violation_status.per_prochot_thrm, MaxUIntegerTypes.UINT64_T, isActivity=True),
        "per_ppt_pwr": _validate_if_max_uint(violation_status.per_ppt_pwr, MaxUIntegerTypes.UINT64_T, isActivity=True),          #PVIOL
        "per_socket_thrm": _validate_if_max_uint(violation_status.per_socket_thrm, MaxUIntegerTypes.UINT64_T, isActivity=True),  #TVIOL
        "per_vr_thrm": _validate_if_max_uint(violation_status.per_vr_thrm, MaxUIntegerTypes.UINT64_T, isActivity=True),
        "per_hbm_thrm": _validate_if_max_uint(violation_status.per_hbm_thrm, MaxUIntegerTypes.UINT64_T, isActivity=True),
        "per_gfx_clk_below_host_limit": _validate_if_max_uint(violation_status.per_gfx_clk_below_host_limit, MaxUIntegerTypes.UINT64_T, isActivity=True),
        "active_prochot_thrm": _validate_if_max_uint(violation_status.active_prochot_thrm, MaxUIntegerTypes.UINT8_T, isBool=True),
        "active_ppt_pwr": _validate_if_max_uint(violation_status.active_ppt_pwr, MaxUIntegerTypes.UINT8_T, isBool=True),         #PVIOL
        "active_socket_thrm": _validate_if_max_uint(violation_status.active_socket_thrm, MaxUIntegerTypes.UINT8_T, isBool=True), #TVIOL
        "active_vr_thrm": _validate_if_max_uint(violation_status.active_vr_thrm, MaxUIntegerTypes.UINT8_T, isBool=True),
        "active_hbm_thrm": _validate_if_max_uint(violation_status.active_hbm_thrm, MaxUIntegerTypes.UINT8_T, isBool=True),
        "active_gfx_clk_below_host_limit": _validate_if_max_uint(violation_status.active_gfx_clk_below_host_limit, MaxUIntegerTypes.UINT8_T, isBool=True),
    }
    return dict_return

def amdsmi_get_gpu_total_ecc_count(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> Dict[str, Any]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    ec = amdsmi_wrapper.amdsmi_error_count_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_total_ecc_count(
            processor_handle, ctypes.byref(ec)
        )
    )

    return {
        "correctable_count": ec.correctable_count,
        "uncorrectable_count": ec.uncorrectable_count,
        "deferred_count": ec.deferred_count,
    }

def notifyTypeToString(notify_type_b):
    idx = 0
    guid = []
    for i in notify_type_b:
        guid.append(format(i, '02x'))
        if idx == 7:
            break
        idx = idx +1
    return "".join(guid[::-1])

def amdsmi_get_gpu_cper_entries(processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    severity_mask: int,
    buffer_size: int = 4*1048576,
    cursor: int = 0
) -> Tuple[List[Dict[str, Any]], int]:

    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    # Allocate a buffer for CPER data.
    buf = ctypes.create_string_buffer(buffer_size)
    buf_size = ctypes.c_uint64(buffer_size)
    entry_count = ctypes.c_uint64(20)
    cur = ctypes.c_uint64(cursor)
    # Allocate a pointer for the CPER header array.
    cper_hdrs_array = (ctypes.POINTER(amdsmi_wrapper.amdsmi_cper_hdr_t) * 20)()
    cper_hdrs = ctypes.cast(cper_hdrs_array, ctypes.POINTER(ctypes.POINTER(amdsmi_wrapper.amdsmi_cper_hdr_t)))

    # Call the underlying AMD-SMI API.
    ret = amdsmi_wrapper.amdsmi_get_gpu_cper_entries(
        processor_handle,
        ctypes.c_uint32(severity_mask),
        buf,
        ctypes.byref(buf_size),
        cper_hdrs,
        ctypes.byref(entry_count),
        ctypes.byref(cur)
    )
    if ret != amdsmi_wrapper.AMDSMI_STATUS_SUCCESS:
        raise AmdSmiLibraryException(ret)

    entries = {}
    cper_data = []
    offset = 0
    # Iterate over each entry using its variable record_length.
    for i in range(entry_count.value):
        entry_address = ctypes.addressof(buf) + offset
        entry_ptr = ctypes.cast(entry_address, ctypes.POINTER(amdsmi_wrapper.amdsmi_cper_hdr_t))
        cper_data.append({
            "bytes":list((entry_ptr.contents.record_length * ctypes.c_byte).from_address(entry_address)),
            "size":entry_ptr.contents.record_length
        })
        # Extract the timestamp fields.
        year = entry_ptr.contents.timestamp.year
        # Adjust the year if it's less than 100. You can tweak this logic based on your expected data.
        if year < 100:
             year += 2000
        formatted_timestamp = (
           f"{year:04d}/"
           f"{entry_ptr.contents.timestamp.month:02d}/"
           f"{entry_ptr.contents.timestamp.day:02d} "
           f"{entry_ptr.contents.timestamp.hours:02d}:"
           f"{entry_ptr.contents.timestamp.minutes:02d}:"
           f"{entry_ptr.contents.timestamp.seconds:02d}"
        )
        cper_entry = {
            "error_severity": amdsmi_wrapper.amdsmi_cper_sev_t__enumvalues.get(entry_ptr.contents.error_severity, "AMDSMI_CPER_SEV_UNUSED").replace("AMDSMI_CPER_SEV_", "").lower(),
            "notify_type": _notifyTypeToString(entry_ptr.contents.notify_type.b),
            "timestamp": formatted_timestamp,
            "signature" : entry_ptr.contents.signature,
            "revision" : entry_ptr.contents.revision,
            "signature_end" : hex(entry_ptr.contents.signature_end),
            "sec_cnt" : entry_ptr.contents.sec_cnt,
            "record_length" : entry_ptr.contents.record_length,
            "platform_id" : entry_ptr.contents.platform_id,
            "creator_id" : entry_ptr.contents.creator_id,
            "record_id" : entry_ptr.contents.record_id,
            "flags" : entry_ptr.contents.flags,
            "persistence_info" : entry_ptr.contents.persistence_info,
            #"reserved" : entry_ptr.contents.reserved
            #"cper_valid_bit" : entry_ptr.contents.cper_valid_bits,
            #"partition_id" : entry_ptr.contents.partition_id,
        }
        entries[i] = cper_entry.copy()
        offset += entry_ptr.contents.record_length  # Use the actual record length to advance the offset

    return entries, cur.value, cper_data


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

    board_info_dict = {
        "model_number": _pad_hex_value(board_info.model_number.decode("utf-8").strip(), 4),
        "product_serial": board_info.product_serial.decode("utf-8").strip(),
        "fru_id": board_info.fru_id.decode("utf-8").strip(),
        "product_name": _pad_hex_value(board_info.product_name.decode("utf-8").strip(), 4),
        "manufacturer_name": board_info.manufacturer_name.decode("utf-8").strip()
    }

    for key, value in board_info_dict.items():
        if value == "":
            board_info_dict[key] = "N/A"
    return board_info_dict


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
            gpu_block.name = "MPIO"
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
) -> List[amdsmi_wrapper.amdsmi_proc_info_t]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    # This will get populated with the number of processes found
    max_processes = ctypes.c_uint32(MAX_NUM_PROCESSES)

    process_list = (amdsmi_wrapper.amdsmi_proc_info_t * max_processes.value)()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_process_list(
            processor_handle, ctypes.byref(max_processes), process_list
        )
    )

    result = []
    for index in range(max_processes.value):
        process_name = process_list[index].name.decode("utf-8").strip()
        if process_name == "":
            process_name = "N/A"
        result.append({
            "name": process_name,
            "pid": process_list[index].pid,
            "mem": process_list[index].mem,
            "engine_usage": {
                "gfx": process_list[index].engine_usage.gfx,
                "enc": process_list[index].engine_usage.enc
            },
            "memory_usage": {
                "gtt_mem": process_list[index].memory_usage.gtt_mem,
                "cpu_mem": process_list[index].memory_usage.cpu_mem,
                "vram_mem": process_list[index].memory_usage.vram_mem,
            },
        })

    return result


def amdsmi_get_gpu_driver_info(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> Dict[str, Any]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    length = ctypes.c_int()
    length.value = AMDSMI_MAX_DRIVER_VERSION_LENGTH

    info = amdsmi_wrapper.amdsmi_driver_info_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_driver_info(
            processor_handle, ctypes.byref(info)
        )
    )

    driver_info = {
        "driver_name": info.driver_name.decode("utf-8"),
        "driver_version": info.driver_version.decode("utf-8"),
        "driver_date": info.driver_date.decode("utf-8")
    }

    for key, value in driver_info.items():
        if value == "":
            driver_info[key] = "N/A"

    return driver_info


# NOTE: this uses amdsmi_get_power_info_v2 under the hood because the C api
# needs to be backwards compatible
def amdsmi_get_power_info(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    sensor_ind: int = 0
) -> Dict[str, ctypes.c_uint32]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    power_measure = amdsmi_wrapper.amdsmi_power_info_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_power_info_v2(
            processor_handle, sensor_ind, ctypes.byref(power_measure)
        )
    )

    power_info_dict = {
        "current_socket_power": power_measure.current_socket_power,
        "average_socket_power": power_measure.average_socket_power,
        "gfx_voltage": power_measure.gfx_voltage,
        "soc_voltage": power_measure.soc_voltage,
        "mem_voltage": power_measure.mem_voltage,
        "power_limit" : power_measure.power_limit,
    }

    for key, value in power_info_dict.items():
        if value == 0xFFFF:
            power_info_dict[key] = "N/A"

    return power_info_dict


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
    _check_res(
        amdsmi_wrapper.amdsmi_get_fw_info(
            processor_handle, ctypes.byref(fw_info)
        )
    )

    # Certain FW blocks are padded with 0s in the front intentionally
    # But the C library converts the hex to an integer which trims the leading 0s
    # Nor do we have a flag that defines the expected format for each FW block
    # We can expect the following blocks to have a padded value and a specified format

    hex_format_fw = [AmdSmiFwBlock.AMDSMI_FW_ID_PSP_SOSDRV,
                     AmdSmiFwBlock.AMDSMI_FW_ID_TA_RAS,
                     AmdSmiFwBlock.AMDSMI_FW_ID_TA_XGMI,
                     AmdSmiFwBlock.AMDSMI_FW_ID_UVD,
                     AmdSmiFwBlock.AMDSMI_FW_ID_VCE,
                     AmdSmiFwBlock.AMDSMI_FW_ID_VCN]

    # PM(AKA: SMC) firmware's hex value looks like 0x12345678
    # However, they are parsed as: int(0x12).int(0x34).int(0x56).int(0x78)
    # Which results in the following: 12.34.56.78
    dec_format_fw = [AmdSmiFwBlock.AMDSMI_FW_ID_PM]

    firmwares = []
    for i in range(0, fw_info.num_fw_info):
        fw_name = AmdSmiFwBlock(fw_info.fw_info_list[i].fw_id)
        fw_version = fw_info.fw_info_list[i].fw_version # This is in int format (base 10)

        if fw_name in hex_format_fw:
            # Convert the fw_version from a int to a hex string padded leading 0s
            fw_version_string = hex(fw_version)[2:].zfill(8)

            # Join every two hex digits with a dot
            fw_version_string = ".".join(re.findall('..?', fw_version_string))
        elif fw_name in dec_format_fw:
            # Convert the fw_version from a int to a hex string padded leading 0s
            fw_version_string = hex(fw_version)[2:].zfill(8)

            # Convert every two hex digits to decimal and join them with a dot
            dec_version_string = ''
            for index, _ in enumerate(fw_version_string):
                if index % 2 != 0:
                    continue
                hex_digits = f"0x{fw_version_string[index:index+2]}"
                dec_version_string += str(int(hex_digits, 16)).zfill(2) + "."
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

    pcie_info_dict = {
        "pcie_static": {
            "max_pcie_width": _validate_if_max_uint(pcie_info.pcie_static.max_pcie_width, MaxUIntegerTypes.UINT16_T),
            "max_pcie_speed": _validate_if_max_uint(pcie_info.pcie_static.max_pcie_speed, MaxUIntegerTypes.UINT32_T),
            "pcie_interface_version": _validate_if_max_uint(pcie_info.pcie_static.pcie_interface_version, MaxUIntegerTypes.UINT32_T),
            "slot_type": pcie_info.pcie_static.slot_type,
            },
        "pcie_metric": {
            "pcie_width": _validate_if_max_uint(pcie_info.pcie_metric.pcie_width, MaxUIntegerTypes.UINT16_T),
            "pcie_speed": _validate_if_max_uint(pcie_info.pcie_metric.pcie_speed, MaxUIntegerTypes.UINT32_T),
            "pcie_bandwidth": _validate_if_max_uint(pcie_info.pcie_metric.pcie_bandwidth, MaxUIntegerTypes.UINT32_T),
            "pcie_replay_count": _validate_if_max_uint(pcie_info.pcie_metric.pcie_replay_count, MaxUIntegerTypes.UINT64_T),
            "pcie_l0_to_recovery_count": _validate_if_max_uint(pcie_info.pcie_metric.pcie_l0_to_recovery_count, MaxUIntegerTypes.UINT64_T),
            "pcie_replay_roll_over_count": _validate_if_max_uint(pcie_info.pcie_metric.pcie_replay_roll_over_count, MaxUIntegerTypes.UINT64_T),
            "pcie_nak_sent_count": _validate_if_max_uint(pcie_info.pcie_metric.pcie_nak_sent_count, MaxUIntegerTypes.UINT64_T),
            "pcie_nak_received_count": _validate_if_max_uint(pcie_info.pcie_metric.pcie_nak_received_count, MaxUIntegerTypes.UINT64_T),
            "pcie_lc_perf_other_end_recovery_count": _validate_if_max_uint(pcie_info.pcie_metric.pcie_lc_perf_other_end_recovery_count, MaxUIntegerTypes.UINT32_T)
        }
    }

    slot_type = pcie_info_dict['pcie_static']['slot_type']
    if isinstance(slot_type, int):
        slot_types = amdsmi_wrapper.amdsmi_card_form_factor_t__enumvalues
        if slot_type in slot_types:
            pcie_info_dict['pcie_static']['slot_type'] = slot_types[slot_type].replace("AMDSMI_CARD_FORM_FACTOR_", "")
        else:
            pcie_info_dict['pcie_static']['slot_type'] = "Unknown"
    else:
        pcie_info_dict['pcie_static']['slot_type'] = "N/A"

    return pcie_info_dict

def amdsmi_get_gpu_xcd_counter(processor_handle: amdsmi_wrapper.amdsmi_processor_handle) -> Dict[str, Any]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(processor_handle, amdsmi_wrapper.amdsmi_processor_handle)

    xcd_counter = ctypes.c_uint16()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_xcd_counter(
            processor_handle, ctypes.byref(xcd_counter)
        )
    )

    return xcd_counter.value

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

    return _pad_hex_value(hex(id.value), 4)


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


def amdsmi_topo_get_p2p_status(
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

    type = ctypes.c_uint32()
    cap = amdsmi_wrapper.struct_amdsmi_p2p_capability_t()

    _check_res(
        amdsmi_wrapper.amdsmi_topo_get_p2p_status(
            processor_handle_src, processor_handle_dst, ctypes.byref(type), ctypes.byref(cap)
        )
    )

    return {
        'type' : type,
        'cap': {
            'is_iolink_coherent': cap.is_iolink_coherent,
            'is_iolink_atomics_32bit': cap.is_iolink_atomics_32bit,
            'is_iolink_atomics_64bit': cap.is_iolink_atomics_64bit,
            'is_iolink_dma': cap.is_iolink_dma,
            'is_iolink_bi_directional': cap.is_iolink_bi_directional
        }
    }


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

def amdsmi_set_gpu_accelerator_partition_profile(processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
                                                 profile_index: int):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    if not isinstance(profile_index, int):
        raise AmdSmiParameterException(profile_index, int)

    _check_res(
        amdsmi_wrapper.amdsmi_set_gpu_accelerator_partition_profile(
            processor_handle, profile_index
        )
    )

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

def amdsmi_get_gpu_memory_partition_config(processor_handle: amdsmi_wrapper.amdsmi_processor_handle):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    config = amdsmi_wrapper.amdsmi_memory_partition_config_t()

    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_memory_partition_config(
            processor_handle, config
        )
    )
    mem_caps_list = []
    if config.partition_caps.nps_flags.nps1_cap == 1:
        mem_caps_list.append("NPS1")
    if config.partition_caps.nps_flags.nps2_cap == 1:
        mem_caps_list.append("NPS2")
    if config.partition_caps.nps_flags.nps4_cap == 1:
        mem_caps_list.append("NPS4")
    if config.partition_caps.nps_flags.nps8_cap == 1:
        mem_caps_list.append("NPS8")

    return_dict = {
        "partition_caps": mem_caps_list,
        "mp_mode": amdsmi_wrapper.amdsmi_memory_partition_type_t__enumvalues[
            config.mp_mode].replace("AMDSMI_MEMORY_PARTITION_", "").replace("UNKNOWN", "N/A"),
        "num_numa_ranges": "N/A",
        "numa_range": "N/A",
    }
    logging.debug("amdsmi_interface.py | amdsmi_get_gpu_memory_partition_config | return_dictionary = \n" + str(json.dumps(return_dict, indent=4)))
    return return_dict


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

def amdsmi_set_gpu_memory_partition_mode(processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
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

def amdsmi_get_gpu_accelerator_partition_profile(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle
    ) -> Dict[str, Any]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    exception_caught = False
    length = 8
    partition_id = [0, 0, 0, 0, 0, 0, 0, 0]
    partition_id_list = (ctypes.c_uint32 * length)(*partition_id)
    profile = amdsmi_wrapper.amdsmi_accelerator_partition_profile_t()
    partition_ids = []
    kPOSITION_OF_PARTITION_ID = 0

    ret = amdsmi_wrapper.amdsmi_get_gpu_accelerator_partition_profile(processor_handle,
                                    ctypes.byref(profile), partition_id_list)
    if ret == amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED:
        #partition_id[0] will contain the partition id of each device
        #BM/Guest will include this logic. Host will only display primary partition ids.
        partition_ids.append(partition_id_list[kPOSITION_OF_PARTITION_ID])

    try:
        _check_res(ret)
    except AmdSmiException as e:
        logging.debug("amdsmi_interface.py | amdsmi_get_gpu_accelerator_partition_profile | exception_caught >> " + str(e))
        partition_profile_dict = {
            "profile_type" : "N/A",
            "num_partitions" : "N/A",
            "profile_index" : "N/A",
            "memory_caps": "N/A",
            "num_resources" : "N/A",
            "resources" : "N/A"
        }
        return_dictionary = {
            "partition_id" : partition_ids,
            "partition_profile" : partition_profile_dict
        }
        if ret == amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED:
            exception_caught = True
        else:
            _check_res(ret) # re-raise the exception if error is anything other than AMDSMI_STATUS_NOT_SUPPORTED
                            # this ensures we can get partition ID even if the profile is not supported.
    finally:
        if exception_caught:
            logging.debug("amdsmi_interface.py | exception_caught >> amdsmi_get_gpu_accelerator_partition_profile | return_dictionary = \n" + str(json.dumps(return_dictionary, indent=4)))
            return return_dictionary
        else:
            profile_type_ret = amdsmi_wrapper.amdsmi_accelerator_partition_type_t__enumvalues[profile.profile_type].replace("AMDSMI_ACCELERATOR_PARTITION_", "")
            profile_type_ret = profile_type_ret.replace("INVALID", "N/A")
            length = profile.num_partitions
            #partition_id[0] will contain the partition id of each device
            #BM/Guest will include this logic. Host will only display primary partition ids.
            partition_ids.append(partition_id_list[kPOSITION_OF_PARTITION_ID])
            mem_caps_list = []
            if profile.memory_caps.nps_flags.nps1_cap == 1:
                mem_caps_list.append("NPS1")
            if profile.memory_caps.nps_flags.nps2_cap == 1:
                mem_caps_list.append("NPS2")
            if profile.memory_caps.nps_flags.nps4_cap == 1:
                mem_caps_list.append("NPS4")
            if profile.memory_caps.nps_flags.nps8_cap == 1:
                mem_caps_list.append("NPS8")
            partition_profile_dict = {
                "profile_type" : profile_type_ret,
                "num_partitions" : profile.num_partitions,
                "profile_index" : profile.profile_index,
                "memory_caps": mem_caps_list,
                "num_resources" : profile.num_resources,
                "resources" : "N/A"
            }
            return_dictionary = {
                "partition_id" : partition_ids,
                "partition_profile" : partition_profile_dict
            }
            logging.debug("amdsmi_interface.py | amdsmi_get_gpu_accelerator_partition_profile | return_dictionary = \n" + str(json.dumps(return_dictionary, indent=4)))
            return return_dictionary

def amdsmi_get_gpu_accelerator_partition_profile_config(processor_handle: amdsmi_wrapper.amdsmi_processor_handle) -> Dict:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    config = amdsmi_wrapper.amdsmi_accelerator_partition_profile_config_t()

    _check_res(amdsmi_wrapper.amdsmi_get_gpu_accelerator_partition_profile_config(processor_handle,
                                                                    ctypes.byref(config)))
    logging.debug("\namdsmi_interface.py | amdsmi_get_gpu_accelerator_partition_profile_config | START - "
          + "config.num_profiles = " + str(config.num_profiles)
          + "\n; config.num_resource_profiles = " + str(config.num_resource_profiles)
          + "\n; config.resource_profiles = " + str(config.resource_profiles)
          + "\n; config.default_profile_index = " + str(config.default_profile_index)
          + "\n; config.profiles = " + str(config.profiles))

    profiles = []
    resource_idx = 0
    for i in range(config.num_profiles):
        profile = config.profiles[i]
        logging.debug("\namdsmi_interface.py | amdsmi_get_gpu_accelerator_partition_profile_config | profile = " + str(profile))
        profile_type_ret = amdsmi_wrapper.amdsmi_accelerator_partition_type_t__enumvalues[
            config.profiles[i].profile_type].replace("AMDSMI_ACCELERATOR_PARTITION_", "")
        profile_type_ret = profile_type_ret.replace("INVALID", "N/A")
        resources = []


        mem_caps_list = []
        if profile.memory_caps.nps_flags.nps1_cap == 1:
            mem_caps_list.append("NPS1")
        if profile.memory_caps.nps_flags.nps2_cap == 1:
            mem_caps_list.append("NPS2")
        if profile.memory_caps.nps_flags.nps4_cap == 1:
            mem_caps_list.append("NPS4")
        if profile.memory_caps.nps_flags.nps8_cap == 1:
            mem_caps_list.append("NPS8")

        for r in range(config.num_resource_profiles):
            logging.debug("\namdsmi_interface.py | amdsmi_get_gpu_accelerator_partition_profile_config | i = " + str(i) + "; r = " + str(r) + "; resource_idx = " + str(resource_idx))
            res_profile = config.resource_profiles[resource_idx]
            resource_profiles_ret = amdsmi_wrapper.amdsmi_accelerator_partition_resource_type_t__enumvalues[
                res_profile.resource_type].replace("AMDSMI_ACCELERATOR_", "")
            resource_profile_dict = {
                "profile_index": res_profile.profile_index,
                "resource_type": resource_profiles_ret,
                "partition_resource": res_profile.partition_resource,
                "num_partitions_share_resource": res_profile.num_partitions_share_resource,
            }
            logging.debug("\namdsmi_interface.py | amdsmi_get_gpu_accelerator_partition_profile_config | resource_profile_dict = " + str(resource_profile_dict))
            resources.append(resource_profile_dict)
            resource_idx += 1

        profile_dict = {
            "profile_type": profile_type_ret,
            "num_partitions": profile.num_partitions,
            "profile_index": profile.profile_index,
            "memory_caps": mem_caps_list,
            "num_resources": profile.num_resources,
            "resources": resources
        }
        profiles.append(profile_dict)

    config_dict = {
        "num_profiles": config.num_profiles,
        "num_resource_profiles": config.num_resource_profiles,
        "resource_profiles": resources,
        "default_profile_index": config.default_profile_index,
        "profiles": profiles,
    }
    logging.debug("\namdsmi_interface.py | amdsmi_get_gpu_accelerator_partition_profile_config | END - config_dict = \n" + str(json.dumps(config_dict, indent=4)))

    return config_dict

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
    clk_type: str,
    freq_bitmask: int,
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if clk_type.lower() == "sclk":
        clk_type_conversion = AmdSmiClkType.SYS
    elif clk_type.lower() == "mclk":
        clk_type_conversion = AmdSmiClkType.MEM
    elif clk_type.lower() == "fclk":
        clk_type_conversion = AmdSmiClkType.DF
    elif clk_type.lower() == "socclk":
        clk_type_conversion = AmdSmiClkType.SOC
    else:
        clk_type_conversion = "N/A"
    if not isinstance(clk_type_conversion, AmdSmiClkType):
        raise AmdSmiParameterException(clk_type_conversion, AmdSmiClkType)
    if not isinstance(freq_bitmask, int):
        raise AmdSmiParameterException(freq_bitmask, int)
    freq_bitmask = ctypes.c_uint64(freq_bitmask)
    _check_res(
        amdsmi_wrapper.amdsmi_set_clk_freq(
            processor_handle, clk_type_conversion, freq_bitmask
        )
    )


def amdsmi_set_soc_pstate(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    policy_id: int,
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    _check_res(
        amdsmi_wrapper.amdsmi_set_soc_pstate(
            processor_handle, policy_id
        )
    )


def amdsmi_set_xgmi_plpd(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    policy_id: int,
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    _check_res(
        amdsmi_wrapper.amdsmi_set_xgmi_plpd(
            processor_handle, policy_id
        )
    )


def amdsmi_set_gpu_process_isolation(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    pisolate: int,
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    _check_res(
        amdsmi_wrapper.amdsmi_set_gpu_process_isolation(
            processor_handle, pisolate
        )
    )


def amdsmi_clean_gpu_local_data(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
):
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    _check_res(
        amdsmi_wrapper.amdsmi_clean_gpu_local_data(
            processor_handle
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

    energy_accumulator= ctypes.c_uint64()
    counter_resolution = ctypes.c_float()
    timestamp = ctypes.c_uint64()

    _check_res(
        amdsmi_wrapper.amdsmi_get_energy_count(processor_handle, ctypes.byref(
            energy_accumulator), ctypes.byref(counter_resolution), ctypes.byref(timestamp))
    )

    return {
        'energy_accumulator': energy_accumulator.value,
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


def amdsmi_set_gpu_clk_limit(
        processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
        clk_type: str,
        limit_type: str,
        value: int
    ) -> None:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )
    if not isinstance(value, int):
        raise AmdSmiParameterException(value, int)
    if clk_type.lower() == "sclk":
        clk_type_conversion = amdsmi_wrapper.AMDSMI_CLK_TYPE_SYS
    elif clk_type.lower() == "mclk":
        clk_type_conversion = amdsmi_wrapper.AMDSMI_CLK_TYPE_MEM
    if limit_type.lower() == "min":
        limit_type_conversion = amdsmi_wrapper.CLK_LIMIT_MIN
    elif limit_type.lower() == "max":
        limit_type_conversion = amdsmi_wrapper.CLK_LIMIT_MAX
    _check_res(
        amdsmi_wrapper.amdsmi_set_gpu_clk_limit(
            processor_handle,
            amdsmi_wrapper.amdsmi_clk_type_t(clk_type_conversion),
            amdsmi_wrapper.amdsmi_clk_limit_type_t(limit_type_conversion),
            ctypes.c_uint64(value),
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

    # Enforce List typing
    if not isinstance(counter_types, list):
        counter_types = [counter_types]

    counter_types = list(set(counter_types))

    # Validate Inputs
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
            counter_type = "AMDSMI_FINE_DECODER_ACTIVITY"
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


def amdsmi_get_gpu_mem_overdrive_level(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> int:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    mem_od_level = ctypes.c_uint32()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_mem_overdrive_level(
            processor_handle, ctypes.byref(mem_od_level)
        )
    )

    return mem_od_level.value


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

    clk_type_str = AmdSmiClkType(clk_type).name

    dict_ret = {
        "num_supported": freq.num_supported,
        "current": freq.current,
        "frequency": list(freq.frequency)[: freq.num_supported],
    }
    logging.debug("amdsmi_interface.py | amdsmi_get_clk_freq | clk_type = " + clk_type_str + " | return_dictionary = \n" + str(json.dumps(dict_ret, indent=4)))
    return dict_ret


def amdsmi_get_soc_pstate(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> Dict[str, Any]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    policy = amdsmi_wrapper.amdsmi_dpm_policy_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_soc_pstate(
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


def amdsmi_get_xgmi_plpd(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> Dict[str, Any]:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    policy = amdsmi_wrapper.amdsmi_dpm_policy_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_xgmi_plpd(
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
        "plpds": polices,
    }


def amdsmi_get_gpu_process_isolation(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
) -> int:
    if not isinstance(processor_handle, amdsmi_wrapper.amdsmi_processor_handle):
        raise AmdSmiParameterException(
            processor_handle, amdsmi_wrapper.amdsmi_processor_handle
        )

    pisolate = ctypes.c_uint32()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_process_isolation(
            processor_handle, ctypes.byref(pisolate)
        )
    )

    return pisolate.value


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
        "temperature_edge": _validate_if_max_uint(gpu_metrics.temperature_edge, MaxUIntegerTypes.UINT16_T),
        "temperature_hotspot": _validate_if_max_uint(gpu_metrics.temperature_hotspot, MaxUIntegerTypes.UINT16_T),
        "temperature_mem": _validate_if_max_uint(gpu_metrics.temperature_mem, MaxUIntegerTypes.UINT16_T),
        "temperature_vrgfx": _validate_if_max_uint(gpu_metrics.temperature_vrgfx, MaxUIntegerTypes.UINT16_T),
        "temperature_vrsoc": _validate_if_max_uint(gpu_metrics.temperature_vrsoc, MaxUIntegerTypes.UINT16_T),
        "temperature_vrmem": _validate_if_max_uint(gpu_metrics.temperature_vrmem, MaxUIntegerTypes.UINT16_T),
        "average_gfx_activity": _validate_if_max_uint(gpu_metrics.average_gfx_activity, MaxUIntegerTypes.UINT16_T, isActivity=True),
        "average_umc_activity": _validate_if_max_uint(gpu_metrics.average_umc_activity, MaxUIntegerTypes.UINT16_T, isActivity=True),
        "average_mm_activity": _validate_if_max_uint(gpu_metrics.average_mm_activity, MaxUIntegerTypes.UINT16_T, isActivity=True),
        "average_socket_power": _validate_if_max_uint(gpu_metrics.average_socket_power, MaxUIntegerTypes.UINT16_T),
        "energy_accumulator": _validate_if_max_uint(gpu_metrics.energy_accumulator, MaxUIntegerTypes.UINT64_T),
        "system_clock_counter": _validate_if_max_uint(gpu_metrics.system_clock_counter, MaxUIntegerTypes.UINT64_T),
        "average_gfxclk_frequency": _validate_if_max_uint(gpu_metrics.average_gfxclk_frequency, MaxUIntegerTypes.UINT16_T),
        "average_socclk_frequency": _validate_if_max_uint(gpu_metrics.average_socclk_frequency, MaxUIntegerTypes.UINT16_T),
        "average_uclk_frequency": _validate_if_max_uint(gpu_metrics.average_uclk_frequency, MaxUIntegerTypes.UINT16_T),
        "average_vclk0_frequency": _validate_if_max_uint(gpu_metrics.average_vclk0_frequency, MaxUIntegerTypes.UINT16_T),
        "average_dclk0_frequency": _validate_if_max_uint(gpu_metrics.average_dclk0_frequency, MaxUIntegerTypes.UINT16_T),
        "average_vclk1_frequency": _validate_if_max_uint(gpu_metrics.average_vclk1_frequency, MaxUIntegerTypes.UINT16_T),
        "average_dclk1_frequency": _validate_if_max_uint(gpu_metrics.average_dclk1_frequency, MaxUIntegerTypes.UINT16_T),
        "current_gfxclk": _validate_if_max_uint(gpu_metrics.current_gfxclk, MaxUIntegerTypes.UINT16_T),
        "current_socclk": _validate_if_max_uint(gpu_metrics.current_socclk, MaxUIntegerTypes.UINT16_T),
        "current_uclk": _validate_if_max_uint(gpu_metrics.current_uclk, MaxUIntegerTypes.UINT16_T),
        "current_vclk0": _validate_if_max_uint(gpu_metrics.current_vclk0, MaxUIntegerTypes.UINT16_T),
        "current_dclk0": _validate_if_max_uint(gpu_metrics.current_dclk0, MaxUIntegerTypes.UINT16_T),
        "current_vclk1": _validate_if_max_uint(gpu_metrics.current_vclk1, MaxUIntegerTypes.UINT16_T),
        "current_dclk1": _validate_if_max_uint(gpu_metrics.current_dclk1, MaxUIntegerTypes.UINT16_T),
        "throttle_status": _validate_if_max_uint(gpu_metrics.throttle_status, MaxUIntegerTypes.UINT32_T, isBool=True),
        "current_fan_speed": _validate_if_max_uint(gpu_metrics.current_fan_speed, MaxUIntegerTypes.UINT16_T),
        "pcie_link_width": _validate_if_max_uint(gpu_metrics.pcie_link_width, MaxUIntegerTypes.UINT16_T),
        "pcie_link_speed": _validate_if_max_uint(gpu_metrics.pcie_link_speed, MaxUIntegerTypes.UINT16_T),
        "gfx_activity_acc": _validate_if_max_uint(gpu_metrics.gfx_activity_acc, MaxUIntegerTypes.UINT32_T),
        "mem_activity_acc": _validate_if_max_uint(gpu_metrics.mem_activity_acc, MaxUIntegerTypes.UINT32_T),
        "temperature_hbm": _validate_if_max_uint(list(gpu_metrics.temperature_hbm), MaxUIntegerTypes.UINT16_T),
        "firmware_timestamp": _validate_if_max_uint(gpu_metrics.firmware_timestamp, MaxUIntegerTypes.UINT64_T),
        "voltage_soc": _validate_if_max_uint(gpu_metrics.voltage_soc, MaxUIntegerTypes.UINT16_T),
        "voltage_gfx": _validate_if_max_uint(gpu_metrics.voltage_gfx, MaxUIntegerTypes.UINT16_T),
        "voltage_mem": _validate_if_max_uint(gpu_metrics.voltage_mem, MaxUIntegerTypes.UINT16_T),
        "indep_throttle_status": _validate_if_max_uint(gpu_metrics.indep_throttle_status, MaxUIntegerTypes.UINT64_T, isBool=True),
        "current_socket_power": _validate_if_max_uint(gpu_metrics.current_socket_power, MaxUIntegerTypes.UINT16_T),
        "vcn_activity": _validate_if_max_uint(list(gpu_metrics.vcn_activity), MaxUIntegerTypes.UINT16_T, isActivity=True),
        "gfxclk_lock_status": _validate_if_max_uint(gpu_metrics.gfxclk_lock_status, MaxUIntegerTypes.UINT32_T),
        "xgmi_link_width": _validate_if_max_uint(gpu_metrics.xgmi_link_width, MaxUIntegerTypes.UINT16_T),
        "xgmi_link_speed": _validate_if_max_uint(gpu_metrics.xgmi_link_speed, MaxUIntegerTypes.UINT16_T),
        "pcie_bandwidth_acc": _validate_if_max_uint(gpu_metrics.pcie_bandwidth_acc, MaxUIntegerTypes.UINT64_T),
        "pcie_bandwidth_inst": _validate_if_max_uint(gpu_metrics.pcie_bandwidth_inst, MaxUIntegerTypes.UINT64_T),
        "pcie_l0_to_recov_count_acc": _validate_if_max_uint(gpu_metrics.pcie_l0_to_recov_count_acc, MaxUIntegerTypes.UINT64_T),
        "pcie_replay_count_acc": _validate_if_max_uint(gpu_metrics.pcie_replay_count_acc, MaxUIntegerTypes.UINT64_T),
        "pcie_replay_rover_count_acc": _validate_if_max_uint(gpu_metrics.pcie_replay_rover_count_acc, MaxUIntegerTypes.UINT64_T),
        "xgmi_read_data_acc": _validate_if_max_uint(list(gpu_metrics.xgmi_read_data_acc), MaxUIntegerTypes.UINT64_T),
        "xgmi_write_data_acc": _validate_if_max_uint(list(gpu_metrics.xgmi_write_data_acc), MaxUIntegerTypes.UINT64_T),
        "current_gfxclks": _validate_if_max_uint(list(gpu_metrics.current_gfxclks), MaxUIntegerTypes.UINT16_T),
        "current_socclks": _validate_if_max_uint(list(gpu_metrics.current_socclks), MaxUIntegerTypes.UINT16_T),
        "current_vclk0s": _validate_if_max_uint(list(gpu_metrics.current_vclk0s), MaxUIntegerTypes.UINT16_T),
        "current_dclk0s": _validate_if_max_uint(list(gpu_metrics.current_dclk0s), MaxUIntegerTypes.UINT16_T),
        "jpeg_activity": _validate_if_max_uint(list(gpu_metrics.jpeg_activity), MaxUIntegerTypes.UINT16_T, isActivity=True),
        "pcie_nak_sent_count_acc": _validate_if_max_uint(gpu_metrics.pcie_nak_sent_count_acc, MaxUIntegerTypes.UINT32_T),
        "pcie_nak_rcvd_count_acc": _validate_if_max_uint(gpu_metrics.pcie_nak_rcvd_count_acc, MaxUIntegerTypes.UINT32_T),
        "accumulation_counter": _validate_if_max_uint(gpu_metrics.accumulation_counter, MaxUIntegerTypes.UINT64_T),
        "prochot_residency_acc": _validate_if_max_uint(gpu_metrics.prochot_residency_acc, MaxUIntegerTypes.UINT64_T),
        "ppt_residency_acc": _validate_if_max_uint(gpu_metrics.ppt_residency_acc, MaxUIntegerTypes.UINT64_T),
        "socket_thm_residency_acc": _validate_if_max_uint(gpu_metrics.socket_thm_residency_acc, MaxUIntegerTypes.UINT64_T),
        "vr_thm_residency_acc": _validate_if_max_uint(gpu_metrics.vr_thm_residency_acc, MaxUIntegerTypes.UINT64_T),
        "hbm_thm_residency_acc": _validate_if_max_uint(gpu_metrics.hbm_thm_residency_acc, MaxUIntegerTypes.UINT64_T),
        "num_partition": _validate_if_max_uint(gpu_metrics.num_partition, MaxUIntegerTypes.UINT16_T),
        "xcp_stats.gfx_busy_inst": list(gpu_metrics.xcp_stats),
        "xcp_stats.jpeg_busy": list(gpu_metrics.xcp_stats),
        "xcp_stats.vcn_busy": list(gpu_metrics.xcp_stats),
        "xcp_stats.gfx_busy_acc": list(gpu_metrics.xcp_stats),
        "xcp_stats.gfx_below_host_limit_acc": list(gpu_metrics.xcp_stats),
        "xcp_stats.gfx_below_host_limit_ppt_acc": list(gpu_metrics.xcp_stats),
        "xcp_stats.gfx_below_host_limit_thm_acc": list(gpu_metrics.xcp_stats),
        "xcp_stats.gfx_low_utilization_acc": list(gpu_metrics.xcp_stats),
        "xcp_stats.gfx_below_host_limit_total_acc": list(gpu_metrics.xcp_stats),
        "pcie_lc_perf_other_end_recovery": _validate_if_max_uint(gpu_metrics.pcie_lc_perf_other_end_recovery, MaxUIntegerTypes.UINT32_T),
        "vram_max_bandwidth": _validate_if_max_uint(gpu_metrics.vram_max_bandwidth, MaxUIntegerTypes.UINT64_T),
        "xgmi_link_status": _validate_if_max_uint(list(gpu_metrics.xgmi_link_status), MaxUIntegerTypes.UINT16_T),
    }

    # Create 2d array with each XCD's stats
    if 'xcp_stats.gfx_busy_inst' in gpu_metrics_output:
        for xcp_index, xcp_metrics in enumerate(gpu_metrics_output['xcp_stats.gfx_busy_inst']):
            xcp_detail = []
            for val in xcp_metrics.gfx_busy_inst:
                xcp_detail.append(_validate_if_max_uint(val, MaxUIntegerTypes.UINT32_T, isActivity=True))
            gpu_metrics_output['xcp_stats.gfx_busy_inst'][xcp_index] = xcp_detail

    if 'xcp_stats.jpeg_busy' in gpu_metrics_output:
        for xcp_index, xcp_metrics in enumerate(gpu_metrics_output['xcp_stats.jpeg_busy']):
            xcp_detail = []
            for val in xcp_metrics.jpeg_busy:
                xcp_detail.append(_validate_if_max_uint(val, MaxUIntegerTypes.UINT16_T, isActivity=True))
            gpu_metrics_output['xcp_stats.jpeg_busy'][xcp_index] = xcp_detail

    if 'xcp_stats.vcn_busy' in gpu_metrics_output:
        for xcp_index, xcp_metrics in enumerate(gpu_metrics_output['xcp_stats.vcn_busy']):
            xcp_detail = []
            for val in xcp_metrics.vcn_busy:
                xcp_detail.append(_validate_if_max_uint(val, MaxUIntegerTypes.UINT16_T, isActivity=True))
            gpu_metrics_output["xcp_stats.vcn_busy"][xcp_index] = xcp_detail

    if 'xcp_stats.gfx_busy_acc' in gpu_metrics_output:
        for xcp_index, xcp_metrics in enumerate(gpu_metrics_output['xcp_stats.gfx_busy_acc']):
            xcp_detail = []
            for val in xcp_metrics.gfx_busy_acc:
                xcp_detail.append(_validate_if_max_uint(val, MaxUIntegerTypes.UINT64_T))
            gpu_metrics_output["xcp_stats.gfx_busy_acc"][xcp_index] = xcp_detail

    if 'xcp_stats.gfx_below_host_limit_acc' in gpu_metrics_output:
        for xcp_index, xcp_metrics in enumerate(gpu_metrics_output['xcp_stats.gfx_below_host_limit_acc']):
            xcp_detail = []
            for val in xcp_metrics.gfx_below_host_limit_acc:
                xcp_detail.append(_validate_if_max_uint(val, MaxUIntegerTypes.UINT64_T))
            gpu_metrics_output['xcp_stats.gfx_below_host_limit_acc'][xcp_index] = xcp_detail
    # new for gpu metrics v1.8
    if 'xcp_stats.gfx_below_host_limit_ppt_acc'  in gpu_metrics_output:
        for xcp_index, xcp_metrics in enumerate(gpu_metrics_output['xcp_stats.gfx_below_host_limit_ppt_acc']):
            xcp_detail = []
            for val in xcp_metrics.gfx_below_host_limit_ppt_acc:
                xcp_detail.append(_validate_if_max_uint(val, MaxUIntegerTypes.UINT64_T))
            gpu_metrics_output['xcp_stats.gfx_below_host_limit_ppt_acc'][xcp_index] = xcp_detail
    if 'xcp_stats.gfx_below_host_limit_thm_acc' in gpu_metrics_output:
        for xcp_index, xcp_metrics in enumerate(gpu_metrics_output['xcp_stats.gfx_below_host_limit_thm_acc']):
            xcp_detail = []
            for val in xcp_metrics.gfx_below_host_limit_thm_acc:
                xcp_detail.append(_validate_if_max_uint(val, MaxUIntegerTypes.UINT64_T))
            gpu_metrics_output['xcp_stats.gfx_below_host_limit_thm_acc'][xcp_index] = xcp_detail
    if 'xcp_stats.gfx_low_utilization_acc' in gpu_metrics_output:
        for xcp_index, xcp_metrics in enumerate(gpu_metrics_output['xcp_stats.gfx_low_utilization_acc']):
            xcp_detail = []
            for val in xcp_metrics.gfx_low_utilization_acc:
                xcp_detail.append(_validate_if_max_uint(val, MaxUIntegerTypes.UINT64_T))
            gpu_metrics_output['xcp_stats.gfx_low_utilization_acc'][xcp_index] = xcp_detail
    if 'xcp_stats.gfx_below_host_limit_total_acc' in gpu_metrics_output:
        for xcp_index, xcp_metrics in enumerate(gpu_metrics_output['xcp_stats.gfx_below_host_limit_total_acc']):
            xcp_detail = []
            for val in xcp_metrics.gfx_below_host_limit_total_acc:
                xcp_detail.append(_validate_if_max_uint(val, MaxUIntegerTypes.UINT64_T))
            gpu_metrics_output['xcp_stats.gfx_below_host_limit_total_acc'][xcp_index] = xcp_detail
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
        "deferred_count": ec.deferred_count,
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
            "pasid": proc.pasid, # Not working in ROCm 6.4+, deprecating in 7.0
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
        "pasid": proc.pasid, # Not working in ROCm 6.4+, deprecating in 7.0
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
    nullptr = ctypes.POINTER(amdsmi_wrapper.amdsmi_retired_page_record_t)()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_memory_reserved_pages(
            processor_handle, ctypes.byref(num_pages), nullptr
        )
    )

    if num_pages.value == 0:
        return []

    mem_reserved_pages = (amdsmi_wrapper.amdsmi_retired_page_record_t * num_pages)()
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_memory_reserved_pages(
            processor_handle, ctypes.byref(num_pages), mem_reserved_pages
        )
    )

    return _format_bad_page_info(mem_reserved_pages, num_pages)


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
            processor_handle, ctypes.byref(header_info)
        )
    )

    return {
        "structure_size": header_info.structure_size,
        "format_revision": header_info.format_revision,
        "content_revision": header_info.content_revision
    }


def amdsmi_get_link_topology_nearest(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle,
    link_type: AmdSmiLinkType,
    )-> Dict[str, Any]:

    topology_nearest_list = amdsmi_wrapper.amdsmi_topology_nearest_t()
    _check_res(
        amdsmi_wrapper.amdsmi_get_link_topology_nearest(
               processor_handle,
               link_type,
               ctypes.byref(topology_nearest_list)
        )
    )

    device_list = []
    for index in range(topology_nearest_list.count):
        device_list.append(topology_nearest_list.processor_list[index])

    return {
        'processor_list': device_list
    }


def amdsmi_get_gpu_virtualization_mode(
    processor_handle: amdsmi_wrapper.amdsmi_processor_handle
    ) -> Dict[str, AmdSmiVirtualizationMode]:

    # make info struct here
    mode = amdsmi_wrapper.amdsmi_virtualization_mode_t()

    # call lib function here
    _check_res(
        amdsmi_wrapper.amdsmi_get_gpu_virtualization_mode(
            processor_handle,
            ctypes.byref(mode)
        )
    )

    return {
        "mode": AmdSmiVirtualizationMode(mode.value)
    }

### Non C-Lib APIs ###

def amdsmi_get_rocm_version()-> Tuple[bool, str]:
    """
    Get the ROCm version for the rocm-core library.

    This function attempts to retrieve the ROCm version by loading the `librocm-core.so` shared library
    and calling its `getROCmVersion` function. The version is returned as a string in the format "major.minor.patch".

    Returns:
        Tuple[bool, str]: A tuple containing a boolean and a string.
            - The boolean indicates whether the operation was successful.
            - The string contains the ROCm version if successful, or an error message if not.

    Raises:
        Exception: If there is an error loading the shared library or calling the function.

    Example:
        rocm_lib_status, version_message = amdsmi_get_rocm_version()
        if rocm_lib_status:
            print(f"ROCm version: {version_message}")
        else:
            print(f"Error: {version_message}")
    """
    # librocm-core.so can be located in found using several different methods.
    # Look for it with below priority:
    # 1. ROCM_HOME/ROCM_PATH environment variables
    #    - ROCM_HOME/lib
    #    - ROCM_PATH/lib (usually set to /opt/rocm/)
    # 2. Decided by the linker
    #    - LD_LIBRARY_PATH env var
    #    - defined path in /etc/ld.so.conf.d/
    # 3. Relative to amdsmi_wrapper.py in /opt/rocm/share/amd_smi
    #    - parent directory

    try:
        possible_locations = list()
        # 1.
        rocm_path = os.getenv("ROCM_HOME", os.getenv("ROCM_PATH"))
        if rocm_path:
            possible_locations.append(os.path.join(rocm_path, "lib/librocm-core.so"))

        # Check if /opt/rocm/lib/librocm-core.so exists and add it to the list
        if os.path.exists("/opt/rocm/lib/librocm-core.so"):
            possible_locations.append("/opt/rocm/lib/librocm-core.so")
        # 2.
        possible_locations.append("librocm-core.so")
        # 3.
        librocm_core_parent_dir =  Path(__file__).resolve().parent.parent.parent / "lib" / "librocm-core.so"
        possible_locations.append(librocm_core_parent_dir)

        for librocm_core_file_path in possible_locations:
            try:
                librocm_core = ctypes.CDLL(librocm_core_file_path)
                VerErrors = ctypes.c_uint32
                get_rocm_core_version = librocm_core.getROCmVersion
                get_rocm_core_version.restype = VerErrors
                get_rocm_core_version.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(ctypes.c_uint32)]

                # call the function
                major =  ctypes.c_uint32()
                minor =  ctypes.c_uint32()
                patch =  ctypes.c_uint32()

                if get_rocm_core_version(ctypes.byref(major), ctypes.byref(minor),ctypes.byref(patch)) == 0:
                    return True, f"{major.value}.{minor.value}.{patch.value}"
                else:
                    return False, "Failed to unpack ROCm version"
            except OSError as e:
                err = e
                continue

        # If we hit here, we were unable to find the librocm-core.so file
        return False, "Could not find librocm-core.so"
    except Exception as e:
        return False, f"Unable to detect ROCm installation, Unknown Error: {e}"
