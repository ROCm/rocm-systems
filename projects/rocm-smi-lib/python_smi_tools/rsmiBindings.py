#!/usr/bin/env python3
"""ROCm_SMI_LIB CLI Tool Python Bindings"""
# NOTE: You MUST call rsmiBindings.initRsmiBindings() when using this library!
# TODO: Get most (or all) of these from rocm_smi.h to avoid mismatches and redundancy

from __future__ import print_function
from ctypes import *
from enum import Enum

import sys

if 'sphinx' in sys.modules:
    path_librocm = str()
    def initRsmiBindings(silent=False):
        # Empty function for document generation
        exit()

    SMI_HASH = '@PKG_VERSION_HASH@'
else:
    from rsmiBindingsInit import *


# Device ID
dv_id = c_uint64()
# GPU ID
gpu_id = c_uint32(0)


# Policy enums
RSMI_MAX_NUM_FREQUENCIES = 33
RSMI_MAX_FAN_SPEED = 255
RSMI_NUM_VOLTAGE_CURVE_POINTS = 3


class rsmi_status_t(c_int):
    RSMI_STATUS_SUCCESS = 0x0
    RSMI_STATUS_INVALID_ARGS = 0x1
    RSMI_STATUS_NOT_SUPPORTED = 0x2
    RSMI_STATUS_FILE_ERROR = 0x3
    RSMI_STATUS_PERMISSION = 0x4
    RSMI_STATUS_OUT_OF_RESOURCES = 0x5
    RSMI_STATUS_INTERNAL_EXCEPTION = 0x6
    RSMI_STATUS_INPUT_OUT_OF_BOUNDS = 0x7
    RSMI_STATUS_INIT_ERROR = 0x8
    RSMI_INITIALIZATION_ERROR = RSMI_STATUS_INIT_ERROR
    RSMI_STATUS_NOT_YET_IMPLEMENTED = 0x9
    RSMI_STATUS_NOT_FOUND = 0xA
    RSMI_STATUS_INSUFFICIENT_SIZE = 0xB
    RSMI_STATUS_INTERRUPT = 0xC
    RSMI_STATUS_UNEXPECTED_SIZE = 0xD
    RSMI_STATUS_NO_DATA = 0xE
    RSMI_STATUS_UNEXPECTED_DATA = 0xF
    RSMI_STATUS_BUSY = 0x10
    RSMI_STATUS_REFCOUNT_OVERFLOW = 0x11
    RSMI_STATUS_SETTING_UNAVAILABLE = 0x12
    RSMI_STATUS_AMDGPU_RESTART_ERR = 0x13
    RSMI_STATUS_DRM_ERROR = 0x14
    RSMI_STATUS_FAIL_LOAD_MODULE = 0x15
    RSMI_STATUS_FAIL_LOAD_SYMBOL = 0x16
    RSMI_STATUS_UNKNOWN_ERROR = 0xFFFFFFFF


#Dictionary of rsmi ret codes and it's verbose output
rsmi_status_verbose_err_out = {
    rsmi_status_t.RSMI_STATUS_SUCCESS: 'Operation was successful',
    rsmi_status_t.RSMI_STATUS_INVALID_ARGS: 'Invalid arguments provided',
    rsmi_status_t.RSMI_STATUS_NOT_SUPPORTED: 'Not supported on the given system',
    rsmi_status_t.RSMI_STATUS_FILE_ERROR: 'Problem accessing a file',
    rsmi_status_t.RSMI_STATUS_PERMISSION: 'Permission denied',
    rsmi_status_t.RSMI_STATUS_OUT_OF_RESOURCES: 'Unable to acquire memory or other resource',
    rsmi_status_t.RSMI_STATUS_INTERNAL_EXCEPTION: 'An internal exception was caught',
    rsmi_status_t.RSMI_STATUS_INPUT_OUT_OF_BOUNDS: 'Provided input is out of allowable or safe range',
    rsmi_status_t.RSMI_INITIALIZATION_ERROR: 'Error occurred during rsmi initialization',
    rsmi_status_t.RSMI_STATUS_NOT_YET_IMPLEMENTED: 'Requested function is not implemented on this setup',
    rsmi_status_t.RSMI_STATUS_NOT_FOUND: 'Item searched for but not found',
    rsmi_status_t.RSMI_STATUS_INSUFFICIENT_SIZE: 'Insufficient resources available',
    rsmi_status_t.RSMI_STATUS_INTERRUPT: 'Interrupt occurred during execution',
    rsmi_status_t.RSMI_STATUS_UNEXPECTED_SIZE: 'Unexpected amount of data read',
    rsmi_status_t.RSMI_STATUS_NO_DATA: 'No data found for the given input',
    rsmi_status_t.RSMI_STATUS_UNEXPECTED_DATA: 'Unexpected data received',
    rsmi_status_t.RSMI_STATUS_BUSY: 'Busy - resources are preventing call the ability to execute',
    rsmi_status_t.RSMI_STATUS_REFCOUNT_OVERFLOW: 'Data overflow - data exceeded INT32_MAX',
    rsmi_status_t.RSMI_STATUS_SETTING_UNAVAILABLE: 'Requested setting is unavailable for current device',
    rsmi_status_t.RSMI_STATUS_AMDGPU_RESTART_ERR: 'Could not successfully restart the amdgpu driver',
    rsmi_status_t.RSMI_STATUS_DRM_ERROR: 'Error when calling libdrm',
    rsmi_status_t.RSMI_STATUS_FAIL_LOAD_MODULE: 'Failed to load a library',
    rsmi_status_t.RSMI_STATUS_FAIL_LOAD_SYMBOL: 'Failed to load a library symbol',
    rsmi_status_t.RSMI_STATUS_UNKNOWN_ERROR: 'Unknown error occured'
}


class rsmi_init_flags_t(c_int):
    RSMI_INIT_FLAG_ALL_GPUS = 0x1


class rsmi_dev_perf_level_t(c_int):
    RSMI_DEV_PERF_LEVEL_AUTO = 0
    RSMI_DEV_PERF_LEVEL_FIRST = RSMI_DEV_PERF_LEVEL_AUTO
    RSMI_DEV_PERF_LEVEL_LOW = 1
    RSMI_DEV_PERF_LEVEL_HIGH = 2
    RSMI_DEV_PERF_LEVEL_MANUAL = 3
    RSMI_DEV_PERF_LEVEL_STABLE_STD = 4
    RSMI_DEV_PERF_LEVEL_STABLE_PEAK = 5
    RSMI_DEV_PERF_LEVEL_STABLE_MIN_MCLK = 6
    RSMI_DEV_PERF_LEVEL_STABLE_MIN_SCLK = 7
    RSMI_DEV_PERF_LEVEL_DETERMINISM = 8
    RSMI_DEV_PERF_LEVEL_LAST = RSMI_DEV_PERF_LEVEL_DETERMINISM
    RSMI_DEV_PERF_LEVEL_UNKNOWN = 0x100


notification_type_names = ['VM_FAULT', 'THERMAL_THROTTLE', 'GPU_PRE_RESET', 'GPU_POST_RESET', 'RING_HANG']


class rsmi_evt_notification_type_t(c_int):
    RSMI_EVT_NOTIF_NONE = 0
    RSMI_EVT_NOTIF_VMFAULT = 1
    RSMI_EVT_NOTIF_FIRST = RSMI_EVT_NOTIF_VMFAULT
    RSMI_EVT_NOTIF_THERMAL_THROTTLE = 2
    RSMI_EVT_NOTIF_GPU_PRE_RESET = 3
    RSMI_EVT_NOTIF_GPU_POST_RESET = 4
    RSMI_EVT_NOTIF_RING_HANG = 5
    RSMI_EVT_NOTIF_LAST = RSMI_EVT_NOTIF_RING_HANG


class rsmi_voltage_metric_t(c_int):
    RSMI_VOLT_CURRENT = 0
    RSMI_VOLT_FIRST = RSMI_VOLT_CURRENT
    RSMI_VOLT_MAX = 1
    RSMI_VOLT_MIN_CRIT = 2
    RSMI_VOLT_MIN = 3
    RSMI_VOLT_MAX_CRIT = 4
    RSMI_VOLT_AVERAGE = 5
    RSMI_VOLT_LOWEST = 6
    RSMI_VOLT_HIGHEST = 7
    RSMI_VOLT_LAST = RSMI_VOLT_HIGHEST
    RSMI_VOLT_UNKNOWN = 0x100


class rsmi_voltage_type_t(c_int):
    RSMI_VOLT_TYPE_FIRST = 0
    RSMI_VOLT_TYPE_VDDGFX = RSMI_VOLT_TYPE_FIRST
    RSMI_VOLT_TYPE_LAST = RSMI_VOLT_TYPE_VDDGFX
    RSMI_VOLT_TYPE_INVALID = 0xFFFFFFFF


# The perf_level_string is correlated to rsmi_dev_perf_level_t
def perf_level_string(i):
    switcher = {
        0:  'AUTO',
        1:  'LOW',
        2:  'HIGH',
        3:  'MANUAL',
        4:  'STABLE_STD',
        5:  'STABLE_PEAK',
        6:  'STABLE_MIN_MCLK',
        7:  'STABLE_MIN_SCLK',
        8:  'PERF_DETERMINISM',
    }
    return switcher.get(i, 'UNKNOWN')


rsmi_dev_perf_level = rsmi_dev_perf_level_t


class rsmi_sw_component_t(c_int):
    RSMI_SW_COMP_FIRST = 0x0
    RSMI_SW_COMP_DRIVER = RSMI_SW_COMP_FIRST
    RSMI_SW_COMP_LAST = RSMI_SW_COMP_DRIVER



rsmi_event_handle_t = POINTER(c_uint)


class rsmi_event_group_t(Enum):
    RSMI_EVNT_GRP_XGMI = 0
    RSMI_EVNT_GRP_XGMI_DATA_OUT =  10
    RSMI_EVNT_GRP_INVALID = 0xFFFFFFFF


class rsmi_event_type_t(c_int):
    RSMI_EVNT_FIRST = rsmi_event_group_t.RSMI_EVNT_GRP_XGMI
    RSMI_EVNT_XGMI_FIRST = rsmi_event_group_t.RSMI_EVNT_GRP_XGMI
    RSMI_EVNT_XGMI_0_NOP_TX = RSMI_EVNT_XGMI_FIRST
    RSMI_EVNT_XGMI_0_REQUEST_TX = 1
    RSMI_EVNT_XGMI_0_RESPONSE_TX = 2
    RSMI_EVNT_XGMI_0_BEATS_TX = 3
    RSMI_EVNT_XGMI_1_NOP_TX = 4
    RSMI_EVNT_XGMI_1_REQUEST_TX = 5
    RSMI_EVNT_XGMI_1_RESPONSE_TX = 6
    RSMI_EVNT_XGMI_1_BEATS_TX = 7
    RSMI_EVNT_XGMI_LAST = RSMI_EVNT_XGMI_1_BEATS_TX

    RSMI_EVNT_XGMI_DATA_OUT_FIRST = rsmi_event_group_t.RSMI_EVNT_GRP_XGMI_DATA_OUT
    RSMI_EVNT_XGMI_DATA_OUT_0 = RSMI_EVNT_XGMI_DATA_OUT_FIRST
    RSMI_EVNT_XGMI_DATA_OUT_1 = 11
    RSMI_EVNT_XGMI_DATA_OUT_2 = 12
    RSMI_EVNT_XGMI_DATA_OUT_3 = 13
    RSMI_EVNT_XGMI_DATA_OUT_4 = 14
    RSMI_EVNT_XGMI_DATA_OUT_5 = 15
    RSMI_EVNT_XGMI_DATA_OUT_LAST = RSMI_EVNT_XGMI_DATA_OUT_5

    RSMI_EVNT_LAST = RSMI_EVNT_XGMI_DATA_OUT_LAST,


class rsmi_counter_command_t(c_int):
    RSMI_CNTR_CMD_START = 0
    RSMI_CNTR_CMD_STOP = 1


class rsmi_counter_value_t(Structure):
    _fields_ = [('value', c_uint64),
                ('time_enabled', c_uint64),
                ('time_running', c_uint64)]


class rsmi_clk_type_t(c_int):
    RSMI_CLK_TYPE_SYS = 0x0
    RSMI_CLK_TYPE_FIRST = RSMI_CLK_TYPE_SYS
    RSMI_CLK_TYPE_DF = 0x1
    RSMI_CLK_TYPE_DCEF = 0x2
    RSMI_CLK_TYPE_SOC = 0x3
    RSMI_CLK_TYPE_MEM = 0x4
    RSMI_CLK_TYPE_LAST = RSMI_CLK_TYPE_MEM
    RSMI_CLK_INVALID = 0xFFFFFFFF


# Clock names here are correlated to the rsmi_clk_type_t values above
clk_type_names = ['sclk', 'sclk', 'fclk', 'dcefclk',\
                  'socclk', 'mclk', 'mclk', 'invalid']
rsmi_clk_type_dict = {'RSMI_CLK_TYPE_SYS': 0x0, 'RSMI_CLK_TYPE_FIRST': 0x0,\
                      'RSMI_CLK_TYPE_DF': 0x1, 'RSMI_CLK_TYPE_DCEF': 0x2,\
                      'RSMI_CLK_TYPE_SOC': 0x3, 'RSMI_CLK_TYPE_MEM': 0x4,\
                      'RSMI_CLK_TYPE_LAST': 0X4, 'RSMI_CLK_INVALID': 0xFFFFFFFF}
rsmi_clk_names_dict = {'sclk': 0x0, 'fclk': 0x1, 'dcefclk': 0x2,\
                       'socclk': 0x3, 'mclk': 0x4}
rsmi_clk_type = rsmi_clk_type_t


class rsmi_temperature_metric_t(c_int):
    RSMI_TEMP_CURRENT = 0x0
    RSMI_TEMP_FIRST = RSMI_TEMP_CURRENT
    RSMI_TEMP_MAX = 0x1
    RSMI_TEMP_MIN = 0x2
    RSMI_TEMP_MAX_HYST = 0x3
    RSMI_TEMP_MIN_HYST = 0x4
    RSMI_TEMP_CRITICAL = 0x5
    RSMI_TEMP_CRITICAL_HYST = 0x6
    RSMI_TEMP_EMERGENCY = 0x7
    RSMI_TEMP_EMERGENCY_HYST = 0x8
    RSMI_TEMP_CRIT_MIN = 0x9
    RSMI_TEMP_CRIT_MIN_HYST = 0xA
    RSMI_TEMP_OFFSET = 0xB
    RSMI_TEMP_LOWEST = 0xC
    RSMI_TEMP_HIGHEST = 0xD
    RSMI_TEMP_LAST = RSMI_TEMP_HIGHEST


rsmi_temperature_metric = rsmi_temperature_metric_t


class rsmi_temperature_type_t(c_int):
    RSMI_TEMP_TYPE_FIRST = 0
    RSMI_TEMP_TYPE_EDGE = RSMI_TEMP_TYPE_FIRST
    RSMI_TEMP_TYPE_JUNCTION = 1
    RSMI_TEMP_TYPE_MEMORY = 2
    RSMI_TEMP_TYPE_HBM_0 = 3
    RSMI_TEMP_TYPE_HBM_1 = 4
    RSMI_TEMP_TYPE_HBM_2 = 5
    RSMI_TEMP_TYPE_HBM_3 = 6
    RSMI_TEMP_TYPE_LAST = RSMI_TEMP_TYPE_HBM_3


# temp_type_lst list correlates to rsmi_temperature_type_t
temp_type_lst = ['edge', 'junction', 'memory', 'HBM 0', 'HBM 1', 'HBM 2', 'HBM 3']


class rsmi_power_profile_preset_masks_t(c_uint64):
    RSMI_PWR_PROF_PRST_CUSTOM_MASK = 0x1
    RSMI_PWR_PROF_PRST_VIDEO_MASK = 0x2
    RSMI_PWR_PROF_PRST_POWER_SAVING_MASK = 0x4
    RSMI_PWR_PROF_PRST_COMPUTE_MASK = 0x8
    RSMI_PWR_PROF_PRST_VR_MASK = 0x10
    RSMI_PWR_PROF_PRST_3D_FULL_SCR_MASK = 0x20
    RSMI_PWR_PROF_PRST_BOOTUP_DEFAULT = 0x40
    RSMI_PWR_PROF_PRST_LAST = RSMI_PWR_PROF_PRST_BOOTUP_DEFAULT
    RSMI_PWR_PROF_PRST_INVALID = 0xFFFFFFFFFFFFFFFF


rsmi_power_profile_preset_masks = rsmi_power_profile_preset_masks_t


class rsmi_gpu_block_t(c_int):
    RSMI_GPU_BLOCK_INVALID = 0x0000000000000000
    RSMI_GPU_BLOCK_FIRST = 0x0000000000000001
    RSMI_GPU_BLOCK_UMC = RSMI_GPU_BLOCK_FIRST
    RSMI_GPU_BLOCK_SDMA = 0x0000000000000002
    RSMI_GPU_BLOCK_GFX = 0x0000000000000004
    RSMI_GPU_BLOCK_MMHUB = 0x0000000000000008
    RSMI_GPU_BLOCK_ATHUB = 0x0000000000000010
    RSMI_GPU_BLOCK_PCIE_BIF = 0x0000000000000020
    RSMI_GPU_BLOCK_HDP = 0x0000000000000040
    RSMI_GPU_BLOCK_XGMI_WAFL = 0x0000000000000080
    RSMI_GPU_BLOCK_DF = 0x0000000000000100
    RSMI_GPU_BLOCK_SMN = 0x0000000000000200
    RSMI_GPU_BLOCK_SEM = 0x0000000000000400
    RSMI_GPU_BLOCK_MP0 = 0x0000000000000800
    RSMI_GPU_BLOCK_MP1 = 0x0000000000001000
    RSMI_GPU_BLOCK_FUSE = 0x0000000000002000
    RSMI_GPU_BLOCK_LAST = RSMI_GPU_BLOCK_FUSE
    RSMI_GPU_BLOCK_RESERVED = 0x8000000000000000


rsmi_gpu_block = rsmi_gpu_block_t


# The following dictionary correlates with rsmi_gpu_block_t enum
rsmi_gpu_block_d = {
    'UMC' :  0x0000000000000001,
    'SDMA' : 0x0000000000000002,
    'GFX' : 0x0000000000000004,
    'MMHUB': 0x0000000000000008,
    'ATHUB': 0x0000000000000010,
    'PCIE_BIF': 0x0000000000000020,
    'HDP': 0x0000000000000040,
    'XGMI_WAFL': 0x0000000000000080,
    'DF': 0x0000000000000100,
    'SMN': 0x0000000000000200,
    'SEM': 0x0000000000000400,
    'MP0': 0x0000000000000800,
    'MP1': 0x0000000000001000,
    'FUSE': 0x0000000000002000
    }


class rsmi_ras_err_state_t(c_int):
    RSMI_RAS_ERR_STATE_NONE = 0
    RSMI_RAS_ERR_STATE_DISABLED = 1
    RSMI_RAS_ERR_STATE_PARITY = 2
    RSMI_RAS_ERR_STATE_SING_C = 3
    RSMI_RAS_ERR_STATE_MULT_UC = 4
    RSMI_RAS_ERR_STATE_POISON = 5
    RSMI_RAS_ERR_STATE_ENABLED = 6
    RSMI_RAS_ERR_STATE_LAST = RSMI_RAS_ERR_STATE_ENABLED
    RSMI_RAS_ERR_STATE_INVALID = 0xFFFFFFFF


# Error type list correlates to rsmi_ras_err_state_t
rsmi_ras_err_stale_readable = ['no errors', 'ECC disabled',
                               'unknown type err', 'single correctable err',
                               'multiple uncorrectable err',
                               'page isolated, treat as uncorrectable err',
                               'ECC enabled', 'status invalid']
rsmi_ras_err_stale_machine = ['none', 'disabled', 'unknown error',
                              'sing', 'mult', 'position', 'enabled']

validRasTypes = ['ue', 'ce']

validRasActions = ['disable', 'enable', 'inject']

validRasBlocks = ['fuse', 'mp1', 'mp0', 'sem', 'smn', 'df', 'xgmi_wafl', 'hdp', 'pcie_bif',

                  'athub', 'mmhub', 'gfx', 'sdma', 'umc']


class rsmi_memory_type_t(c_int):
    RSMI_MEM_TYPE_FIRST = 0
    RSMI_MEM_TYPE_VRAM = RSMI_MEM_TYPE_FIRST
    RSMI_MEM_TYPE_VIS_VRAM = 1
    RSMI_MEM_TYPE_GTT = 2
    RSMI_MEM_TYPE_LAST = RSMI_MEM_TYPE_GTT


# memory_type_l includes names for with rsmi_memory_type_t
# Usage example to get corresponding names:
# memory_type_l[rsmi_memory_type_t.RSMI_MEM_TYPE_VRAM] will return string 'vram'
memory_type_l = ['VRAM', 'VIS_VRAM', 'GTT']


class rsmi_freq_ind_t(c_int):
    RSMI_FREQ_IND_MIN = 0
    RSMI_FREQ_IND_MAX = 1
    RSMI_FREQ_IND_INVALID = 0xFFFFFFFF


rsmi_freq_ind = rsmi_freq_ind_t


class rsmi_fw_block_t(c_int):
    RSMI_FW_BLOCK_FIRST = 0
    RSMI_FW_BLOCK_ASD = RSMI_FW_BLOCK_FIRST
    RSMI_FW_BLOCK_CE = 1
    RSMI_FW_BLOCK_DMCU = 2
    RSMI_FW_BLOCK_MC = 3
    RSMI_FW_BLOCK_ME = 4
    RSMI_FW_BLOCK_MEC = 5
    RSMI_FW_BLOCK_MEC2 = 6
    RSMI_FW_BLOCK_MES = 7
    RSMI_FW_BLOCK_MES_KIQ = 8
    RSMI_FW_BLOCK_PFP = 9
    RSMI_FW_BLOCK_RLC = 10
    RSMI_FW_BLOCK_RLC_SRLC = 11
    RSMI_FW_BLOCK_RLC_SRLG = 12
    RSMI_FW_BLOCK_RLC_SRLS = 13
    RSMI_FW_BLOCK_SDMA = 14
    RSMI_FW_BLOCK_SDMA2 = 15
    RSMI_FW_BLOCK_SMC = 16
    RSMI_FW_BLOCK_SOS = 17
    RSMI_FW_BLOCK_TA_RAS = 18
    RSMI_FW_BLOCK_TA_XGMI = 19
    RSMI_FW_BLOCK_UVD = 20
    RSMI_FW_BLOCK_VCE = 21
    RSMI_FW_BLOCK_VCN = 22
    RSMI_FW_BLOCK_LAST = RSMI_FW_BLOCK_VCN


# The following list correlated to the rsmi_fw_block_t
fw_block_names_l = ['ASD', 'CE', 'DMCU', 'MC', 'ME', 'MEC', 'MEC2', 'MES', 'MES KIQ', 'PFP',\
                    'RLC', 'RLC SRLC', 'RLC SRLG', 'RLC SRLS', 'SDMA', 'SDMA2',\
                    'SMC', 'SOS', 'TA RAS', 'TA XGMI', 'UVD', 'VCE', 'VCN']


rsmi_bit_field_t = c_uint64()
rsmi_bit_field = rsmi_bit_field_t

class rsmi_utilization_counter_type(c_int):
    RSMI_UTILIZATION_COUNTER_FIRST = 0
    RSMI_COARSE_GRAIN_GFX_ACTIVITY  = RSMI_UTILIZATION_COUNTER_FIRST
    RSMI_COARSE_GRAIN_MEM_ACTIVITY = 1
    RSMI_UTILIZATION_COUNTER_LAST = RSMI_COARSE_GRAIN_MEM_ACTIVITY

utilization_counter_name = ['GFX Activity', 'Memory Activity']

class rsmi_utilization_counter_t(Structure):
    _fields_ = [('type', c_int),
                ('val', c_uint64)]


class rsmi_xgmi_status_t(c_int):
    RSMI_XGMI_STATUS_NO_ERRORS = 0
    RSMI_XGMI_STATUS_ERROR = 1
    RSMI_XGMI_STATUS_MULTIPLE_ERRORS = 2


class rsmi_memory_page_status_t(c_int):
    RSMI_MEM_PAGE_STATUS_RESERVED = 0
    RSMI_MEM_PAGE_STATUS_PENDING = 1
    RSMI_MEM_PAGE_STATUS_UNRESERVABLE = 2


memory_page_status_l = ['reserved', 'pending', 'unreservable']


class rsmi_retired_page_record_t(Structure):
    _fields_ = [('page_address', c_uint64),
                ('page_size', c_uint64),
                ('status', c_int)]


RSMI_MAX_NUM_POWER_PROFILES = (sizeof(rsmi_bit_field_t) * 8)


class rsmi_power_profile_status_t(Structure):
    _fields_ = [('available_profiles', c_uint32),
                ('current', c_uint64),
                ('num_profiles', c_uint32)]


rsmi_power_profile_status = rsmi_power_profile_status_t


class rsmi_frequencies_t(Structure):
    _fields_ = [('has_deep_sleep', c_bool),
                ('num_supported', c_int32),
                ('current', c_uint32),
                ('frequency', c_uint64 * RSMI_MAX_NUM_FREQUENCIES)]


rsmi_frequencies = rsmi_frequencies_t


class rsmi_pcie_bandwidth_t(Structure):
    _fields_ = [('transfer_rate', rsmi_frequencies_t),
                ('lanes', c_uint32 * RSMI_MAX_NUM_FREQUENCIES)]


rsmi_pcie_bandwidth = rsmi_pcie_bandwidth_t


class rsmi_version_t(Structure):
    _fields_ = [('major', c_uint32),
                ('minor', c_uint32),
                ('patch', c_uint32),
                ('build', c_char_p)]


rsmi_version = rsmi_version_t


class rsmi_range_t(Structure):
    _fields_ = [('lower_bound', c_uint64),
                ('upper_bound', c_uint64)]


rsmi_range = rsmi_range_t


class rsmi_od_vddc_point_t(Structure):
    _fields_ = [('frequency', c_uint64),
                ('voltage', c_uint64)]


rsmi_od_vddc_point = rsmi_od_vddc_point_t


class rsmi_freq_volt_region_t(Structure):
    _fields_ = [('freq_range', rsmi_range_t),
                ('volt_range', rsmi_range_t)]


rsmi_freq_volt_region = rsmi_freq_volt_region_t


class rsmi_od_volt_curve_t(Structure):
    _fields_ = [('vc_points', rsmi_od_vddc_point_t *\
                RSMI_NUM_VOLTAGE_CURVE_POINTS)]


rsmi_od_volt_curve = rsmi_od_volt_curve_t


class rsmi_od_volt_freq_data_t(Structure):
    _fields_ = [('curr_sclk_range', rsmi_range_t),
                ('curr_mclk_range', rsmi_range_t),
                ('sclk_freq_limits', rsmi_range_t),
                ('mclk_freq_limits', rsmi_range_t),
                ('curve', rsmi_od_volt_curve_t),
                ('num_regions', c_uint32)]


rsmi_od_volt_freq_data = rsmi_od_volt_freq_data_t


class rsmi_error_count_t(Structure):
    _fields_ = [('correctable_err', c_uint64),
                ('uncorrectable_err', c_uint64)]


class rsmi_evt_notification_data_t(Structure):
    _fields_ = [('dv_ind', c_uint32),
                ('event', rsmi_evt_notification_type_t),
                ('message', c_char*64)]


class rsmi_process_info_t(Structure):
    _fields_ = [('process_id', c_uint32),
                ('pasid', c_uint32),
                ('vram_usage', c_uint64),
                ('sdma_usage', c_uint64),
                ('cu_occupancy', c_uint32)]


class rsmi_func_id_iter_handle(Structure):
    _fields_ = [('func_id_iter', POINTER(c_uint)),
                ('container_ptr', POINTER(c_uint)),
                ('id_type', c_uint32)]


rsmi_func_id_iter_handle_t = POINTER(rsmi_func_id_iter_handle)


RSMI_DEFAULT_VARIANT = 0xFFFFFFFFFFFFFFFF


class submodule_union(Union):
    _fields_ = [('memory_type', c_int),      #    rsmi_memory_type_t,
                ('temp_metric', c_int),      #    rsmi_temperature_metric_t,
                ('evnt_type', c_int),        #    rsmi_event_type_t,
                ('evnt_group', c_int),       #    rsmi_event_group_t,
                ('clk_type', c_int),         #    rsmi_clk_type_t,
                ('fw_block', c_int),         #    rsmi_fw_block_t,
                ('gpu_block_type', c_int)]   #    rsmi_gpu_block_t


class rsmi_func_id_value_t(Union):
    _fields_ = [('id', c_uint64),
                ('name', c_char_p),
                ('submodule', submodule_union)]

class rsmi_compute_partition_type_t(c_int):
    RSMI_COMPUTE_PARTITION_INVALID = 0
    RSMI_COMPUTE_PARTITION_SPX = 1
    RSMI_COMPUTE_PARTITION_DPX = 2
    RSMI_COMPUTE_PARTITION_TPX = 3
    RSMI_COMPUTE_PARTITION_QPX = 4
    RSMI_COMPUTE_PARTITION_CPX = 5

rsmi_compute_partition_type_dict = {
    #'RSMI_COMPUTE_PARTITION_INVALID': 0,
    'SPX': 1,
    'DPX': 2,
    'TPX': 3,
    'QPX': 4,
    'CPX': 5,
}

rsmi_compute_partition_type = rsmi_compute_partition_type_t

# compute_partition_type_l includes string names for the rsmi_compute_partition_type_t
# Usage example to get corresponding names:
# compute_partition_type_l[rsmi_compute_partition_type_t.RSMI_COMPUTE_PARTITION_CPX]
# will return string 'CPX'
compute_partition_type_l = ['SPX', 'DPX', 'TPX', 'QPX', 'CPX']

class rsmi_memory_partition_type_t(c_int):
    RSMI_MEMORY_PARTITION_UNKNOWN = 0
    RSMI_MEMORY_PARTITION_NPS1 = 1
    RSMI_MEMORY_PARTITION_NPS2 = 2
    RSMI_MEMORY_PARTITION_NPS4 = 3
    RSMI_MEMORY_PARTITION_NPS8 = 4

rsmi_memory_partition_type_dict = {
    'NPS1': 1,
    'NPS2': 2,
    'NPS4': 3,
    'NPS8': 4
}

rsmi_memory_partition_type = rsmi_memory_partition_type_t

# memory_partition_type_l includes string names for the rsmi_compute_partition_type_t
# Usage example to get corresponding names:
# memory_partition_type_l[rsmi_memory_partition_type_t.RSMI_MEMORY_PARTITION_NPS2]
# will return string 'NPS2'
memory_partition_type_l = ['NPS1', 'NPS2', 'NPS4', 'NPS8']

class rsmi_power_label(str, Enum):
    AVG_POWER = '(Avg)'
    CURRENT_SOCKET_POWER = '(Socket)'

class rsmi_power_type_t(c_int):
  RSMI_AVERAGE_POWER = 0,
  RSMI_CURRENT_POWER = 1,
  RSMI_INVALID_POWER = 0xFFFFFFFF

rsmi_power_type_dict = {
    0: 'AVERAGE',
    1: 'CURRENT SOCKET',
    0xFFFFFFFF: 'INVALID_POWER_TYPE'
}

class metrics_table_header_t(Structure):
    pass

# metrics_table_header_t._pack_ = 1 # source:False
metrics_table_header_t._fields_ = [
    ('structure_size', c_uint16),
    ('format_revision', c_uint8),
    ('content_revision', c_uint8),
]
amd_metrics_table_header_t = metrics_table_header_t

class amdgpu_xcp_metrics_t(Structure):
    pass

# amdgpu_xcp_metrics_t._pack_ = 1 # source:False
amdgpu_xcp_metrics_t._fields_ = [
    ('gfx_busy_inst', c_uint32 * 8),
    ('jpeg_busy', c_uint16 * 40),
    ('vcn_busy', c_uint16 * 4),
    ('gfx_busy_acc', c_uint64 * 8),
    ('gfx_below_host_limit_acc', c_uint64 * 8),
    ('gfx_below_host_limit_ppt_acc', c_uint64 * 8),
    ('gfx_below_host_limit_thm_acc', c_uint64 * 8),
    ('gfx_low_utilization_acc', c_uint64 * 8),
    ('gfx_below_host_limit_total_acc', c_uint64 * 8),
]
xcp_stats_t = amdgpu_xcp_metrics_t

class rsmi_gpu_metrics_t(Structure):
    pass


# rsmi_gpu_metrics_t._pack_ = 1  # source:False
rsmi_gpu_metrics_t._fields_ = [
    ('common_header', amd_metrics_table_header_t),
    ('temperature_edge', c_uint16),
    ('temperature_hotspot', c_uint16),
    ('temperature_mem', c_uint16),
    ('temperature_vrgfx', c_uint16),
    ('temperature_vrsoc', c_uint16),
    ('temperature_vrmem', c_uint16),
    ('average_gfx_activity', c_uint16),
    ('average_umc_activity', c_uint16),
    ('average_mm_activity', c_uint16),
    ('average_socket_power', c_uint16),
    ('energy_accumulator', c_uint64),
    ('system_clock_counter', c_uint64),
    ('average_gfxclk_frequency', c_uint16),
    ('average_socclk_frequency', c_uint16),
    ('average_uclk_frequency', c_uint16),
    ('average_vclk0_frequency', c_uint16),
    ('average_dclk0_frequency', c_uint16),
    ('average_vclk1_frequency', c_uint16),
    ('average_dclk1_frequency', c_uint16),
    ('current_gfxclk', c_uint16),
    ('current_socclk', c_uint16),
    ('current_uclk', c_uint16),
    ('current_vclk0', c_uint16),
    ('current_dclk0', c_uint16),
    ('current_vclk1', c_uint16),
    ('current_dclk1', c_uint16),
    ('throttle_status', c_uint32),
    ('current_fan_speed', c_uint16),
    ('pcie_link_width', c_uint16),
    ('pcie_link_speed', c_uint16),
    ('gfx_activity_acc', c_uint32),
    ('mem_activity_acc', c_uint32),
    ('temperature_hbm', c_uint16 * 4),
    ('firmware_timestamp', c_uint64),
    ('voltage_soc', c_uint16),
    ('voltage_gfx', c_uint16),
    ('voltage_mem', c_uint16),
    ('indep_throttle_status', c_uint64),
    ('current_socket_power', c_uint16),
    ('vcn_activity', c_uint16 * 4),
    ('gfxclk_lock_status', c_uint32),
    ('xgmi_link_width', c_uint16),
    ('xgmi_link_speed', c_uint16),
    ('pcie_bandwidth_acc', c_uint64),
    ('pcie_bandwidth_inst', c_uint64),
    ('pcie_l0_to_recov_count_acc', c_uint64),
    ('pcie_replay_count_acc', c_uint64),
    ('pcie_replay_rover_count_acc', c_uint64),
    ('xgmi_read_data_acc', c_uint64 * 8),
    ('xgmi_write_data_acc', c_uint64 * 8),
    ('current_gfxclks', c_uint16 * 8),
    ('current_socclks', c_uint16 * 4),
    ('current_vclk0s', c_uint16 * 4),
    ('current_dclk0s', c_uint16 * 4),
    ('jpeg_activity', c_uint16 * 32),
    ('pcie_nak_sent_count_acc', c_uint32),
    ('pcie_nak_rcvd_count_acc', c_uint32),
    ('accumulation_counter', c_uint64),
    ('prochot_residency_acc', c_uint64),
    ('ppt_residency_acc', c_uint64),
    ('socket_thm_residency_acc', c_uint64),
    ('vr_thm_residency_acc', c_uint64),
    ('hbm_thm_residency_acc', c_uint64),
    ('num_partition', c_uint16),
    ('xcp_stats', xcp_stats_t * 8),
    ('pcie_lc_perf_other_end_recovery', c_uint32),
    ('vram_max_bandwidth', c_uint64),
    ('xgmi_link_status', c_uint16 * 8),
]
amdsmi_gpu_metrics_t = rsmi_gpu_metrics_t
