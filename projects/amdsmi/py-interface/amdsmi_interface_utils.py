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

"""Internal utility helpers for amdsmi_interface.

These are private helpers (prefixed with ``_``) shared across the amdsmi
Python interface.  They are **not** part of the public API.
"""

import ctypes
import re
from enum import IntEnum
from typing import Dict, List, Union

from . import amdsmi_wrapper
from .amdsmi_exception import (
    AmdSmiLibraryException,
    AmdSmiParameterException,
    AmdSmiRetryException,
    AmdSmiTimeoutException,
)

# __all__ is required so that `from .amdsmi_interface_utils import *` picks up
# the underscore-prefixed private helpers that amdsmi_interface.py needs.
__all__ = [
    "AMDSMI_MAX_STRING_LENGTH",
    "MaxUIntegerTypes",
    "NO_OF_32BITS",
    "NO_OF_64BITS",
    "_amdsmi_init_enum_flag_is_valid",
    "_check_msb_32",
    "_check_msb_64",
    "_check_res",
    "_format_bad_page_info",
    "_format_bdf",
    "_format_transfer_rate",
    "_get_name_value",
    "_make_amdsmi_bdf_from_list",
    "_NA_amdsmi_get_gpu_metrics_info",
    "_notifyTypeToString",
    "_pad_hex_value",
    "_parse_bdf",
    "_validate_if_max_uint",
]

# Re-export the constant used by _get_name_value so callers don't need a
# separate import.
AMDSMI_MAX_STRING_LENGTH = 256


class MaxUIntegerTypes(IntEnum):
    UINT8_T = 0xFF
    UINT16_T = 0xFFFF
    UINT32_T = 0xFFFFFFFF
    UINT64_T = 0xFFFFFFFFFFFFFFFF


NO_OF_32BITS = ctypes.sizeof(ctypes.c_uint32) * 8
NO_OF_64BITS = ctypes.sizeof(ctypes.c_uint64) * 8


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


def _format_bdf(amdsmi_bdf: Union[amdsmi_wrapper.amdsmi_bdf_t, amdsmi_wrapper.struct_bdf_]) -> str:
    """
    Format BDF struct to readable data.

    Parameters:
        amdsmi_bdf(`amdsmi_bdf_t`): Struct containing BDF data that
        will be formatted.

    Returns:
        `str`: String containing BDF data in a readable format.
    """
    try:
        struct = amdsmi_bdf.bdf
    except AttributeError:
        struct = amdsmi_bdf

    domain = hex(struct.domain_number)[2:].zfill(4)
    bus = hex(struct.bus_number)[2:].zfill(2)
    device = hex(struct.device_number)[2:].zfill(2)
    function = hex(struct.function_number)[2:]
    return domain + ":" + bus + ":" + device + "." + function


def _parse_bdf(bdf):
    if bdf is None:
        return None
    extended_regex = re.compile(r"^([0-9a-fA-F]{4}):([0-9a-fA-F]{2}):([0-1][0-9a-fA-F])\.([0-7])$")
    if extended_regex.match(bdf) is None:
        simple_regex = re.compile(r"^([0-9a-fA-F]{2}):([0-1][0-9a-fA-F])\.([0-7])$")
        if simple_regex.match(bdf) is None:
            return None
        else:
            match = simple_regex.match(bdf)
            if match:
                return [0] + [int(x, 16) for x in match.groups()]
            else:
                return None
    else:
        match = extended_regex.match(bdf)
        if match:
            return [int(x, 16) for x in match.groups()]
        return None


def _make_amdsmi_bdf_from_list(bdf):
    if len(bdf) != 4:
        return None
    amdsmi_bdf = amdsmi_wrapper.amdsmi_bdf_t()
    amdsmi_bdf.bdf.function_number = bdf[3]
    amdsmi_bdf.bdf.device_number = bdf[2]
    amdsmi_bdf.bdf.bus_number = bdf[1]
    amdsmi_bdf.bdf.domain_number = bdf[0]
    return amdsmi_bdf


def _pad_hex_value(value, length) -> str:
    """Pad a hexadecimal value with a given length of zeros

    :param value: A hexadecimal value to be padded with zeros
    :param length: Number of zeros to pad the hexadecimal value
    :param return original string string or
        padded hex of confirmed hex output (using length provided)
    """
    # Ensure value entered meets the minimum length and is hexadecimal
    if (
        len(value) > 2
        and length > 1
        and value[:2].lower() == "0x"
        and all(c in "0123456789abcdefABCDEF" for c in value[2:])
    ):
        # Pad with zeros after '0x' prefix
        return "0x" + value[2:].zfill(length)
    return value


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
            raise AmdSmiParameterException(bad_page, amdsmi_wrapper.amdsmi_retired_page_record_t)

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


def _get_name_value(num, data) -> List[Dict[str, int]]:
    """
    Extracts a list of name-value pairs from a ctypes array buffer.

    This function works around a ctypes array issue where direct field access
    to the `amdsmi_name_value_t` structure is unreliable. Instead, it uses
    memory operations to extract the 'name' (a 64-byte char array) and 'value'
    (a uint64) from each structure in the array.

    Parameters:
        num (ctypes.c_uint32): Number of elements in the array.
        data (ctypes.c_void_p): Pointer to the start of the array buffer containing
            `amdsmi_name_value_t` structures.

    Returns:
        List[Dict[str, int]]: A list of dictionaries, each with keys 'name' (str)
            and 'value' (int) extracted from the buffer.

    Workaround:
        Direct access to the fields of the ctypes array is broken, so the function
        uses memory alignment and pointer arithmetic to extract the fields manually.
    """

    # Work around ctypes array issue by using memory access
    # Use 4 byte alignment for amdsmi_name_value_t.name char array,  64=256/4
    # Use 8 bytes for amdsmi_name_value_t.value uint64
    aligned_name_size = int(AMDSMI_MAX_STRING_LENGTH / 4)
    value_size_bytes = 8
    struct_alignment = aligned_name_size + value_size_bytes

    # Access name,value field using memory operations since direct access is broken
    struct_ptr = ctypes.cast(data, ctypes.POINTER(ctypes.c_char * struct_alignment))

    results = []
    for i in range(num.value):
        # Offset into structure array
        current_struct = struct_ptr[i]

        # Cast address for name member with max chars to read
        name_ptr = ctypes.cast(
            ctypes.addressof(current_struct),
            ctypes.POINTER(ctypes.c_char * AMDSMI_MAX_STRING_LENGTH),
        )
        # Data buffer in bytes
        name_bytes = ctypes.string_at(name_ptr.contents)
        # Get string
        name_str = name_bytes.rstrip(b"\x00").decode("utf-8", errors="replace")

        # Address for value member
        addr_value = ctypes.addressof(current_struct) + struct_alignment
        # Cast data buffer to a uint64
        int64_ptr = ctypes.cast(addr_value, ctypes.POINTER(ctypes.c_uint64))
        # Get value
        value = int64_ptr.contents.value

        item = {"name": name_str, "value": value}
        results.append(item)

    return results


def _format_transfer_rate(transfer_rate):
    return {
        "num_supported": transfer_rate.num_supported,
        "current": transfer_rate.current,
        "frequency": list(transfer_rate.frequency),
    }


def _NA_amdsmi_get_gpu_metrics_info() -> Dict[str, str]:
    """
    Get 'N/A' metric values for gpu_metric, used for exception handling.

    Parameters:
        None

    Returns:
        Dict[str, str]: A dictionary with keys as metric names and values as 'N/A'.
        This is used to indicate that the metric is not available or applicable.

    Raises:
        N/A
    """
    na_gpu_metrics_info = {
        "common_header.structure_size": "N/A",
        "common_header.format_revision": "N/A",
        "common_header.content_revision": "N/A",
        "temperature_edge": "N/A",
        "temperature_hotspot": "N/A",
        "temperature_mem": "N/A",
        "temperature_vrgfx": "N/A",
        "temperature_vrsoc": "N/A",
        "temperature_vrmem": "N/A",
        "average_gfx_activity": "N/A",
        "average_umc_activity": "N/A",
        "average_mm_activity": "N/A",
        "average_socket_power": "N/A",
        "energy_accumulator": "N/A",
        "system_clock_counter": "N/A",
        "average_gfxclk_frequency": "N/A",
        "average_socclk_frequency": "N/A",
        "average_uclk_frequency": "N/A",
        "average_vclk0_frequency": "N/A",
        "average_dclk0_frequency": "N/A",
        "average_vclk1_frequency": "N/A",
        "average_dclk1_frequency": "N/A",
        "current_gfxclk": "N/A",
        "current_socclk": "N/A",
        "current_uclk": "N/A",
        "current_vclk0": "N/A",
        "current_dclk0": "N/A",
        "current_vclk1": "N/A",
        "current_dclk1": "N/A",
        "throttle_status": "N/A",
        "current_fan_speed": "N/A",
        "pcie_link_width": "N/A",
        "pcie_link_speed": "N/A",
        "gfx_activity_acc": "N/A",
        "mem_activity_acc": "N/A",
        "temperature_hbm": "N/A",
        "firmware_timestamp": "N/A",
        "voltage_soc": "N/A",
        "voltage_gfx": "N/A",
        "voltage_mem": "N/A",
        "indep_throttle_status": "N/A",
        "current_socket_power": "N/A",
        "vcn_activity": "N/A",
        "gfxclk_lock_status": "N/A",
        "xgmi_link_width": "N/A",
        "xgmi_link_speed": "N/A",
        "pcie_bandwidth_acc": "N/A",
        "pcie_bandwidth_inst": "N/A",
        "pcie_l0_to_recov_count_acc": "N/A",
        "pcie_replay_count_acc": "N/A",
        "pcie_replay_rover_count_acc": "N/A",
        "xgmi_read_data_acc": "N/A",
        "xgmi_write_data_acc": "N/A",
        "current_gfxclks": "N/A",
        "current_socclks": "N/A",
        "current_vclk0s": "N/A",
        "current_dclk0s": "N/A",
        "jpeg_activity": "N/A",
        "pcie_nak_sent_count_acc": "N/A",
        "pcie_nak_rcvd_count_acc": "N/A",
        "accumulation_counter": "N/A",
        "prochot_residency_acc": "N/A",
        "ppt_residency_acc": "N/A",
        "socket_thm_residency_acc": "N/A",
        "vr_thm_residency_acc": "N/A",
        "hbm_thm_residency_acc": "N/A",
        "num_partition": "N/A",
        "xcp_stats.gfx_busy_inst": "N/A",
        "xcp_stats.jpeg_busy": "N/A",
        "xcp_stats.vcn_busy": "N/A",
        "xcp_stats.gfx_busy_acc": "N/A",
        "xcp_stats.gfx_below_host_limit_acc": "N/A",
        "xcp_stats.gfx_below_host_limit_ppt_acc": "N/A",
        "xcp_stats.gfx_below_host_limit_thm_acc": "N/A",
        "xcp_stats.gfx_low_utilization_acc": "N/A",
        "xcp_stats.gfx_below_host_limit_total_acc": "N/A",
        "pcie_lc_perf_other_end_recovery": "N/A",
        "vram_max_bandwidth": "N/A",
        "xgmi_link_status": "N/A",
        "temperature_hbm_stacks": "N/A",
        "temperature_mid": "N/A",
        "temperature_aid": "N/A",
        "current_uclk_aid": "N/A",
        "current_socclks_mid": "N/A",
        "xcp_stats.temperature_xcd": "N/A",
        "apu_metrics.temperature_gfx": "N/A",
        "apu_metrics.temperature_soc": "N/A",
        "apu_metrics.temperature_core": "N/A",
        "apu_metrics.temperature_l3": "N/A",
        "apu_metrics.temperature_skin": "N/A",
        "apu_metrics.average_gfx_activity": "N/A",
        "apu_metrics.average_mm_activity": "N/A",
        "apu_metrics.average_vcn_activity": "N/A",
        "apu_metrics.average_ipu_activity": "N/A",
        "apu_metrics.average_core_c0_activity": "N/A",
        "apu_metrics.average_dram_reads": "N/A",
        "apu_metrics.average_dram_writes": "N/A",
        "apu_metrics.average_ipu_reads": "N/A",
        "apu_metrics.average_ipu_writes": "N/A",
        "apu_metrics.average_socket_power": "N/A",
        "apu_metrics.average_cpu_power": "N/A",
        "apu_metrics.average_soc_power": "N/A",
        "apu_metrics.average_gfx_power": "N/A",
        "apu_metrics.average_core_power": "N/A",
        "apu_metrics.average_ipu_power": "N/A",
        "apu_metrics.average_apu_power": "N/A",
        "apu_metrics.average_dgpu_power": "N/A",
        "apu_metrics.average_all_core_power": "N/A",
        "apu_metrics.average_sys_power": "N/A",
        "apu_metrics.stapm_power_limit": "N/A",
        "apu_metrics.current_stapm_power_limit": "N/A",
        "apu_metrics.average_gfxclk_frequency": "N/A",
        "apu_metrics.average_socclk_frequency": "N/A",
        "apu_metrics.average_uclk_frequency": "N/A",
        "apu_metrics.average_fclk_frequency": "N/A",
        "apu_metrics.average_vclk_frequency": "N/A",
        "apu_metrics.average_dclk_frequency": "N/A",
        "apu_metrics.average_vpeclk_frequency": "N/A",
        "apu_metrics.average_ipuclk_frequency": "N/A",
        "apu_metrics.average_mpipu_frequency": "N/A",
        "apu_metrics.current_gfxclk": "N/A",
        "apu_metrics.current_socclk": "N/A",
        "apu_metrics.current_uclk": "N/A",
        "apu_metrics.current_fclk": "N/A",
        "apu_metrics.current_vclk": "N/A",
        "apu_metrics.current_dclk": "N/A",
        "apu_metrics.current_coreclk": "N/A",
        "apu_metrics.current_l3clk": "N/A",
        "apu_metrics.current_core_maxfreq": "N/A",
        "apu_metrics.current_gfx_maxfreq": "N/A",
        "apu_metrics.throttle_status": "N/A",
        "apu_metrics.indep_throttle_status": "N/A",
        "apu_metrics.throttle_residency_prochot": "N/A",
        "apu_metrics.throttle_residency_spl": "N/A",
        "apu_metrics.throttle_residency_fppt": "N/A",
        "apu_metrics.throttle_residency_sppt": "N/A",
        "apu_metrics.throttle_residency_thm_core": "N/A",
        "apu_metrics.throttle_residency_thm_gfx": "N/A",
        "apu_metrics.throttle_residency_thm_soc": "N/A",
        "apu_metrics.fan_pwm": "N/A",
        "apu_metrics.average_temperature_gfx": "N/A",
        "apu_metrics.average_temperature_soc": "N/A",
        "apu_metrics.average_temperature_core": "N/A",
        "apu_metrics.average_temperature_l3": "N/A",
        "apu_metrics.average_cpu_voltage": "N/A",
        "apu_metrics.average_soc_voltage": "N/A",
        "apu_metrics.average_gfx_voltage": "N/A",
        "apu_metrics.average_cpu_current": "N/A",
        "apu_metrics.average_soc_current": "N/A",
        "apu_metrics.average_gfx_current": "N/A",
        "apu_metrics.time_filter_alphavalue": "N/A",
        "is_apu": False,
    }
    return na_gpu_metrics_info


def _validate_if_max_uint(
    value, uint_type: MaxUIntegerTypes, isActivity=False, isBool=False
) -> Union[str, bool, int, list]:
    return_val = "N/A"
    if not isinstance(value, list):
        if (value == uint_type) or (isActivity and value > 100):
            return return_val
        if isBool:
            return bool(value)
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
    return return_val


def _notifyTypeToString(notify_type_b):
    # Late import to avoid circular dependency with amdsmi_interface
    from .amdsmi_interface import AmdSmiCperNotifyType

    guid = []
    # Iterate over only the first 8 bytes, but backwards
    for i in notify_type_b[7::-1]:
        guid.append(format(i, "02x"))
    hex_string = "".join(guid)
    hex_value = int(hex_string, 16)
    if hex_value in AmdSmiCperNotifyType._value2member_map_:
        # Convert to the corresponding enum name
        return AmdSmiCperNotifyType(hex_value).name
    else:
        return "Unknown"


# Get 2's complement of 32 bit unsigned integer
def _check_msb_32(num):
    msb = 1 << (NO_OF_32BITS - 1)

    # If msb = 1 , then take 2's complement of the number
    if num & msb:
        num = ~num + 1
    return num


# Get 2's complement of 64 bit unsigned integer
def _check_msb_64(num):
    msb = 1 << (NO_OF_64BITS - 1)

    # If msb = 1 , then take 2's complement of the number
    if num & msb:
        num = ~num + 1
    return num


def _amdsmi_init_enum_flag_is_valid(flag):
    """Validate that flag contains only valid initialization bits."""
    # Late import to avoid circular dependency with amdsmi_interface
    from .amdsmi_interface import AmdSmiInitFlags

    if flag == amdsmi_wrapper.AMDSMI_INIT_ALL_PROCESSORS:
        return True

    # Build mask of all valid flags (excluding ALL_PROCESSORS)
    valid_mask = 0
    for enum_flag in AmdSmiInitFlags:
        if enum_flag != AmdSmiInitFlags.INIT_ALL_PROCESSORS:
            valid_mask |= enum_flag.value
    # Check if flag contains only valid bits and is not zero
    return flag != 0 and (flag & ~valid_mask) == 0
