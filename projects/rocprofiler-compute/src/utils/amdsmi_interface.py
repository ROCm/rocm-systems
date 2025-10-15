##############################################################################
# MIT License
#
# Copyright (c) 2021 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################

import os
import sys
from contextlib import contextmanager
from typing import Iterator

from utils.logger import (
    console_debug,
    console_warning,
)

PYTHON_LIB_PATH = os.getenv("ROCM_PATH", "/opt/rocm") + "/share/amd_smi"
sys.path.append(PYTHON_LIB_PATH)
# If the python library is installed, it will overwrite the path above

try:
    import amdsmi
except ImportError as e:
    console_warning(f"Unhandled import error: {e}")
    console_warning("Failed to import the amdsmi Python library.")
    sys.exit(1)


@contextmanager
def amdsmi_ctx() -> Iterator[None]:
    """Context manager to initialize and shutdown amdsmi."""
    try:
        amdsmi.amdsmi_init()
        yield
    except Exception as e:
        console_warning(f"amd-smi init failed: {e}")
    finally:
        try:
            amdsmi.amdsmi_shut_down()
        except Exception as e:
            console_warning(f"amd-smi shutdown failed: {e}")


def get_device_handle() -> "amdsmi.ProcessorHandle | None":
    """Get the first AMD device handle."""
    try:
        devices = amdsmi.amdsmi_get_processor_handles()
        if len(devices) == 0:
            console_warning("No AMD GPU detected!")
            return None
        console_debug(f"Found {len(devices)} AMD device(s).")
        return devices[0]
    except Exception as e:
        console_warning(f"Error getting device handle: {e}")
        return None


def get_mem_max_clock() -> float:
    """Get the maximum memory clock of the device."""
    try:
        # Extract max memory clock frequency
        amd_smi_mclk = amdsmi.amdsmi_get_gpu_od_volt_info(get_device_handle())[
            "curr_sclk_range"
        ]["upper_bound"]
        # 100 Mhz -> 100
        return amd_smi_mclk / 10**6
    except Exception as e:
        console_warning(f"Error getting memory clocks: {e}")
        return 0.0


def get_gpu_model() -> str:
    """Get the GPU model name."""
    try:
        board_info = amdsmi.amdsmi_get_gpu_board_info(get_device_handle())
        gpu_model = board_info["product_name"]

        console_debug(f"GPU Model: {gpu_model}")

        return gpu_model
    except Exception as e:
        console_warning(f"Error getting GPU model: {e}")
        return "N/A"


def get_gpu_vbios_part_number() -> str:
    """Get the GPU VBIOS part number."""
    try:
        vbios_part_number = amdsmi.amdsmi_get_gpu_vbios_info(get_device_handle())[
            "part_number"
        ]

        console_debug(f"GPU VBIOS Part Number: {vbios_part_number}")

        return vbios_part_number
    except Exception as e:
        console_warning(f"Error getting GPU VBIOS part number: {e}")
        return "N/A"


def get_gpu_compute_partition() -> str:
    """Get the GPU compute partition."""
    try:
        compute_partition = amdsmi.amdsmi_get_gpu_compute_partition(get_device_handle())

        console_debug(f"GPU Compute Partition: {compute_partition}")

        return compute_partition
    except Exception as e:
        console_warning(f"Error getting GPU compute partition: {e}")
        return "N/A"


def get_gpu_memory_partition() -> str:
    """Get the GPU memory partition."""
    try:
        memory_partition = amdsmi.amdsmi_get_gpu_memory_partition(get_device_handle())

        console_debug(f"GPU Memory Partition: {memory_partition}")

        return memory_partition
    except Exception as e:
        console_warning(f"Error getting GPU memory partition: {e}")
        return "N/A"
