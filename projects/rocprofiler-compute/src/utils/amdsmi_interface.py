##############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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
from collections.abc import Iterator
from contextlib import contextmanager

from utils.logger import (
    console_debug,
    console_error,
    console_warning,
)

_amdsmi_module = None


# Ignore undefined name amdsmi since it's dynamically imported
def import_amdsmi_module() -> "amdsmi":  # noqa: F821
    """
    Dynamically import the amdsmi module because we only
    want profile time dependency on amdsmi.
    Uses global cache to avoid repeated imports.
    """
    global _amdsmi_module

    if not _amdsmi_module:
        sys.path.insert(0, os.getenv("ROCM_PATH", "/opt/rocm") + "/share/amd_smi")
        try:
            import amdsmi

            _amdsmi_module = amdsmi
        except ImportError as e:
            console_warning(f"Unhandled import error: {e}")
            console_error("Failed to import the amdsmi Python library.")

    return _amdsmi_module


@contextmanager
def amdsmi_ctx() -> Iterator[None]:
    """Context manager to initialize and shutdown amdsmi."""
    amdsmi = import_amdsmi_module()
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


# Ignore undefined name amdsmi since it's dynamically imported
def get_device_handles() -> "list[amdsmi.ProcessorHandle]":  # noqa: F821
    """
    Get all AMD device handles.
    We query all handles since some handles cannot be
    used as they are hidden by ROCR or HIP environment variables.
    """
    amdsmi = import_amdsmi_module()
    try:
        devices = amdsmi.amdsmi_get_processor_handles()
        if not devices:
            console_warning("No AMD device(s) detected!")
            return []
        console_debug(f"Found {len(devices)} AMD device(s).")
        return devices
    except Exception as e:
        console_warning(f"Error getting device handles: {e}")
        return []


def get_mem_max_clock() -> float:
    """Get the maximum memory clock of the device."""
    amdsmi = import_amdsmi_module()
    error = None
    for device in get_device_handles():
        try:
            return amdsmi.amdsmi_get_clock_info(device, amdsmi.AmdSmiClkType.MEM)[
                "max_clk"
            ]
        except Exception as e:
            error = e
    console_warning(f"Error getting max memory clock: {error}")
    return 0.0


def get_gpu_model() -> tuple[str, str, str]:
    """Get GPU model related names."""
    amdsmi = import_amdsmi_module()
    error = None
    for device in get_device_handles():
        try:
            gpu_model_info = (
                amdsmi.amdsmi_get_gpu_board_info(device)["product_name"],
                amdsmi.amdsmi_get_gpu_asic_info(device)["market_name"],
                amdsmi.amdsmi_get_gpu_vbios_info(device)["name"],
            )
            console_debug(f"gpu model info: {str(gpu_model_info)}")
            return gpu_model_info
        except Exception as e:
            error = e
    console_warning(f"Error getting gpu model info: {error}")
    return ("N/A", "N/A", "N/A")


def get_gpu_vbios_part_number() -> str:
    """Get the GPU VBIOS part number."""
    amdsmi = import_amdsmi_module()
    error = None
    for device in get_device_handles():
        try:
            vbios_part_number = amdsmi.amdsmi_get_gpu_vbios_info(device)["part_number"]
            console_debug(f"GPU VBIOS Part Number: {vbios_part_number}")
            return vbios_part_number
        except Exception as e:
            error = e
    console_warning(f"Error getting GPU VBIOS part number: {error}")
    return "N/A"


def get_gpu_compute_partition() -> str:
    """Get the GPU compute partition."""
    amdsmi = import_amdsmi_module()
    error = None
    for device in get_device_handles():
        try:
            compute_partition = amdsmi.amdsmi_get_gpu_compute_partition(device)
            console_debug(f"GPU Compute Partition: {compute_partition}")
            return compute_partition
        except Exception as e:
            error = e
    console_warning(f"Error getting GPU compute partition: {error}")
    return "N/A"


def get_gpu_memory_partition() -> str:
    """Get the GPU memory partition."""
    amdsmi = import_amdsmi_module()
    error = None
    for device in get_device_handles():
        try:
            memory_partition = amdsmi.amdsmi_get_gpu_memory_partition(device)
            console_debug(f"GPU Memory Partition: {memory_partition}")
            return memory_partition
        except Exception as e:
            error = e
    console_warning(f"Error getting GPU memory partition: {error}")
    return "N/A"


def get_amdgpu_driver_version() -> str:
    """Get the AMDGPU driver version."""
    amdsmi = import_amdsmi_module()
    error = None
    for device in get_device_handles():
        try:
            driver_info = amdsmi.amdsmi_get_gpu_driver_info(device)
            driver_version = driver_info["driver_version"]
            console_debug(f"AMDGPU Driver Version: {driver_version}")
            return driver_version
        except Exception as e:
            error = e
    console_warning(f"Error getting AMDGPU driver version: {error}")
    return "N/A"


def get_gpu_vram_size() -> str:
    """Get the GPU VRAM size in KB."""
    amdsmi = import_amdsmi_module()
    error = None
    for device in get_device_handles():
        try:
            vram_info = amdsmi.amdsmi_get_gpu_vram_info(device)
            vram_size = str(int(vram_info["vram_size"]) * 1024)  # MB -> KB
            console_debug(f"GPU VRAM Size: {vram_size} KB")
            return vram_size
        except Exception as e:
            error = e
    console_warning(f"Error getting GPU VRAM size: {error}")
    return "0"
