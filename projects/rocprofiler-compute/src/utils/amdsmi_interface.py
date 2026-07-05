# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import functools
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Callable

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


def _per_device_query(
    fn: Callable,
    *,
    default_return: object,
    warning_label: str,
) -> Callable:
    """
    Try fn(device, amdsmi) on each device; return first success.
    On all-failure, warn with warning_label and return default_return.
    """

    @functools.wraps(fn)
    def wrapper() -> object:
        amdsmi = import_amdsmi_module()
        error = None
        for device in get_device_handles():
            try:
                return fn(device, amdsmi)
            except Exception as e:
                error = e
        console_warning(f"Error getting {warning_label}: {error}")
        return default_return

    return wrapper


@functools.partial(
    _per_device_query, default_return=0.0, warning_label="max memory clock"
)
def get_mem_max_clock(device: Any, amdsmi: Any) -> float:  # noqa: ANN401
    """Get the maximum memory clock of the device."""
    return amdsmi.amdsmi_get_clock_info(device, amdsmi.AmdSmiClkType.MEM)["max_clk"]


@functools.partial(
    _per_device_query,
    default_return=("N/A", "N/A", "N/A"),
    warning_label="gpu model info",
)
def get_gpu_model(device: Any, amdsmi: Any) -> tuple[str, str, str]:  # noqa: ANN401
    """Get GPU model related names."""
    gpu_model_info = (
        amdsmi.amdsmi_get_gpu_board_info(device)["product_name"],
        amdsmi.amdsmi_get_gpu_asic_info(device)["market_name"],
        amdsmi.amdsmi_get_gpu_vbios_info(device)["name"],
    )
    console_debug(f"gpu model info: {str(gpu_model_info)}")
    return gpu_model_info


@functools.partial(
    _per_device_query,
    default_return="N/A",
    warning_label="GPU VBIOS part number",
)
def get_gpu_vbios_part_number(device: Any, amdsmi: Any) -> str:  # noqa: ANN401
    """Get the GPU VBIOS part number."""
    vbios_part_number = amdsmi.amdsmi_get_gpu_vbios_info(device)["part_number"]
    console_debug(f"GPU VBIOS Part Number: {vbios_part_number}")
    return vbios_part_number


@functools.partial(
    _per_device_query,
    default_return="N/A",
    warning_label="GPU compute partition",
)
def get_gpu_compute_partition(device: Any, amdsmi: Any) -> str:  # noqa: ANN401
    """Get the GPU compute partition."""
    compute_partition = amdsmi.amdsmi_get_gpu_compute_partition(device)
    console_debug(f"GPU Compute Partition: {compute_partition}")
    return compute_partition


@functools.partial(
    _per_device_query,
    default_return="N/A",
    warning_label="GPU memory partition",
)
def get_gpu_memory_partition(device: Any, amdsmi: Any) -> str:  # noqa: ANN401
    """Get the GPU memory partition."""
    memory_partition = amdsmi.amdsmi_get_gpu_memory_partition(device)
    console_debug(f"GPU Memory Partition: {memory_partition}")
    return memory_partition


@functools.partial(
    _per_device_query,
    default_return="N/A",
    warning_label="AMDGPU driver version",
)
def get_amdgpu_driver_version(device: Any, amdsmi: Any) -> str:  # noqa: ANN401
    """Get the AMDGPU driver version."""
    driver_version = amdsmi.amdsmi_get_gpu_driver_info(device)["driver_version"]
    console_debug(f"AMDGPU Driver Version: {driver_version}")
    return driver_version


@functools.partial(_per_device_query, default_return="0", warning_label="GPU VRAM size")
def get_gpu_vram_size(device: Any, amdsmi: Any) -> str:  # noqa: ANN401
    """Get the GPU VRAM size in KB."""
    vram_info = amdsmi.amdsmi_get_gpu_vram_info(device)
    vram_size = str(int(vram_info["vram_size"]) * 1024)  # MB -> KB
    console_debug(f"GPU VRAM Size: {vram_size} KB")
    return vram_size


@functools.partial(
    _per_device_query, default_return=None, warning_label="GPU cache info"
)
def get_gpu_cache_info(device: Any, amdsmi: Any) -> dict[str, Any]:  # noqa: ANN401
    """Get the GPU cache level information."""
    cache_info = amdsmi.amdsmi_get_gpu_cache_info(device)
    console_debug(f"GPU Cache Info: {cache_info}")
    return cache_info


@functools.partial(
    _per_device_query,
    default_return=0,
    warning_label="GPU compute unit count",
)
def get_gpu_num_compute_units(device: Any, amdsmi: Any) -> int:  # noqa: ANN401
    """Get the GPU's number of compute units."""
    cu_count = int(amdsmi.amdsmi_get_gpu_asic_info(device)["num_compute_units"])
    console_debug(f"GPU compute units count: {cu_count}")
    return cu_count
