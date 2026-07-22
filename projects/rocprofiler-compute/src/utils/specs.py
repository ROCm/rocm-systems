# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Get host/gpu specs.

This module provides:
- MachineSpecs dataclass hierarchy (base, CDNA, RDNA) for GPU specifications
- Functions to extract machine, GPU, and SoC information
- generate_machine_specs() as the main entry point for spec generation
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import socket
import subprocess
from dataclasses import dataclass, field, fields
from datetime import datetime
from math import ceil
from pathlib import Path as path
from typing import Any, Optional, TypeVar

import config
from utils import amdsmi_interface
from utils.logger import (
    console_debug,
    console_error,
    console_log,
    console_warning,
    demarcate,
)
from utils.mi_gpu_spec import MIGPUSpecs, mi_gpu_specs
from utils.utils_common import format_table_ascii, get_version

T = TypeVar("T")

VERSION_LOC: list[str] = [
    "version",
    "version-dev",
    "version-hip-libraries",
    "version-hiprt",
    "version-hiprt-devel",
    "version-hip-sdk",
    "version-libs",
    "version-utils",
]


def spec_family_for_arch(gpu_arch: Optional[str]) -> type[MachineSpecs]:
    """Return the MachineSpecs subclass for a GPU arch (by series).

    The "RDNA3.5" series maps to RDNA 3.5; everything else (mi100, mi200,
    mi300, etc.) maps to CDNA.
    """
    series = mi_gpu_specs.get_gpu_series(gpu_arch) if gpu_arch else None
    if series and series.upper() == "RDNA3.5":
        return MachineSpecsRDNA35
    return MachineSpecsCDNA


def generate_machine_specs(
    args: Optional[argparse.Namespace], sysinfo: Optional[dict[str, Any]] = None
) -> MachineSpecs:
    """Generate machine specifications from sysinfo dict or live hardware probing.

    If sysinfo is provided, reconstructs MachineSpecs from the saved data.
    Otherwise, probes the current system for hardware information.
    """
    if sysinfo is not None:
        return _reconstruct_specs_from_sysinfo(sysinfo)

    return _probe_live_machine_specs(args)


def get_rocm_ver() -> str:
    """Detect the installed ROCm version from filesystem or environment."""
    rocm_base_path = path(os.getenv("ROCM_PATH", "/opt/rocm"))

    for version_file_name in VERSION_LOC:
        version_file_path = rocm_base_path / ".info" / version_file_name
        if version_file_path.exists():
            return version_file_path.read_text(encoding="utf-8").strip()

    rocm_ver_user = os.getenv("ROCM_VER")
    if rocm_ver_user:
        console_log(
            "profiling",
            f"Overriding missing ROCm version detection with "
            f"ROCM_VER = {rocm_ver_user}",
        )
        return rocm_ver_user

    console_warning("Unable to detect a complete local ROCm installation.")
    console_warning(
        f"The expected {rocm_base_path}/.info/ versioning directory is missing."
    )
    console_error("Ensure you have valid ROCm installation.", exit=False)
    return ""


def total_sqc(archname: str, num_compute_units: str, num_shader_engines: str) -> int:
    """Calculate total SQC (scalar cache) count for an architecture."""
    cu_per_se = float(num_compute_units) / float(num_shader_engines)
    sq_per_se = cu_per_se / 2.0
    if archname.lower() in ["mi50", "mi100"]:
        sq_per_se = cu_per_se / 3
    sq_per_se = ceil(sq_per_se)
    return int(sq_per_se) * int(num_shader_engines)


def totall2_banks(
    gpu_arch: Optional[str] = None,
    gpu_model: Optional[str] = None,
    L2banks: Optional[str] = None,
    compute_partition: Optional[str] = None,
) -> Optional[str]:
    """Calculate total L2 banks across all XCDs."""
    xcd_count = mi_gpu_specs.get_num_xcds(gpu_arch, gpu_model, compute_partition)

    if L2banks is not None and xcd_count is not None:
        return str(int(L2banks) * int(xcd_count))
    return None


def set_cache_sizes(
    gpu_model: str,
    num_cu: int,
    cache_info: dict,
    num_dies: int,
    num_se: str | int,
    num_sa_se: str | int,
) -> dict[str, int]:
    """Extrapolate cache sizes from AMD-SMI cache info output."""
    if num_cu == 0:
        console_error("Failed to determine GPU compute unit count from AMD-SMI.")
    if not cache_info:
        console_error("Failed to retrieve GPU cache information from AMD-SMI.")

    cache_sizes: dict[str, int] = {}
    memory_levels = mi_gpu_specs.get_memory_levels(gpu_model)

    l1_data_caches = [
        cache_values
        for cache_values in cache_info["cache"]
        if cache_values["cache_level"] == 1
        and "DATA_CACHE" in cache_values["cache_properties"]
    ]
    if l1_data_caches:
        vl1d = max(l1_data_caches, key=lambda cache: cache["num_cache_instance"])
        l1_key = "L0" if "L0" in memory_levels else "L1"
        cache_sizes[l1_key] = int(vl1d["cache_size"]) * 1024

    if "L0" in memory_levels and "L1" in memory_levels:
        if num_sa_se is None:
            console_warning(
                "Shader Arrays per Shader Engine spec is not available; "
                "skipping GL1 cache size detection"
            )
        else:
            sa_level_cache = [
                c
                for c in l1_data_caches
                if c["num_cache_instance"] == int(num_se) * int(num_sa_se)
            ]
            if sa_level_cache:
                cache_sizes["L1"] = int(sa_level_cache[0]["cache_size"]) * 1024

    for cache_values in cache_info["cache"]:
        if cache_values["cache_level"] == 2:
            cache_sizes["L2"] = int(cache_values["cache_size"]) * 1024
        elif cache_values["cache_level"] == 3 and num_dies > 0:
            cache_sizes["MALL"] = int(cache_values["cache_size"]) * 1024 // num_dies

    return cache_sizes


def _kw_only(cls: T) -> T:
    """Enforce keyword-only dataclass initialization (Python 3.9 compatible)."""

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        if args:
            raise TypeError(
                f"{cls.__name__}() takes keyword arguments only, "
                f"got {len(args)} positional argument(s)"
            )
        for name, value in kwargs.items():
            setattr(self, name, value)

    cls.__init__ = __init__  # type: ignore
    return cls


def _run_command(cmd: list[str]) -> Optional[str]:
    """Run a command and return stdout, aborting on execution failures."""
    cmd_str = " ".join(cmd)
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        console_error(f"Required command not found: {cmd_str} ({exc})")
        return None
    except OSError as exc:
        console_error(f"Failed to execute command: {cmd_str} ({exc})")
        return None

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        message = f"Command failed with exit code {completed.returncode}: {cmd_str}"
        if stderr:
            message += f". stderr: {stderr}"
        console_error(message)
        return None

    return completed.stdout


def search_pattern(pattern: str, string: str) -> Optional[str]:
    """Return the first multiline regex capture group, if present."""
    match = re.search(pattern, string, re.MULTILINE)
    if match is not None:
        return match.group(1)
    return None


def _detect_arch(rocminfo_lines: list[str]) -> Optional[tuple[str, int]]:
    """Detect GPU architecture from rocminfo output."""
    supported_gpu_arch = mi_gpu_specs.get_gpu_series_dict()
    unsupported_gpu_arch: set[str] = set()

    for idx, line_text in enumerate(rocminfo_lines):
        gpu_arch = search_pattern(
            r"^\s*Name\s*:\s* ([Gg][Ff][Xx][a-zA-Z0-9]+).*\s*$", line_text
        )
        if not gpu_arch:
            continue

        if gpu_arch in supported_gpu_arch:
            return (gpu_arch, idx)

        if gpu_arch not in unsupported_gpu_arch:
            unsupported_gpu_arch.add(gpu_arch)
            console_warning(
                f"Detected GPU architecture: {gpu_arch} is currently NOT supported "
                "by the profile mode."
            )

    if unsupported_gpu_arch:
        console_log(f"Supported architectures: {list(supported_gpu_arch.keys())}")

    console_error("Cannot find a supported arch in rocminfo.")
    return None


def _detect_gpu_chip_id(rocminfo_lines: list[str]) -> Optional[str]:
    """Detect GPU chip ID from rocminfo output."""
    chip_id_dict = mi_gpu_specs.get_chip_id_dict()
    unknown_chips: list[str] = []

    for line_text in rocminfo_lines:
        chip_id = search_pattern(r"^\s*Chip ID\s*:\s* ([0-9]+).*\s*$", line_text)
        if chip_id:
            if chip_id in chip_id_dict or int(chip_id) in chip_id_dict:
                return chip_id
            unknown_chips.append(chip_id)

    if unknown_chips:
        for chip_id in unknown_chips:
            console_warning(f"Unknown Chip ID(s) detected: {chip_id}")
    else:
        console_warning("No Chip ID detected")

    return None


def _reconstruct_specs_from_sysinfo(sysinfo: dict[str, Any]) -> MachineSpecs:
    """Reconstruct MachineSpecs from saved sysinfo dictionary."""
    try:
        sysinfo_ver = str(sysinfo["version"])
        version = get_version(config.rocprof_compute_home)["version"]
        curr_ver = version[: version.find(".")]

        if sysinfo_ver != curr_ver:
            console_warning(
                "Detected mismatch in sysinfo versioning. "
                "You need to reprofile to update data."
            )

        spec_cls = spec_family_for_arch(sysinfo.get("gpu_arch"))
        return spec_cls(**sysinfo)

    except KeyError:
        console_error(
            "Detected mismatch in sysinfo versioning. "
            "You need to reprofile to update data."
        )
        raise


def _probe_live_machine_specs(args: Optional[argparse.Namespace]) -> MachineSpecs:
    """Probe live hardware to generate MachineSpecs."""
    now = datetime.now()
    local_now = now.astimezone()
    local_tzname = local_now.tzinfo.tzname(local_now)  # type: ignore
    timestamp = f"{now.strftime('%c')} ({local_tzname})"

    version = get_version(config.rocprof_compute_home)["version"]
    specs_version = version[: version.find(".")]

    machine_info = _extract_machine_info()
    soc_info = _extract_soc_info()
    gpu_info = _extract_gpu_info(gpu_arch=soc_info["gpu_arch"])

    with amdsmi_interface.amdsmi_ctx():
        specs = spec_family_for_arch(soc_info["gpu_arch"])(
            version=specs_version,
            timestamp=timestamp,
            rocminfo_lines=soc_info["rocminfo_lines"],
            hostname=socket.gethostname(),
            cpu_model=machine_info["cpu_model"],
            sbios=machine_info["sbios"],
            linux_kernel_version=machine_info["linux_kernel_version"],
            amd_gpu_kernel_version=amdsmi_interface.get_amdgpu_driver_version(),
            cpu_memory=machine_info["cpu_memory"],
            gpu_memory=amdsmi_interface.get_gpu_vram_size(),
            linux_distro=machine_info["linux_distro"],
            rocm_version=get_rocm_ver().strip(),
            vbios=gpu_info["vbios"],
            gpu_arch=soc_info["gpu_arch"],
            gpu_chip_id=soc_info["gpu_chip_id"],
        )

    _load_soc_module(args, specs)

    if specs.rocminfo_lines is None:
        specs.gpu_model = specs.gpu_model or ""
    else:
        specs.gpu_model = (
            mi_gpu_specs.get_gpu_model(specs.gpu_arch, specs.gpu_chip_id) or ""
        )

    specs.finalize_soc_fields(gpu_info)

    return specs


def _load_soc_module(args: Optional[argparse.Namespace], specs: MachineSpecs) -> None:
    """Load the architecture-specific SoC module."""
    try:
        soc_module = importlib.import_module(
            f"rocprof_compute_soc.soc_{specs.gpu_arch}"
        )
        soc_class = getattr(soc_module, f"{specs.gpu_arch}_soc")
        soc_class(args, specs)
    except ModuleNotFoundError as e:
        console_error(
            f"Arch {specs.gpu_arch} marked as supported, "
            f"but couldn't find class implementation {e}."
        )


@demarcate
def _extract_machine_info() -> dict[str, Any]:
    """Extract machine information from /proc and /sys filesystems."""
    result: dict[str, Optional[str]] = {
        "cpu_model": None,
        "sbios": None,
        "linux_kernel_version": None,
        "cpu_memory": None,
        "linux_distro": None,
    }

    try:
        cpuinfo = path("/proc/cpuinfo").read_text(encoding="utf-8")
        meminfo = path("/proc/meminfo").read_text(encoding="utf-8")
        version = path("/proc/version").read_text(encoding="utf-8")
        os_release = path("/etc/os-release").read_text(encoding="utf-8")

        result["cpu_model"] = search_pattern(r"^model name\s*: (.*?)$", cpuinfo)
        result["sbios"] = (
            path("/sys/class/dmi/id/bios_vendor").read_text(encoding="utf-8").strip()
            + path("/sys/class/dmi/id/bios_version").read_text(encoding="utf-8").strip()
        )
        result["linux_kernel_version"] = search_pattern(r"version (\S*)", version)
        result["cpu_memory"] = search_pattern(r"MemTotal:\s*(\S*)", meminfo)
        result["linux_distro"] = (
            search_pattern(r'PRETTY_NAME="(.*?)"', os_release) or ""
        )

    except OSError as e:
        console_warning(f"Could not read system files: {e}")

    return result


@demarcate
def _extract_gpu_info(gpu_arch: Optional[str]) -> dict[str, Any]:
    """Extract GPU information from amd-smi."""
    is_partition_supported = gpu_arch and MIGPUSpecs.is_partition_supported(
        gpu_arch=gpu_arch, gpu_model=None
    )

    result: dict[str, Any] = {
        "vbios": None,
        "compute_partition": None,
        "memory_partition": None,
        "num_compute_units": None,
        "gpu_cache_info": None,
        "vram_bit_width": None,
    }

    with amdsmi_interface.amdsmi_ctx():
        result["vbios"] = amdsmi_interface.get_gpu_vbios_part_number()
        if is_partition_supported:
            result["compute_partition"] = amdsmi_interface.get_gpu_compute_partition()
            result["memory_partition"] = amdsmi_interface.get_gpu_memory_partition()
        else:
            result["compute_partition"] = "N/A"
            result["memory_partition"] = "N/A"

        result["num_compute_units"] = amdsmi_interface.get_gpu_num_compute_units()
        result["gpu_cache_info"] = amdsmi_interface.get_gpu_cache_info() or {}
        result["vram_bit_width"] = amdsmi_interface.get_gpu_vram_bit_width()

    if is_partition_supported:
        if result["compute_partition"] == "N/A" or not result["compute_partition"]:
            console_warning("Cannot detect accelerator partition from amd-smi.")
            console_warning("Applying default accelerator partition: SPX")
            result["compute_partition"] = "SPX"

        if result["memory_partition"] == "N/A" or not result["memory_partition"]:
            console_warning("Cannot detect memory partition from amd-smi.")

    console_debug(
        f"vbios is {result['vbios']}, compute partition is "
        f"{result['compute_partition']}, memory partition is "
        f"{result['memory_partition']}"
    )

    return result


@demarcate
def _extract_soc_info() -> dict[str, Any]:
    """Extract SoC information from rocminfo."""
    result: dict[str, Any] = {
        "rocminfo_lines": None,
        "gpu_arch": None,
        "gpu_chip_id": None,
    }

    rocminfo_full = _run_command(["rocminfo"])
    if rocminfo_full is None:
        return result

    rocminfo_lines = rocminfo_full.split("\n")
    arch_result = _detect_arch(rocminfo_lines)

    if arch_result is None:
        return result

    result["gpu_arch"], arch_idx = arch_result
    result["rocminfo_lines"] = rocminfo_lines[arch_idx + 1 :]
    result["gpu_chip_id"] = _detect_gpu_chip_id(rocminfo_lines)

    return result


@_kw_only
@dataclass
class MachineSpecs:
    """Base class for machine specifications.

    Contains fields common to all GPU families (CDNA and RDNA).
    Subclasses add family-specific fields.
    """

    # Workload / Spec info
    workload_path: Optional[str] = field(
        default=None,
        metadata={
            "doc": "Path to the workload data directory.",
            "name": "Workload Path",
            "optional": True,
            "show_in_table": True,
        },
    )
    command: Optional[str] = field(
        default=None,
        metadata={
            "doc": "The command the workload was executed with.",
            "name": "Command",
            "optional": True,
            "show_in_table": True,
        },
    )
    ip_blocks: Optional[str] = field(
        default=None,
        metadata={
            "doc": "The hardware blocks profiling information was collected for.",
            "name": "IP Blocks",
            "optional": True,
        },
    )
    timestamp: Optional[str] = field(
        default=None,
        metadata={
            "doc": "The time (in local system time) when data was collected",
            "name": "Timestamp",
            "show_in_table": True,
        },
    )
    version: Optional[str] = field(
        default=None,
        metadata={
            "doc": "The version of the machine specification file format.",
            "name": "MachineSpecs Version",
            "intable": False,
            "show_in_table": True,
        },
    )
    rocminfo_lines: Optional[list] = field(
        default=None, metadata={"show_in_table": False}
    )

    # Machine specs
    hostname: Optional[str] = field(
        default=None,
        metadata={
            "doc": "The hostname of the machine.",
            "name": "Hostname",
            "show_in_table": True,
        },
    )
    cpu_model: Optional[str] = field(
        default=None,
        metadata={
            "doc": "The model name of the CPU used.",
            "name": "CPU Model",
            "show_in_table": True,
        },
    )
    sbios: Optional[str] = field(
        default=None,
        metadata={
            "doc": "The system management bios version and vendor.",
            "name": "SBIOS",
            "show_in_table": True,
        },
    )
    linux_distro: Optional[str] = field(
        default=None,
        metadata={
            "doc": "The Linux distribution installed on the machine.",
            "name": "Linux Distribution",
            "show_in_table": True,
        },
    )
    linux_kernel_version: Optional[str] = field(
        default=None,
        metadata={
            "doc": "The Linux kernel version running on the machine.",
            "name": "Linux Kernel Version",
            "show_in_table": True,
        },
    )
    amd_gpu_kernel_version: Optional[str] = field(
        default=None,
        metadata={
            "doc": "The version of the AMDGPU driver installed on the machine.",
            "name": "AMD GPU Kernel Version",
            "show_in_table": True,
        },
    )
    cpu_memory: Optional[str] = field(
        default=None,
        metadata={
            "doc": "The total amount of memory available to the CPU.",
            "unit": "KB",
            "name": "CPU Memory",
            "show_in_table": True,
        },
    )
    gpu_memory: Optional[str] = field(
        default=None,
        metadata={
            "doc": (
                "The total amount of memory available to accelerators/GPUs "
                "in the system."
            ),
            "unit": "KB",
            "name": "GPU Memory",
            "show_in_table": True,
        },
    )
    rocm_version: Optional[str] = field(
        default=None,
        metadata={
            "doc": "The ROCm version used during data-collection.",
            "name": "ROCm Version",
            "show_in_table": True,
        },
    )
    vbios: Optional[str] = field(
        default=None,
        metadata={
            "doc": "The version of the accelerators/GPUs video bios in the system.",
            "name": "VBIOS",
            "show_in_table": True,
        },
    )

    # SoC specs
    gpu_series: Optional[str] = field(
        default=None,
        metadata={
            "doc": "The series of the accelerators/GPUs in the system.",
            "name": "GPU Series",
            "show_in_table": True,
        },
    )
    gpu_model: Optional[str] = field(
        default=None,
        metadata={
            "doc": "The product name of the accelerators/GPUs in the system.",
            "name": "GPU Model",
            "show_in_table": True,
        },
    )
    gpu_arch: Optional[str] = field(
        default=None,
        metadata={
            "doc": "The architecture name of the accelerators/GPUs in the system,\n"
            "as used by (e.g.,) the AMDGPU backed of LLVM.",
            "name": "GPU Arch",
            "show_in_table": True,
        },
    )
    gpu_chip_id: Optional[str] = field(
        default=None,
        metadata={
            "doc": "The Chip ID of the accelerators/GPUs in the system.",
            "name": "Chip ID",
            "optional": True,
            "show_in_table": True,
        },
    )
    gpu_l1: Optional[str] = field(
        default=None,
        metadata={
            "doc": (
                "The size of the vL1D cache (per compute-unit) on the "
                "accelerators/GPUs."
            ),
            "name": "GPU L1",
            "unit": "KiB",
            "show_in_table": True,
        },
    )
    gpu_l2: Optional[str] = field(
        default=None,
        metadata={
            "doc": ("The size of the L2 cache (total) on the accelerators/GPUs."),
            "name": "GPU L2",
            "unit": "KiB",
            "show_in_table": True,
        },
    )
    cu_per_gpu: Optional[str] = field(
        default=None,
        metadata={
            "doc": (
                "The total number of compute units per accelerator/GPU in the system. "
                "On systems with configurable partitioning, (e.g., MI300) this is "
                "the total number of compute units in a partition."
            ),
            "name": "CU per GPU",
            "show_in_table": True,
        },
    )
    simd_per_cu: Optional[str] = field(
        default=None,
        metadata={
            "doc": (
                "The number of SIMD processors in a compute unit for the "
                "accelerators/GPUs in the system."
            ),
            "name": "SIMD per CU",
            "show_in_table": True,
        },
    )
    se_per_gpu: Optional[str] = field(
        default=None,
        metadata={
            "doc": (
                "The number of shader engines on the accelerators/GPUs in the system. "
                "On systems with configurable partitioning, (e.g., MI300) this is "
                "the total number of shader engines in a partition."
            ),
            "name": "SE per GPU",
            "show_in_table": True,
        },
    )
    sa_per_se: Optional[str] = field(
        default=None,
        metadata={
            "doc": (
                "The number of shader arrays per shader engine on the accelerators/GPUs"
                " in the system."
            ),
            "name": "SA per SE",
            "show_in_table": True,
        },
    )
    wave_size: Optional[str] = field(
        default=None,
        metadata={
            "doc": (
                "The number work-items in a wavefront on the accelerators/GPUs in "
                "the system."
            ),
            "name": "Wave Size",
            "show_in_table": True,
        },
    )
    workgroup_max_size: Optional[str] = field(
        default=None,
        metadata={
            "doc": (
                "The maximum number of work-items in a workgroup on the "
                "accelerators/GPUs in the system."
            ),
            "name": "Workgroup Max Size",
            "show_in_table": True,
        },
    )
    max_waves_per_cu: Optional[str] = field(
        default=None,
        metadata={
            "doc": (
                "The maximum number of wavefronts that can be resident on a "
                "compute unit on the accelerators/GPUs in the system"
            ),
            "name": "Max Waves per CU",
            "show_in_table": True,
        },
    )
    max_sclk: Optional[str] = field(
        default=None,
        metadata={
            "doc": (
                "The maximum engine (compute-unit) clock rate of the "
                "accelerators/GPUs in the system."
            ),
            "name": "Max SCLK",
            "unit": "MHz",
            "show_in_table": True,
        },
    )
    max_mclk: Optional[str] = field(
        default=None,
        metadata={
            "doc": (
                "The maximum memory clock rate of the accelerators/GPUs in the system."
            ),
            "name": "Max MCLK",
            "unit": "MHz",
            "show_in_table": True,
        },
    )
    cur_sclk: Optional[str] = field(
        default=None,
        metadata={
            "doc": (
                "[RESERVED] The current engine (compute unit) clock rate of the "
                "accelerators/GPUs in the system. Unused."
            ),
            "name": "Cur SCLK",
            "unit": "MHz",
            "show_in_table": True,
        },
    )
    cur_mclk: Optional[str] = field(
        default=None,
        metadata={
            "doc": (
                "[RESERVED] The current memory clock rate of the accelerators/GPUs "
                "in the system. Unused."
            ),
            "name": "Cur MCLK",
            "unit": "MHz",
            "show_in_table": True,
        },
    )
    l2_banks: Optional[str] = field(
        default=None,
        metadata={"show_in_table": True},
    )
    total_l2_chan: Optional[str] = field(
        default=None,
        metadata={
            "doc": (
                "The maximum number of L2 cache channels on the accelerators/GPUs "
                "in the system. On systems with configurable partitioning, "
                "(e.g., MI300) this is the total number of L2 cache channels "
                "in a partition."
            ),
            "name": "Total L2 Channels",
            "show_in_table": True,
        },
    )
    lds_banks_per_cu: Optional[str] = field(
        default=None,
        metadata={
            "doc": (
                "The number of banks in the LDS for a compute unit on the "
                "accelerators/GPUs in the system."
            ),
            "name": "LDS Banks per CU",
            "show_in_table": True,
        },
    )
    sqc_per_gpu: Optional[str] = field(
        default=None,
        metadata={
            "doc": (
                "The number of L1I/sL1D caches on the accelerators/GPUs in the "
                "system. On systems with configurable partitioning, (e.g., MI300) "
                "this is the total number of L1I/sL1D caches in a partition."
            ),
            "name": "SQC per GPU",
            "show_in_table": True,
        },
    )
    pipes_per_gpu: Optional[str] = field(
        default=None,
        metadata={
            "doc": (
                "The number of scheduler-pipes on the accelerators/GPUs in the system."
            ),
            "name": "Pipes per GPU",
            "show_in_table": True,
        },
    )
    num_memory_channels: Optional[str] = field(
        default=None,
        metadata={
            "doc": "Number of memory channels (HBM for CDNA, LPDDR for RDNA 3.5)",
            "name": "Memory Channels",
            "show_in_table": True,
        },
    )
    num_dies: Optional[int] = field(
        default=None,
        metadata={
            "doc": "Number of logical dies present on the model. For Instinct products "
            "it refers to the number of logical AID partitions, for Radeon products it "
            "is the memory die (*dGPU only, otherwise set to 1 partition).",
            "name": "Number of logical dies",
            "show_in_table": False,
        },
    )
    cache_sizes: Optional[dict[str, int]] = field(
        default=None,
        metadata={
            "doc": "Size of cache at each level present on the GPU",
            "name": "Cache sizes",
            "show_in_table": False,
        },
    )

    def finalize_soc_fields(self, gpu_info: dict[str, Any]) -> None:
        """Derive shared SoC-dependent fields. Subclasses override and call super()."""
        self.total_l2_chan = totall2_banks(
            self.gpu_arch,
            self.gpu_model,
            self.l2_banks,
            getattr(self, "compute_partition", None),
        )
        self.num_dies = mi_gpu_specs.get_num_dies(self.gpu_arch, self.gpu_model)
        self.cache_sizes = set_cache_sizes(
            self.gpu_model,
            gpu_info["num_compute_units"],
            gpu_info["gpu_cache_info"],
            self.num_dies,
            self.se_per_gpu,
            self.sa_per_se,
        )

    def get_class_members(self) -> dict[str, Any]:
        """Return class members as a dictionary."""
        data: dict[str, Any] = {}
        missing_required_fields: list[str] = []

        for class_field in fields(self):
            if not class_field.metadata.get("show_in_table", True):
                continue

            name = class_field.name
            value = getattr(self, name)
            data[name] = value

            if value is None and not class_field.metadata.get("optional", False):
                missing_required_fields.append(name)

        if missing_required_fields:
            for field_name in missing_required_fields:
                console_warning(
                    f"Incomplete class definition for {self.gpu_arch}. "
                    f"Expecting populated {field_name} but detected None."
                )
            console_warning(f"Missing specs fields for {self.gpu_arch}")

        return data

    def __repr__(self) -> str:
        topstr = (
            "Machine Specifications: describing the state of the machine that "
            "ROCm Compute Profiler data was collected on.\n"
        )
        data: list[dict[str, Any]] = []
        has_description = False
        has_unit = False

        for class_field in fields(self):
            name = class_field.name
            if class_field.metadata.get("show_in_table", True):
                _data: dict[str, Any] = {}
                value = getattr(self, name)
                if class_field.metadata:
                    if (
                        "intable" in class_field.metadata
                        and not class_field.metadata["intable"]
                    ):
                        if name == "version":
                            topstr += f"Output version: {value}\n"
                        else:
                            console_error(
                                f"Unknown out of table printing field: {name}"
                            )
                        continue
                    if "name" in class_field.metadata:
                        name = class_field.metadata["name"]
                    if "unit" in class_field.metadata:
                        _data["Unit"] = class_field.metadata["unit"]
                        has_unit = True
                    if "doc" in class_field.metadata:
                        _data["Description"] = class_field.metadata["doc"]
                        has_description = True
                _data["Spec"] = name
                _data["Value"] = value if value is not None else ""
                data.append(_data)

        columns = ["Spec", "Value"]
        if has_description:
            columns.append("Description")
        if has_unit:
            columns.append("Unit")

        return topstr + format_table_ascii(data, columns, decimal=2)


@_kw_only
@dataclass(repr=False)
class MachineSpecsCDNA(MachineSpecs):
    """CDNA specs: adds compute/memory partition and XCD fields."""

    compute_partition: Optional[str] = field(
        default=None,
        metadata={
            "doc": (
                "The compute partitioning mode active on the accelerators/GPUs in the "
                "system (MI300 only)."
            ),
            "name": "Compute Partition",
            "show_in_table": True,
        },
    )
    memory_partition: Optional[str] = field(
        default=None,
        metadata={
            "doc": (
                "The memory partitioning mode active on the accelerators/GPUs in the "
                "system (MI300 only)."
            ),
            "name": "Memory Partition",
            "show_in_table": True,
        },
    )
    num_xcd: Optional[str] = field(
        default=None,
        metadata={
            "doc": (
                "The total number of accelerator complex dies in a compute partition "
                "on the accelerators/GPUs in the system. For accelerators without "
                "partitioning (i.e., pre-MI300), this is considered to be one."
            ),
            "name": "Num XCDs",
            "unit": "XCDs",
            "show_in_table": True,
        },
    )

    def _get_hbm_channels(self) -> Optional[str]:
        """HBM channel count, adjusted for the MI300 NPS memory partition."""
        partition = self.memory_partition or ""
        if partition.lower().startswith("nps"):
            channels = 128
            if partition.lower() == "nps4":
                channels //= 4
            elif partition.lower() == "nps8":
                channels //= 8
            return str(channels)
        return self.total_l2_chan

    def finalize_soc_fields(self, gpu_info: dict[str, Any]) -> None:
        self.compute_partition = gpu_info["compute_partition"]
        self.memory_partition = gpu_info["memory_partition"]
        self.num_xcd = str(
            mi_gpu_specs.get_num_xcds(
                self.gpu_arch, self.gpu_model or None, self.compute_partition
            )
        )
        super().finalize_soc_fields(gpu_info)
        self.num_memory_channels = self._get_hbm_channels()


@_kw_only
@dataclass(repr=False)
class MachineSpecsRDNA35(MachineSpecs):
    """RDNA 3.5 specs: adds GL1 cache count field."""

    num_gl1c: Optional[str] = field(
        default=None,
        metadata={
            "doc": (
                "The number of GL1 caches (one per Shader Array) on the GPU. "
                "Used for GL1 bandwidth-ceiling calculations in analysis configs."
            ),
            "name": "Num GL1 Caches",
            "show_in_table": True,
        },
    )

    def finalize_soc_fields(self, gpu_info: dict[str, Any]) -> None:
        if self.se_per_gpu is None or self.sa_per_se is None:
            self.num_gl1c = None
        else:
            self.num_gl1c = str(int(self.se_per_gpu) * int(self.sa_per_se))

        super().finalize_soc_fields(gpu_info)

        bit_width = gpu_info.get("vram_bit_width")
        if bit_width:
            self.num_memory_channels = str(int(bit_width) // 32)
        else:
            self.num_memory_channels = self.total_l2_chan


if __name__ == "__main__":
    specs = generate_machine_specs(None, None)
    if specs:
        print(specs)
    else:
        console_error("specs", "Failed to generate machine specifications", exit=False)
