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
"""Get host/gpu specs."""

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

import pandas as pd

import config
from utils.amdsmi_interface import (
    amdsmi_ctx,
    get_gpu_compute_partition,
    get_gpu_memory_partition,
    get_gpu_vbios_part_number,
)
from utils.logger import (
    console_debug,
    console_error,
    console_log,
    console_warning,
    demarcate,
)
from utils.mi_gpu_spec import mi_gpu_specs
from utils.tty import get_table_string
from utils.utils import get_version

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


def detect_arch(rocminfo_lines: list[str]) -> Optional[tuple[str, int]]:
    for idx1, line_text in enumerate(rocminfo_lines):
        gpu_arch = search(
            r"^\s*Name\s*:\s* ([Gg][Ff][Xx][a-zA-Z0-9]+).*\s*$", line_text
        )
        if gpu_arch and gpu_arch in mi_gpu_specs.get_gpu_series_dict():
            return (gpu_arch, idx1)

    console_error("Cannot find a supported arch in rocminfo")


def detect_gpu_chip_id(rocminfo_lines: list[str]) -> Optional[str]:
    chip_id_dict = mi_gpu_specs.get_chip_id_dict()
    unknown_chips: list[str] = []

    for idx, line_text in enumerate(rocminfo_lines):
        chip_id = search(r"^\s*Chip ID\s*:\s* ([0-9]+).*\s*$", line_text)
        if chip_id:
            # Check if this chip ID is valid (known)
            if chip_id in chip_id_dict or int(chip_id) in chip_id_dict:
                return chip_id  # Return first valid chip ID found
            else:
                unknown_chips.append(chip_id)

    # Exhausted all lines - handle the cases where no valid chip was found
    if unknown_chips:
        for chip_id in unknown_chips:
            console_warning(f"Unknown Chip ID(s) detected: {chip_id}")
    else:
        console_warning("No Chip ID detected")

    return None


# Custom decorator to mimic the behavior of kw_only found in Python 3.10
def kw_only(cls: T) -> T:
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        for name, value in kwargs.items():
            setattr(self, name, value)

    cls.__init__ = __init__  # type: ignore
    return cls


def generate_machine_specs(
    args: Optional[argparse.Namespace], sysinfo: Optional[dict[str, Any]] = None
) -> MachineSpecs:
    if sysinfo is not None:
        try:
            sysinfo_ver = str(sysinfo["version"])
            version = get_version(config.rocprof_compute_home)["version"]
            curr_ver = version[: version.find(".")]

            if sysinfo_ver != curr_ver:
                console_warning(
                    "Detected mismatch in sysinfo versioning. "
                    "You need to reprofile to update data."
                )

            return MachineSpecs(**sysinfo)
        except KeyError:
            console_error(
                "Detected mismatch in sysinfo versioning. You need to reprofile "
                "to update data."
            )

    # read timestamp info
    now = datetime.now()
    local_now = now.astimezone()
    local_tzname = local_now.tzinfo.tzname(local_now)  # type: ignore
    timestamp = f"{now.strftime('%c')} ({local_tzname})"

    # set specs version
    version = get_version(config.rocprof_compute_home)["version"]
    # NB: Just taking major as specs version.
    # May want to make this more specific in the future
    # version will always follow 'major.minor.patch' format
    specs_version = version[: version.find(".")]

    ##########################################
    ## A. Machine Specs
    ##########################################
    machine_info = extract_machine_info()

    # FIXME: use device
    # Load amd-smi data
    gpu_info = extract_gpu_info()

    ##########################################
    ## B. SoC Specs
    ##########################################
    soc_info = extract_soc_info()

    # Combine all specifications
    specs = MachineSpecs(
        version=specs_version,
        timestamp=timestamp,
        rocminfo_lines=soc_info["rocminfo_lines"],
        hostname=socket.gethostname(),
        cpu_model=machine_info["cpu_model"],
        sbios=machine_info["sbios"],
        linux_kernel_version=machine_info["linux_kernel_version"],
        amd_gpu_kernel_version="",
        cpu_memory=machine_info["cpu_memory"],
        gpu_memory="",
        linux_distro=machine_info["linux_distro"],
        rocm_version=get_rocm_ver().strip(),
        vbios=gpu_info["vbios"],
        compute_partition=gpu_info["compute_partition"],
        memory_partition=gpu_info["memory_partition"],
        gpu_arch=soc_info["gpu_arch"],
        gpu_chip_id=soc_info["gpu_chip_id"],
    )

    # Load above SoC specs via module import
    try:
        soc_module = importlib.import_module(
            f"rocprof_compute_soc.soc_{specs.gpu_arch}"
        )
        soc_class = getattr(soc_module, f"{specs.gpu_arch}_soc")
        soc_obj = soc_class(args, specs)  # noqa: F841
    except ModuleNotFoundError as e:
        console_error(
            f"Arch {specs.gpu_arch} marked as supported,"
            f"but couldn't find class implementation {e}."
        )

    # Update arch specific specs
    specs.gpu_model = (
        mi_gpu_specs.get_gpu_model(specs.gpu_arch, specs.gpu_chip_id) or ""
    )
    specs.num_xcd = str(
        mi_gpu_specs.get_num_xcds(
            specs.gpu_arch, specs.gpu_model, specs.compute_partition
        )
    )
    specs.total_l2_chan = totall2_banks(
        specs.gpu_arch,
        specs.gpu_model,
        specs.l2_banks,
        specs.compute_partition,
    )
    specs.num_hbm_channels = str(specs.get_hbm_channels())

    return specs


@demarcate
def extract_machine_info() -> dict[str, Any]:
    result: dict[str, Optional[str]] = {
        "cpu_model": None,
        "sbios": None,
        "linux_kernel_version": None,
        "cpu_memory": None,
        "linux_distro": None,
    }

    try:
        cpuinfo = path("/proc/cpuinfo").read_text()
        meminfo = path("/proc/meminfo").read_text()
        version = path("/proc/version").read_text()
        os_release = path("/etc/os-release").read_text()

        result["cpu_model"] = search(r"^model name\s*: (.*?)$", cpuinfo)
        result["sbios"] = (
            path("/sys/class/dmi/id/bios_vendor").read_text().strip()
            + path("/sys/class/dmi/id/bios_version").read_text().strip()
        )
        result["linux_kernel_version"] = search(r"version (\S*)", version)
        result["cpu_memory"] = search(r"MemTotal:\s*(\S*)", meminfo)
        result["linux_distro"] = search(r'PRETTY_NAME="(.*?)"', os_release) or ""

    except OSError as e:
        console_warning(f"Could not read system files: {e}")
    return result


@demarcate
def extract_gpu_info() -> dict[str, Any]:
    result: dict[str, Optional[str]] = {
        "vbios": None,
        "compute_partition": None,
        "memory_partition": None,
    }

    with amdsmi_ctx():
        result["vbios"] = get_gpu_vbios_part_number()
        result["compute_partition"] = get_gpu_compute_partition()
        result["memory_partition"] = get_gpu_memory_partition()

    # Apply defaults and warnings
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
def extract_soc_info() -> dict[str, Any]:
    result: dict[str, Any] = {
        "rocminfo_lines": None,
        "gpu_arch": None,
        "gpu_chip_id": None,
    }

    # Read rocminfo
    rocminfo_full = run(["rocminfo"])
    if rocminfo_full is None:
        return result

    rocminfo_lines = rocminfo_full.split("\n")
    arch_result = detect_arch(rocminfo_lines)

    if arch_result is None:
        return result

    result["gpu_arch"], arch_idx = arch_result
    result["rocminfo_lines"] = rocminfo_lines[
        arch_idx + 1 :
    ]  # update rocminfo for target section
    result["gpu_chip_id"] = detect_gpu_chip_id(rocminfo_lines)

    return result


@kw_only
@dataclass
class MachineSpecs:
    ##########################################
    ## A. Workload / Spec info
    ##########################################

    # these three fields are special in that they're not included
    # when you use (e.g.,) --specs to view the machinespecs, but they
    # _are_ included in profiling/analysis, so we mark them as 'optional'
    # in the metadata to avoid erroring out on missing fields on
    # serialization
    workload_name: Optional[str] = field(
        default=None,
        metadata={
            "doc": "The name of the workload data was collected for.",
            "name": "Workload Name",
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
    timestamp: Optional[str] = field(
        default=None,
        metadata={
            "doc": "The time (in local system time) when data was collected",
            "name": "Timestamp",
            "show_in_table": True,
        },
    )
    rocminfo_lines: Optional[list] = field(
        default=None, metadata={"show_in_table": False}
    )
    ##########################################
    ## A. Machine Specs
    ##########################################
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
            "doc": (
                "[RESERVED] The version of the AMDGPU driver installed on the machine. "
                "Unimplemented."
            ),
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
                "[RESERVED] The total amount of memory available to accelerators/GPUs "
                "in the system. Unimplemented."
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

    ##########################################
    ## B. SoC Specs
    ##########################################
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
            "doc": (
                "The size of the vL1D cache (per compute-unit) on the "
                "accelerators/GPUs."
            ),
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
        metadata={
            "show_in_table": True,
        },
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
    num_hbm_channels: Optional[str] = field(
        default=None,
        metadata={
            "doc": "Number of HBM channels",
            "name": "HBM channels",
            "show_in_table": True,
        },
    )

    def get_hbm_channels(self) -> Optional[str]:
        if self.memory_partition and self.memory_partition.lower().startswith("nps"):
            hbmchannels = 128
            if self.memory_partition.lower() == "nps4":
                hbmchannels //= 4
            elif self.memory_partition.lower() == "nps8":
                hbmchannels //= 8
            return str(hbmchannels)
        else:
            return self.total_l2_chan

    def get_class_members(self) -> pd.DataFrame:
        data = {}
        missing_required_fields = []

        for class_field in fields(self):
            if not class_field.metadata.get("show_in_table", True):
                continue

            name = class_field.name
            value = getattr(self, name)
            data[name] = value

            # Check for missing required fields
            if value is None and not class_field.metadata.get("optional", False):
                missing_required_fields.append(name)

        # Handle warnings after processing all fields
        if missing_required_fields:
            for field_name in missing_required_fields:
                console_warning(
                    f"Incomplete class definition for {self.gpu_arch}. "
                    f"Expecting populated {field_name} but detected None."
                )
            console_warning(f"Missing specs fields for {self.gpu_arch}")

        return pd.DataFrame(data, index=[0])

    def __repr__(self) -> str:
        topstr = (
            "Machine Specifications: describing the state of the machine that "
            "ROCm Compute Profiler data was collected on.\n"
        )
        data = []
        for class_field in fields(self):
            name = class_field.name
            if class_field.metadata.get("show_in_table", True):
                _data = {}
                value = getattr(self, name)
                if class_field.metadata:
                    # check out of table before any re-naming for pretty-printing
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
                    if "doc" in class_field.metadata:
                        _data["Description"] = class_field.metadata["doc"]
                _data["Spec"] = name
                _data["Value"] = value
                data.append(_data)
        df = pd.DataFrame(data)
        columns = ["Spec", "Value"]
        if "Description" in df.columns:
            columns += ["Description"]
        if "Unit" in df.columns:
            columns += ["Unit"]
        df = df[columns]
        df = df.fillna("")
        return topstr + get_table_string(df, transpose=False, decimal=2)


def get_rocm_ver() -> str:
    # Check for version files in ROCm installation
    rocm_base_path = path(os.getenv("ROCM_PATH", "/opt/rocm"))

    for version_file_name in VERSION_LOC:
        version_file_path = rocm_base_path / ".info" / version_file_name
        if version_file_path.exists():
            return version_file_path.read_text().strip()

    # Fallback to environment variable
    ROCM_VER_USER = os.getenv("ROCM_VER")
    if ROCM_VER_USER:
        console_log(
            "profiling",
            "Overriding missing ROCm version detection with "
            f"ROCM_VER = {ROCM_VER_USER}",
        )
        return ROCM_VER_USER

    # No version found - log error and return empty string
    console_warning("Unable to detect a complete local ROCm installation.")
    console_warning(
        f"The expected {rocm_base_path}/.info/ versioning directory is missing."
    )
    console_error("Ensure you have valid ROCm installation.", exit=False)
    return ""


def run(cmd: list[str], exit_on_error: bool = False) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True)
    except FileNotFoundError as e:
        console_error(
            f"Unable to parse specs. Can't find ROCm asset: {e.filename}\n"
            'Try passing a path to an existing workload results in "analyze" mode.'
        )

    if exit_on_error and p.returncode != 0:  # type: ignore
        console_error(f"Command {cmd} failed with non-zero exit code")
    return p.stdout.decode("utf-8")  # type: ignore


def search(pattern: str, string: str) -> Optional[str]:
    m = re.search(pattern, string, re.MULTILINE)
    if m is not None:
        return m.group(1)
    return None


def total_sqc(archname: str, num_compute_units: str, num_shader_engines: str) -> int:
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
    xcd_count = mi_gpu_specs.get_num_xcds(gpu_arch, gpu_model, compute_partition)

    # TODO: MachineSpecs and OmniSoC mspec should converge...
    if L2banks is not None and xcd_count is not None:
        return str(int(L2banks) * int(xcd_count))
    return None


if __name__ == "__main__":
    specs = generate_machine_specs(None, None)
    if specs:
        print(specs)
    else:
        console_error("specs", "Failed to generate machine specifications", exit=False)
