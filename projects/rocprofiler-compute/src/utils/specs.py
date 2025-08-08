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

import importlib
import json
import os
import re
import socket
import subprocess
from dataclasses import dataclass, field, fields
from datetime import datetime
from math import ceil
from pathlib import Path as path

import pandas as pd

import config
from utils.logger import console_debug, console_error, console_log, console_warning
from utils.mi_gpu_spec import mi_gpu_specs
from utils.tty import get_table_string
from utils.utils import get_version

VERSION_LOC = [
    "version",
    "version-dev",
    "version-hip-libraries",
    "version-hiprt",
    "version-hiprt-devel",
    "version-hip-sdk",
    "version-libs",
    "version-utils",
]


def detect_arch(_rocminfo):
    for idx1, linetext in enumerate(_rocminfo):
        # NOTE: currently supported socs are gfx archs only
        gpu_arch = search(r"^\s*Name\s*:\s* ([Gg][Ff][Xx][a-zA-Z0-9]+).*\s*$", linetext)
        if gpu_arch in mi_gpu_specs.get_gpu_series_dict().keys():
            break
        if str(gpu_arch) in mi_gpu_specs.get_gpu_series_dict().keys():
            gpu_arch = str(gpu_arch)
            break
    if not gpu_arch in mi_gpu_specs.get_gpu_series_dict().keys():
        console_error("Cannot find a supported arch in rocminfo: " + str(gpu_arch))
    else:
        return (gpu_arch, idx1)


def detect_gpu_chip_id(_rocminfo):
    gpu_chip_id = None

    for idx1, linetext in enumerate(_rocminfo):
        # NOTE: current supported socs only have numbers in Chip ID
        chip_found = search(r"^\s*Chip ID\s*:\s* ([0-9]+).*\s*$", linetext)
        if chip_found:
            gpu_chip_id = str(chip_found)
            break

    if not gpu_chip_id:
        console_warning("No Chip ID detected: " + str(gpu_chip_id))
    elif (
        gpu_chip_id not in mi_gpu_specs.get_chip_id_dict().keys()
        and int(gpu_chip_id) not in mi_gpu_specs.get_chip_id_dict().keys()
    ):
        console_warning("Unknown Chip ID detected: " + str(gpu_chip_id))
    return gpu_chip_id


# Custom decorator to mimic the behavior of kw_only found in Python 3.10
def kw_only(cls):
    def __init__(self, *args, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)

    cls.__init__ = __init__
    return cls


def generate_machine_specs(args, sysinfo: dict = None):
    if sysinfo is not None:
        try:
            sysinfo_ver = str(sysinfo["version"])
        except KeyError:
            console_error(
                "Detected mismatch in sysinfo versioning. You need to reprofile "
                "to update data."
            )
        version = get_version(config.rocprof_compute_home)["version"]
        if sysinfo_ver != version[: version.find(".")]:
            console_warning(
                "Detected mismatch in sysinfo versioning. You need to reprofile "
                "to update data."
            )
        return MachineSpecs(**sysinfo)
    # read timestamp info
    now = datetime.now()
    local_now = now.astimezone()
    local_tz = local_now.tzinfo
    local_tzname = local_tz.tzname(local_now)
    timestamp = now.strftime("%c") + " (" + local_tzname + ")"
    hostname = socket.gethostname()

    # set specs version
    vData = get_version(config.rocprof_compute_home)
    version = vData["version"]
    # NB: Just taking major as specs version.
    # May want to make this more specific in the future
    specs_version = version[
        : version.find(".")
    ]  # version will always follow 'major.minor.patch' format

    ##########################################
    ## A. Machine Specs
    ##########################################
    cpuinfo = path("/proc/cpuinfo").read_text()
    meminfo = path("/proc/meminfo").read_text()
    version = path("/proc/version").read_text()
    os_release = path("/etc/os-release").read_text()
    cpu_model = search(r"^model name\s*: (.*?)$", cpuinfo)
    sbios = (
        path("/sys/class/dmi/id/bios_vendor").read_text().strip()
        + path("/sys/class/dmi/id/bios_version").read_text().strip()
    )
    linux_kernel_version = search(r"version (\S*)", version)
    amd_gpu_kernel_version = ""  # TODO: Extract amdgpu kernel version
    cpu_memory = search(r"MemTotal:\s*(\S*)", meminfo)
    gpu_memory = ""  # TODO: Extract gpu memory
    linux_distro = search(r'PRETTY_NAME="(.*?)"', os_release)
    if linux_distro is None:
        linux_distro = ""
    rocm_version = get_rocm_ver().strip()
    # FIXME: use device

    # Parse json from amd-smi
    amd_smi_static_json = json.loads(
        run(["amd-smi", "static", "--json"], exit_on_error=True)
    )

    # NOTE: Need to obtain partition information explicitly from amd-smi partition
    #       https://github.com/ROCm/amdsmi/commit/88473b7fd0d34790bca09fcc3607ea79e3a837e0
    amd_smi_partition_json = json.loads(
        run(["amd-smi", "partition", "--json"], exit_on_error=True)
    )

    gpu_data = amd_smi_static_json.get("gpu_data", [])
    vbios = gpu_data[0].get("vbios", {}).get("part_number", None) if gpu_data else None

    partition_data = amd_smi_partition_json.get("current_partition", [])
    compute_partition = (
        partition_data[0].get("accelerator_type", None) if partition_data else None
    )
    memory_partition = partition_data[0].get("memory", None) if partition_data else None

    if compute_partition is None:
        console_warning("Can not detect accelerator partition from amd-smi.")
        console_warning("Applying default accelerator partition: SPX")
        # Apply default compute partition
        compute_partition = "SPX"

    if memory_partition is None:
        console_warning("Can not detect memory partition from amd-smi.")

    console_debug(
        "vbios is {}, compute partition is {}, memory partition is {}".format(
            vbios, compute_partition, memory_partition
        )
    )

    ##########################################
    ## B. SoC Specs
    ##########################################
    # read rocminfo
    rocminfo_full = run(["rocminfo"])
    _rocminfo = rocminfo_full.split("\n")
    gpu_arch, idx = detect_arch(_rocminfo)
    _rocminfo = _rocminfo[idx + 1 :]  # update rocminfo for target section
    gpu_chip_id = detect_gpu_chip_id(_rocminfo)
    specs = MachineSpecs(
        version=specs_version,
        timestamp=timestamp,
        _rocminfo=_rocminfo,
        hostname=hostname,
        cpu_model=cpu_model,
        sbios=sbios,
        linux_kernel_version=linux_kernel_version,
        amd_gpu_kernel_version=amd_gpu_kernel_version,
        cpu_memory=cpu_memory,
        gpu_memory=gpu_memory,
        linux_distro=linux_distro,
        rocm_version=rocm_version,
        vbios=vbios,
        compute_partition=compute_partition,
        memory_partition=memory_partition,
        gpu_arch=gpu_arch,
        gpu_chip_id=gpu_chip_id,
    )

    # Load above SoC specs via module import
    try:
        soc_module = importlib.import_module("rocprof_compute_soc.soc_" + specs.gpu_arch)
    except ModuleNotFoundError as e:
        console_error(
            "Arch %s marked as supported, but couldn't find class implementation %s."
            % (specs.gpu_arch, e)
        )
    soc_class = getattr(soc_module, specs.gpu_arch + "_soc")
    soc_obj = soc_class(args, specs)  # noqa: F841
    # Update arch specific specs
    specs.gpu_model = mi_gpu_specs.get_gpu_model(specs.gpu_arch, specs.gpu_chip_id)
    specs.num_xcd = mi_gpu_specs.get_num_xcds(
        specs.gpu_arch, specs.gpu_model, specs.compute_partition
    )
    specs.total_l2_chan: str = total_l2_banks(
        specs.gpu_arch, specs.gpu_model, specs._l2_banks, specs.compute_partition
    )
    specs.num_hbm_channels: str = str(specs.get_hbm_channels())
    return specs


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
    workload_name: str = field(
        default=None,
        metadata={
            "doc": "The name of the workload data was collected for.",
            "name": "Workload Name",
            "optional": True,
        },
    )
    command: str = field(
        default=None,
        metadata={
            "doc": "The command the workload was executed with.",
            "name": "Command",
            "optional": True,
        },
    )
    ip_blocks: str = field(
        default=None,
        metadata={
            "doc": "The hardware blocks profiling information was collected for.",
            "name": "IP Blocks",
            "optional": True,
        },
    )
    timestamp: str = field(
        default=None,
        metadata={
            "doc": "The time (in local system time) when data was collected",
            "name": "Timestamp",
        },
    )
    version: str = field(
        default=None,
        metadata={
            "doc": "The version of the machine specification file format.",
            "name": "MachineSpecs Version",
            "intable": False,
        },
    )
    timestamp: str = field(
        default=None,
        metadata={
            "doc": "The time (in local system time) when data was collected",
            "name": "Timestamp",
        },
    )
    _rocminfo: list = field(default=None)
    ##########################################
    ## A. Machine Specs
    ##########################################
    hostname: str = field(
        default=None,
        metadata={"doc": "The hostname of the machine.", "name": "Hostname"},
    )
    cpu_model: str = field(
        default=None,
        metadata={"doc": "The model name of the CPU used.", "name": "CPU Model"},
    )
    sbios: str = field(
        default=None,
        metadata={
            "doc": "The system management bios version and vendor.",
            "name": "SBIOS",
        },
    )
    linux_distro: str = field(
        default=None,
        metadata={
            "doc": "The Linux distribution installed on the machine.",
            "name": "Linux Distribution",
        },
    )
    linux_kernel_version: str = field(
        default=None,
        metadata={
            "doc": "The Linux kernel version running on the machine.",
            "name": "Linux Kernel Version",
        },
    )
    amd_gpu_kernel_version: str = field(
        default=None,
        metadata={
            "doc": (
                "[RESERVED] The version of the AMDGPU driver installed on the machine. "
                "Unimplemented."
            ),
            "name": "AMD GPU Kernel Version",
        },
    )
    cpu_memory: str = field(
        default=None,
        metadata={
            "doc": "The total amount of memory available to the CPU.",
            "unit": "KB",
            "name": "CPU Memory",
        },
    )
    gpu_memory: str = field(
        default=None,
        metadata={
            "doc": (
                "[RESERVED] The total amount of memory available to accelerators/GPUs "
                "in the system. Unimplemented."
            ),
            "unit": "KB",
            "name": "GPU Memory",
        },
    )
    rocm_version: str = field(
        default=None,
        metadata={
            "doc": "The ROCm version used during data-collection.",
            "name": "ROCm Version",
        },
    )
    vbios: str = field(
        default=None,
        metadata={
            "doc": "The version of the accelerators/GPUs video bios in the system.",
            "name": "VBIOS",
        },
    )
    compute_partition: str = field(
        default=None,
        metadata={
            "doc": (
                "The compute partitioning mode active on the accelerators/GPUs in the "
                "system (MI300 only)."
            ),
            "name": "Compute Partition",
        },
    )
    memory_partition: str = field(
        default=None,
        metadata={
            "doc": (
                "The memory partitioning mode active on the accelerators/GPUs in the "
                "system (MI300 only)."
            ),
            "name": "Memory Partition",
        },
    )

    ##########################################
    ## B. SoC Specs
    ##########################################
    gpu_series: str = field(
        default=None,
        metadata={
            "doc": "The series of the accelerators/GPUs in the system.",
            "name": "GPU Series",
        },
    )
    gpu_model: str = field(
        default=None,
        metadata={
            "doc": "The product name of the accelerators/GPUs in the system.",
            "name": "GPU Model",
        },
    )
    gpu_arch: str = field(
        default=None,
        metadata={
            "doc": "The architecture name of the accelerators/GPUs in the system,\n"
            "as used by (e.g.,) the AMDGPU backed of LLVM.",
            "name": "GPU Arch",
        },
    )
    gpu_chip_id: str = field(
        default=None,
        metadata={
            "doc": "The Chip ID of the accelerators/GPUs in the system.",
            "name": "Chip ID",
            "optional": True,
        },
    )
    gpu_l1: str = field(
        default=None,
        metadata={
            "doc": (
                "The size of the vL1D cache (per compute-unit) on the "
                "accelerators/GPUs."
            ),
            "name": "GPU L1",
            "unit": "KiB",
        },
    )
    gpu_l2: str = field(
        default=None,
        metadata={
            "doc": (
                "The size of the vL1D cache (per compute-unit) on the "
                "accelerators/GPUs."
            ),
            "name": "GPU L2",
            "unit": "KiB",
        },
    )
    cu_per_gpu: str = field(
        default=None,
        metadata={
            "doc": (
                "The total number of compute units per accelerator/GPU in the system. "
                "On systems with configurable partitioning, (e.g., MI300) this is "
                "the total number of compute units in a partition."
            ),
            "name": "CU per GPU",
        },
    )
    simd_per_cu: str = field(
        default=None,
        metadata={
            "doc": (
                "The number of SIMD processors in a compute unit for the "
                "accelerators/GPUs in the system."
            ),
            "name": "SIMD per CU",
        },
    )
    se_per_gpu: str = field(
        default=None,
        metadata={
            "doc": (
                "The number of shader engines on the accelerators/GPUs in the system. "
                "On systems with configurable partitioning, (e.g., MI300) this is "
                "the total number of shader engines in a partition."
            ),
            "name": "SE per GPU",
        },
    )
    wave_size: str = field(
        default=None,
        metadata={
            "doc": (
                "The number work-items in a wavefront on the accelerators/GPUs in "
                "the system."
            ),
            "name": "Wave Size",
        },
    )
    workgroup_max_size: str = field(
        default=None,
        metadata={
            "doc": (
                "The maximum number of work-items in a workgroup on the "
                "accelerators/GPUs in the system."
            ),
            "name": "Workgroup Max Size",
        },
    )
    max_waves_per_cu: str = field(
        default=None,
        metadata={
            "doc": (
                "The maximum number of wavefronts that can be resident on a "
                "compute unit on the accelerators/GPUs in the system"
            ),
            "name": "Max Waves per CU",
        },
    )
    max_sclk: str = field(
        default=None,
        metadata={
            "doc": (
                "The maximum engine (compute-unit) clock rate of the "
                "accelerators/GPUs in the system."
            ),
            "name": "Max SCLK",
            "unit": "MHz",
        },
    )
    max_mclk: str = field(
        default=None,
        metadata={
            "doc": (
                "The maximum memory clock rate of the accelerators/GPUs in the system."
            ),
            "name": "Max MCLK",
            "unit": "MHz",
        },
    )
    cur_sclk: str = field(
        default=None,
        metadata={
            "doc": (
                "[RESERVED] The current engine (compute unit) clock rate of the "
                "accelerators/GPUs in the system. Unused."
            ),
            "name": "Cur SCLK",
            "unit": "MHz",
        },
    )
    cur_mclk: str = field(
        default=None,
        metadata={
            "doc": (
                "[RESERVED] The current memory clock rate of the accelerators/GPUs "
                "in the system. Unused."
            ),
            "name": "Cur MCLK",
            "unit": "MHz",
        },
    )
    _l2_banks: str = None  # NB: Only used in flatten_tcc_info_across_hbm_stacks()
    total_l2_chan: str = field(
        default=None,
        metadata={
            "doc": (
                "The maximum number of L2 cache channels on the accelerators/GPUs "
                "in the system. On systems with configurable partitioning, "
                "(e.g., MI300) this is the total number of L2 cache channels "
                "in a partition."
            ),
            "name": "Total L2 Channels",
        },
    )
    lds_banks_per_cu: str = field(
        default=None,
        metadata={
            "doc": (
                "The number of banks in the LDS for a compute unit on the "
                "accelerators/GPUs in the system."
            ),
            "name": "LDS Banks per CU",
        },
    )
    sqc_per_gpu: str = field(
        default=None,
        metadata={
            "doc": (
                "The number of L1I/sL1D caches on the accelerators/GPUs in the "
                "system. On systems with configurable partitioning, (e.g., MI300) "
                "this is the total number of L1I/sL1D caches in a partition."
            ),
            "name": "SQC per GPU",
        },
    )
    pipes_per_gpu: str = field(
        default=None,
        metadata={
            "doc": (
                "The number of scheduler-pipes on the accelerators/GPUs in the system."
            ),
            "name": "Pipes per GPU",
        },
    )
    num_xcd: str = field(
        default=None,
        metadata={
            "doc": (
                "The total number of accelerator complex dies in a compute partition "
                "on the accelerators/GPUs in the system. For accelerators without "
                "partitioning (i.e., pre-MI300), this is considered to be one."
            ),
            "name": "Num XCDs",
            "unit": "XCDs",
        },
    )
    num_hbm_channels: str = field(
        default=None,
        metadata={
            "doc": "Number of HBM channels",
            "name": "HBM channels",
        },
    )

    def get_hbm_channels(self):
        if self.memory_partition.lower().startswith("nps"):
            hbmchannels = 128
            if self.memory_partition.lower() == "nps4":
                hbmchannels /= 4
            elif self.memory_partition.lower() == "nps8":
                hbmchannels /= 8
            return hbmchannels
        else:
            return int(self.total_l2_chan)

    def get_class_members(self):
        all_populated = True
        data = {}
        # dataclass uses an OrderedDict for member variables, ensuring order consistency
        for class_field in fields(self):
            name = class_field.name
            if not name.startswith("_"):
                value = getattr(self, name)
                if value is None:
                    # check if we've marked it optional
                    if (
                        class_field.metadata
                        and "optional" in class_field.metadata
                        and class_field.metadata["optional"]
                    ):
                        pass
                    else:
                        console_warning(
                            f"Incomplete class definition for {self.gpu_arch}. "
                            f"Expecting populated {name} but detected None."
                        )
                        all_populated = False
                data[name] = value

        if not all_populated:
            console_warning("Missing specs fields for %s" % self.gpu_arch)
        return pd.DataFrame(data, index=[0])

    def __repr__(self):
        topstr = (
            "Machine Specifications: describing the state of the machine that "
            "ROCm Compute Profiler data was collected on.\n"
        )
        data = []
        for class_field in fields(self):
            name = class_field.name
            if not name.startswith("_"):
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
                            console_error(f"Unknown out of table printing field: {name}")
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


def get_rocm_ver():
    rocm_found = False
    for itr in VERSION_LOC:
        _path = str(path(os.getenv("ROCM_PATH", "/opt/rocm")).joinpath(".info", itr))
        if path(_path).exists():
            rocm_ver = path(_path).read_text()
            rocm_found = True
        break
    if not rocm_found:
        # check if ROCM_VER is supplied externally
        ROCM_VER_USER = os.getenv("ROCM_VER")
        if ROCM_VER_USER is not None:
            console_log(
                "profiling",
                "Overriding missing ROCm version detection with ROCM_VER = %s"
                % ROCM_VER_USER,
            )
            rocm_ver = ROCM_VER_USER
        else:
            _rocm_path = os.getenv("ROCM_PATH", "/opt/rocm")
            console_warning("Unable to detect a complete local ROCm installation.")
            console_warning(
                "The expected %s/.info/ versioning directory is missing." % _rocm_path
            )
            console_error("Ensure you have valid ROCm installation.")
    return rocm_ver


def run(cmd, exit_on_error=False):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as e:
        console_error(
            (
                f"Unable to parse specs. Can't find ROCm asset: {e.filename}\n"
                "Try passing a path to an existing workload results in 'analyze' mode."
            )
        )

    if exit_on_error:
        if cmd[0] == "amd-smi":
            if p.returncode != 2 and p.returncode != 0:
                console_error("No GPU detected. Unable to load amd-smi")
        elif p.returncode != 0:
            console_error("Command [%s] failed with non-zero exit code" % cmd)
    return p.stdout.decode("utf-8")


def search(pattern, string):
    m = re.search(pattern, string, re.MULTILINE)
    if m is not None:
        return m.group(1)
    return None


def total_sqc(archname, numCUs, numSEs):
    cu_per_se = float(numCUs) / float(numSEs)
    sq_per_se = cu_per_se / 2
    if archname.lower() in ["mi50", "mi100"]:
        sq_per_se = cu_per_se / 3
    sq_per_se = ceil(sq_per_se)
    return int(sq_per_se) * int(numSEs)


def total_l2_banks(gpu_arch, gpu_model, L2Banks, compute_partition):
    xcd_count = mi_gpu_specs.get_num_xcds(gpu_arch, gpu_model, compute_partition)

    # TODO: MachineSpecs and OmniSoC mspec should converge...
    if L2Banks is not None and xcd_count is not None:
        return int(L2Banks) * int(xcd_count)
    return None


if __name__ == "__main__":
    print(generate_machine_specs())
