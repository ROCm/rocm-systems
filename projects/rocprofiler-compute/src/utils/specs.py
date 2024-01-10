"""Get host/gpu specs."""

##############################################################################bl
# MIT License
#
# Copyright (c) 2021 - 2023 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################el

import os
import re
import sys
import socket
import subprocess

from dataclasses import dataclass
from pathlib import Path as path
from textwrap import dedent

@dataclass
class MachineSpecs:
    hostname: str
    CPU: str
    sbios: str
    kernel_version: str
    ram: str
    distro: str
    rocm_version: str
    GPU: str
    arch: str
    vbios: str
    L1: str
    L2: str
    CU: str
    SIMD: str
    SE: str
    wave_size: str
    workgroup_max_size: str
    max_sclk: str
    cur_sclk: str
    cur_mclk: str
    max_waves_per_cu: str
    L2Banks: str
    LDSBanks: str
    numSQC: str
    hbmBW: str
    compute_partition: str
    memory_partition: str

    def __str__(self):
        return dedent(
            f"""\
        Host info:
            hostname:       {self.hostname}
            CPU:            {self.CPU}
            sbios:          {self.sbios}
            ram:            {self.ram}
            distro:         {self.distro}
            kernel_version: {self.kernel_version}
            rocm_version:   {self.rocm_version}
        Device info:
            GPU:                {self.GPU}
            arch:               {self.arch}
            vbios:              {self.vbios}
            L1:                 {self.L1} KB
            L2:                 {self.L2} KB
            max_sclk:           {self.max_sclk} MHz
            cur_sclk:           {self.cur_sclk} MHz
            cur_mclk:           {self.cur_mclk} MHz
            CU:                 {self.CU}
            SIMD:               {self.SIMD}
            SE:                 {self.SE}
            wave_size:          {self.wave_size}
            workgroup_max_size: {self.workgroup_max_size}
            max_waves_per_cu:   {self.max_waves_per_cu}
            L2Banks:            {self.L2Banks}
            LDSBanks:           {self.LDSBanks}
            numSQC:             {self.numSQC}
            hbmBW:              {self.hbmBW} MB/s
            compute_partition:  {self.compute_partition}
            memory_partition:   {self.memory_partition}
        """
        )


def gpuinfo():
    # Local var only for rocminfo searching
    gpu_list = {"gfx906", "gfx908", "gfx90a", "gfx940", "gfx941", "gfx942"}

    # Fixme: find better way to differentiate cards, GPU vs APU, etc.
    rocminfo_full = run(["rocminfo"])
    rocminfo = rocminfo_full.split("\n")

    for idx1, linetext in enumerate(rocminfo):
        gpu_arch = search(r"^\s*Name\s*:\s+ ([a-zA-Z0-9]+)\s*$", linetext)
        if gpu_arch in gpu_list:
            break
        if str(gpu_arch) in gpu_list:
            gpu_arch = str(gpu_arch)
            break
    if not gpu_arch in gpu_list:
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    L1, L2 = "", ""
    for idx2, linetext in enumerate(rocminfo[idx1 + 1 :]):
        key = search(r"^\s*L1:\s+ ([a-zA-Z0-9]+)\s*", linetext)
        if key != None:
            L1 = key
            continue

        key = search(r"^\s*L2:\s+ ([a-zA-Z0-9]+)\s*", linetext)
        if key != None:
            L2 = key
            continue

        key = search(r"^\s*Max Clock Freq\. \(MHz\):\s+([0-9]+)", linetext)
        if key != None:
            max_sclk = key
            continue

        key = search(r"^\s*Compute Unit:\s+ ([a-zA-Z0-9]+)\s*", linetext)
        if key != None:
            num_CU = key
            continue

        key = search(r"^\s*SIMDs per CU:\s+ ([a-zA-Z0-9]+)\s*", linetext)
        if key != None:
            num_SIMD = key
            continue

        key = search(r"^\s*Shader Engines:\s+ ([a-zA-Z0-9]+)\s*", linetext)
        if key != None:
            num_SE = key
            continue

        key = search(r"^\s*Wavefront Size:\s+ ([a-zA-Z0-9]+)\s*", linetext)
        if key != None:
            wave_size = key
            continue

        key = search(r"^\s*Workgroup Max Size:\s+ ([a-zA-Z0-9]+)\s*", linetext)
        if key != None:
            grp_size = key
            continue

        key = search(r"^\s*Max Waves Per CU:\s+ ([a-zA-Z0-9]+)\s*", linetext)
        if key != None:
            max_waves_per_cu = key
            break

    gpu_name = ""
    L2Banks = ""
    LDSBanks = "32"
    numSQC = ""

    if gpu_arch == "gfx906":
        gpu_name = "MI50"
        L2Banks = "16"
        numSQC = str(int(num_CU) // 4)
    elif gpu_arch == "gfx908":
        gpu_name = "MI100"
        L2Banks = "32"
        numSQC = "48"
    elif gpu_arch == "gfx90a":
        L2Banks = "32"
        gpu_name = "MI200"
        numSQC = "56"
    elif gpu_arch == "gfx940":
        gpu_name = "MI300A_A0"
        L2Banks = "16"
        numSQC = "56"
    elif gpu_arch == "gfx941":
        gpu_name = "MI300X_A0"
        L2Banks = "16"
        numSQC = "56"
    elif (gpu_arch == "gfx942") and ("MI300A" in rocminfo_full):
        gpu_name = "MI300A_A1"
        L2Banks = "16"
        numSQC = "56"
    elif (gpu_arch == "gfx942") and ("MI300A" not in rocminfo_full):
        gpu_name = "MI300X_A1"
        L2Banks = "16"
        numSQC = "56"
    else:
        print("\nInvalid SoC")
        sys.exit(0)

    compute_partition = ""
    memory_partition = ""
    return (
        gpu_name,
        gpu_arch,
        L1,
        L2,
        max_sclk,
        num_CU,
        num_SIMD,
        num_SE,
        wave_size,
        grp_size,
        max_waves_per_cu,
        L2Banks,
        LDSBanks,
        numSQC,
        compute_partition,
        memory_partition,
    )


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cmd[0] == "rocm-smi" and p.returncode == 8:
        print("ERROR: No GPU detected. Unable to load rocm-smi")
        sys.exit(1)
    return p.stdout.decode("ascii")


def search(pattern, string):
    m = re.search(pattern, string, re.MULTILINE)
    if m is not None:
        return m.group(1)
    return None


def get_machine_specs(devicenum):
    cpuinfo = path("/proc/cpuinfo").read_text()
    meminfo = path("/proc/meminfo").read_text()
    version = path("/proc/version").read_text()
    os_release = path("/etc/os-release").read_text()

    version_loc = [
        "version",
        "version-dev",
        "version-hip-libraries",
        "version-hiprt",
        "version-hiprt-devel",
        "version-hip-sdk",
        "version-libs",
        "version-utils",
    ]

    rocmFound = False
    for itr in version_loc:
        _path = os.path.join(os.getenv("ROCM_PATH", "/opt/rocm"), ".info", itr)
        if os.path.exists(_path):
            rocm_ver = path(_path).read_text()
            rocmFound = True
            break

    if not rocmFound:
        # check if ROCM_VER is supplied externally
        ROCM_VER_USER = os.getenv("ROCM_VER")
        if ROCM_VER_USER is not None:
            print(
                "Overriding missing ROCm version detection with ROCM_VER = %s"
                % ROCM_VER_USER
            )
            rocm_ver = ROCM_VER_USER
        else:
            _rocm_path = os.getenv("ROCM_PATH", "/opt/rocm")
            print("Error: Unable to detect a complete local ROCm installation.")
            print(
                "\nThe expected %s/.info/ versioning directory is missing. Please"
                % _rocm_path
            )
            print("ensure you have valid ROCm installation.")
            sys.exit(1)

    (
        gpu_name,
        gpu_arch,
        L1,
        L2,
        max_sclk,
        num_CU,
        num_SIMD,
        num_SE,
        wave_size,
        grp_size,
        max_waves_per_cu,
        L2Banks,
        LDSBanks,
        numSQC,
        compute_partition,
        memory_partition,
    ) = gpuinfo()

    rocm_smi = run(["rocm-smi"])

    device = rf"^\s*{devicenum}(.*)"

    hostname = socket.gethostname()
    sbios = (
        path("/sys/class/dmi/id/bios_vendor").read_text().strip()
        + path("/sys/class/dmi/id/bios_version").read_text().strip()
    )
    CPU = search(r"^model name\s*: (.*?)$", cpuinfo)
    kernel_version = search(r"version (\S*)", version)
    ram = search(r"MemTotal:\s*(\S*)", meminfo)
    distro = search(r'PRETTY_NAME="(.*?)"', os_release)
    if distro is None:
        distro = ""

    rocm_version = rocm_ver.strip()

    freq = search(device, rocm_smi).split()
    cur_sclk = search(r"([0-9]+)", freq[2])
    if cur_sclk is None:
        cur_sclk = ""

    cur_mclk = search(r"([0-9]+)", freq[3])
    if cur_mclk is None:
        cur_mclk = 0

    # FIXME with device
    vbios = search(r"VBIOS version: (.*?)$", run(["rocm-smi", "-v"]))

    # FIXME with spec
    hbmBW = str(int(cur_mclk) / 1000 * 4096 / 8 * 2)

    compute_partition = search(
        r"Compute Partition:\s*(\w+)", run(["rocm-smi", "--showcomputepartition"])
    )
    if compute_partition == None:
        compute_partition = "NA"

    memory_partition = search(
        r"Memory Partition:\s*(\w+)", run(["rocm-smi", "--showmemorypartition"])
    )
    if memory_partition == None:
        memory_partition = "NA"

    return MachineSpecs(
        hostname,
        CPU,
        sbios,
        kernel_version,
        ram,
        distro,
        rocm_version,
        gpu_name,
        gpu_arch,
        vbios,
        L1,
        L2,
        num_CU,
        num_SIMD,
        num_SE,
        wave_size,
        grp_size,
        max_sclk,
        cur_sclk,
        cur_mclk,
        max_waves_per_cu,
        L2Banks,
        LDSBanks,
        numSQC,
        hbmBW,
        compute_partition,
        memory_partition,
    )


if __name__ == "__main__":
    print(get_machine_specs(0))
