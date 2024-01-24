"""Get host/gpu specs."""

##############################################################################bl
# MIT License
#
# Copyright (c) 2021 - 2024 Advanced Micro Devices, Inc. All Rights Reserved.
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
import importlib
import logging

from dataclasses import dataclass
from pathlib import Path as path
from textwrap import dedent
from utils.utils import error

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
    from omniperf_base import SUPPORTED_ARCHS

    gpu_info = {
        "gpu_name": None,
        "gpu_arch": None,
        "L1": None,
        "L2": None,
        "max_sclk": None,
        "num_CU": None,
        "num_SIMD": None,
        "num_SE": None,
        "wave_size": None,
        "grp_size": None,
        "max_waves_per_cu": None,
        "L2Banks": None,
        "LDSBanks": None,
        "numSQC": None,
        "compute_partition": None,
        "memory_partition": None,
    }

    # Fixme: find better way to differentiate cards, GPU vs APU, etc.
    rocminfo_full = run(["rocminfo"])
    rocminfo = rocminfo_full.split("\n")

    for idx1, linetext in enumerate(rocminfo):
        gpu_arch = search(r"^\s*Name\s*:\s+ ([a-zA-Z0-9]+)\s*$", linetext)
        if gpu_arch in SUPPORTED_ARCHS.keys():
            break
        if str(gpu_arch) in SUPPORTED_ARCHS.keys():
            gpu_arch = str(gpu_arch)
            break
    if not gpu_arch in SUPPORTED_ARCHS.keys():
        return gpu_info

    gpu_info['L1'], gpu_info['L1'] = "", ""
    for idx2, linetext in enumerate(rocminfo[idx1 + 1 :]):
        key = search(r"^\s*L1:\s+ ([a-zA-Z0-9]+)\s*", linetext)
        if key != None:
            gpu_info['L1'] = key
            continue

        key = search(r"^\s*L2:\s+ ([a-zA-Z0-9]+)\s*", linetext)
        if key != None:
            gpu_info['L2'] = key
            continue

        key = search(r"^\s*Max Clock Freq\. \(MHz\):\s+([0-9]+)", linetext)
        if key != None:
            gpu_info['max_sclk'] = key
            continue

        key = search(r"^\s*Compute Unit:\s+ ([a-zA-Z0-9]+)\s*", linetext)
        if key != None:
            gpu_info['num_CU'] = key
            continue

        key = search(r"^\s*SIMDs per CU:\s+ ([a-zA-Z0-9]+)\s*", linetext)
        if key != None:
            gpu_info['num_SIMD'] = key
            continue

        key = search(r"^\s*Shader Engines:\s+ ([a-zA-Z0-9]+)\s*", linetext)
        if key != None:
            gpu_info['num_SE'] = key
            continue

        key = search(r"^\s*Wavefront Size:\s+ ([a-zA-Z0-9]+)\s*", linetext)
        if key != None:
            gpu_info['wave_size'] = key
            continue

        key = search(r"^\s*Workgroup Max Size:\s+ ([a-zA-Z0-9]+)\s*", linetext)
        if key != None:
            gpu_info['grp_size'] = key
            continue

        key = search(r"^\s*Max Waves Per CU:\s+ ([a-zA-Z0-9]+)\s*", linetext)
        if key != None:
            gpu_info['max_waves_per_cu'] = key
            break

    try:
        soc_module = importlib.import_module('omniperf_soc.soc_'+gpu_arch)
    except ModuleNotFoundError as e:
        error("Arch %s marked as supported, but couldn't find class implementation %s." % (gpu_arch, e))
    
    # load arch specific info
    try:
        gpu_name = list(SUPPORTED_ARCHS[gpu_arch].keys())[0].upper()
        gpu_info['L2Banks'] = str(soc_module.SOC_PARAM['L2Banks'])
        gpu_info['numSQC'] = str(soc_module.SOC_PARAM['numSQC'])
        gpu_info['LDSBanks'] = str(soc_module.SOC_PARAM['LDSBanks'])
    except KeyError as e:
        error("Incomplete class definition for %s. Expected a field for %s in SOC_PARAM." % (gpu_arch, e))\
    
    # specify gpu name for gfx942 hardware
    if gpu_name == "MI300":
        gpu_name = list(SUPPORTED_ARCHS[gpu_arch].values())[0][0]
    if (gpu_info['gpu_arch'] == "gfx942") and ("MI300A" in rocminfo_full):
        gpu_name = "MI300A_A1"
    if (gpu_arch == "gfx942") and ("MI300A" not in rocminfo_full):
        gpu_name = "MI300X_A1"
    

    gpu_info['gpu_name'] = gpu_name
    gpu_info['gpu_arch'] = gpu_arch
    gpu_info['compute_partition'] = ""
    gpu_info['memory_partition'] = ""

    # verify all fields are filled
    for key, value in gpu_info.items():
        if value is None:
            logging.info("Warning: %s is missing from gpu_info dictionary." % key)

    return gpu_info


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cmd[0] == "rocm-smi" and p.returncode == 8:
        print("ERROR: No GPU detected. Unable to load rocm-smi")
        sys.exit(1)
    return p.stdout.decode("utf-8")


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

    gpu_info = gpuinfo()

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
        gpu_info['gpu_name'],
        gpu_info['gpu_arch'],
        vbios,
        gpu_info['L1'],
        gpu_info['L2'],
        gpu_info['num_CU'],
        gpu_info['num_SIMD'],
        gpu_info['num_SE'],
        gpu_info['wave_size'],
        gpu_info['grp_size'],
        gpu_info['max_sclk'],
        cur_sclk,
        cur_mclk,
        gpu_info['max_waves_per_cu'],
        gpu_info['L2Banks'],
        gpu_info['LDSBanks'],
        gpu_info['numSQC'],
        hbmBW,
        compute_partition,
        memory_partition,
    )


if __name__ == "__main__":
    print(get_machine_specs(0))
