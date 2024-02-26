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
import pandas as pd

from datetime import datetime
from math import ceil
from dataclasses import dataclass
from pathlib import Path as path
from textwrap import dedent
from utils.utils import error, get_hbm_stack_num

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

@dataclass
class MachineSpecs:
    def __init__(self, args, sysinfo=None):
        if not sysinfo is None:
            self.gpu_arch = sysinfo.iloc[0]["gpu_arch"]
            return
        # read timestamp info
        now = datetime.now()
        local_now = now.astimezone()
        local_tz = local_now.tzinfo
        local_tzname = local_tz.tzname(local_now)
        self.timestamp = now.strftime("%c") + " (" + local_tzname + ")"

        # read rocminfo
        rocminfo_full = run(["rocminfo"])
        self._rocminfo = rocminfo_full.split("\n")

        ##########################################
        ## A. Machine Specs
        ##########################################
        cpuinfo = path("/proc/cpuinfo").read_text()
        meminfo = path("/proc/meminfo").read_text()
        version = path("/proc/version").read_text()
        os_release = path("/etc/os-release").read_text()
        
        self.hostname: str = socket.gethostname()
        self.cpu_model: str = search(r"^model name\s*: (.*?)$", cpuinfo)
        self.sbios: str = (
            path("/sys/class/dmi/id/bios_vendor").read_text().strip()
            + path("/sys/class/dmi/id/bios_version").read_text().strip()
        )
        self.linux_kernel_version: str = search(r"version (\S*)", version)
        self.amd_gpu_kernel_version: str = None
        if self.amd_gpu_kernel_version is None: #TODO: Extract amdgpu kernel version
            self.amd_gpu_kernel_version = ""
        self.cpu_memory: str = search(r"MemTotal:\s*(\S*)", meminfo)
        self.gpu_memory: str = None #TODO: Extract total gpu memory
        if self.gpu_memory is None:
            self.gpu_memory = ""
        self.linux_distro: str = search(r'PRETTY_NAME="(.*?)"', os_release)
        if self.linux_distro is None:
            self.linux_distro = ""
        self.rocm_version: str = get_rocm_ver().strip()
        #FIXME: use device
        self.vbios: str = search(
            r"VBIOS version: (.*?)$", run(["rocm-smi", "-v"], exit_on_error=True)
        )
        self.compute_partition: str = search(
            r"Compute Partition:\s*(\w+)", run(["rocm-smi", "--showcomputepartition"])
        )
        if self.compute_partition is None:
            self.compute_partition = "NA"
        self.memory_partition: str = search(
            r"Memory Partition:\s*(\w+)", run(["rocm-smi", "--showmemorypartition"])
        )
        if self.memory_partition is None:
            self.memory_partition = "NA"
        
        ##########################################
        ## B. SoC Specs
        ##########################################
        self.gpu_arch: str = self.detect_arch()[0]
        self.gpu_l1: str = None
        self.gpu_l2: str = None
        self.cu_per_gpu: str = None
        self.simd_per_cu: str = None
        self.se_per_gpu: str = None
        self.wave_size: str = None
        self.workgroup_max_size: str = None
        self.max_sclk: str = None
        self.max_mclk: str = None
        self.cur_sclk: str = None
        self.cur_mclk: str = None
        self.max_waves_per_cu: str = None
        self.gpu_model: str = None
        self._l2_banks: str = None #NB: This only used in flatten_tcc_info_across_hbm_stacks()
        self.lds_banks_per_cu: str = None 
        self.sqc_per_gpu: str = None
        self.pipes_per_gpu: str = None
        self.total_l2_chan: str = None
        self.hbm_bw: str = None
        self.num_xcd: str = None
        # Load above SoC specs via module import
        try:
            soc_module = importlib.import_module('omniperf_soc.soc_'+ self.gpu_arch)
        except ModuleNotFoundError as e:
            error("Arch %s marked as supported, but couldn't find class implementation %s." % (self.gpu_arch, e))
        soc_class = getattr(soc_module, self.gpu_arch+'_soc')
        self._rocminfo = self._rocminfo[self.detect_arch()[1] + 1 :] # update rocminfo for target section
        soc_obj = soc_class(args, self)
        # Update arch specific specs
        self.total_l2_chan: str = total_l2_banks(
            self.gpu_model, int(self._l2_banks), self.memory_partition
        )
        self.hbm_bw: str = str(int(self.max_mclk) / 1000 * 32 * self.get_hbm_channels())


    def detect_arch(self):
        from omniperf_base import SUPPORTED_ARCHS

        for idx1, linetext in enumerate(self._rocminfo):
            gpu_arch = search(r"^\s*Name\s*:\s+ ([a-zA-Z0-9]+)\s*$", linetext)
            if gpu_arch in SUPPORTED_ARCHS.keys():
                break
            if str(gpu_arch) in SUPPORTED_ARCHS.keys():
                gpu_arch = str(gpu_arch)
                break
        if not gpu_arch in SUPPORTED_ARCHS.keys():
            error("[profiling] Cannot find a supported arch in rocminfo")
        else:
            return (gpu_arch, idx1)
        
    def get_hbm_channels(self):
        hbmchannels = int(self.total_l2_chan)
        if (
            self.gpu_model.lower() == "mi300a_a0"
            or self.gpu_model.lower() == "mi300a_a1"
        ) and self.memory_partition.lower() == "nps1":
            # we have an extra 32 channels for the CCD
            hbmchannels += 32
        return hbmchannels
    
    def get_class_members(self):
        all_populated = True
        data = {}
        # dataclass uses an OrderedDict for member variables, ensuring order consistency
        for attr_name in self.__dict__.keys():
            if not attr_name.startswith("_"):
                attr_value = getattr(self, attr_name)
                if attr_value is None:
                    #TODO: use proper logging function when that's merged
                    logging.warning(f"WARNING: Incomplete class definition for {self.gpu_arch}. Expecting populated {attr_name} but detected None.")
                    all_populated = False
                data[attr_name] = attr_value

        if not all_populated:
            error("Missing specs fields for %s" % self.gpu_arch)
        return pd.DataFrame(data, index=[0])
    
    
    def __str__(self):
        return dedent(
            f"""\
        Host info:
            Hostname:                   {self.hostname}
            CPU Model:                  {self.cpu_model}
            sbios:                      {self.sbios}
            CPU Memory:                 {self.cpu_memory}
            GPU Memory:                 {self.gpu_memory}
            Linux Distro:               {self.linux_distro}
            Linux Kernel Version:       {self.linux_kernel_version}
            AMD GPU Kernel Version:     {self.amd_gpu_kernel_version}
            ROCm Version:               {self.rocm_version}
        Device info:
            GPU Model:                  {self.gpu_model}
            GPU Arch:                   {self.gpu_arch}
            vbios:                      {self.vbios}
            GPU L1:                     {self.gpu_l1} KB
            GPU L2:                     {self.gpu_l2} KB
            Max SCLK:                   {self.max_sclk} MHz
            Max MCLK:                   {self.max_mclk} MHz
            Cur SCLK:                   {self.cur_sclk} MHz
            Cur MCLK:                   {self.cur_mclk} MHz
            CU per GPU:                 {self.cu_per_gpu}
            SIMD per CU:                {self.simd_per_cu}
            SE per GPU:                 {self.se_per_gpu}
            Wave Size:                  {self.wave_size}
            Workgroup Max Size:         {self.workgroup_max_size}
            Max Waves per CU:           {self.max_waves_per_cu}
            Total L2 Channels:          {self.total_l2_chan}
            LDS Banks per CU:           {self.lds_banks_per_cu}
            SQC per GPU:                {self.sqc_per_gpu}
            Pipes per GPU:              {self.pipes_per_gpu}
            HBM BW:                     {self.hbm_bw} MB/s
            Compute Partition:          {self.compute_partition}
            Memory Partition:           {self.memory_partition}
            Num XCDs:                   {self.num_xcd}
        """
        )


def get_rocm_ver():
    rocm_found = False
    for itr in VERSION_LOC:
        _path = os.path.join(os.getenv("ROCM_PATH", "/opt/rocm"), ".info", itr)
        if os.path.exists(_path):
            rocm_ver = path(_path).read_text()
            rocm_found = True
        break
    if not rocm_found:
        # check if ROCM_VER is supplied externally
        ROCM_VER_USER = os.getenv("ROCM_VER")
        if ROCM_VER_USER is not None:
            logging.info(
                "Overriding missing ROCm version detection with ROCM_VER = %s"
                % ROCM_VER_USER
            )
            rocm_ver = ROCM_VER_USER
        else:
            _rocm_path = os.getenv("ROCM_PATH", "/opt/rocm")
            error("Unable to detect a complete local ROCm installation.\nThe expected %s/.info/ versioning directory is missing. Please ensure you have valid ROCm installation." % _rocm_path)
    return rocm_ver

def run(cmd, exit_on_error=False):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if exit_on_error:
        if cmd[0] == "rocm-smi":
            if p.returncode != 2 and p.returncode != 0:
                logging.error("ERROR: No GPU detected. Unable to load rocm-smi")
                sys.exit(1)
        elif p.returncode != 0:
            logging.error("ERROR: command [%s] failed with non-zero exit code" % cmd)
            sys.exit(1)
    return p.stdout.decode("utf-8")


def search(pattern, string):
    m = re.search(pattern, string, re.MULTILINE)
    if m is not None:
        return m.group(1)
    return None


def total_l2_banks(archname, L2Banks, memory_partition):
    # Fixme: support all supported partitioning mode
    # Fixme: "name" is a bad name!
    totalL2Banks = L2Banks
    if (
        archname.lower() == "mi300a_a0"
        or archname.lower() == "mi300a_a1"
    ):
        totalL2Banks = L2Banks * get_hbm_stack_num(
            archname, memory_partition)
    elif (
        archname.lower() == "mi300x_a0"
        or archname.lower() == "mi300x_a1"
    ):
        totalL2Banks = L2Banks * get_hbm_stack_num(
            archname, memory_partition)
    return str(totalL2Banks)

def total_sqc(archname, numCUs, numSEs):
    cu_per_se = float(numCUs) / float(numSEs)
    sq_per_se = cu_per_se / 2
    if archname.lower() in ['mi50', 'mi100']:
        sq_per_se = cu_per_se / 3
    sq_per_se = ceil(sq_per_se)
    return int(sq_per_se) * int(numSEs)

def total_xcds(archname, compute_partition):
    # check MI300 has a valid compute partition
    mi300a_archs = ["mi300a_a0", "mi300a_a1"]
    mi300x_archs = ["mi300x_a0", "mi300x_a1"]
    if archname.lower() in mi300a_archs + mi300x_archs \
        and compute_partition == "NA":
            error("Invalid compute partition found for {}".format(archname))
    if archname.lower() not in mi300a_archs + mi300x_archs:
        return 1
    # from the whitepaper
    # https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
    if compute_partition.lower() == "spx":
        if archname.lower() in mi300a_archs:
            return 6
        if archname.lower() in mi300x_archs:
            return 8
    if compute_partition.lower() == "tpx":
        if archname.lower() in mi300a_archs:
            return 2
    if compute_partition.lower() == "dpx":
        if archname.lower() in mi300x_archs:
            return 4
    if compute_partition.lower() == "qpx":
        if archname.lower() in mi300x_archs:
            return 2
    if compute_partition.lower() == "cpx":
        if archname.lower() in mi300x_archs:
            return 2
    error("Unknown compute partition / arch found for {} / {}".format(
        compute_partition, archname))


if __name__ == "__main__":
    print(MachineSpecs())
