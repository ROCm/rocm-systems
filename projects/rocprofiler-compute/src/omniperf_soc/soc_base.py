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

from abc import ABC, abstractmethod
import logging
import os
import math
import shutil
import glob
import re
import numpy as np
from utils.utils import demarcate
from pathlib import Path

class OmniSoC_Base():
    def __init__(self,args):
        self.__args = args
        self.__soc = None
        self.__perfmon_dir = None
        self.__perfmon_config = {} # Per IP block max number of simulutaneous counters. GFX IP Blocks
        self.__soc_params = {} # SoC specifications
        if self.__args.path == os.path.join(os.getcwd(), "workloads"):
            self.__workload_dir = os.path.join(self.__args.path, self.__args.name, self.__args.target)
        else:
            self.__workload_dir = self.__args.path
    
    def __hash__(self):
        return hash(self.__soc)
    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.__soc == other.get_soc()

    def set_perfmon_dir(self, path:str):
        self.__perfmon_dir = path
    def set_perfmon_config(self, config: dict):
        self.__perfmon_config = config
    def set_soc_param(self, param: dict):
        self.__soc_params = param
    def get_workload_perfmon_dir(self):
        return str(Path(self.__perfmon_dir).parent.absolute())
    def get_soc_param(self):
        return self.__soc_params
    def set_soc(self, soc: str):
        self.__soc = soc
    def get_soc(self):
        return self.__soc
    def get_args(self):
        return self.__args
    
    @demarcate
    def get_profiler_options(self):
        """Fetch any SoC specific arguments required by the profiler
        """
        # assume no SoC specific options and return empty list by default
        return []
    
    @demarcate
    def perfmon_filter(self, roofline_perfmon_only: bool):
        """Filter default performance counter set based on user arguments
        """
        if roofline_perfmon_only and os.path.isfile(os.path.join(self.get_args().path, "pmc_perf.csv")):
            return
        workload_perfmon_dir = self.__workload_dir + "/perfmon"

        # Initialize directories
        if not os.path.isdir(self.__workload_dir):
            os.makedirs(self.__workload_dir)
        else:
            shutil.rmtree(self.__workload_dir)

        os.makedirs(workload_perfmon_dir)

        if not roofline_perfmon_only:
            ref_pmc_files_list = glob.glob(self.__perfmon_dir + "/" + "pmc_*perf*.txt")
            ref_pmc_files_list += glob.glob(self.__perfmon_dir + "/" + self.__soc + "/pmc_*_perf*.txt")

            # Perfmon list filtering
            if self.__args.ipblocks != None:
                for i in range(len(self.__args.ipblocks)):
                    self.__args.ipblocks[i] = self.__args.ipblocks[i].lower()
                mpattern = "pmc_([a-zA-Z0-9_]+)_perf*"

                pmc_files_list = []
                for fname in ref_pmc_files_list:
                    fbase = os.path.splitext(os.path.basename(fname))[0]
                    ip = re.match(mpattern, fbase).group(1)
                    if ip in self.__args.ipblocks:
                        pmc_files_list.append(fname)
                        logging.info("fname: " + fbase + ": Added")
                    else:
                        logging.info("fname: " + fbase + ": Skipped")

            else:
                # default: take all perfmons
                pmc_files_list = ref_pmc_files_list
        else:
            ref_pmc_files_list = glob.glob(self.__perfmon_dir + "/" + "pmc_roof_perf.txt")
            pmc_files_list = ref_pmc_files_list


        # Coalesce and writeback workload specific perfmon
        pmc_list = perfmon_coalesce(pmc_files_list, self.__perfmon_config, self.__workload_dir)
        perfmon_emit(pmc_list, self.__perfmon_config, self.__workload_dir)

    #----------------------------------------------------
    # Required methods to be implemented by child classes
    #----------------------------------------------------
    @abstractmethod
    def profiling_setup(self):
        """Perform any SoC-specific setup prior to profiling.
        """
        logging.debug("[profiling] perform SoC profiling setup for %s" % self.__soc)


    @abstractmethod
    def post_profiling(self):
        """Perform any SoC-specific post profiling activities.
        """
        logging.debug("[profiling] perform SoC post processing for %s" % self.__soc)

    @abstractmethod
    def analysis_setup(self):
        """Perform any SoC-specific setup prior to analysis.
        """
        logging.debug("[analysis] perform SoC analysis setup for %s" % self.__soc)
        

@demarcate
def perfmon_coalesce(pmc_files_list, perfmon_config, workload_dir):
    """Sort and bucket all related performance counters to minimize required application passes
    """
    workload_perfmon_dir = workload_dir + "/perfmon"

    # match pattern for pmc counters
    mpattern = r"^pmc:(.*)"
    pmc_list = dict(
        [
            ("SQ", []),
            ("GRBM", []),
            ("TCP", []),
            ("TA", []),
            ("TD", []),
            ("TCC", []),
            ("SPI", []),
            ("CPC", []),
            ("CPF", []),
            ("GDS", []),
            ("TCC2", {}),  # per-channel TCC perfmon
        ]
    )
    for ch in range(perfmon_config["TCC_channels"]):
        pmc_list["TCC2"][str(ch)] = []

    # Extract all PMC counters and store in separate buckets
    for fname in pmc_files_list:
        lines = open(fname, "r").read().splitlines()

        for line in lines:
            # Strip all comements, skip empty lines
            stext = line.split("#")[0].strip()
            if not stext:
                continue

            # all pmc counters start with  "pmc:"
            m = re.match(mpattern, stext)
            if m is None:
                continue

            # we have found all the counters, store them in buckets
            counters = m.group(1).split()

            # Utilitze helper function once a list of counters has be extracted
            save_file = True
            pmc_list = update_pmc_bucket(
                counters, save_file, perfmon_config, pmc_list, stext, workload_perfmon_dir
            )

    # add a timestamp file
    fd = open(workload_perfmon_dir + "/timestamps.txt", "w")
    fd.write("pmc:\n\n")
    fd.write("gpu:\n")
    fd.write("range:\n")
    fd.write("kernel:\n")
    fd.close()

    # sort the per channel counter, so that same counter in all channels can be aligned
    for ch in range(perfmon_config["TCC_channels"]):
        pmc_list["TCC2"][str(ch)].sort()

    return pmc_list

@demarcate
def update_pmc_bucket(
    counters, save_file, perfmon_config, pmc_list=None, stext=None, workload_perfmon_dir=None
):
    # Verify inputs.
    # If save_file is True, we're being called internally, from perfmon_coalesce
    # Else we're being called externally, from rocomni
    detected_external_call = False
    if save_file and (stext is None or workload_perfmon_dir is None):
        raise ValueError(
            "stext and workload_perfmon_dir must be specified if save_file is True"
        )
    if pmc_list is None:
        detected_external_call = True
        pmc_list = dict(
            [
                ("SQ", []),
                ("GRBM", []),
                ("TCP", []),
                ("TA", []),
                ("TD", []),
                ("TCC", []),
                ("SPI", []),
                ("CPC", []),
                ("CPF", []),
                ("GDS", []),
                ("TCC2", {}),  # per-channel TCC perfmon
            ]
        )
        for ch in range(perfmon_config["TCC_channels"]):
            pmc_list["TCC2"][str(ch)] = []

    if "SQ_ACCUM_PREV_HIRES" in counters and not detected_external_call:
        # save  all level counters separately
        nindex = counters.index("SQ_ACCUM_PREV_HIRES")
        level_counter = counters[nindex - 1]

        if save_file:
            # Save to level counter file, file name = level counter name
            fd = open(workload_perfmon_dir + "/" + level_counter + ".txt", "w")
            fd.write(stext + "\n\n")
            fd.write("gpu:\n")
            fd.write("range:\n")
            fd.write("kernel:\n")
            fd.close()

        return pmc_list

    # save normal pmc counters in matching buckets
    for counter in counters:
        IP_block = counter.split(sep="_")[0].upper()
        # SQC and SQ belong to the IP block, coalesce them
        if IP_block == "SQC":
            IP_block = "SQ"

        if IP_block != "TCC":
            # Insert unique pmc counters into its bucket
            if counter not in pmc_list[IP_block]:
                pmc_list[IP_block].append(counter)

        else:
            # TCC counters processing
            m = re.match(r"[\s\S]+\[(\d+)\]", counter)
            if m is None:
                # Aggregated TCC counters
                if counter not in pmc_list[IP_block]:
                    pmc_list[IP_block].append(counter)

            else:
                # TCC channel ID
                ch = m.group(1)

                # fake IP block for per channel TCC
                if str(ch) in pmc_list["TCC2"]:
                    # append unique counter into the channel
                    if counter not in pmc_list["TCC2"][str(ch)]:
                        pmc_list["TCC2"][str(ch)].append(counter)
                else:
                    # initial counter in this channel
                    pmc_list["TCC2"][str(ch)] = [counter]

    if detected_external_call:
        # sort the per channel counter, so that same counter in all channels can be aligned
        for ch in range(perfmon_config["TCC_channels"]):
            pmc_list["TCC2"][str(ch)].sort()
    return pmc_list

@demarcate
def perfmon_emit(pmc_list, perfmon_config, workload_dir=None):
    # Calculate the minimum number of iteration to save the pmc counters
    # non-TCC counters
    pmc_cnt = [
        len(pmc_list[key]) / perfmon_config[key]
        for key in pmc_list
        if key not in ["TCC", "TCC2"]
    ]

    # TCC counters
    tcc_channels = perfmon_config["TCC_channels"]

    tcc_cnt = len(pmc_list["TCC"]) / perfmon_config["TCC"]
    tcc2_cnt = (
        np.array([len(pmc_list["TCC2"][str(ch)]) for ch in range(tcc_channels)])
        / perfmon_config["TCC"]
    )

    # Total number iterations to write pmc: counters line, except TCC2
    niter = max(math.ceil(max(pmc_cnt)), math.ceil(tcc_cnt))

    # Emit PMC counters into pmc config file
    if workload_dir:
        workload_perfmon_dir = workload_dir + "/perfmon"
        fd = open(workload_perfmon_dir + "/pmc_perf.txt", "w")
    else:
        batches = []

    for iter in range(niter):
        # Prefix
        line = "pmc: "

        # Add all non-TCC counters
        for key in pmc_list:
            if key not in ["TCC", "TCC2"]:
                N = perfmon_config[key]
                ip_counters = pmc_list[key][iter * N : iter * N + N]
                if ip_counters:
                    line = line + " " + " ".join(ip_counters)

        # Add TCC counters
        N = perfmon_config["TCC"]
        tcc_counters = pmc_list["TCC"][iter * N : iter * N + N]

        # TCC aggregated counters
        line = line + " " + " ".join(tcc_counters)
        if workload_dir:
            fd.write(line + "\n")
        else:
            b = line.split()
            b.remove("pmc:")
            batches.append(b)

    # TCC2, handle TCC per channel counters separatly
    tcc2_index = 0
    niter = math.ceil(max(tcc2_cnt))
    for iter in range(niter):
        # Prefix
        line = "pmc: "
        
        N = perfmon_config["TCC"]
        # TCC per-channel counters
        tcc_counters = []
        for ch in range(perfmon_config["TCC_channels"]):
            tcc_counters = pmc_list["TCC2"][str(ch)][tcc2_index * N : tcc2_index * N + N]
        
        tcc2_index += 1

        # TCC2 aggregated counters
        line = line + " " + " ".join(tcc_counters)
        if workload_dir:
            fd.write(line + "\n")
        else:
            b = line.split()
            b.remove("pmc:")
            batches.append(b)

    if workload_dir:
        fd.write("\ngpu:\n")
        fd.write("range:\n")
        fd.write("kernel:\n")
        fd.close()
    else:
        return batches