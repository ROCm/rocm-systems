################################################################################
# Copyright (c) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
################################################################################

import sys, os, pathlib, shutil, subprocess, argparse, glob, re
import numpy as np
import math

prog = "omniperf"

# Per IP block max number of simulutaneous counters
# GFX IP Blocks
perfmon_config = {
    "vega10": {
        "SQ": 8,
        "TA": 2,
        "TD": 2,
        "TCP": 4,
        "TCC": 4,
        "CPC": 2,
        "CPF": 2,
        "SPI": 2,
        "GRBM": 2,
        "GDS": 4,
        "TCC_channels": 16,
    },
    "mi50": {
        "SQ": 8,
        "TA": 2,
        "TD": 2,
        "TCP": 4,
        "TCC": 4,
        "CPC": 2,
        "CPF": 2,
        "SPI": 2,
        "GRBM": 2,
        "GDS": 4,
        "TCC_channels": 16,
    },
    "mi100": {
        "SQ": 8,
        "TA": 2,
        "TD": 2,
        "TCP": 4,
        "TCC": 4,
        "CPC": 2,
        "CPF": 2,
        "SPI": 2,
        "GRBM": 2,
        "GDS": 4,
        "TCC_channels": 32,
    },
    "mi200": {
        "SQ": 8,
        "TA": 2,
        "TD": 2,
        "TCP": 4,
        "TCC": 4,
        "CPC": 2,
        "CPF": 2,
        "SPI": 2,
        "GRBM": 2,
        "GDS": 4,
        "TCC_channels": 32,
    },
}


def perfmon_coalesce(pmc_files_list, workload_dir, soc):
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
    for ch in range(perfmon_config[soc]["TCC_channels"]):
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
            if "SQ_ACCUM_PREV_HIRES" in counters:
                # save  all level counters separately

                nindex = counters.index("SQ_ACCUM_PREV_HIRES")
                level_counter = counters[nindex - 1]

                # Save to level counter file, file name = level counter name
                fd = open(workload_perfmon_dir + "/" + level_counter + ".txt", "w")
                fd.write(stext + "\n\n")
                fd.write("gpu:\n")
                fd.write("range:\n")
                fd.write("kernel:\n")
                fd.close()

                continue

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

    # sort the per channel counter, so that same counter in all channels can be aligned
    for ch in range(perfmon_config[soc]["TCC_channels"]):
        pmc_list["TCC2"][str(ch)].sort()

    return pmc_list


def perfmon_emit(pmc_list, workload_dir, soc):
    workload_perfmon_dir = workload_dir + "/perfmon"

    # Calculate the minimum number of iteration to save the pmc counters
    # non-TCC counters
    pmc_cnt = [
        len(pmc_list[key]) / perfmon_config[soc][key]
        for key in pmc_list
        if key not in ["TCC", "TCC2"]
    ]

    # TCC counters
    tcc_channels = perfmon_config[soc]["TCC_channels"]

    tcc_cnt = len(pmc_list["TCC"]) / perfmon_config[soc]["TCC"]
    tcc2_cnt = (
        np.array([len(pmc_list["TCC2"][str(ch)]) for ch in range(tcc_channels)])
        / perfmon_config[soc]["TCC"]
    )

    # Total number iterations to write pmc: counters line
    niter = max(math.ceil(max(pmc_cnt)), math.ceil(tcc_cnt) + math.ceil(max(tcc2_cnt)))

    # Emit PMC counters into pmc config file
    fd = open(workload_perfmon_dir + "/pmc_perf.txt", "w")

    tcc2_index = 0
    for iter in range(niter):
        # Prefix
        line = "pmc: "

        # Add all non-TCC counters
        for key in pmc_list:
            if key not in ["TCC", "TCC2"]:
                N = perfmon_config[soc][key]
                ip_counters = pmc_list[key][iter * N : iter * N + N]
                if ip_counters:
                    line = line + " " + " ".join(ip_counters)

        # Add TCC counters
        N = perfmon_config[soc]["TCC"]
        tcc_counters = pmc_list["TCC"][iter * N : iter * N + N]

        if not tcc_counters:
            # TCC per-channel counters
            for ch in range(perfmon_config[soc]["TCC_channels"]):
                tcc_counters += pmc_list["TCC2"][str(ch)][
                    tcc2_index * N : tcc2_index * N + N
                ]

            tcc2_index += 1

        # TCC aggregated counters
        line = line + " " + " ".join(tcc_counters)
        fd.write(line + "\n")

    fd.write("\ngpu:\n")
    fd.write("range:\n")
    fd.write("kernel:\n")
    fd.close()


def perfmon_filter(workload_dir, perfmon_dir, args):
    workload_perfmon_dir = workload_dir + "/perfmon"
    soc = args.target

    # Initialize directories
    # TODO: Modify this so that data is appended to previous?
    if not os.path.isdir(workload_dir):
        os.makedirs(workload_dir)
    else:
        shutil.rmtree(workload_dir)

    os.makedirs(workload_perfmon_dir)

    ref_pmc_files_list = glob.glob(perfmon_dir + "/" + "pmc_*perf*.txt")
    ref_pmc_files_list += glob.glob(perfmon_dir + "/" + soc + "/pmc_*_perf*.txt")

    # Perfmon list filtering
    if args.ipblocks != None:
        for i in range(len(args.ipblocks)):
            args.ipblocks[i] = args.ipblocks[i].lower()
        mpattern = "pmc_([a-zA-Z0-9_]+)_perf*"

        pmc_files_list = []
        for fname in ref_pmc_files_list:
            fbase = os.path.splitext(os.path.basename(fname))[0]
            ip = re.match(mpattern, fbase).group(1)
            if ip in args.ipblocks:
                pmc_files_list.append(fname)
                print("fname: " + fbase + ": Added")
            else:
                print("fname: " + fbase + ": Skipped")

    else:
        # default: take all perfmons
        pmc_files_list = ref_pmc_files_list

    # Coalesce and writeback workload specific perfmon
    pmc_list = perfmon_coalesce(pmc_files_list, workload_dir, soc)
    perfmon_emit(pmc_list, workload_dir, soc)


def pmc_filter(workload_dir, perfmon_dir, soc):
    workload_perfmon_dir = workload_dir + "/perfmon"

    if not os.path.isdir(workload_perfmon_dir):
        os.makedirs(workload_perfmon_dir)
    else:
        shutil.rmtree(workload_perfmon_dir)

    ref_pmc_files_list = glob.glob(perfmon_dir + "/roofline/" + "pmc_roof_perf.txt")
    # ref_pmc_files_list += glob.glob(perfmon_dir + "/" + soc + "/pmc_*_perf*.txt")

    pmc_files_list = ref_pmc_files_list

    # Coalesce and writeback workload specific perfmon
    pmc_list = perfmon_coalesce(pmc_files_list, workload_dir, soc)
    perfmon_emit(pmc_list, workload_dir, soc)
