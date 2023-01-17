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

from linecache import cache
import os
import sys
from pathlib import Path

import numpy
import matplotlib

try:

    import matplotlib.pyplot as plt
except ImportError:
    # other non-interactive options:
    #   cairo, pdf, pgf, ps, svg, template
    matplotlib.use("agg", force=True)
    import matplotlib.pyplot as plt

from matplotlib.pyplot import get, text
from math import log, pi, sqrt
import pandas as pd
import pylab

from dataclasses import dataclass
import csv


################################################
# Global vars
################################################

IMGNAME = "empirRoof"

L2_BANKS = 32  # default assuming mi200

XMIN = 0.01
XMAX = 1000

FONT_SIZE = 16
FONT_COLOR = "black"
FONT_WEIGHT = "bold"

SUPPORTED_SOC = ["mi200"]

################################################
# Helper funcs
################################################
@dataclass
class AI_Data:
    KernelName: str
    numCalls: float

    total_flops: float
    valu_flops: float
    mfma_flops_f16: float
    mfma_flops_bf16: float
    mfma_flops_f32: float
    mfma_flops_f64: float
    lds_data: float
    L1cache_data: float
    L2cache_data: float
    hbm_data: float

    totalDuration: float
    avgDuration: float


def get_font():
    return {
        "size": FONT_SIZE,
        "color": FONT_COLOR,
        "weight": FONT_WEIGHT,
        "family": "serif",
    }


def get_color(catagory):
    if catagory == "curr_ai_l1":
        return "green"
    elif catagory == "curr_ai_l2":
        return "blue"
    elif catagory == "curr_ai_hbm":
        return "red"
    else:
        raise RuntimeError("Invalid catagory passed to get_color()")


# -------------------------------------------------------------------------------------
#                           Plot BW at each cache level
# -------------------------------------------------------------------------------------
def plot_roof(inputs, roof_data):
    cacheHierarchy = []
    if inputs["mem"] == "ALL":
        cacheHierarchy += ["HBM", "L2", "L1", "LDS"]
    else:
        cacheHierarchy.append(inputs["mem"])
    targ_dtype = (
        "FP32"
        if float(roof_data["FP32Flops"][0]) > float(roof_data["FP64Flops"][0])
        else "FP64"
    )
    print("Dtype: ", targ_dtype)
    print(inputs["mem"])
    x1 = y1 = x2 = y2 = -1
    x1_mfma = y1_mfma = x2_mfma = y2_mfma = -1
    target_precision = targ_dtype[2:]

    peakOps = float(roof_data[targ_dtype + "Flops"][0])
    for i in range(0, len(cacheHierarchy)):
        # Plot BW line
        # print("Current cache level: {}".format(cacheHierarchy[i]))
        curr_bw = cacheHierarchy[i] + "Bw"
        peakBw = float(roof_data[curr_bw][0])

        peakMFMA = float(roof_data["MFMAF{}Flops".format(target_precision)][0])

        x1 = float(XMIN)
        y1 = float(XMIN) * peakBw

        x2 = peakOps / peakBw
        y2 = peakOps

        plt.plot([x1, x2], [y1, y2], color="magenta")
        # print("Mem Points: [{}, {}], [{}, {}]".format(x1, x2, y1, y2))

        # Plot MFMA lines (NOTE: Assuming MI200 soc)
        x1_mfma = peakOps / peakBw
        y1_mfma = peakOps

        x2_mfma = peakMFMA / peakBw
        y2_mfma = peakMFMA

        plt.plot([x1_mfma, x2_mfma], [y1_mfma, y2_mfma], color="blue")
        # print("Extend BW Points: [{}, {}], [{}, {}]".format(x1_mfma, x2_mfma, y1_mfma, y2_mfma))

        # These are the points to use:
        # print("x = [{}, {}]".format(x1,x2_mfma))
        # print("y = [{}, {}]".format(y1, y2_mfma))

        # Plot BW label
        x1log = log(x1) / log(10)
        x2log = log(x2) / log(10)
        y1log = log(y1) / log(10)
        y2log = log(y2) / log(10)
        x_text = 10 ** ((x1log + x2log) / 2)
        y_text = 10 ** ((y1log + y2log) / 2)

        fig = plt.gcf()
        size = fig.get_size_inches() * fig.dpi
        fig_x, fig_y = size

        # dx = log(x2) - log(x1)
        # dy = log(y2) - log(y1)
        # x_min, x_max = plt.xlim()
        # y_min, y_max = plt.ylim()
        # Dx = dx * fig_x / (log(x_max) - log(x_min))
        # Dy = dy * fig_y / (log(y_max) - log(y_min))
        # #fdiv = 0.7 #TODO: improve accuracy of text angle (tilt)
        # angle = (180.0 / pi) * numpy.arctan(Dy / Dx )#/fdiv)

        dx = abs(log(x2) - log(x1))
        dy = abs(log(y2) - log(y1))
        angle = (180.0 / pi) * numpy.arctan(dy / dx)
        # If user isn't zooming in, print bw labels normally
        if not inputs["axes"]:
            text(
                x_text,
                y_text,
                "{} vL1D GB/s".format(int(peakBw))
                if cacheHierarchy[i].upper() == "L1"
                else "{} {} GB/s".format(int(peakBw), cacheHierarchy[i].upper()),
                rotation=angle,
                rotation_mode="anchor",
                **get_font(),
            )
        else:
            # if bw line isn't being cut out then plot bw
            print("if {} < {}".format(inputs["axes"][0], 10**x2log))
            if inputs["axes"][0] < 10**x2log:
                text(
                    10**x2log,
                    10**y2log,
                    "{} {} GB/s".format(int(peakBw), cacheHierarchy[i].upper()),
                    rotation=angle,
                    rotation_mode="anchor",
                    **get_font(),
                )

    # -------------------------------------------------------------------------------------
    #                                     Plot computing roof
    # -------------------------------------------------------------------------------------
    # Plot FMA roof
    x0 = XMAX
    if x2 < x0:
        x0 = x2

    temp_label = "{} VALU GFLOP/sec".format(int(peakOps))
    plt.plot([x0, XMAX], [peakOps, peakOps], color="magenta")
    # print("FMA Points: [{}, {}], [{},{}]".format(x0, XMAX, peakOps, peakOps))
    text(
        XMAX if not inputs["axes"] else inputs["axes"][1],
        peakOps - 4000,  # should i keep this fixed at 4000?
        temp_label,
        horizontalalignment="right",
        **get_font(),
    )

    # Plot MFMA roof
    if x1_mfma != -1:  # assert that mfma has been assigned
        x0_mfma = XMAX
        if x2_mfma < x0_mfma:
            x0_mfma = x2_mfma

        peakMFMA = float(roof_data["MFMAF{}Flops".format(target_precision)][0])
        temp_label = "{} MFMA GFLOP/sec".format(int(peakMFMA))
        plt.plot([x0_mfma, XMAX], [peakMFMA, peakMFMA], color="blue")
        # print("MFMA Points: [{}, {}], [{},{}]".format(x0_mfma, XMAX, peakMFMA, peakMFMA))
        text(
            XMAX if not inputs["axes"] else inputs["axes"][1],
            peakMFMA + 1000,
            temp_label,
            horizontalalignment="right",
            **get_font(),
        )

    return targ_dtype


# -------------------------------------------------------------------------------------
#                              Overlay application performance
# -------------------------------------------------------------------------------------
# Calculate relevent metrics for ai calculation
def plot_application(inputs, verbose):

    df = pd.read_csv(inputs["path"] + "/pmc_perf.csv")
    # Sort by top kernels or top dispatches?
    df = df.sort_values(by=["KernelName"])
    df = df.reset_index(drop=True)

    total_flops = (
        valu_flops
    ) = (
        mfma_flops_bf16
    ) = (
        mfma_flops_f16
    ) = (
        mfma_iops_i8
    ) = (
        mfma_flops_f32
    ) = (
        mfma_flops_f64
    ) = (
        lds_data
    ) = L1cache_data = L2cache_data = hbm_data = calls = totalDuration = avgDuration = 0.0
    kernelName = ""

    myList = []
    for index, row in df.iterrows():
        # CASE: Top kernels
        if inputs["sort"] == "kernels" and (
            (row["KernelName"] != kernelName and kernelName != "")
            or index == df.shape[0] - 1
        ):
            if df.shape[0] - 1 == index:
                calls += 1
            myList.append(
                AI_Data(
                    kernelName,
                    calls,
                    total_flops / calls,
                    valu_flops / calls,
                    mfma_flops_f16 / calls,
                    mfma_flops_bf16 / calls,
                    mfma_flops_f32 / calls,
                    mfma_flops_f64 / calls,
                    lds_data / calls,
                    L1cache_data / calls,
                    L2cache_data / calls,
                    hbm_data / calls,
                    totalDuration,
                    avgDuration / calls,
                )
            )
            if verbose >= 2:
                print(
                    "Just added {} to AI_Data at index {}. # of calls: {}".format(
                        kernelName, index, calls
                    )
                )
            total_flops = (
                valu_flops
            ) = (
                mfma_flops_bf16
            ) = (
                mfma_flops_f16
            ) = (
                mfma_iops_i8
            ) = (
                mfma_flops_f32
            ) = (
                mfma_flops_f64
            ) = (
                lds_data
            ) = (
                L1cache_data
            ) = L2cache_data = hbm_data = calls = totalDuration = avgDuration = 0.0

        kernelName = row["KernelName"]
        try:
            total_flops += (
                (
                    64
                    * (
                        row["SQ_INSTS_VALU_ADD_F16"]
                        + row["SQ_INSTS_VALU_MUL_F16"]
                        + (2 * row["SQ_INSTS_VALU_FMA_F16"])
                        + row["SQ_INSTS_VALU_TRANS_F16"]
                    )
                )
                + (
                    64
                    * (
                        row["SQ_INSTS_VALU_ADD_F32"]
                        + row["SQ_INSTS_VALU_MUL_F32"]
                        + (2 * row["SQ_INSTS_VALU_FMA_F32"])
                        + row["SQ_INSTS_VALU_TRANS_F32"]
                    )
                )
                + (
                    64
                    * (
                        row["SQ_INSTS_VALU_ADD_F64"]
                        + row["SQ_INSTS_VALU_MUL_F64"]
                        + (2 * row["SQ_INSTS_VALU_FMA_F64"])
                        + row["SQ_INSTS_VALU_TRANS_F64"]
                    )
                )
                + (row["SQ_INSTS_VALU_MFMA_MOPS_F16"] * 512)
                + (row["SQ_INSTS_VALU_MFMA_MOPS_BF16"] * 512)
                + (row["SQ_INSTS_VALU_MFMA_MOPS_F32"] * 512)
                + (row["SQ_INSTS_VALU_MFMA_MOPS_F64"] * 512)
            )
        except KeyError:
            if verbose >= 2:
                print("Skipped total_flops at index {}".format(index))
            pass
        try:
            valu_flops += (
                64
                * (
                    row["SQ_INSTS_VALU_ADD_F16"]
                    + row["SQ_INSTS_VALU_MUL_F16"]
                    + (2 * row["SQ_INSTS_VALU_FMA_F16"])
                    + row["SQ_INSTS_VALU_TRANS_F16"]
                )
                + 64
                * (
                    row["SQ_INSTS_VALU_ADD_F32"]
                    + row["SQ_INSTS_VALU_MUL_F32"]
                    + (2 * row["SQ_INSTS_VALU_FMA_F32"])
                    + row["SQ_INSTS_VALU_TRANS_F32"]
                )
                + 64
                * (
                    row["SQ_INSTS_VALU_ADD_F64"]
                    + row["SQ_INSTS_VALU_MUL_F64"]
                    + (2 * row["SQ_INSTS_VALU_FMA_F64"])
                    + row["SQ_INSTS_VALU_TRANS_F64"]
                )
            )
        except KeyError:
            if verbose >= 2:
                print("Skipped valu_flops at index {}".format(index))
            pass

        try:
            mfma_flops_f16 += row["SQ_INSTS_VALU_MFMA_MOPS_F16"] * 512
            mfma_flops_bf16 += row["SQ_INSTS_VALU_MFMA_MOPS_BF16"] * 512
            mfma_flops_f32 += row["SQ_INSTS_VALU_MFMA_MOPS_F32"] * 512
            mfma_flops_f64 += row["SQ_INSTS_VALU_MFMA_MOPS_F64"] * 512
            mfma_iops_i8 += row["SQ_INSTS_VALU_MFMA_MOPS_I8"] * 512
        except KeyError:
            if verbose >= 2:
                print("Skipped mfma ops at index {}".format(index))
            pass

        try:
            lds_data += (
                (row["SQ_LDS_IDX_ACTIVE"] - row["SQ_LDS_BANK_CONFLICT"]) * 4 * L2_BANKS
            )  # L2_BANKS = 32 (since assuming mi200)
        except KeyError:
            if verbose >= 2:
                print("Skipped lds_data at index {}".format(index))
            pass

        try:
            L1cache_data += row["TCP_TOTAL_CACHE_ACCESSES_sum"] * 64
        except KeyError:
            if verbose >= 2:
                print("Skipped L1cache_data at index {}".format(index))
            pass

        try:
            L2cache_data += (
                row["TCP_TCC_WRITE_REQ_sum"] * 64
                + row["TCP_TCC_ATOMIC_WITH_RET_REQ_sum"] * 64
                + row["TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum"] * 64
                + row["TCP_TCC_READ_REQ_sum"] * 64
            )
        except KeyError:
            if verbose >= 2:
                print("Skipped L2cache_data at index {}".format(index))
            pass
        try:
            hbm_data += (
                (row["TCC_EA_RDREQ_32B_sum"] * 32)
                + ((row["TCC_EA_RDREQ_sum"] - row["TCC_EA_RDREQ_32B_sum"]) * 64)
                + (row["TCC_EA_WRREQ_64B_sum"] * 64)
                + ((row["TCC_EA_WRREQ_sum"] - row["TCC_EA_WRREQ_64B_sum"]) * 32)
            )
        except KeyError:
            if verbose >= 2:
                print("Skipped hbm_data at index {}".format(index))
            pass

        totalDuration += row["EndNs"] - row["BeginNs"]

        avgDuration += row["EndNs"] - row["BeginNs"]

        calls += 1
        if inputs["sort"] == "dispatches":
            myList.append(
                AI_Data(
                    kernelName,
                    calls,
                    total_flops,
                    valu_flops,
                    mfma_flops_f16,
                    mfma_flops_bf16,
                    mfma_flops_f32,
                    mfma_flops_f64,
                    mfma_iops_i8,
                    lds_data,
                    L1cache_data,
                    L2cache_data,
                    hbm_data,
                    totalDuration,
                    avgDuration,
                )
            )
            total_flops = (
                valu_flops
            ) = (
                mfma_flops_bf16
            ) = (
                mfma_flops_f16
            ) = (
                mfma_iops_i8
            ) = (
                mfma_flops_f32
            ) = (
                mfma_flops_f64
            ) = (
                lds_data
            ) = (
                L1cache_data
            ) = L2cache_data = hbm_data = calls = totalDuration = avgDuration = 0.0

    myList.sort(key=lambda x: x.totalDuration, reverse=True)

    print("Top 10 intensities ('{}')...".format(inputs["sort"]))
    intensities = {"curr_ai_l1": [], "curr_ai_l2": [], "curr_ai_hbm": []}
    curr_perf = []
    i = 0
    # Create list of top 5 intensities
    while i <= 9 and i != len(myList):
        intensities["curr_ai_l1"].append(
            myList[i].total_flops / myList[i].L1cache_data
        ) if myList[i].L1cache_data else intensities["curr_ai_l1"].append(0)
        # print("cur_ai_L1", myList[i].total_flops/myList[i].L1cache_data) if myList[i].L1cache_data else print("null")
        # print()
        intensities["curr_ai_l2"].append(
            myList[i].total_flops / myList[i].L2cache_data
        ) if myList[i].L2cache_data else intensities["curr_ai_l2"].append(0)
        # print("cur_ai_L2", myList[i].total_flops/myList[i].L2cache_data) if myList[i].L2cache_data else print("null")
        # print()
        intensities["curr_ai_hbm"].append(
            myList[i].total_flops / myList[i].hbm_data
        ) if myList[i].hbm_data else intensities["curr_ai_hbm"].append(0)
        # print("cur_ai_hbm", myList[i].total_flops/myList[i].hbm_data) if myList[i].hbm_data else print("null")
        # print()
        curr_perf.append(myList[i].total_flops / myList[i].avgDuration) if myList[
            i
        ].avgDuration else curr_perf.append(0)
        # print("cur_perf", myList[i].total_flops/myList[i].avgDuration) if myList[i].avgDuration else print("null")

        i += 1

    print(intensities)

    plotted_spots = []
    labels = []
    for i in intensities:
        values = intensities[i]
        color = get_color(i)
        x = []
        y = []
        for entryIndx in range(0, len(values)):
            x.append(values[entryIndx])
            y.append(curr_perf[entryIndx])
        myScatter = plt.scatter(x, y, c=color, marker="o")
        plotted_spots.append(myScatter)
        label = i
        labels.append(label)

    try:
        pylab.legend(
            plotted_spots,
            labels,
            prop={"size": (FONT_SIZE - 2)},
            bbox_to_anchor=(1.04, 1),
            loc="upper left",
            title="Top {}".format(inputs["sort"]),
            title_fontsize=FONT_SIZE,
        )
    except Exception as e:
        sys.stderr.write(f"{e}\n")
        pylab.legend(
            plotted_spots,
            labels,
            prop={"size": (FONT_SIZE - 2)},
        )


def empirical_roof(args):
    soc = args.target
    inputs = {
        "path": str,
        "cmd": str,
        "sort": str,
        "mem": str,
        "axes": list,
        "device": int,
        # "workgroups": int,
        # "wsize": int,
        # "dataset": int,
        # "experiments": int,
        # "iter": int
    }

    inputs["sort"] = args.sort.lower()
    inputs["mem"] = args.mem_level.upper()

    if inputs["sort"] != "kernels" and inputs["sort"] != "dispatches":
        sys.exit("Invalid sort. Must be either 'kernels' or 'dispatches'")
    if (
        inputs["mem"] != "HBM"
        and inputs["mem"] != "VL1D"
        and inputs["mem"] != "L2"
        and inputs["mem"] != "LDS"
        and inputs["mem"] != "ALL"
    ):
        sys.exit(
            "Invalid mem-level. Must be one of these option 'LDS', 'L2', 'vL1D', or 'HBM'"
        )
    if inputs["mem"] == "VL1D":
        inputs["mem"] = "L1"

    inputs["device"] = int(args.device)
    # inputs["workgroups"] = int(args.workgroups)
    # inputs["wsize"] = int(args.wsize)
    # inputs["dataset"] = int(args.dataset)
    # inputs["experiments"] = int(args.experiments)
    # inputs["iter"] = int(args.iter)
    inputs["path"] = args.path
    inputs["cmd"] = args.remaining
    inputs["axes"] = args.axes

    # device_list = [int(item) for item in args.device.split(',')]

    if soc not in SUPPORTED_SOC:
        sys.exit("SoC not yet supported for Roofline Analysis")

    # Basic Info
    print("Path: ", inputs["path"])
    print("Target: ", soc)
    print("Memory Level: ", inputs["mem"])

    roofPath = inputs["path"] + "/roofline.csv"
    # -----------------------------------------------------
    # Initialize roofline data dictionary from roofline.csv
    # -----------------------------------------------------
    roof_data = (
        {}
    )  # TODO: consider changing this to an ordered dict for consistency over py versions
    headers = []
    with open(roofPath, "r") as csvfile:
        csvReader = csv.reader(csvfile, delimiter=",")
        rowCount = 0
        for row in csvReader:
            row.pop(0)  # remove devID
            if rowCount == 0:
                headers = row
                for i in headers:
                    roof_data[i] = []
            else:
                for i, key in enumerate(headers):
                    roof_data[key].append(row[i])

            rowCount += 1
    csvfile.close()

    # Initalize plot
    f = plt.figure(figsize=(1600 / 100, 1200 / 100), dpi=100)
    f.add_subplot(111)

    _title_font = get_font()
    _title_font["size"] += 8

    plt.title("Empirical Roofline", **_title_font)
    plt.xlabel("Arithmetic Intensity (FLOP/Byte)", **get_font())
    plt.ylabel("Performance (GFLOP/sec)", **get_font())
    plt.grid(True, which="major", ls="--", lw=1)
    plt.grid(True, which="minor", ls="--", lw=0.5)
    plt.yscale("log")
    plt.xscale("log")
    # Adjust axes if instructed
    if inputs["axes"]:
        plt.xlim(inputs["axes"][0], inputs["axes"][1])
        plt.ylim(inputs["axes"][2], inputs["axes"][3])

    # ------------------
    #  Generate Roofline
    # ------------------
    dtype = plot_roof(inputs, roof_data)  # Also returns chosen dtype
    plot_application(inputs, args.verbose)

    if inputs["device"] == -1:
        dev_id = "ALL"
    else:
        dev_id = str(inputs["device"])

    filename = IMGNAME + "_gpu-" + dev_id + "_{}".format(dtype) + ".pdf"

    full_path = os.path.abspath(inputs["path"])
    path_to_output = full_path + "/" + filename

    print('Saving plot: "{}"...'.format(filename))
    plt.savefig(path_to_output, bbox_inches="tight", format="pdf")
    print('File saved to: "{}"'.format(path_to_output))
    plt.close()
