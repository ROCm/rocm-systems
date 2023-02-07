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

import sys

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
    mfma_iops_i8: float
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
    if catagory == "ai_l1":
        return "green"
    elif catagory == "ai_l2":
        return "blue"
    elif catagory == "ai_hbm":
        return "red"
    else:
        raise RuntimeError("Invalid catagory passed to get_color()")


# -------------------------------------------------------------------------------------
#                           Plot BW at each cache level
# -------------------------------------------------------------------------------------
def plot_roof(roof_details, roof_data, mem_level, verbose):
    # TODO: This is where filtering by memory level will need to occur for standalone
    graphPoints = {"hbm": [], "l2": [], "l1": [], "lds": [], "valu": [], "mfma": []}

    if mem_level == "ALL":
        cacheHierarchy = ["HBM", "L2", "L1", "LDS"]
    else:
        cacheHierarchy = mem_level

    x1 = y1 = x2 = y2 = -1
    x1_mfma = y1_mfma = x2_mfma = y2_mfma = -1
    target_precision = roof_details["dtype"][2:]

    if roof_details["dtype"] != "FP16" and roof_details["dtype"] != "I8":
        peakOps = float(
            roof_data[roof_details["dtype"] + "Flops"][roof_details["device"]]
        )
    for i in range(0, len(cacheHierarchy)):
        # Plot BW line
        if verbose >= 3:
            print("Current cache level is ", cacheHierarchy[i])
        curr_bw = cacheHierarchy[i] + "Bw"
        peakBw = float(roof_data[curr_bw][roof_details["device"]])

        if roof_details["dtype"] == "I8":
            peakMFMA = float(roof_data["MFMAI8Ops"][roof_details["device"]])
        else:
            peakMFMA = float(
                roof_data["MFMAF{}Flops".format(target_precision)][roof_details["device"]]
            )

        x1 = float(XMIN)
        y1 = float(XMIN) * peakBw
        # Note: No reg peakOps for FP16 or INT8
        if roof_details["dtype"] != "FP16" and roof_details["dtype"] != "I8":
            x2 = peakOps / peakBw
            y2 = peakOps

            # Plot MFMA lines (NOTE: Assuming MI200 soc)
            x1_mfma = peakOps / peakBw
            y1_mfma = peakOps

        x2_mfma = peakMFMA / peakBw
        y2_mfma = peakMFMA

        # These are the points to use:
        if verbose >= 3:
            print("x = [{}, {}]".format(x1, x2_mfma))
            print("y = [{}, {}]".format(y1, y2_mfma))

        graphPoints[cacheHierarchy[i].lower()].append([x1, x2_mfma])
        graphPoints[cacheHierarchy[i].lower()].append([y1, y2_mfma])
        graphPoints[cacheHierarchy[i].lower()].append(peakBw)

    # -------------------------------------------------------------------------------------
    #                                     Plot computing roof
    # -------------------------------------------------------------------------------------
    # Note: No FMA roof for FP16 or INT8
    if roof_details["dtype"] != "FP16" and roof_details["dtype"] != "I8":
        # Plot FMA roof
        x0 = XMAX
        if x2 < x0:
            x0 = x2

        if verbose >= 3:
            print("FMA ROOF [{}, {}], [{},{}]".format(x0, XMAX, peakOps, peakOps))
        graphPoints["valu"].append([x0, XMAX])
        graphPoints["valu"].append([peakOps, peakOps])
        graphPoints["valu"].append(peakOps)

    # Plot MFMA roof
    if (
        x1_mfma != -1 or roof_details["dtype"] == "FP16" or roof_details["dtype"] == "I8"
    ):  # assert that mfma has been assigned
        x0_mfma = XMAX
        if x2_mfma < x0_mfma:
            x0_mfma = x2_mfma

        if verbose >= 3:
            print("MFMA ROOF [{}, {}], [{},{}]".format(x0_mfma, XMAX, peakMFMA, peakMFMA))
        graphPoints["mfma"].append([x0_mfma, XMAX])
        graphPoints["mfma"].append([peakMFMA, peakMFMA])
        graphPoints["mfma"].append(peakMFMA)

    return graphPoints


# -------------------------------------------------------------------------------------
#                              Overlay application performance
# -------------------------------------------------------------------------------------
# Calculate relevent metrics for ai calculation
def plot_application(sortType, ret_df, verbose):
    df = ret_df["pmc_perf"]
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
        # Calculate + append AI data if
        # a) current KernelName is different than previous OR
        # b) We've reached the end of list
        if sortType == "kernels" and (
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
                    mfma_iops_i8 / calls,
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
        if sortType == "dispatches":
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

    # print("Top 5 intensities ('{}')...".format(roof_details["sort"]))
    intensities = {"ai_l1": [], "ai_l2": [], "ai_hbm": []}
    curr_perf = []
    i = 0
    # Create list of top 5 intensities
    while i <= 9 and i != len(myList):
        intensities["ai_l1"].append(
            myList[i].total_flops / myList[i].L1cache_data
        ) if myList[i].L1cache_data else intensities["ai_l1"].append(0)
        # print("cur_ai_L1", myList[i].total_flops/myList[i].L1cache_data) if myList[i].L1cache_data else print("null")
        # print()
        intensities["ai_l2"].append(
            myList[i].total_flops / myList[i].L2cache_data
        ) if myList[i].L2cache_data else intensities["ai_l2"].append(0)
        # print("cur_ai_L2", myList[i].total_flops/myList[i].L2cache_data) if myList[i].L2cache_data else print("null")
        # print()
        intensities["ai_hbm"].append(
            myList[i].total_flops / myList[i].hbm_data
        ) if myList[i].hbm_data else intensities["ai_hbm"].append(0)
        # print("cur_ai_hbm", myList[i].total_flops/myList[i].hbm_data) if myList[i].hbm_data else print("null")
        # print()
        curr_perf.append(myList[i].total_flops / myList[i].avgDuration) if myList[
            i
        ].avgDuration else curr_perf.append(0)
        # print("cur_perf", myList[i].total_flops/myList[i].avgDuration) if myList[i].avgDuration else print("null")

        i += 1

    intensityPoints = {"ai_l1": [], "ai_l2": [], "ai_hbm": []}

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

        intensityPoints[i].append(x)
        intensityPoints[i].append(y)

    return intensityPoints


def empirical_roof(roof_info, mem_level, verbose):
    roofPath = roof_info["path"] + "/roofline.csv"
    # -----------------------------------------------------
    # Initialize roofline data dictionary from roofline.csv
    # -----------------------------------------------------
    roof_data = (
        {}
    )  # TODO: consider changing this to an ordered dict for consistency over py versions
    headers = []
    try:
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
    except:
        graphPoints = {
            "hbm": [None, None, None],
            "l2": [None, None, None],
            "l1": [None, None, None],
            "lds": [None, None, None],
            "valu": [None, None, None],
            "mfma": [None, None, None],
        }
        return graphPoints

    # ------------------
    #  Generate Roofline
    # ------------------
    results = plot_roof(roof_info, roof_data, mem_level, verbose)
    # for key in results:
    #     print(key, "->", results[key])

    return results
