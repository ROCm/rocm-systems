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

import logging
import os
import pandas as pd
import glob
import sys
import re
from omniperf_profile.profiler_base import OmniProfiler_Base
from utils.utils import demarcate, replace_timestamps
from utils.csv_processor import kernel_name_shortener


class rocprof_v1_profiler(OmniProfiler_Base):
    def __init__(self,profiling_args,profiler_mode,soc):
        super().__init__(profiling_args,profiler_mode,soc)
        self.ready_to_run = (self.get_args().roof_only and not os.path.isfile(os.path.join(self.get_args().path, "pmc_perf.csv"))
                            or not self.get_args().roof_only)

    #-----------------------
    # Required child methods
    #-----------------------
    @demarcate
    def pre_processing(self):
        """Perform any pre-processing steps prior to profiling.
        """
        super().pre_processing()
        if self.ready_to_run:
            pmc_perf_split(self.get_args().path)

    @demarcate
    def run_profiling(self, version:str, prog:str):
        """Run profiling.
        """
        if self.ready_to_run:
            if self.get_args().roof_only:
                logging.info("[roofline] Generating pmc_perf.csv")
            super().run_profiling(version, prog)
        else:
            logging.info("[roofline] Detected existing pmc_perf.csv")
        
    @demarcate
    def post_processing(self):
        """Perform any post-processing steps prior to profiling.
        """
        super().post_processing()
        if self.ready_to_run:
            # Manually join each pmc_perf*.csv output
            join_prof(self.get_args().path, self.get_args().join_type, self.get_args().verbose)
            # Replace timestamp data to solve a known rocprof bug
            replace_timestamps(self.get_args().path)
            # Demangle and overwrite original KernelNames
            kernel_name_shortener(self.get_args().path, self.get_args().kernel_verbose)

@demarcate
def pmc_perf_split(workload_dir):
    """Avoid default rocprof join utility by spliting each line into a separate input file
    """
    workload_perfmon_dir = os.path.join(workload_dir, "perfmon")
    lines = open(os.path.join(workload_perfmon_dir, "pmc_perf.txt"), "r").read().splitlines()

    # Iterate over each line in pmc_perf.txt
    mpattern = r"^pmc:(.*)"
    i = 0
    for line in lines:
        # Verify no comments
        stext = line.split("#")[0].strip()
        if not stext:
            continue

        # all pmc counters start with  "pmc:"
        m = re.match(mpattern, stext)
        if m is None:
            continue

        # Create separate file for each line
        fd = open(workload_perfmon_dir + "/pmc_perf_" + str(i) + ".txt", "w")
        fd.write(stext + "\n\n")
        fd.write("gpu:\n")
        fd.write("range:\n")
        fd.write("kernel:\n")
        fd.close()

        i += 1

    # Remove old pmc_perf.txt input from perfmon dir
    os.remove(workload_perfmon_dir + "/pmc_perf.txt")

def test_df_column_equality(df):
    return df.eq(df.iloc[:, 0], axis=0).all(1).all()

# joins disparate runs less dumbly than rocprof
@demarcate
def join_prof(workload_dir, join_type, verbose, out=None):
    """Manually join separated rocprof runs
    """
    # Set default output directory if not specified
    if type(workload_dir) == str:
        if out is None:
            out = workload_dir + "/pmc_perf.csv"
        files = glob.glob(workload_dir + "/" + "pmc_perf_*.csv")
    elif type(workload_dir) == list:
        files = workload_dir
    else:
        logging.error("ERROR: Invalid workload_dir")
        sys.exit(1)

    df = None
    for i, file in enumerate(files):
        _df = pd.read_csv(file) if type(workload_dir) == str else file
        if join_type == "kernel":
            key = _df.groupby("KernelName").cumcount()
            _df["key"] = _df.KernelName + " - " + key.astype(str)
        elif join_type == "grid":
            key = _df.groupby(["KernelName", "grd"]).cumcount()
            _df["key"] = (
                _df.KernelName + " - " + _df.grd.astype(str) + " - " + key.astype(str)
            )
        else:
            print("ERROR: Unrecognized --join-type")
            sys.exit(1)

        if df is None:
            df = _df
        else:
            # join by unique index of kernel
            df = pd.merge(df, _df, how="inner", on="key", suffixes=("", f"_{i}"))

    # TODO: check for any mismatch in joins
    duplicate_cols = {
        "gpu": [col for col in df.columns if "gpu" in col],
        "grd": [col for col in df.columns if "grd" in col],
        "wgr": [col for col in df.columns if "wgr" in col],
        "lds": [col for col in df.columns if "lds" in col],
        "scr": [col for col in df.columns if "scr" in col],
        "spgr": [col for col in df.columns if "sgpr" in col],
    }
    # Check for vgpr counter in ROCm < 5.3
    if "vgpr" in df.columns:
        duplicate_cols["vgpr"] = [col for col in df.columns if "vgpr" in col]
    # Check for vgpr counter in ROCm >= 5.3
    else:
        duplicate_cols["arch_vgpr"] = [col for col in df.columns if "arch_vgpr" in col]
        duplicate_cols["accum_vgpr"] = [col for col in df.columns if "accum_vgpr" in col]
    for key, cols in duplicate_cols.items():
        _df = df[cols]
        if not test_df_column_equality(_df):
            msg = (
                "WARNING: Detected differing {} values while joining pmc_perf.csv".format(
                    key
                )
            )
            logging.warning(msg + "\n")
        else:
            msg = "Successfully joined {} in pmc_perf.csv".format(key)
            logging.debug(msg + "\n")
        if test_df_column_equality(_df) and verbose:
            logging.info(msg)

    # now, we can:
    #   A) throw away any of the "boring" duplicats
    df = df[
        [
            k
            for k in df.keys()
            if not any(
                check in k
                for check in [
                    # removed merged counters, keep original
                    "gpu-id_",
                    "grd_",
                    "wgr_",
                    "lds_",
                    "scr_",
                    "vgpr_",
                    "sgpr_",
                    "Index_",
                    # un-mergable, remove all
                    "queue-id",
                    "queue-index",
                    "pid",
                    "tid",
                    "fbar",
                    "sig",
                    "obj",
                    # rocscope specific merged counters, keep original
                    "dispatch_",
                ]
            )
        ]
    ]
    #   B) any timestamps that are _not_ the duration, which is the one we care
    #   about
    df = df[
        [
            k
            for k in df.keys()
            if not any(
                check in k
                for check in [
                    "DispatchNs",
                    "CompleteNs",
                    # rocscope specific timestamp
                    "HostDuration",
                ]
            )
        ]
    ]
    #   C) sanity check the name and key
    namekeys = [k for k in df.keys() if "KernelName" in k]
    assert len(namekeys)
    for k in namekeys[1:]:
        assert (df[namekeys[0]] == df[k]).all()
    df = df.drop(columns=namekeys[1:])
    # now take the median of the durations
    bkeys = []
    ekeys = []
    for k in df.keys():
        if "Begin" in k:
            bkeys.append(k)
        if "End" in k:
            ekeys.append(k)
    # compute mean begin and end timestamps
    endNs = df[ekeys].mean(axis=1)
    beginNs = df[bkeys].mean(axis=1)
    # and replace
    df = df.drop(columns=bkeys)
    df = df.drop(columns=ekeys)
    df["BeginNs"] = beginNs
    df["EndNs"] = endNs
    # finally, join the drop key
    df = df.drop(columns=["key"])
    # save to file and delete old file(s), skip if we're being called outside of Omniperf
    if type(workload_dir) == str:
        df.to_csv(out, index=False)
        if not verbose:
            for file in files:
                os.remove(file)
    else:
        return df