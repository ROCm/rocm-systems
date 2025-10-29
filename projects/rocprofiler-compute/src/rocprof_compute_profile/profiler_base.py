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

import argparse
import csv
import shlex
import shutil
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
import yaml

from rocprof_compute_soc.soc_base import OmniSoC_Base
from utils.logger import (
    console_debug,
    console_error,
    console_log,
    console_warning,
    demarcate,
)
from utils.utils import (
    capture_subprocess_output,
    format_time,
    gen_sysinfo,
    pc_sampling_prof,
    print_status,
    run_prof,
)


class RocProfCompute_Base:
    def __init__(
        self,
        args: argparse.Namespace,
        profiler_mode: str,
        soc: OmniSoC_Base,
    ) -> None:
        self.__args = args
        self.__profiler = profiler_mode
        self._soc = soc  # OmniSoC obj

    def get_args(self) -> argparse.Namespace:
        return self.__args

    def get_profiler_options(
        self, fname: str, soc: OmniSoC_Base
    ) -> Union[list[str], dict[str, Any]]:
        """Fetch any version specific arguments required by profiler"""
        # assume no SoC specific options and return empty list by default
        return []

    @demarcate
    def sanitize(self) -> None:
        """Perform sanitization of inputs"""
        args = self.get_args()

        if (
            sum((
                bool(args.filter_blocks),
                bool(args.set_selected),
                bool(args.roof_only),
            ))
            > 1
        ):
            console_error(
                "--block, --set, and --roof-only are mutually exclusive options. "
                "Please use only one of them."
            )

        # verify not accessing parent directories
        if ".." in str(args.path):
            console_error(
                "Access denied. Cannot access parent directories in path (i.e. ../)"
            )

        # verify correct formatting for application binary
        args.remaining = args.remaining[1:]
        if args.remaining:
            # Ensure that command points to an executable
            if not shutil.which(args.remaining[0]):
                console_error(
                    f"Your command {args.remaining[0]} doesn't point to a executable. "
                    "Please verify."
                )
            args.remaining = " ".join(args.remaining)
        elif not args.attach_pid:
            console_error(
                "Profiling command required. Pass application executable after -- "
                "at the end of options.\n"
                "\ti.e. rocprof-compute profile -n vcopy -- "
                "./vcopy -n 1048576 -b 256"
            )

    @demarcate
    def join_prof(self, out: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Manually join separated rocprof runs"""
        args = self.get_args()
        output_file = out or f"{args.path}/pmc_perf.csv"

        # handle rocpd format
        if args.format_rocprof_output == "rocpd":
            # Vertically concat (by rows) results_*.csv into pmc_perf.csv
            result_files = list(Path(args.path).glob("results_*.csv"))

            with open(output_file, "w", newline="") as outfile:
                writer = None
                for file in result_files:
                    with open(file, newline="") as infile:
                        reader = csv.reader(infile)
                        header = next(reader)
                        # Write header only once
                        if writer is None:
                            writer = csv.writer(outfile)
                            writer.writerow(header)
                        for row in reader:
                            writer.writerow(row)

            console_debug(f"Created file: {output_file}")

            # Delete results_*.csv files
            for file in result_files:
                Path(file).unlink()
                console_debug(f"Deleted file: {file}")
            return None

        # Collect files to process - normalize to Path objects
        files: list[Path] = []

        # Set default output directory if not specified
        if isinstance(args.path, str):
            csv_patterns = ["pmc_perf_*.csv", "SQ_*.csv", "SQC_*.csv"]
            files = [
                file
                for pattern in csv_patterns
                for file in Path(args.path).glob(pattern)
            ]

            if args.hip_trace:
                # remove hip api trace outputs from this list
                files = [f for f in files if not f.name.endswith("_hip_api_trace.csv")]

            if args.kokkos_trace:
                # remove marker api trace outputs from this list
                files = [
                    f for f in files if not f.name.endswith("_marker_api_trace.csv")
                ]
        elif isinstance(args.path, list):
            files = [Path(path) for path in args.path]
        else:
            console_error(f"Invalid workload directory. Cannot resolve {args.path}")

        # Process files and create joined dataframe
        df = None
        for i, file in enumerate(files):
            current_df = pd.read_csv(file)

            if current_df.empty:
                console_warning("join_prof", f"Empty dataframe from {file}")
                continue

            if args.join_type == "kernel":
                key = current_df.groupby("Kernel_Name").cumcount()
                current_df["key"] = current_df.Kernel_Name + " - " + key.astype(str)
            elif args.join_type == "grid":
                key = current_df.groupby(["Kernel_Name", "Grid_Size"]).cumcount()
                current_df["key"] = (
                    current_df["Kernel_Name"].astype(str)
                    + " - "
                    + current_df["Grid_Size"].astype(str)
                    + " - "
                    + key.astype(str)
                )
            else:
                console_error(
                    "join_prof",
                    f"{args.join_type} is an unrecognized option for --join-type",
                )

            if df is None:
                df = current_df
            else:
                # join by unique index of kernel
                df = pd.merge(
                    df, current_df, how="inner", on="key", suffixes=("", f"_{i}")
                )

        if df is None or df.empty:
            console_warning("join_prof", "No data available after processing all files")
            return None

        # TODO: check for any mismatch in joins
        duplicate_cols = {
            "GPU_ID": [col for col in df.columns if col.startswith("GPU_ID")],
            "Grid_Size": [col for col in df.columns if col.startswith("Grid_Size")],
            "Workgroup_Size": [
                col for col in df.columns if col.startswith("Workgroup_Size")
            ],
            "LDS_Per_Workgroup": [
                col for col in df.columns if col.startswith("LDS_Per_Workgroup")
            ],
            "Scratch_Per_Workitem": [
                col for col in df.columns if col.startswith("Scratch_Per_Workitem")
            ],
            "SGPR": [col for col in df.columns if col.startswith("SGPR")],
        }

        # Check for vgpr counter in ROCm < 5.3
        if "vgpr" in df.columns:
            duplicate_cols["vgpr"] = [
                col for col in df.columns if col.startswith("vgpr")
            ]
        # Check for vgpr counter in ROCm >= 5.3
        else:
            duplicate_cols["Arch_VGPR"] = [
                col for col in df.columns if col.startswith("Arch_VGPR")
            ]
            duplicate_cols["Accum_VGPR"] = [
                col for col in df.columns if col.startswith("Accum_VGPR")
            ]

        for key, cols in duplicate_cols.items():
            current_df = df[cols]
            if not test_df_column_equality(current_df):
                console_warning(
                    "join_prof",
                    f"Detected differing {key} values while joining pmc_perf.csv",
                )
            else:
                console_debug("join_prof", f"Successfully joined {key} in pmc_perf.csv")

        # now, we can:
        #   A) throw away any of the "boring" duplicates
        columns_to_remove = [
            # rocprofv2 headers
            "GPU_ID_",
            "Grid_Size_",
            "Workgroup_Size_",
            "LDS_Per_Workgroup_",
            "Scratch_Per_Workitem_",
            "vgpr_",
            "Arch_VGPR_",
            "Accum_VGPR_",
            "SGPR_",
            "Dispatch_ID_",
            "Queue_ID",
            "Queue_Index",
            "PID",
            "TID",
            "SIG",
            "OBJ",
            "Correlation_ID_",
            "Wave_Size_",
            # rocscope specific merged counters, keep original
            "dispatch_",
            # extras
            "sig",
            "queue-id",
            "queue-index",
            "pid",
            "tid",
            "fbar",
        ]

        df = df[
            [
                col
                for col in df.columns
                if not any(col.startswith(prefix) for prefix in columns_to_remove)
            ]
        ]

        #   B) any timestamps that are _not_ the duration,
        #      which is the one we care about
        timestamp_patterns = ["DispatchNs", "CompleteNs", "HostDuration"]

        df = df[
            [
                col
                for col in df.columns
                if not any(pattern in col for pattern in timestamp_patterns)
            ]
        ]

        #   C) sanity check the name and key
        name_cols = [col for col in df.columns if "Kernel_Name" in col]
        if not name_cols:
            return df

        for col in name_cols[1:]:
            assert (df[name_cols[0]] == df[col]).all()

        df = df.drop(columns=name_cols[1:])

        # now take the median of the durations
        start_cols = [col for col in df.columns if "Start_Timestamp" in col]
        end_cols = [col for col in df.columns if "End_Timestamp" in col]

        # compute mean mean timestamps
        if start_cols and end_cols:
            mean_start = df[start_cols].mean(axis=1)
            mean_end = df[end_cols].mean(axis=1)

            # Replace with consolidated timestamps
            df = df.drop(columns=start_cols + end_cols)
            df["Start_Timestamp"] = mean_start
            df["End_Timestamp"] = mean_end

        # finally, join the drop key
        if "key" in df.columns:
            df = df.drop(columns=["key"])

        # save to file and delete old file(s)
        # skip if we're being called outside of rocprof-compute
        if isinstance(args.path, str):
            df.to_csv(output_file, index=False)
            if not args.verbose:
                for file in files:
                    # Do not remove accumulate counter files
                    if "SQ_" not in file.name or "SQC_" not in file.name:
                        file.unlink()
            return None
        else:
            return df

    # ----------------------------------------------------
    # Required methods to be implemented by child classes
    # ----------------------------------------------------
    @abstractmethod
    def pre_processing(self) -> None:
        """Perform any pre-processing steps prior to profiling."""
        args = self.get_args()
        console_debug("profiling", f"pre-processing using {self.__profiler} profiler")

        if args.attach_pid:
            args.remaining = ""

        self._filter_blocks = self._soc.profiling_setup()

        # Write profiling configuration as yaml file
        with open(f"{self.__args.path}/profiling_config.yaml", "w") as f:
            args_dict = vars(self.__args)
            # Override filter_blocks when writing profiling config yaml
            args_dict["filter_blocks"] = self._filter_blocks
            args_dict["config_dir"] = str(args_dict["config_dir"])
            yaml.dump(args_dict, f)

        # verify soc compatibility
        if self.__profiler not in self._soc.get_compatible_profilers():
            console_error(
                f"{self._soc.get_arch()} is not enabled in {self.__profiler}. "
                f"Available profilers include: {self._soc.get_compatible_profilers()}"
            )

        gen_sysinfo(
            workload_name=args.name,
            workload_dir=args.path,
            app_cmd=args.remaining,
            skip_roof=args.no_roof,
            mspec=self._soc._mspec,
            soc=self._soc,
        )

    @abstractmethod
    def run_profiling(self, version: str, prog: str) -> None:
        """Run profiling."""
        console_debug(
            "profiling", f"performing profiling using {self.__profiler} profiler"
        )
        args = self.get_args()

        # log basic info
        console_log(f"{str(prog).title()} version: {version}")
        console_log(f"Profiler choice: {self.__profiler}")
        console_log(f"Path: {Path(self.__args.path).absolute().resolve()}")
        console_log(f"Target: {self._soc._mspec.gpu_model}")
        console_log(f"Command: {args.remaining}")
        console_log(f"Kernel Selection: {args.kernel}")
        console_log(f"Dispatch Selection: {args.dispatch}")
        if self._filter_blocks:
            console_log(f"Filtered sections: {str(self._filter_blocks)}")
        else:
            console_log("Filtered sections: All")

        msg = "Collecting Performance Counters"
        status_msg = f"{msg} (Roofline Only)" if self.__args.roof_only else msg
        print_status(status_msg)

        # Run profiling on each input file
        input_files = sorted(Path(args.path).glob("perfmon/*.txt"))
        total_runs = len(input_files)
        total_profiling_time = 0.0

        for i, fname in enumerate(input_files):
            run_number = i + 1

            # Log progress and time estimation
            if i > 0:
                avg_time = total_profiling_time / i
                time_left_seconds = (total_runs - run_number) * avg_time
                time_left = format_time(time_left_seconds)
                console_log(
                    f"[Run {run_number}/{total_runs}]"
                    f"[Approximate profiling time left: {time_left}]..."
                )
            else:
                console_log(
                    f"[Run {run_number}/{total_runs}]"
                    "[Approximate profiling time left: "
                    "pending first measurement...]"
                )

            # Kernel filtering (in-place replacement)
            if not args.kernel == None:
                success, output = capture_subprocess_output([
                    "sed",
                    "-i",
                    "-r",
                    f"s%^(kernel:).*%kernel: {','.join(self.__args.kernel)}%g",
                    str(fname),
                ])
                # log output from profile filtering
                if not success:
                    console_error(output)
                else:
                    console_debug(output)

            # Dispatch filtering (inplace replacement)
            if not args.dispatch == None:
                success, output = capture_subprocess_output([
                    "sed",
                    "-i",
                    "-r",
                    f"s%^(range:).*%range: {' '.join(self.__args.dispatch)}%g",
                    str(fname),
                ])
                # log output from profile filtering
                if not success:
                    console_error(output)
                else:
                    console_debug(output)

            console_log("profiling", f"Current input file: {fname}")
            options = self.get_profiler_options(str(fname), self._soc)
            start_time = time.time()
            if self.__profiler == "rocprofv3" or self.__profiler == "rocprofiler-sdk":
                # Only 1-run case is permitted for attach/detach
                if (isinstance(options, list) and "--pid" in options) or (
                    isinstance(options, dict)
                    and (options.get("ROCPROF_ATTACH_PID") is not None)
                ):
                    if total_runs > 1:
                        console_error(
                            f"Cannot attach process for profiling as the requested "
                            f"performance counters exceed the collection capacity of "
                            f"single pass counter collection. The current setup of "
                            f"requested counter blocks needs {total_runs} number of "
                            f'passes. Please use "--block" or "--set" '
                            f"to adjust or reduce the requested performance metrics!"
                        )
                run_prof(
                    fname=str(fname),
                    profiler_options=options,
                    workload_dir=args.path,
                    mspec=self._soc._mspec,
                    loglevel=args.loglevel,
                    format_rocprof_output=args.format_rocprof_output,
                    retain_rocpd_output=args.retain_rocpd_output,
                )
                end_time = time.time()
                duration = end_time - start_time
                total_profiling_time += duration

                console_debug(
                    f"The time of run_prof of {fname} is {int(duration / 60)} min"
                    f" {duration % 60} sec"
                )
            else:
                console_error("Profiler not supported")

        # PC sampling data is only collected when block "21" is specified
        if not "21" in args.filter_blocks:
            console_warning(
                "PC sampling data collection skipped as block 21 is not specified."
            )
            return

        total_runs = len(list(Path(args.path).glob("perfmon/*.txt")))

        console_log(f"[Run {total_runs + 1}/{total_runs + 1}][PC sampling profile run]")

        start_time = time.time()
        pc_sampling_prof(
            method=args.pc_sampling_method,
            interval=args.pc_sampling_interval,
            workload_dir=args.path,
            appcmd=shlex.split(
                args.remaining
            ),  # FIXME: the right solution is applying it when argparsing once!
            rocprofiler_sdk_library_path=args.rocprofiler_sdk_library_path,
        )
        end_time = time.time()

        duration = end_time - start_time
        console_debug(
            "profiling",
            f"The time of pc sampling profiling is {int(duration / 60)} m "
            f"{duration % 60} sec",
        )

    @abstractmethod
    def post_processing(self) -> None:
        """Perform any post-processing steps prior to profiling."""
        console_debug(
            "profiling", f"performing post-processing using {self.__profiler} profiler"
        )
        self._soc.post_profiling()


def test_df_column_equality(df: pd.DataFrame) -> bool:
    return df.eq(df.iloc[:, 0], axis=0).all(1).all()
