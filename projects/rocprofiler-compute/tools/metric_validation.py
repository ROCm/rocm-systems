#!/usr/bin/env python3
##############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

##############################################################################
# This script reads counter values of workloads and computes metrics
# per dispatch based on the counter values. The computed metrics and counter
# values are dumped to CSV files.
##############################################################################
import argparse
import copy
import sys
from pathlib import Path

import pandas as pd

current_path = Path(__file__).resolve().parent
additional_path = current_path / ".." / "src"
sys.path.insert(0, str(additional_path.resolve()))

from argparser import omniarg_parser  # noqa: E402
from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base  # noqa: E402
from utils import file_io, parser  # noqa: E402
from utils.mi_gpu_spec import mi_gpu_specs  # noqa: E402
from utils.utils import merge_counters_iteration_multiplex  # noqa: E402


class Colors:
    """ANSI color codes as class attributes for easy use."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    ENDC = "\033[0m"  # Resets the color


class Analyzer(OmniAnalyze_Base):
    """Analyzer class for dumping raw counter and metric values."""

    def __init__(
        self, args: argparse.Namespace, supported_archs: dict[str, str]
    ) -> None:
        super().__init__(args, supported_archs)

    def dump_values(self) -> None:
        """Dump raw counter and/or metric values to CSV files."""
        args = self.get_args()

        # Columns to drop from the metric dataframes
        cols_to_drop = [
            "Kernel_Name",
            "Count",
            "Sum(ns)",
            "Mean(ns)",
            "Median(ns)",
            "Pct",
            "Dispatch_ID",
            "GPU_ID",
            "Info",
            "coll_level",
            "from_csv",
        ]

        # Define the order of columns for the final output of metrics
        start_columns = [
            "Dispatch_ID",
            "GPU_ID",
            "Kernel_Name",
            "Metric",
            "Channel",
        ]

        # Define the end columns for the final output of metrics
        end_columns = [
            "Description",
        ]

        # Keep track of written CSV paths to avoid overwriting
        written_csv_paths = []

        print(
            f"{Colors.RED}Dumping values takes a long time for workloads with "
            f"large number of dispatches.{Colors.ENDC}"
        )

        # FIXME: Currently only supports single path input
        if len(args.path[0]) > 1:
            print(
                f"{Colors.RED}Warning: Multiple paths provided. "
                f"Only the first path will be processed for dumping values."
                f"{Colors.ENDC}"
            )

        for path_info in args.path:
            # create 'mega dataframe'
            raw_pmc = file_io.create_df_pmc(
                path_info[0],
                args.nodes,
                args.spatial_multiplexing,
                args.kernel_verbose,
                args.verbose,
                self._profiling_config,
            )

            path_suffix_base = "_".join(Path(path_info[0]).parts[-2:])
            path_suffix = path_suffix_base
            counter_index = 1
            # Ensure unique file names
            while path_suffix in written_csv_paths:
                path_suffix = f"{path_suffix_base}_{counter_index}"
                counter_index += 1

            written_csv_paths.append(path_suffix)

            # Dump counter values if requested
            if args.dump_values in ("counter", "all"):
                counter_csv_path = f"{args.output_dir}/{path_suffix}_counters.csv"
                print(
                    f"{Colors.GREEN}Writing raw counter values to "
                    f"{counter_csv_path}{Colors.ENDC}"
                )
                # Dump the raw counters to CSV
                raw_pmc["pmc_perf"].set_index("Dispatch_ID").to_csv(counter_csv_path)

            # Dump metric values if requested
            if args.dump_values in ("metric", "all"):
                dfs = []
                coll_levels = ["pmc_perf"]

                # Handle iteration multiplexing if specified
                if policy := self._profiling_config.get("iteration_multiplexing"):
                    raw_pmc = merge_counters_iteration_multiplex(raw_pmc, policy)

                base_workload = self._runs[path_info[0]]
                # Make a copy of the dataframe to process
                df_new = raw_pmc["pmc_perf"].copy()

                # Process each dispatch individually
                for i in range(len(df_new)):
                    workload = copy.deepcopy(base_workload)
                    df = df_new.loc[[i]].copy()
                    df.reset_index(drop=True, inplace=True)
                    pmc_dfs = [df]
                    final_df = pd.concat(
                        pmc_dfs, keys=coll_levels, axis=1, join="inner", copy=False
                    )
                    workload.raw_pmc = final_df

                    # create the loaded table
                    parser.load_table_data(
                        workload=workload,
                        dir_path=path_info[0],
                        is_gui=False,
                        args=args,
                        config=self._profiling_config,
                        skip_kernel_top=False,
                    )

                    for _, value in workload.dfs.items():
                        value.drop(columns=cols_to_drop, inplace=True, errors="ignore")

                        # Check if the dataframe is not empty after dropping NaNs
                        if not value.empty:
                            value = value.dropna(how="all")

                            # Insert identifying columns
                            value.insert(
                                0,
                                "Dispatch_ID",
                                df.at[0, "Dispatch_ID"]
                                if "Dispatch_ID" in df.columns
                                else 0,
                            )
                            value.insert(1, "GPU_ID", df.at[0, "GPU_ID"])
                            value.insert(2, "Kernel_Name", df.at[0, "Kernel_Name"])

                            # Append to list of dataframes to merge
                            dfs.append(value)

                merged_df = pd.concat(dfs, ignore_index=True)
                # reorder columns
                reordered_cols = (
                    start_columns
                    + [
                        col
                        for col in merged_df.columns
                        if col not in start_columns + end_columns
                    ]
                    + end_columns
                )
                merged_df = merged_df.reindex(columns=reordered_cols)
                metric_csv_path = f"{args.output_dir}/{path_suffix}_metrics.csv"
                print(
                    f"{Colors.GREEN}Writing metric values to "
                    f"{metric_csv_path}{Colors.ENDC}"
                )
                merged_df.set_index(start_columns).to_csv(metric_csv_path)

    def pre_processing(self) -> None:
        """Perform any pre-processing steps prior to analysis."""
        args = self.get_args()

        # Read profiling config
        self._profiling_config = file_io.load_profiling_config(args.path[0][0])

        # initalize runs
        self._runs = self.initalize_runs()


def add_parser_args(parser_obj: argparse.ArgumentParser) -> None:
    """Add arguments to the parser object."""
    parser_obj.add_argument(
        "--dump-values",
        dest="dump_values",
        type=str,
        choices=["counter", "metric", "all"],
        default="all",
        required=False,
        help="Dump raw counter and/or metric values to CSV files.",
    )
    parser_obj.add_argument(
        "-p",
        "--path",
        dest="path",
        required=False,
        metavar="",
        help="\t\tSpecify the directory of profiling data.",
    )
    parser_obj.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        required=False,
        metavar="",
        help="\t\tSpecify the directory for writing values.",
        default=".",
    )


def copy_actions(
    src_parser: argparse.ArgumentParser,
    dst_parser: argparse.ArgumentParser,
    exclude=(
        "--help",
        "-h",
        "-V",
        "--verbose",
        "-q",
        "--quiet",
        "--list-metrics",
        "--list-blocks",
        "--config-dir",
        "-s",
        "--specs",
        "-p",
        "--path",
    ),
) -> None:
    """Copy actions from src_parser to dst_parser, excluding specified options."""
    for action in src_parser._actions:
        # Skip general group commands and subparser actions
        if any(s in exclude for s in action.option_strings):
            continue
        if isinstance(action, argparse._SubParsersAction):
            continue

        # Build kwargs for add_argument
        kwargs = {
            "dest": action.dest,
            "default": action.default,
            "type": getattr(action, "type", None),
            "choices": getattr(action, "choices", None),
            "required": getattr(action, "required", False),
            "help": argparse.SUPPRESS,
            "metavar": getattr(action, "metavar", None),
        }

        # Handle special actions (flags, counts, append, etc.)
        if action.option_strings:
            # Optional-style action
            if isinstance(action, argparse._StoreTrueAction):
                kwargs["action"] = "store_true"
                kwargs.pop("type", None)
            elif isinstance(action, argparse._StoreFalseAction):
                kwargs["action"] = "store_false"
                kwargs.pop("type", None)
            elif isinstance(action, argparse._CountAction):
                kwargs["action"] = "count"
                kwargs.pop("type", None)
            elif isinstance(action, argparse._AppendAction):
                kwargs["action"] = "append"
            elif isinstance(action, argparse._AppendConstAction):
                kwargs["action"] = "append_const"
                kwargs["const"] = action.const
                kwargs.pop("type", None)
            elif isinstance(action, argparse._StoreConstAction):
                kwargs["action"] = "store_const"
                kwargs["const"] = action.const
                kwargs.pop("type", None)
            elif isinstance(action, argparse._VersionAction):
                # skip version, or add as needed
                continue

            # Clean None values from kwargs
            for k in list(kwargs.keys()):
                if kwargs[k] is None:
                    del kwargs[k]

            dst_parser.add_argument(*action.option_strings, **kwargs)
        else:
            # Positional-style action
            pos_kwargs = dict(kwargs)
            for k in list(pos_kwargs.keys()):
                if pos_kwargs[k] is None:
                    del pos_kwargs[k]
            dst_parser.add_argument(action.dest, **pos_kwargs)


def remove_subparsers(parser: argparse.ArgumentParser) -> None:
    """Remove subparsers from the parser object."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            parser._remove_action(action)


def get_subparsers(
    parser: argparse.ArgumentParser,
) -> dict[str, argparse.ArgumentParser]:
    """Get subparsers from the parser object."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices  # dict: {name: ArgumentParser}
    return {}  # no subparsers defined


def main() -> None:
    rocprof_version = {"ver": None, "ver_pretty": None}
    rocprof_compute_path = additional_path.resolve()

    supported_archs = mi_gpu_specs.get_gpu_series_dict()

    parser_obj = argparse.ArgumentParser(description="Metric validation tool.")

    omniarg_parser(parser_obj, rocprof_compute_path, supported_archs, rocprof_version)

    # Move analyze subparser actions to main parser for argument initialization
    subparsers = get_subparsers(parser_obj)
    analyze_subparser = subparsers["analyze"]
    copy_actions(analyze_subparser, parser_obj)

    # Suppress help for all actions in the rocprof-compute parser and copied actions
    for action in parser_obj._actions:
        if not isinstance(action, argparse._HelpAction):
            action.help = argparse.SUPPRESS

    remove_subparsers(parser_obj)
    add_parser_args(parser_obj)

    args = parser_obj.parse_args()

    if not args.path:
        parser_obj.print_help()
        sys.exit(1)

    # Convert paths to absolute paths
    args.path = [[str(Path(args.path).absolute())]]

    analyzer = Analyzer(args, supported_archs)
    analyzer.pre_processing()
    analyzer.dump_values()


if __name__ == "__main__":
    main()
