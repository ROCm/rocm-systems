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
import argparse
import copy
import sys
from pathlib import Path

import pandas as pd

current_path = Path(__file__).resolve().parent
additional_path = current_path / ".." / "src"
sys.path.insert(0, str(additional_path.resolve()))

from argparser import omniarg_parser
from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base
from utils import file_io, parser
from utils.mi_gpu_spec import mi_gpu_specs


class Analyzer(OmniAnalyze_Base):
    def __init__(
        self, args: argparse.Namespace, supported_archs: dict[str, str]
    ) -> None:
        super().__init__(args, supported_archs)

    def dump_raw_values(self) -> None:
        args = self.get_args()
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
        ]

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

            if args.dump_raw_values in ("counter", "all"):
                raw_pmc["pmc_perf"].to_csv(f"{path_info[0]}/counter_values.csv")

            if args.dump_raw_values in ("metric", "all"):
                dfs = []
                coll_levels = ["pmc_perf"]

                base_workload = self._runs[path_info[0]]
                df_new = raw_pmc["pmc_perf"].copy()

                for i in range(len(df_new)):
                    # for i in range(1):
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
                    value = value.dropna(how="all")
                    dfs.append(value)

                merged_df = pd.concat(dfs, ignore_index=True)
                merged_df.to_csv(f"{path_info[0]}/metric_values.csv")

    def pre_processing(self) -> None:
        """Perform any pre-processing steps prior to analysis."""
        args = self.get_args()

        # Read profiling config
        self._profiling_config = file_io.load_profiling_config(args.path[0][0])

        # initalize runs
        self._runs = self.initalize_runs()


def add_parser_args(parser_obj: argparse.ArgumentParser) -> None:
    parser_obj.add_argument(
        "--dump-values",
        dest="dump_raw_values",
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
        nargs="+",
        action="append",
        help="\t\tSpecify the raw data root dirs or desired results directory.",
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
            "help": action.help,
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


def get_subparsers(
    parser: argparse.ArgumentParser,
) -> dict[str, argparse.ArgumentParser]:
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
    subparsers = get_subparsers(parser_obj)
    analyze_subparser = subparsers["analyze"]
    copy_actions(analyze_subparser, parser_obj)
    add_parser_args(parser_obj)
    args = parser_obj.parse_args()
    args.path = [
        list(map(lambda x: str(Path(x).absolute()), path)) for path in args.path
    ]

    analyzer = Analyzer(args, supported_archs)
    analyzer.pre_processing()
    analyzer.dump_raw_values()


if __name__ == "__main__":
    main()
