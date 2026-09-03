# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import copy
import csv
import re
import sys
from abc import abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, TextIO

import pandas as pd

import config
from rocprof_compute_soc.soc_base import OmniSoC_Base
from utils import csv_compression, file_io, parser, schema
from utils.inject_roctx.constants import KNOWN_ML_API_BACKENDS
from utils.logger import (
    console_debug,
    console_error,
    console_log,
    console_warning,
    demarcate,
)
from utils.metrics.expression import build_metric_value_string
from utils.utils_analysis import (
    impute_counters_iteration_multiplex,
    validate_workload,
)
from utils.utils_common import (
    PC_SAMPLING_BLOCK_IDS,
    canonical_config_arch,
    get_uuid,
    is_only_pc_sampling,
    load_panel_configs,
    validate_roofline_csv,
)

# the build-in config to list kernel names purpose only
TOP_STATS_BUILD_IN_CONFIG: OrderedDict[int, dict[str, Any]] = OrderedDict([
    (
        0,
        {
            "id": 0,
            "title": "Top Kernels",
            "data source": [
                {"raw_csv_table": {"id": 1, "source": "pmc_kernel_top.csv"}}
            ],
        },
    ),
    (
        1,
        {
            "id": 1,
            "title": "Dispatch List",
            "data source": [
                {"raw_csv_table": {"id": 2, "source": "pmc_dispatch_info.csv"}}
            ],
        },
    ),
])


class OmniAnalyze_Base:
    def __init__(
        self, args: argparse.Namespace, supported_archs: dict[str, str]
    ) -> None:
        self.__args = args
        self._runs: OrderedDict[str, schema.Workload] = OrderedDict()
        self._arch_configs: dict[str, schema.ArchConfig] = {}
        self.__supported_archs = supported_archs
        self._output: Optional[TextIO] = None
        self.__socs: Optional[dict[str, OmniSoC_Base]] = None

    def get_args(self) -> argparse.Namespace:
        return self.__args

    def get_profiling_config(self) -> dict[str, Any]:
        return self._profiling_config

    def pc_sampling_collected(self) -> bool:
        """True when PC sampling is among the collected blocks."""
        config = getattr(self, "_profiling_config", {})
        return any(
            block in PC_SAMPLING_BLOCK_IDS for block in config.get("filter_blocks", [])
        )

    def pc_sampling_only(self) -> bool:
        """True when every collected block is PC sampling."""
        config = getattr(self, "_profiling_config", {})
        return is_only_pc_sampling(config.get("filter_blocks", []))

    def load_pc_sampling_tool_data(self, workload_path: str) -> list[dict[str, Any]]:
        """Return parsed PC sampling tool records, or an empty list."""
        if not self.pc_sampling_collected():
            return []
        return file_io.load_pc_sampling_results(str(workload_path))

    def build_pc_sampling_only_workload(
        self,
        workload: schema.Workload,
        dir_path: str,
        args: argparse.Namespace,
        tool_data: list[dict[str, Any]],
    ) -> None:
        """Build dispatch scaffolding and tables for a run without counters."""
        workload.raw_pmc = file_io.process_pc_sampling_kernel_traces(tool_data)
        workload.raw_pmc = workload.raw_pmc.rename(
            columns={"Dispatch_Id": "Dispatch_ID"}
        )
        kernel_top_df, dispatch_info_df = file_io.create_df_kernel_top_stats(
            df_in=workload.raw_pmc,
            raw_data_dir=str(dir_path),
            filter_gpu_ids=workload.filter_gpu_ids,
            filter_dispatch_ids=workload.filter_dispatch_ids,
            time_unit=args.time_unit,
        )
        workload.dfs[parser.PMC_KERNEL_TOP_TABLE_ID] = kernel_top_df
        workload.dfs[parser.PMC_DISPATCH_INFO_TABLE_ID] = dispatch_info_df
        parser.load_non_mertrics_table(
            workload, dir_path, args, pc_sampling_tool_data=tool_data
        )
        parser.nullify_unevaluated_metric_values(workload)

    def set_soc(self, omni_socs: dict[str, OmniSoC_Base]) -> None:
        self.__socs = omni_socs

    def get_socs(self) -> Optional[dict[str, OmniSoC_Base]]:
        return self.__socs

    @demarcate
    def iteration_multiplex_impute_counters(
        self, df: pd.DataFrame, policy: str, workload_dir: Path
    ) -> pd.DataFrame:
        return impute_counters_iteration_multiplex(df, policy, workload_dir)

    @demarcate
    def generate_configs(
        self,
        arch: str,
        config_dir: str,
        list_stats: bool,
        filter_metrics: Optional[list[str]],
        sys_info: pd.Series,
        profiling_config: dict[str, Any],
    ) -> dict[str, schema.ArchConfig]:
        single_panel_config = file_io.is_single_panel_config(
            config_dir, self.__supported_archs
        )

        ac = schema.ArchConfig()
        if list_stats:
            ac.panel_configs = TOP_STATS_BUILD_IN_CONFIG
        else:
            config_arch = canonical_config_arch(arch) or arch
            arch_panel_config = [
                config_dir
                if single_panel_config
                else str(Path(config_dir) / config_arch)
            ]
            # Use restructured perf metrics in TUI analyze mode
            if self.get_args().tui and arch in ["gfx942", "gfx950"]:
                arch_panel_config.append(
                    str(
                        config.rocprof_compute_home
                        / "rocprof_compute_tui"
                        / "utils"
                        / arch
                    )
                )
            ac.panel_configs = load_panel_configs(arch_panel_config)

        parser.build_dfs(
            arch_configs=ac,
            filter_metrics=filter_metrics,
            sys_info=sys_info,
            profiling_config=profiling_config,
            arch=arch,
        )
        self._arch_configs[arch] = ac
        return self._arch_configs

    @demarcate
    def load_options(self, normalization_filter: Optional[str]) -> None:
        args = self.get_args()
        target_filter = normalization_filter or args.normal_unit

        for arch_config in self._arch_configs.values():
            build_metric_value_string(
                arch_config.dfs,
                arch_config.dfs_type,
                target_filter,
            )
        # Error checking for multiple runs and multiple kernel filters
        if args.gpu_kernel and (len(args.path) != len(args.gpu_kernel)):
            if len(args.gpu_kernel) == 1:
                args.gpu_kernel *= len(args.path)
            else:
                console_error(
                    "analysis"
                    "The number of -k/--kernel doesn't match the number of --dir."
                )

    @demarcate
    def initalize_runs(
        self, normalization_filter: Optional[str] = None
    ) -> OrderedDict[str, schema.Workload]:
        args = self.get_args()

        # load required configs
        for path_info in args.path:
            sysinfo_path = path_info[0]
            if sysinfo_path:
                sys_info = pd.read_csv(f"{sysinfo_path}/sysinfo.csv")
                arch = sys_info.iloc[0]["gpu_arch"]
                self.generate_configs(
                    arch,
                    args.config_dir,
                    args.list_stats,
                    args.filter_metrics,
                    sys_info.iloc[0],
                    getattr(self, "_profiling_config", {}),
                )

        self.load_options(normalization_filter)

        for path_info in args.path:
            w = schema.Workload()
            sysinfo_path = path_info[0]
            if sysinfo_path:
                w.sys_info = pd.read_csv(f"{sysinfo_path}/sysinfo.csv")
                if not getattr(args, "no_roof", False):
                    # Validate roofline CSV before loading

                    is_valid, error_msg = validate_roofline_csv(sysinfo_path)

                    if is_valid:
                        try:
                            roofline_df = pd.read_csv(f"{sysinfo_path}/roofline.csv")
                            w.roofline_peaks = roofline_df
                        except Exception as e:
                            console_error(
                                "roofline",
                                f"Failed to load roofline.csv: {e}",
                                exit=False,
                            )
                            w.roofline_peaks = pd.DataFrame()
                    else:
                        console_log(
                            "roofline",
                            f"Roofline analysis skipped: {error_msg}",
                        )
                        w.roofline_peaks = pd.DataFrame()
                else:
                    w.roofline_peaks = pd.DataFrame()

                arch = w.sys_info.iloc[0]["gpu_arch"]
                socs = self.get_socs()
                if socs and arch in socs:
                    mspec = socs[arch]._mspec
                    if args.specs_correction:
                        w.sys_info = parser.correct_sys_info(
                            mspec, args.specs_correction
                        )
                w.avail_ips = w.sys_info["ip_blocks"].item().split("|")
                w.dfs = copy.deepcopy(self._arch_configs[arch].dfs)
                w.dfs_type = self._arch_configs[arch].dfs_type
                self._runs[path_info[0]] = w

        return self._runs

    @demarcate
    def sanitize(self) -> None:
        """Perform sanitization of inputs"""
        args = self.get_args()

        if args.tui:
            return

        if not args.path:
            console_error("The following arguments are required: -p/--path")

        # verify not accessing parent directories
        if ".." in str(args.path):
            console_error(
                "Access denied. Cannot access parent directories in path (i.e. ../)"
            )

        # ensure absolute path
        seen_paths: set[str] = set()
        seen_workload_names: set[tuple[str, ...]] = set()
        for dir_info in args.path:
            full_path = Path(dir_info[0]).absolute().resolve()
            dir_info[0] = str(full_path)

            if not full_path.is_dir():
                console_error(
                    "analysis", f"Invalid directory {full_path}\nPlease try again."
                )
            # validate profiling data

            if dir_info[0] in seen_paths:
                console_error("analysis", "You cannot provide the same path twice.")
            seen_paths.add(dir_info[0])

            # The pair names the workload's row and its source export folder.
            workload_name = full_path.parts[-2:]
            if workload_name in seen_workload_names:
                console_error(
                    "analysis",
                    f"{full_path} reuses the workload name "
                    f"{'/'.join(workload_name)}. Paths must differ in their "
                    "last two components.",
                )
            seen_workload_names.add(workload_name)

        self._profiling_config: dict[str, Any] = file_io.load_profiling_config(
            args.path[0][0]
        )
        profiling_config = self.get_profiling_config()

        # --ml-api-trace enables every backend.
        ml_api_trace = profiling_config.get("ml_api_trace", False)
        for backend in KNOWN_ML_API_BACKENDS:
            needs_trace = getattr(
                args, f"{backend}_operator", None
            ) is not None or getattr(args, f"list_{backend}_operators", False)
            if needs_trace and not (
                profiling_config.get(f"{backend}_trace", False) or ml_api_trace
            ):
                console_error(
                    "ml api trace",
                    f'Workload was not profiled with "--{backend}-trace" or '
                    '"--ml-api-trace". '
                    f"Cannot use --{backend}-operator or "
                    f"--list-{backend}-operators.",
                )

        for dir_info in args.path:
            if not any([
                profiling_config.get("iteration_multiplexing"),
                self.pc_sampling_only(),
            ]):
                validate_workload(dir_info[0])

        # Ensure analysis output does not overwrite existing files
        if args.output_name:
            if not re.match(r"^[A-Za-z0-9_-]+$", args.output_name):
                console_error(
                    "analysis",
                    "Analysis output file/folder name must "
                    "contain only alphanumeric characters "
                    "or underscores (_), hyphens (-).",
                )

            path_to_check = args.output_name
            if args.output_format in ("txt", "db"):
                path_to_check += f".{args.output_format}"

            if Path(path_to_check).exists():
                console_error(
                    f"Analysis output file/folder {path_to_check} already exists. "
                    "Please choose a different name."
                )

        if profiling_config.get("iteration_multiplexing") is not None:
            console_log(
                "analysis",
                (
                    "Profiling data was collected using iteration multiplexing.\n\t"
                    "Metrics are calculated based on partially available counter data."
                ),
            )

    @demarcate
    def concat_result_csvs(self, result_files: list[Path], output_file: Path) -> None:
        """Vertically concatenate rocpd ``results_*.csv.gz`` files into one CSV.

        Every file shares the long-form header rocpd writes, so the header is
        taken from the first non-empty file and the remaining rows are appended.

        Args:
            result_files: The results_*.csv.gz files to concatenate
            output_file: Destination gzip CSV
        """
        console_warning(
            "Reading intermediate results_*.csv.gz files is deprecated and "
            "will be removed in a future release."
        )

        rows_written = 0
        with csv_compression.open_gzip_csv_write(output_file) as outfile:
            writer = None
            for file in result_files:
                # Corruption comes from the source read, not the gzip write.
                try:
                    with csv_compression.open_gzip_csv_read(file) as infile:
                        reader = csv.reader(infile)
                        header = next(reader, None)
                        if header is None:
                            console_warning(f"Skipping empty {file}")
                            continue
                        if "Counter_Name" not in header:
                            output_file.unlink(missing_ok=True)
                            console_error(
                                f"{file} is not in the supported rocpd format. "
                                "Please re-profile this workload with a current "
                                "release."
                            )
                        if writer is None:
                            writer = csv.writer(outfile)
                            writer.writerow(header)
                        for row in reader:
                            writer.writerow(row)
                            rows_written += 1
                except csv_compression.CORRUPT_CSV_ERRORS as e:
                    # Drop the partial output built from earlier files.
                    output_file.unlink(missing_ok=True)
                    console_error(
                        f"{file} is truncated or corrupt: {e}\n"
                        "A profile run killed mid-write leaves this behind; "
                        "re-run 'rocprof-compute profile' to regenerate the "
                        "workload."
                    )

        # A header-only output would be reused by later analyze runs.
        if rows_written == 0:
            output_file.unlink(missing_ok=True)
            console_error(
                f"No counter data in results_*.csv.gz under {output_file.parent}.\n"
                f"Please re-run 'rocprof-compute profile'."
            )

        console_debug(f"Created file: {output_file} ({rows_written} counter rows)")

    def join_workload_csvs(self, workload_dir: Path) -> None:
        """Concatenate results_*.csv.gz source files into pmc_perf.csv.gz if needed.

        Args:
            workload_dir: Path to the workload directory
        """
        pmc_perf = csv_compression.compressed_name(
            workload_dir / f"{schema.PMC_PERF_FILE_PREFIX}.csv"
        )
        results_glob = f"results_*.csv{csv_compression.GZIP_SUFFIX}"
        result_files = sorted(workload_dir.glob(results_glob))

        if pmc_perf.exists() and pmc_perf.stat().st_size > 0:
            console_debug(f"Using existing {pmc_perf}")
        elif result_files:
            console_log(f"Joining {results_glob} for {workload_dir}...")
            self.concat_result_csvs(result_files, pmc_perf)
            console_log(f"Created {pmc_perf}")
        else:
            console_error(
                f"No profiling data found in {workload_dir}.\n"
                f"Expected: {pmc_perf.name} or {results_glob}\n"
                f"Please run 'rocprof-compute profile' first."
            )

    # ----------------------------------------------------
    # Required methods to be implemented by child classes
    # ----------------------------------------------------
    @abstractmethod
    def pre_processing(self) -> None:
        """Perform initialization prior to analysis."""
        console_debug("analysis", "prepping to do some analysis")
        console_log("analysis", "deriving rocprofiler-compute metrics...")
        args = self.get_args()

        # initalize output file
        if args.output_format == "txt":
            output_filename = args.output_name or f"rocprof_compute_{get_uuid()}"
            output_filename += ".txt"
            self._output = open(output_filename, "w+", encoding="utf-8")
            console_warning("analysis", f"Created file: {output_filename}")
        elif args.output_format == "stdout":
            self._output = sys.stdout

        # initalize runs
        self._runs = self.initalize_runs()

        # set filters
        filter_configs = [
            (args.gpu_kernel, "filter_kernel_ids"),
            (args.gpu_id, "filter_gpu_ids"),
            (args.gpu_dispatch_id, "filter_dispatch_ids"),
        ]

        for filter_list, attr_name in filter_configs:
            if not filter_list:
                continue

            # Extend single filter to match all paths
            if len(filter_list) == 1 and len(args.path) > 1:
                filter_list *= len(args.path)

            # Apply filters to workloads
            for path_info, filter_value in zip(args.path, filter_list):
                setattr(self._runs[path_info[0]], attr_name, filter_value)

        if not self.pc_sampling_only():
            # Join results_*.csv.gz source files into pmc_perf.csv.gz if needed
            for path_info in args.path:
                workload_dir = Path(path_info[0])
                self.join_workload_csvs(workload_dir)

    @abstractmethod
    def run_analysis(self) -> None:
        """Run analysis."""
        console_debug("analysis", "generating analysis")
