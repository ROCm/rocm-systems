# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import importlib
import shutil
import socket
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import config
from argparser import omniarg_parser
from rocprof_compute_soc.soc_base import OmniSoC_Base
from roofline.run_benchmark import run_roofline_benchmark
from utils.logger import (
    console_debug,
    console_error,
    console_log,
    console_warning,
    demarcate,
    setup_console_handler,
    setup_file_handler,
    setup_logging_priority,
)
from utils.mi_gpu_spec import mi_gpu_specs
from utils.specs import (
    MachineSpecs,
    generate_machine_specs,
)
from utils.utils_common import (
    build_metric_list,
    canonical_config_arch,
    detect_rocprof,
    get_arch_panel_id_to_alias,
    get_job_rank_and_size,
    get_version,
    get_version_display,
    load_panel_configs,
    parse_sets_yaml,
    reconfigure_stdio_utf8,
    replace_env,
    replace_rank,
    resolve_rocm_library_path,
    validate_roofline_csv,
)
from utils.utils_exceptions import WorkloadCommandError
from utils.utils_profile import get_submodules


class RocProfCompute:
    def __init__(self) -> None:
        self.__args: Optional[argparse.Namespace] = None
        self.__profiler_mode = None
        self.__analyze_mode = None
        self.__soc: dict[str, OmniSoC_Base] = {}
        self.__version: dict[str, Optional[str]] = {"ver": None, "ver_pretty": None}
        self.__supported_archs = mi_gpu_specs.get_gpu_series_dict()
        self.__mspec: MachineSpecs  # to be initialized in load_soc_specs()
        self.__parser: argparse.ArgumentParser  # to be initialized in parse_args()

        setup_console_handler()
        self.set_version()
        self.parse_args()
        assert self.__args is not None
        self.__mode = self.__args.mode

        gui_value = getattr(self.__args, "gui", None)
        self.__loglevel = setup_logging_priority(
            self.__args.verbose, self.__args.quiet, self.__mode, gui_value
        )
        setattr(self.__args, "loglevel", self.__loglevel)
        reconfigure_stdio_utf8()

        self.sanitize()

        # Skip machine specs generation for general list options that don't need it
        # or will generate it themselves
        skip_machine_specs = (
            getattr(self.__args, "specs", False)
            or self.__args.list_metrics is not None
            or self.__args.list_blocks is not None
        )

        if self.__mode != "analyze" and not skip_machine_specs:
            self.generate_machine_specs()

        self.handle_list_args()

        if self.__mode == "profile":
            if not getattr(self.__args, "bench_only", False):
                self.detect_profiler()
        elif self.__mode == "analyze":
            self.detect_analyze()

        console_debug(f"Execution mode = {self.__mode}")

    def print_graphic(self) -> None:
        print(
            r"""
                                 __                                       _
 _ __ ___   ___ _ __  _ __ ___  / _|       ___ ___  _ __ ___  _ __  _   _| |_ ___
| '__/ _ \ / __| '_ \| '__/ _ \| |_ _____ / __/ _ \| '_ ` _ \| '_ \| | | | __/ _ \
| | | (_) | (__| |_) | | | (_) |  _|_____| (_| (_) | | | | | | |_) | |_| | ||  __/
|_|  \___/ \___| .__/|_|  \___/|_|        \___\___/|_| |_| |_| .__/ \__,_|\__\___|
               |_|                                           |_|
"""
        )

    def get_mode(self) -> Optional[str]:
        return self.__mode

    def set_version(self) -> None:
        vData = get_version(config.rocprof_compute_home)
        self.__version["ver"] = vData["version"]
        self.__version["ver_pretty"] = get_version_display(
            vData["version"], vData["sha"], vData["mode"]
        )

    def detect_profiler(self) -> None:
        profiler_mode = detect_rocprof(self.__args)
        if str(profiler_mode).endswith("rocprofv3"):
            self.__profiler_mode = "rocprofv3"
        elif str(profiler_mode) == "rocprofiler-sdk":
            self.__profiler_mode = "rocprofiler-sdk"
        else:
            console_error(
                f"Incompatible profiler: {profiler_mode}. Supported profilers "
                f"include: {get_submodules('rocprof_compute_profile')}"
            )

    def detect_analyze(self) -> None:
        if self.__args.gui:
            self.__analyze_mode = "web_ui"
        elif self.__args.tui:
            self.__analyze_mode = "tui"
        elif self.__args.output_format in ("db", "csv"):
            self.__analyze_mode = "db"
        else:
            self.__analyze_mode = "cli"

    def sanitize(self) -> None:
        if self.__args.mode is None and not (
            getattr(self.__args, "list_metrics", False)
            or getattr(self.__args, "list_blocks", False)
            or getattr(self.__args, "specs", False)
        ):
            self.__parser.print_help(sys.stderr)
            console_error(
                "rocprof-compute requires you to pass a valid mode. Detected None."
            )

        self._validate_list_option_exclusions()

        # Validate block 30 / block 21 require their respective experimental flags
        filter_list: list[str] = []
        if hasattr(self.__args, "filter_blocks") and self.__args.filter_blocks:
            filter_list = self.__args.filter_blocks
        elif hasattr(self.__args, "filter_metrics") and self.__args.filter_metrics:
            filter_list = self.__args.filter_metrics

        for block_input in filter_list:
            # Check if this is block 30 (starts with "30" or "30.")
            if block_input.startswith("30") and (
                len(block_input) == 2 or block_input[2] == "."
            ):
                if not self.__args.membw_analysis or not self.__args.experimental:
                    console_error(
                        "Block 30 (Memory Bandwidth Analysis) is an experimental "
                        "feature.\n"
                        f'To use "-b {block_input}", you must also specify: '
                        "--membw-analysis --experimental"
                    )
            # Block 21 (PC sampling) is profile-only; analyze auto-detects it
            # from the profiling config yaml.
            if self.__mode == "profile" and block_input in ("21", "pc_sampling"):
                if not self.__args.pc_sampling or not self.__args.experimental:
                    console_error(
                        "Block 21 (PC Sampling) is an experimental feature.\n"
                        f'To use "-b {block_input}", you must also specify: '
                        "--pc-sampling --experimental"
                    )

        # When --pc-sampling is set, inject "21" into filter_blocks so the
        # profiling config yaml records it and downstream code is unchanged.
        if self.__mode == "profile" and self.__args.pc_sampling:
            current = list(self.__args.filter_blocks or [])
            if "21" not in current:
                current.append("21")
            self.__args.filter_blocks = current

        if self.__mode == "profile":
            self._validate_profile_mode_arguments()
            self._resolve_pc_sampling_interval()

        # fallback to csv output format, if rocpd public api not available
        if self.__mode == "profile" and self.__args.format_rocprof_output == "rocpd":
            rocpd_path = resolve_rocm_library_path(
                str(
                    Path(self.__args.rocprofiler_sdk_tool_path).parents[1]
                    / "librocprofiler-sdk-rocpd.so"
                )
            )
            if not Path(rocpd_path).exists():
                console_warning(
                    "rocpd output format is not supported with the "
                    "current rocprofiler-sdk version. "
                    "Falling back to csv output format."
                )
                self.__args.format_rocprof_output = "csv"

        # Validate name and output directory arguments in profiling mode
        # Skip validation if only listing metrics or sets
        if self.__mode == "profile" and (
            self.__args.list_metrics is None
            and not getattr(self.__args, "list_available_metrics", False)
            and not getattr(self.__args, "list_sets", False)
            and not getattr(self.__args, "list_blocks", False)
            and not getattr(self.__args, "specs", False)
        ):
            if self.__args.name is None and self.__args.output_directory == str(
                Path.cwd() / "workloads"
            ):
                console_error("Either --output-directory or --name is required")

            if self.__args.name is not None and "/" in self.__args.name:
                console_error('"/" is not permitted in profile name')

    def replace_parameters_in_output_directory(self) -> None:
        """Replace parameters in output directory path"""
        # Add --name to output directory if --output-directory is not given
        if self.__args.output_directory == str(Path.cwd() / "workloads"):
            self.__args.output_directory = str(
                Path(self.__args.output_directory) / self.__args.name
            )

            # Add MPI rank to workload path if available
            mpi_rank, _ = get_job_rank_and_size()
            if mpi_rank is not None:
                self.__args.output_directory = str(
                    Path(self.__args.output_directory) / f"{mpi_rank}"
                )
            # OR, Add gpu model name to workload path
            else:
                self.__args.output_directory = str(
                    Path(self.__args.output_directory) / self.__mspec.gpu_model
                )
            return  # exit function after this block
        elif self.__args.name is not None:
            console_warning(
                "--name is ignored when --output-directory is explicitly specified."
            )

        # Add MPI rank to workload path if %rank% is not present in output directory
        # and rank is available
        mpi_rank, _ = get_job_rank_and_size()
        if "%rank%" not in self.__args.output_directory and mpi_rank is not None:
            self.__args.output_directory = str(
                Path(self.__args.output_directory) / f"{mpi_rank}"
            )

        # Replace parameters with actual values in workload path
        self.__args.output_directory = self.__args.output_directory.replace(
            "%hostname%", socket.gethostname()
        ).replace("%gpumodel%", self.__mspec.gpu_model)

        # Replace environment variables in workload path
        self.__args.output_directory = replace_env(self.__args.output_directory)

        # Replace %rank% with actual rank value in workload path
        if "%rank%" in self.__args.output_directory and mpi_rank is None:
            console_warning(
                "Ignoring %%rank%% placeholder in output directory"
                " since no MPI rank was detected."
            )
        self.__args.output_directory = replace_rank(self.__args.output_directory)

    @demarcate
    def generate_machine_specs(self) -> None:
        """Generate MachineSpecs for RocProfCompute"""
        self.__mspec = generate_machine_specs(self.__args)

    @demarcate
    def load_soc_specs(self, sysinfo: Optional[dict] = None) -> None:
        """
        Load OmniSoC instance for RocProfCompute run

        If sysinfo is provided (e.g., in analyze mode from sysinfo.csv),
        regenerate the MachineSpecs from that data instead of using the
        current host's characteristics.
        """
        if sysinfo is not None:
            # Regenerate machine specs based on the provided sysinfo rather than
            # the current host. This is important for analyze mode where we may
            # be running on a different machine than the one that produced the data.
            self.__mspec = generate_machine_specs(self.__args, sysinfo)

        arch = self.__mspec.gpu_arch
        soc_module = importlib.import_module(f"rocprof_compute_soc.soc_{arch}")
        soc_class = getattr(soc_module, f"{arch}_soc")
        self.__soc[arch] = soc_class(self.__args, self.__mspec)

    def parse_args(self) -> None:
        # Detect if --experimental flag is present (for help text control)
        prelim_parser = argparse.ArgumentParser(add_help=False)
        prelim_parser.add_argument("--experimental", action="store_true", default=False)

        # Parse only known args (respects -- separator)
        prelim_args, _ = prelim_parser.parse_known_args()
        experimental_requested: bool = prelim_args.experimental

        # Build full parser with experimental knowledge
        self.__parser = argparse.ArgumentParser(
            description=(
                "Command line interface for AMD's GPU profiler, ROCm Compute Profiler"
            ),
            prog="tool",
            formatter_class=lambda prog: argparse.RawTextHelpFormatter(
                prog, max_help_position=30
            ),
            usage="rocprof-compute [mode] [options]",
        )
        omniarg_parser(
            self.__parser,
            config.rocprof_compute_home,
            self.__supported_archs,
            self.__version,
            experimental_requested,
        )
        self.__args = self.__parser.parse_args()

        if self.__args.mode == "profile":
            self.handle_profile_args()
        elif self.__args.mode == "analyze":
            self.handle_analyze_args()

    def handle_profile_args(self) -> None:
        # Handle list operations first - these are independent and exit immediately
        if getattr(self.__args, "list_sets", False):
            return
        if getattr(self.__args, "list_available_metrics", False):
            return

    def handle_analyze_args(self) -> None:
        """Handle analyze-specific argument processing"""
        args = self.__args
        torch_operator = args.torch_operator
        list_torch_operators = args.list_torch_operators

        if torch_operator is not None or list_torch_operators:
            if args.gui is not None:
                console_error(
                    "torch trace",
                    "--torch-operator and --list-torch-operators are not "
                    "supported in --gui mode. Please remove --gui or run "
                    "without the torch-operator flags.",
                )
            if args.tui:
                console_error(
                    "torch trace",
                    "--torch-operator and --list-torch-operators are not "
                    "supported in --tui mode. Please remove --tui or run "
                    "without the torch-operator flags.",
                )
            if args.spatial_multiplexing:
                console_error(
                    "torch trace",
                    "--torch-operator and --list-torch-operators do not yet "
                    "support multi-node analysis via --spatial-multiplexing. "
                    "Please remove one of these options.",
                )
            if args.output_format != "stdout":
                console_error(
                    "torch trace",
                    "--torch-operator and --list-torch-operators are only "
                    "supported with --output-format stdout (the default). "
                    "The matched operator call tree is printed directly to "
                    "stdout and is not captured in txt, csv, or db output. "
                    "Remove the --output-format option or drop the "
                    "torch-operator flags.",
                )

            if torch_operator is not None:
                if args.list_stats:
                    console_warning(
                        "torch trace",
                        "--torch-operator is ignored by --list-stats; the "
                        "full kernel stats table will be shown regardless "
                        "of the operator filter.",
                    )
                if args.list_nodes:
                    console_warning(
                        "torch trace",
                        "--torch-operator is ignored by --list-nodes; the "
                        "node enumeration does not respect the operator "
                        "filter.",
                    )
                if list_torch_operators:
                    console_warning(
                        "torch trace",
                        "--torch-operator is ignored when "
                        "--list-torch-operators is used; the full operator "
                        "tree will be shown. Drop --list-torch-operators to "
                        "apply the operator filter to the analysis, or drop "
                        "--torch-operator to list all operators.",
                    )

        # Block all filters during spatial-multiplexing
        if self.__args.spatial_multiplexing:
            self.__args.gpu_id = None
            self.__args.gpu_kernel = None
            self.__args.gpu_dispatch_id = None
            self.__args.nodes = None

    @demarcate
    def handle_list_args(self) -> None:
        if self.__args.specs:
            print(generate_machine_specs(self.__args))
            sys.exit(0)
        elif self.__args.list_metrics is not None:
            self.list_metrics()
        elif self.__args.list_blocks is not None:
            self.list_blocks()

        if self.__mode == "profile":
            if self.__args.list_sets:
                self.list_sets()
            elif self.__args.list_available_metrics:
                self.list_metrics()

    @demarcate
    def list_metrics(self) -> None:
        for_current_arch = getattr(self.__args, "list_available_metrics", False)
        arch = self.__mspec.gpu_arch if for_current_arch else self.__args.list_metrics
        config_arch = canonical_config_arch(arch) or arch
        config_root = Path(self.__args.config_dir) / config_arch

        if arch in self.__supported_archs.keys() or config_root.is_dir():
            sys_info = self.__mspec.get_class_members() if for_current_arch else None
            metric_list = self._build_arch_metric_list(arch, sys_info)
            for key, value in metric_list.items():
                prefix = "\t" * min(key.count("."), 2)
                print(f"{prefix}{key} -> {value}")
            sys.exit(0)
        else:
            console_error("Unsupported arch")

    @demarcate
    def list_blocks(self) -> None:
        arch = self.__args.list_blocks
        config_arch = canonical_config_arch(arch) or arch
        config_root = Path(self.__args.config_dir) / config_arch

        if arch in self.__supported_archs.keys() or config_root.is_dir():
            metric_list = self._build_arch_metric_list(arch, sys_info=None)
            top_panels = {k: v for k, v in metric_list.items() if "." not in k}
            panel_alias_dict = get_arch_panel_id_to_alias(arch)
            print(f"{'INDEX':<8} {'BLOCK ALIAS':<16} {'BLOCK NAME'}")
            for key, value in top_panels.items():
                alias = panel_alias_dict.get(key, "")
                print(f"{key:<8} {alias:<16} {value}")
            sys.exit(0)
        else:
            console_error("Unsupported arch")

    def _build_arch_metric_list(
        self,
        arch: str,
        sys_info: Optional[dict[str, Any]],
    ) -> dict[str, str]:
        """Load panel configs for arch and build metric_list.
        Returns the metric_list dictionary."""
        config_arch = canonical_config_arch(arch) or arch
        panel_configs = load_panel_configs([
            str(Path(self.__args.config_dir) / config_arch)
        ])
        return build_metric_list(panel_configs, sys_info)

    @demarcate
    def list_sets(self) -> None:
        sets_info = parse_sets_yaml(self.__mspec.gpu_arch)

        if not sets_info:
            console_error("No sets configuration found.")

        print("\nAvailable Sets:")
        print("=" * 115)

        # Print header
        print(
            f"{'Set Option':<35} {'Set Title':<35}"
            f" {'Metric Name':<30} {'Metric ID':<10}"
        )
        print("-" * 115)

        # Print data grouped by set
        for set_option, set_data in sets_info.items():
            title = set_data.get("title", set_option)
            metrics = set_data.get("metric", [])

            first_row = True
            for metric in metrics:
                if isinstance(metric, dict) and metric:
                    metric_id = next(iter(metric.keys()))
                    metric_name = next(iter(metric.values()))

                    # Only show set info on first row of each set
                    set_display = set_option if first_row else ""
                    title_display = title if first_row else ""

                    print(
                        f"{set_display:<35} {title_display:<35}"
                        f" {metric_name:<30} {metric_id:<10}"
                    )
                    first_row = False
            # Empty line between sets
            print()

        print("Usage Examples:")
        if sets_info:
            first_set = next(iter(sets_info.keys()))
            print(f"  rocprof-compute profile --set {first_set}  # Profile this set")
        print("  rocprof-compute profile --list-sets        # Show this help")
        print()

        sys.exit(0)

    def create_profiler(self) -> object:
        profiler_classes = {
            "rocprofv3": (
                "rocprof_compute_profile.profiler_rocprof_v3",
                "rocprof_v3_profiler",
            ),
            "rocprofiler-sdk": (
                "rocprof_compute_profile.profiler_rocprofiler_sdk",
                "rocprofiler_sdk_profiler",
            ),
        }

        if self.__profiler_mode not in profiler_classes:
            console_error("Unsupported profiler")

        module_name, class_name = profiler_classes[self.__profiler_mode]
        module = importlib.import_module(module_name)
        profiler_class = getattr(module, class_name)

        return profiler_class(
            self.__args,
            self.__profiler_mode,
            self.__soc[self.__mspec.gpu_arch],
        )

    @demarcate
    def run_profiler(self) -> None:
        self.print_graphic()

        self.replace_parameters_in_output_directory()

        if self.__args.bench_only:
            self._run_bench_only()
            return

        self.load_soc_specs()

        # instantiate desired profiler
        profiler = self.create_profiler()
        try:
            profiler.sanitize()
        except WorkloadCommandError as e:
            console_error(str(e))

        # Create workload directory if it does not exist
        p = Path(self.__args.output_directory)
        if not p.exists():
            try:
                p.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                console_error("Directory already exists.")

        # enable file-based logging
        setup_file_handler(self.__args.loglevel, self.__args.output_directory)

        profiler.pre_processing()

        console_debug('starting "run_profiling" and about to start rocprof\'s workload')

        time_start_prof = time.time()
        profiler.run_profiling(self.__version["ver"], config.PROJECT_NAME)
        time_end_prof = time.time()

        prof_duration = time_end_prof - time_start_prof
        console_debug(
            f'finished "run_profiling" and finished rocprof\'s workload, '
            f"time taken was {int(prof_duration / 60)} m {prof_duration % 60} sec"
        )

        profiler.post_processing()
        time_end_post = time.time()

        post_duration = int(time_end_post - time_end_prof)
        console_debug(f'time taken for "post_processing" was {post_duration} seconds')

    def _validate_profile_mode_arguments(self) -> None:
        """Validate that the profile-mode invocation is internally consistent.

        Covers the mutual exclusion among action-selection flags
        (--block, --set, --roof-only, --bench-only) and the
        --bench-only / --no-roof conflict.
        """
        args = self.__args
        if (
            sum((
                bool(getattr(args, "filter_blocks", None)),
                bool(getattr(args, "set_selected", None)),
                bool(getattr(args, "roof_only", False)),
                bool(getattr(args, "bench_only", False)),
            ))
            > 1
        ):
            console_error(
                "--block, --set, --roof-only, and --bench-only"
                " are mutually exclusive options."
                " Please use only one of them."
            )

        if getattr(args, "bench_only", False) and getattr(args, "no_roof", False):
            console_error("--bench-only cannot be used with --no-roof.")

    def _resolve_pc_sampling_interval(self) -> None:
        """Apply the method-aware default for --pc-sampling-interval and
        validate a user-supplied value."""
        args = self.__args
        if not getattr(args, "pc_sampling", False):
            return

        stochastic_default_interval_in_cycles = 1048576
        stochastic_min_interval_in_cycles = 65536
        host_trap_default_interval_in_microseconds = 512

        method = args.pc_sampling_method
        if args.pc_sampling_interval is None:
            if method == "stochastic":
                args.pc_sampling_interval = stochastic_default_interval_in_cycles
            else:
                args.pc_sampling_interval = host_trap_default_interval_in_microseconds
            return

        interval = args.pc_sampling_interval
        if method == "stochastic":
            is_power_of_two = interval > 0 and interval & (interval - 1) == 0
            if not is_power_of_two or interval < stochastic_min_interval_in_cycles:
                console_error(
                    "--pc-sampling-interval for stochastic sampling must be a "
                    f"power of 2 and at least {stochastic_min_interval_in_cycles} "
                    f"(got {interval})."
                )
        elif interval <= 0:
            console_error(
                "--pc-sampling-interval for host_trap sampling must be a "
                f"positive integer (got {interval})."
            )

    def _validate_list_option_exclusions(self) -> None:
        """Validate that list/discovery options aren't combined with --block.
        Applies to both profile and analyze mode.
        """
        args = self.__args
        block_active = bool(
            getattr(args, "filter_blocks", None)
            or getattr(args, "filter_metrics", None)
        )
        if not block_active:
            return

        if args.list_metrics is not None:
            console_error("Cannot use --list-metrics with --blocks")
        if args.list_blocks is not None:
            console_error("Cannot use --list-blocks with --blocks")
        if getattr(args, "list_available_metrics", False):
            console_error("Cannot use --list-available-metrics with --blocks")

    @demarcate
    def _run_bench_only(self) -> None:
        """Run standalone roofline microbenchmark execution.

        The microbenchmark is written to a temp location first and only
        promoted to the workload directory after passing validation.
        """
        output_dir = Path(self.__args.output_directory)
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        setup_file_handler(self.__args.loglevel, str(output_dir))

        roofline_csv = output_dir / "roofline.csv"
        existing_roofline = roofline_csv.is_file()
        console_log(
            "roofline",
            f"Running roofline microbenchmark on device {self.__args.device}",
        )

        with tempfile.TemporaryDirectory(prefix="rocprof_bench_") as tmp_dir:
            tmp_csv = Path(tmp_dir) / "roofline.csv"
            try:
                run_roofline_benchmark(
                    self.__args.device, tmp_csv, self.__mspec.cache_sizes
                )
            except Exception as e:
                console_error(f"Benchmark execution failed: {e}")

            is_valid, error_message = validate_roofline_csv(tmp_dir)
            if not is_valid:
                console_error(
                    f"Invalid roofline.csv: {error_message}",
                    exit=False,
                )
                return

            shutil.move(str(tmp_csv), str(roofline_csv))

        if existing_roofline:
            console_warning(f"Overwrote existing {roofline_csv}")

        console_log(
            "roofline",
            f"Roofline data saved to {roofline_csv}",
        )

    @demarcate
    def run_analysis(self) -> None:
        # Lazy import pandas and file_io since they are only used in analysis
        # mode. This keeps analysis deps out of the profile path.
        import pandas as pd

        from utils import file_io

        self.print_graphic()
        console_log(f"Analysis mode = {self.__analyze_mode}")

        if self.__analyze_mode == "cli":
            from rocprof_compute_analyze.analysis_cli import cli_analysis

            analyzer = cli_analysis(self.__args, self.__supported_archs)
        elif self.__analyze_mode == "web_ui":
            from rocprof_compute_analyze.analysis_webui import webui_analysis

            analyzer = webui_analysis(self.__args, self.__supported_archs)
        elif self.__analyze_mode == "tui":
            from rocprof_compute_tui.tui_app import run_tui

            run_tui(self.__args, self.__supported_archs)
            return
        elif self.__analyze_mode == "db":
            from rocprof_compute_analyze.analysis_db import db_analysis

            analyzer = db_analysis(self.__args, self.__supported_archs)
        else:
            console_error(f"Unsupported analysis mode -> {self.__analyze_mode}")

        # -----------------------
        # run analysis workflow
        # -----------------------
        analyzer.sanitize()

        # Load required SoC(s) from input
        for path_list in analyzer.get_args().path:
            base_path = path_list[0] if isinstance(path_list, list) else path_list

            # Determine sysinfo path
            if (
                analyzer.get_args().nodes is None
                and not analyzer.get_args().spatial_multiplexing
            ):
                sysinfo_path = base_path
            else:
                sysinfo_path = file_io.find_1st_sub_dir(base_path)

            sys_info = pd.read_csv(f"{sysinfo_path}/sysinfo.csv")
            sys_info_dict = {
                key: value[0] for key, value in sys_info.to_dict("list").items()
            }
            self.load_soc_specs(sys_info_dict)

        if getattr(self.__args, "list_available_metrics", False):
            self.list_metrics()

        analyzer.set_soc(self.__soc)
        analyzer.pre_processing()
        analyzer.run_analysis()
