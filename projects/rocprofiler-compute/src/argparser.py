# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.inject_roctx.constants import KNOWN_ML_API_BACKENDS, OPERATOR_BACKENDS
from utils.logger import console_warning
from utils.utils_common import METRIC_ID_RE, resolve_rocm_library_path

# Deprecated options by argparse dest: the option and what to use instead
DEPRECATED_OPTIONS: Dict[str, Tuple[str, str]] = {
    "list_blocks": ("--list-blocks", "--list-metrics"),
    "list_available_metrics": ("--list-available-metrics", "--list-metrics"),
}


def warn_deprecated_options(args: argparse.Namespace) -> None:
    """Warn about each deprecated option the command line uses."""
    for dest, (option, replacement) in DEPRECATED_OPTIONS.items():
        if getattr(args, dest, None) not in (None, False):
            console_warning(
                f"{option} is deprecated and will be removed in a future release. "
                f"Use {replacement} instead."
            )


def validate_block(value: str) -> str:
    if METRIC_ID_RE.match(value):
        return value
    raise argparse.ArgumentTypeError(f"Invalid metric id: {value}")


def block_token_or_alias(s: str) -> str:
    try:
        return validate_block(s)
    except argparse.ArgumentTypeError:
        s = (s or "").strip()
        if not s:
            raise argparse.ArgumentTypeError("empty token for --block")
        return s


def non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}")
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"must be a non-negative integer (0 means all), got {parsed}"
        )
    return parsed


def add_general_group(
    parser: argparse.ArgumentParser,
    rocprof_compute_home: Path,
    supported_archs: Dict[str, str],
    rocprof_compute_version: Dict[str, Optional[str]],
) -> None:
    general_group = parser.add_argument_group("General Options")

    general_group.add_argument(
        "-v",
        "--version",
        action="version",
        version=rocprof_compute_version["ver_pretty"],
    )
    general_group.add_argument(
        "-V",
        "--verbose",
        help="Increase output verbosity (use multiple times for higher levels).",
        action="count",
        default=0,
    )
    general_group.add_argument(
        "-q", "--quiet", action="store_true", help="Reduce output and run quietly."
    )
    arch_values = ", ".join(supported_archs)
    general_group.add_argument(
        "--list-metrics",
        dest="list_metrics",
        metavar="id",
        nargs="*",
        help=(
            "List the blocks of the current GPU.\n"
            "Add block or metric [id]s or aliases to list their metrics with units\n"
            'and descriptions, or "all" to list every metric. Each id or alias adds\n'
            "to the list: --list-metrics sol 12.1 lists all of block 2 (sol) and\n"
            "table 12.1.\n"
            "With analyze -p <workload>, list the workload's blocks and metrics."
        ),
    )
    general_group.add_argument(
        "--list-metrics-gfx-id",
        dest="list_metrics_gfx_id",
        metavar="<arch>",
        choices=supported_archs.keys(),
        help=(
            "List the blocks of GPU <arch> instead of the current GPU.\n"
            "Add --list-metrics [id]... to list their metrics.\n"
            f"Values: {arch_values}"
        ),
    )
    general_group.add_argument(
        "--list-blocks",
        dest="list_blocks",
        metavar="<arch>",
        choices=supported_archs.keys(),
        help=(
            "DEPRECATED: use --list-metrics instead.\n"
            "List available blocks and their aliases for specified GPU <arch>.\n"
            f"Values: {arch_values}"
        ),
    )
    general_group.add_argument(
        "--config-dir",
        dest="config_dir",
        metavar="<dir>",
        help=(
            "Specify the directory of customized report section configs "
            "(Default: built-in configs)."
        ),
        default=rocprof_compute_home / "rocprof_compute_soc/analysis_configs/",
    )
    general_group.add_argument(
        "-s", "--specs", action="store_true", help="Print system specs and exit."
    )

    general_group.add_argument(
        "--experimental",
        action="store_true",
        default=False,
        help=(
            "Enable experimental feature(s):\n"
            "   ML API trace (--trace-ml-api, --list-operators, --operator)\n"
            "   Memory Bandwidth Analysis (--membw-analysis)\n"
            "   PC Sampling (--pc-sampling, --pc-sampling-method, "
            "--pc-sampling-interval)\n"
        ),
    )


def experimental_requested(argv: List[str]) -> bool:
    """Return True if --experimental comes before the "--" that starts the workload."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--experimental", action="store_true")
    return parser.parse_known_args(argv)[0].experimental


def build_parser(
    rocprof_compute_home: Path,
    supported_archs: Dict[str, str],
    rocprof_compute_version: Dict[str, Optional[str]],
    experimental_enabled: bool = False,
) -> argparse.ArgumentParser:
    """Return the rocprof-compute command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Command line interface for AMD's GPU profiler, ROCm Compute Profiler"
        ),
        prog="rocprof-compute",
        allow_abbrev=False,
        formatter_class=lambda prog: CliHelpFormatter(prog, max_help_position=30),
        usage="rocprof-compute [mode] [options]",
    )
    omniarg_parser(
        parser,
        rocprof_compute_home,
        supported_archs,
        rocprof_compute_version,
        experimental_enabled,
    )
    return parser


def omniarg_parser(
    parser: argparse.ArgumentParser,
    rocprof_compute_home: Path,
    supported_archs: Dict[str, str],
    rocprof_compute_version: Dict[str, Optional[str]],
    experimental_enabled: bool = False,
) -> None:
    # -----------------------------------------
    # Parse arguments (dependent on mode)
    # -----------------------------------------

    ## General Command Line Options
    ## ----------------------------
    add_general_group(
        parser,
        rocprof_compute_home,
        supported_archs,
        rocprof_compute_version,
    )
    parser._positionals.title = "Modes"
    parser._optionals.title = "Help"
    skills_note = _skills_note(rocprof_compute_home)
    if skills_note is not None:
        parser.description = f"{parser.description}\n\n{skills_note}"

    subparsers = parser.add_subparsers(
        dest="mode", help="Select mode of interaction with the target application:"
    )

    ## Profile Command Line Options
    ## ----------------------------
    profile_parser = subparsers.add_parser(
        "profile",
        help="Profile the target application",
        description=skills_note,
        usage="""

`rocprof-compute profile --name <workload_name> [profile options] [roofline options] -- <workload_cmd>`

---------------------------------------------------------------------------------
Examples:
\trocprof-compute profile -n vcopy_all -- ./vcopy -n 1048576 -b 256
\trocprof-compute profile -n vcopy_blocks -b sol -- ./vcopy -n 1048576 -b 256
\trocprof-compute profile -n vcopy_kernel -k vecCopy -- ./vcopy -n 1048576 -b 256
\trocprof-compute profile -n vcopy_iter --kernel-iteration-range 1 -- ./vcopy -n 1048576 -b 256
\trocprof-compute profile -n vcopy_roof --roof-only -- ./vcopy -n 1048576 -b 256
\trocprof-compute profile -n my_bench --bench-only
---------------------------------------------------------------------------------
        """,  # noqa: E501
        prog="rocprof-compute",
        allow_abbrev=False,
        formatter_class=CliHelpFormatter,
    )
    profile_parser._optionals.title = "Help"

    add_general_group(
        profile_parser,
        rocprof_compute_home,
        supported_archs,
        rocprof_compute_version,
    )
    profile_group = profile_parser.add_argument_group("Profile Options")
    roofline_group = profile_parser.add_argument_group("Roofline Options")

    profile_group.add_argument(
        "-n",
        "--name",
        type=str,
        metavar="<name>",
        dest="name",
        help=(
            "Assign a name to workload.\n"
            "--name will be ignored if used together with --output-directory.\n"
            "Use --overwrite to re-profile into an existing directory."
        ),
    )
    profile_group.add_argument(
        "--attach-pid",
        type=str,
        dest="attach_pid",
        metavar="<pid>",
        default=None,
        required=False,
        help="Process id to be attached for profiling.\nImplies --no-native-tool.",
    )
    profile_group.add_argument(
        "--attach-duration-msec",
        type=str,
        dest="attach_duration_msec",
        metavar="<msec>",
        default=None,
        required=False,
        help=(
            "When --attach-pid is used, it specifies the attach duration\n"
            "in milliseconds. If not set, detachment occurs when\n"
            '"Enter" key is pressed.'
        ),
    )
    profile_group.add_argument(
        "-d",
        "--output-directory",
        metavar="<dir>",
        type=str,
        dest="output_directory",
        default=str(Path.cwd() / "workloads"),
        required=False,
        help=(
            "Specify output directory to save workload.\n"
            "Output directory can also be parameterized with the following keywords:\n"
            "   %%hostname%%: Host name\n"
            "   %%gpumodel%%: GPU model\n"
            "   %%rank%%: MPI process rank\n"
            '   %%env{NAME}%%: Environment variable "NAME"\n'
            "Use --overwrite to re-profile into an existing directory.\n"
            "(Default: <current-working-directory>/workloads/<name>/%%gpumodel%% "
            "without MPI,\n"
            " <current-working-directory>/workloads/<name>/%%rank%% with MPI)"
        ),
    )
    profile_group.add_argument(
        "--overwrite",
        dest="overwrite",
        required=False,
        default=False,
        action="store_true",
        help=(
            "Overwrite an existing workload directory.\n"
            "Without it, profiling into a non-empty directory fails\n"
            "instead of mixing runs. Use a fresh directory per run;\n"
            "pass this flag only to re-profile in place."
        ),
    )
    profile_group.add_argument(
        "-k",
        "--kernel",
        type=str,
        dest="kernel",
        metavar="<regexes>",
        required=False,
        nargs="+",
        default=None,
        help="Profile only kernels whose names match one of the regular expressions.",
    )
    profile_group.add_argument(
        "--kernel-iteration-range",
        metavar="<ranges>",
        nargs="+",
        dest="kernel_iteration_range",
        required=False,
        help=(
            "Which iterations of each kernel to profile\n"
            "(1-based; positive integer or 'start:end'/'start-end'\n"
            "range, e.g. 1 3:5 captures 1st, 3rd, 4th and 5th\n"
            "iterations)."
        ),
    )
    profile_group.add_argument(
        "--iteration-multiplexing",
        type=str,
        dest="iteration_multiplexing",
        metavar="policy",
        required=False,
        nargs="?",
        choices=[
            "kernel",
            "kernel_launch_params",
        ],
        const="kernel_launch_params",
        help=(
            "Choose the iteration multiplexing policy:\n"
            "   kernel (i.e. Round robin counters over kernel calls with "
            "unique kernel names.)\n"
            "   kernel_launch_params (i.e. Round robin counters over "
            "kernel calls with unique kernel and launch parameters)\n"
            "(Default when given without a value: kernel_launch_params)\n"
            "Values: kernel, kernel_launch_params"
        ),
    )

    profile_group.add_argument(
        "--list-available-metrics",
        dest="list_available_metrics",
        help=(
            "DEPRECATED: use --list-metrics instead.\n"
            "List all available metrics for analysis on current arch."
        ),
        action="store_true",
    )
    profile_group.add_argument(
        "-b",
        "--block",
        dest="filter_blocks",
        metavar="<ids>",
        nargs="+",
        type=block_token_or_alias,
        required=False,
        default=[],
        help=(
            "Specify metric id(s) from --list-metrics for filtering "
            "(e.g. 12 12.1 12.1.1).\n"
            "Alternatively, specify block id(s) for filtering "
            "(e.g. 12 13 14).\n"
            "Alternatively, specify block alias(es) for filtering.\n"
            "Aliases are arch-specific; run --list-metrics to see\n"
            "all valid block ids and aliases.\n"
            "Give all ids after one -b; a repeated -b keeps only the last list.\n"
            "Cannot be used with --set, --roof-only or --bench-only."
        ),
    )
    profile_group.add_argument(
        "--list-sets",
        action="store_true",
        help="Display available metric sets and their descriptions.",
    )
    profile_group.add_argument(
        "--set",
        default=None,
        dest="set_selected",
        metavar="<set>",
        help=(
            "Profile a set of metrics of topic of interest by collecting "
            "counters in a single pass.\n"
            "For available sets, see --list-sets.\n"
            "Cannot be used with --block, --roof-only or --bench-only."
        ),
    )
    profile_group.add_argument(
        "--no-roof",
        required=False,
        default=False,
        action="store_true",
        help="Profile without collecting roofline data.",
    )
    profile_group.add_argument(
        "remaining",
        metavar="-- <workload_cmd>",
        default=None,
        nargs=argparse.REMAINDER,
        help="Provide command for profiling after double dash.",
    )
    default_sdk_tool_path = (
        Path(os.getenv("ROCM_PATH", "/opt/rocm"))
        / "lib/rocprofiler-sdk/librocprofiler-sdk-tool.so"
    )
    profile_group.add_argument(
        "--rocprofiler-sdk-tool-path",
        type=resolve_rocm_library_path,
        dest="rocprofiler_sdk_tool_path",
        metavar="<path>",
        required=False,
        default=resolve_rocm_library_path(str(default_sdk_tool_path)),
        help=(
            "Set the path to rocprofiler-sdk tool.\n"
            "(Default: $ROCM_PATH/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so)"
        ),
    )
    profile_group.add_argument(
        "--no-native-tool",
        required=False,
        default=False,
        action="store_true",
        help=(
            "Do not use the native counter collection tool (advanced).\n"
            "Native tool is not used if ROCPROF env. var. is set "
            "and not equal to rocprofiler-sdk.\n"
            "Native tool is not used for ROCm version < 7.x.x.\n"
            "Native tool is not used in attach/detach scenarios."
        ),
    )
    profile_group.add_argument(
        "--retain-rocpd-output",
        required=False,
        default=False,
        action="store_true",
        help=(
            "DEPRECATED: .db files will be retained automatically in a "
            "future release.\n"
            "Retain the large raw rocpd database in workload directory."
        ),
    )

    ## Roofline Command Line Options
    roofline_group.add_argument(
        "--roof-only",
        dest="roof_only",
        required=False,
        default=False,
        action="store_true",
        help=(
            "Profile roofline data only.\n"
            "Cannot be used with --block, --set or --bench-only."
        ),
    )
    roofline_group.add_argument(
        "--bench-only",
        dest="bench_only",
        required=False,
        default=False,
        action="store_true",
        help=(
            "Run roofline microbenchmark only.\n"
            "No application profiling or counter collection.\n"
            "No application run is required.\n"
            "Cannot be used with --block, --set, --roof-only or --no-roof."
        ),
    )
    roofline_group.add_argument(
        "--device",
        dest="device",
        metavar="<id>",
        required=False,
        default=0,
        type=int,
        help=(
            "Target GPU device ID per amd-smi for roofline benchmarking (Default: 0)."
        ),
    )

    ## ----------------------------
    # Experimental Features
    ## ----------------------------

    profile_group.add_argument(
        "--trace-ml-api",
        dest="trace_ml_api",
        required=False,
        default=None,
        nargs="*",
        choices=KNOWN_ML_API_BACKENDS,
        metavar="framework",
        base_action="store",
        action=ExperimentalAction,
        experimental_enabled=experimental_enabled,
        feature_label="ML API trace",
        help=(
            "Trace ML APIs. User ROCTx regions are always included.\n"
            f"With no arguments, traces {' and '.join(KNOWN_ML_API_BACKENDS)}.\n"
            f"Values: {', '.join(KNOWN_ML_API_BACKENDS)}"
        ),
    )
    profile_group.add_argument(
        "--membw-analysis",
        dest="membw_analysis",
        required=False,
        default=False,
        base_action="store_const",
        action=ExperimentalAction,
        experimental_enabled=experimental_enabled,
        feature_label="Memory Bandwidth Analysis",
        nargs=0,
        const=True,
        help="Enable Memory Bandwidth Analysis counters (block 30).",
    )

    profile_group.add_argument(
        "--pc-sampling",
        dest="pc_sampling",
        required=False,
        default=False,
        base_action="store_const",
        action=ExperimentalAction,
        experimental_enabled=experimental_enabled,
        feature_label="PC Sampling",
        nargs=0,
        const=True,
        help="Enable PC sampling (block 21) for profile mode.",
    )
    profile_group.add_argument(
        "--pc-sampling-method",
        required=False,
        metavar="<method>",
        dest="pc_sampling_method",
        default="stochastic",
        choices=["stochastic", "host_trap"],
        base_action="store",
        action=ExperimentalAction,
        experimental_enabled=experimental_enabled,
        feature_label="PC Sampling",
        help=(
            "Set the method of pc sampling. stochastic requires MI300 or newer "
            "(Default: stochastic).\n"
            "Values: stochastic, host_trap"
        ),
    )
    profile_group.add_argument(
        "--pc-sampling-interval",
        required=False,
        metavar="<interval>",
        dest="pc_sampling_interval",
        default=None,
        type=int,
        base_action="store",
        action=ExperimentalAction,
        experimental_enabled=experimental_enabled,
        feature_label="PC Sampling",
        help=(
            "Set the interval of pc sampling.\n"
            "   stochastic: in cycles, must be a power of 2.\n"
            "   host_trap: in microseconds.\n"
            "(Default: 1048576 for stochastic, 512 for host_trap)\n"
            "Values: the range the device reports "
            "(see 'rocprofv3-avail info --pc-sampling'),\n"
            "or 1 to 1048576 when the device cannot be queried."
        ),
    )

    ## Analyze Command Line Options
    ## ----------------------------
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze existing profiling results at command line",
        description=skills_note,
        usage="""
rocprof-compute analyze --path <workload_path> [analyze options]

-----------------------------------------------------------------------------------
Examples:
\trocprof-compute analyze -p workloads/vcopy/mi200/ --list-metrics
\trocprof-compute analyze -p workloads/mixbench/mi200/ --dispatch 12 34 --decimal 3
-----------------------------------------------------------------------------------
        """,
        prog="rocprof-compute",
        allow_abbrev=False,
        formatter_class=CliHelpFormatter,
    )
    analyze_parser._optionals.title = "Help"

    add_general_group(
        analyze_parser,
        rocprof_compute_home,
        supported_archs,
        rocprof_compute_version,
    )
    analyze_group = analyze_parser.add_argument_group("Analyze Options")
    analyze_advanced_group = analyze_parser.add_argument_group("Advanced Options")

    analyze_group.add_argument(
        "-p",
        "--path",
        dest="path",
        required=False,
        metavar="<paths>",
        nargs="+",
        action="append",
        help="Specify the raw data root dirs or desired results directory.",
    )
    analyze_group.add_argument(
        "--verify-deps",
        dest="verify_deps",
        action="store_true",
        help="Check the Python dependencies analyze mode needs, then exit.",
    )
    analyze_group.add_argument(
        "--list-stats",
        action="store_true",
        help=(
            "List all detected kernels and kernel dispatches.\n"
            "Cannot be used with --output-format csv or db.\n"
            "With --operator, lists only the kernels and dispatches\n"
            "of the matching operators."
        ),
    )
    analyze_group.add_argument(
        "--list-available-metrics",
        dest="list_available_metrics",
        help=(
            "DEPRECATED: use --list-metrics instead.\n"
            "List all available metrics for analysis on the workload arch."
        ),
        action="store_true",
    )
    analyze_group.add_argument(
        "--list-operators",
        dest="list_operators",
        default=None,
        nargs="*",
        choices=OPERATOR_BACKENDS,
        metavar="framework",
        base_action="store",
        action=ExperimentalAction,
        experimental_enabled=experimental_enabled,
        feature_label="List operators",
        help=(
            "List collected operators as a call tree.\n"
            "With no arguments, lists every collected operator.\n"
            "Ignores --operator.\n"
            f"Values: {', '.join(OPERATOR_BACKENDS)}"
        ),
    )
    analyze_group.add_argument(
        "--operator",
        metavar="pattern",
        type=str,
        dest="operator",
        nargs="*",
        default=None,
        base_action="store",
        action=ExperimentalAction,
        experimental_enabled=experimental_enabled,
        feature_label="Operator filter",
        help=(
            "Filter operators with glob patterns.\n"
            "A pattern without a prefix uses the prefix before it.\n"
            "Before any prefix, a pattern matches every traced\n"
            "framework and user regions.\n"
            "Example: --operator 'torch:*conv*' '*relu*' 'triton:*matmul*'\n"
            "Example: --operator 'user:*UserDefined*'\n"
            "With no arguments, matches every collected operator.\n"
            f"Prefixes: {', '.join(OPERATOR_BACKENDS)}"
        ),
    )
    analyze_group.add_argument(
        "-k",
        "--kernel",
        metavar="<ids>",
        dest="gpu_kernel",
        type=int,
        nargs="+",
        action="append",
        help=(
            "Specify kernel id(s) from --list-stats for filtering.\n"
            "With an operator filter, use the kernel ids shown in the\n"
            "operator tree."
        ),
    )
    analyze_group.add_argument(
        "-d",
        "--dispatch",
        dest="gpu_dispatch_id",
        metavar="<ids>",
        nargs="+",
        action="append",
        help="Specify dispatch id(s) for filtering (1-based).",
    )
    analyze_group.add_argument(
        "-b",
        "--block",
        dest="filter_metrics",
        metavar="<ids>",
        nargs="+",
        type=block_token_or_alias,
        help=(
            "Specify metric id(s) or block alias(es) from --list-metrics "
            "for filtering.\n"
            "Give all ids after one -b; a repeated -b keeps only the last list."
        ),
    )
    analyze_group.add_argument(
        "--gpu-id",
        dest="gpu_id",
        metavar="<ids>",
        nargs="+",
        help="Specify GPU id(s) for filtering.",
    )
    analyze_group.add_argument(
        "--output-format",
        metavar="<format>",
        dest="output_format",
        choices=["stdout", "txt", "csv", "db"],
        default="stdout",
        help=(
            "Format of the analysis output:\n"
            "   stdout - print report to the terminal (no file/folder created).\n"
            "            Required by --list-operators and --operator.\n"
            "   txt    - write report to <name>.txt; disables terminal output.\n"
            "   csv    - write one CSV per analysis view into a folder <name>/.\n"
            "            Requires profiles collected in rocpd format. "
            "Disables terminal output.\n"
            "   db     - write a SQLite database <name>.db (see analysis\n"
            "            database schema in the docs). Requires profiles\n"
            "            collected in rocpd format.\n"
            "            Disables terminal output.\n"
            "<name> is rocprof_compute_<uuid> unless --output-name is given "
            "(Default: stdout).\n"
            "Values: stdout, txt, csv, db"
        ),
    )
    analyze_group.add_argument(
        "--output-name",
        metavar="<name>",
        dest="output_name",
        help=(
            "Override the default output file name rocprof_compute_<uuid> "
            "with the specified name.\n"
            "This is only applicable when --output-format txt/csv/db is used."
        ),
    )
    analyze_group.add_argument(
        "--pc-sampling-rows",
        required=False,
        metavar="<rows>",
        dest="pc_sampling_rows",
        default=10,
        type=non_negative_int,
        help="Maximum number of kernels shown in the PC sampling table; "
        "use 0 to show all kernels (Default: 10).",
    )

    analyze_advanced_group.add_argument(
        "--max-stat-num",
        dest="max_stat_num",
        metavar="<num>",
        type=int,
        default=10,
        help="Specify the maximum number of stats shown in "
        '"Top Stats" tables (Default: 10).',
    )
    analyze_advanced_group.add_argument(
        "-n",
        "--normal-unit",
        dest="normal_unit",
        metavar="<unit>",
        default="per_kernel",
        choices=["per_wave", "per_cycle", "per_second", "per_kernel"],
        help="Specify the normalization unit (Default: per_kernel).\n"
        "Values: per_wave, per_cycle, per_second, per_kernel",
    )
    analyze_advanced_group.add_argument(
        "-t",
        "--time-unit",
        dest="time_unit",
        metavar="<unit>",
        default="ns",
        choices=["s", "ms", "us", "ns"],
        help="Specify display time unit (Default: ns).\nValues: s, ms, us, ns",
    )
    analyze_advanced_group.add_argument(
        "--decimal",
        type=int,
        metavar="<digits>",
        default=2,
        help="Specify desired decimal precision of analysis results (Default: 2).",
    )
    analyze_advanced_group.add_argument(
        "-g", dest="debug", action="store_true", help="Debug single metric."
    )
    analyze_advanced_group.add_argument(
        "--view",
        dest="view",
        metavar="<view>",
        choices=["table"],  # future: e.g. "bar" for additional TTY views
        default=None,
        help=(
            "TTY output view. "
            "table: force plain tables and ignore cli_style from YAML\n"
            "(e.g. mem_chart, per-kernel Roofline layout as tables). "
            "Additional views may be added in future releases\n"
            "(Default: use cli_style from the analysis config).\n"
            "Values: table"
        ),
    )
    analyze_advanced_group.add_argument(
        "--specs-correction",
        metavar="<name:value>",
        nargs="+",
        help=(
            "Correct the workload's system specs with name:value pairs.\n"
            "The spec names are the column names in the workload's sysinfo.csv.\n"
            "For example: --specs-correction num_xcd:4 cu_per_gpu:64"
        ),
    )

    ## ----------------------------
    # Experimental Features
    ## ----------------------------


def _skills_note(rocprof_compute_home: Path) -> Optional[str]:
    """Return the help line that points to the Agent Skills README, if shipped."""
    # Source checkout first, then the install's share directory.
    for skills_dir in (
        rocprof_compute_home.parent / "skills",
        rocprof_compute_home.parent.parent / "share" / "rocprofiler-compute" / "skills",
    ):
        readme = skills_dir / "README.md"
        if readme.is_file():
            return f"Agent Skills: see {readme} to install them."
    return None


class ExperimentalAction(argparse.Action):
    """
    Custom action that enforces experimental feature gating.
    - Suppresses help text when experimental mode is disabled
    - Errors if feature used without --experimental flag
    - Warns when experimental feature is used
    - Delegates to inner action for proper value storage
    """

    def __init__(
        self,
        option_strings: List[str],
        help: str,
        **kwargs,
    ) -> None:
        self.experimental_enabled = kwargs.pop("experimental_enabled", False)
        self.feature_label = kwargs.pop("feature_label", None)

        # Extract the base_action
        base_action = kwargs.pop("base_action", None)
        if base_action is None:
            raise ValueError(
                "base_action is required for ExperimentalAction. "
                "Specify one of: store, store_const, store_true, store_false, "
                "append, append_const, count, extend"
            )

        if self.experimental_enabled:
            leading_whitespace = help[: len(help) - len(help.lstrip())]
            help_content = help.lstrip()
            help = f"{leading_whitespace}EXPERIMENTAL: {help_content}"
        else:
            help = argparse.SUPPRESS

        super().__init__(
            option_strings=option_strings,
            help=help,
            **kwargs,
        )

        # Map of action types to their __call__ methods
        action_map = {
            "store": argparse._StoreAction.__call__,
            "store_const": argparse._StoreConstAction.__call__,
            "store_true": argparse._StoreTrueAction.__call__,
            "store_false": argparse._StoreFalseAction.__call__,
            "append": argparse._AppendAction.__call__,
            "append_const": argparse._AppendConstAction.__call__,
            "count": argparse._CountAction.__call__,
            "extend": argparse._ExtendAction.__call__,
        }

        if base_action not in action_map:
            raise ValueError(f"Unsupported base_action: {base_action}")

        self._base_action_call = action_map[base_action]

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values,  # noqa: ANN001
        option_string: Optional[str] = None,
    ) -> None:
        # Error if experimental feature used without --experimental flag
        if not self.experimental_enabled:
            parser.error(
                f"{self.feature_label} is an experimental feature. "
                f"Use --experimental to enable it."
            )

        console_warning(
            f"{self.feature_label} is experimental and may change in future releases."
        )

        self._base_action_call(self, parser, namespace, values, option_string)


class CliHelpFormatter(argparse.RawTextHelpFormatter):
    """Show options as "-b, --block <ids>...", per CONTRIBUTING.md."""

    def __init__(self, prog: str, max_help_position: int = 40) -> None:
        super().__init__(prog, max_help_position=max_help_position)

    def _format_args(self, action: argparse.Action, default_metavar: str) -> str:
        metavar = self._metavar_formatter(action, default_metavar)(1)[0]
        if action.nargs == argparse.ONE_OR_MORE:
            return f"{metavar}..."
        if action.nargs == argparse.ZERO_OR_MORE:
            return f"[{metavar}]..."
        return super()._format_args(action, default_metavar)

    def _format_action_invocation(self, action: argparse.Action) -> str:
        if not action.option_strings or action.nargs == 0:
            return super()._format_action_invocation(action)
        default_metavar = self._get_default_metavar_for_optional(action)
        args_string = self._format_args(action, default_metavar)
        return f"{', '.join(action.option_strings)} {args_string}"
