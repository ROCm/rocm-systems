# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from utils.inject_roctx.constants import KNOWN_ML_API_BACKENDS, OPERATOR_BACKENDS
from utils.logger import console_warning
from utils.utils_common import METRIC_ID_RE, resolve_rocm_library_path

# Standard argparse actions that the CustomAction subclasses delegate to
BASE_ACTIONS: Dict[str, Type[argparse.Action]] = {
    "store": argparse._StoreAction,
    "store_const": argparse._StoreConstAction,
    "store_true": argparse._StoreTrueAction,
    "store_false": argparse._StoreFalseAction,
    "append": argparse._AppendAction,
    "append_const": argparse._AppendConstAction,
    "count": argparse._CountAction,
    "extend": argparse._ExtendAction,
}


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
    general_group.add_argument(
        "--list-metrics",
        dest="list_metrics",
        metavar="ids",
        nargs="*",
        help=(
            "List the blocks of the current GPU.\n"
            "Add block or metric [ids] or aliases to list their metrics with units\n"
            'and descriptions, or "all" to list every metric. Each id or alias adds\n'
            "to the list: --list-metrics sol 12.1 lists all of block 2 (sol) and\n"
            "table 12.1.\n"
            "With analyze -p <workload>, list the workload's blocks and metrics."
        ),
    )
    general_group.add_argument(
        "--list-blocks",
        dest="list_blocks",
        metavar="<arch>",
        choices=supported_archs.keys(),
        action=DeprecatedAction,
        base_action="store",
        advice="Use --list-metrics instead.",
        help=(
            "List available blocks and their aliases for specified GPU <arch>.\n"
            "Values: %(choices)s"
        ),
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
    general_group.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Show the debug options in --help:\n"
            "   Top level and profile: --debug-list-metrics-gfx-id\n"
            "   Analyze: --debug-metric-expression, --debug-table-view, "
            "--debug-verify-deps\n"
        ),
    )


def build_parser(
    rocprof_compute_home: Path,
    supported_archs: Dict[str, str],
    rocprof_compute_version: Dict[str, Optional[str]],
    argv: List[str],
) -> argparse.ArgumentParser:
    """Return the rocprof-compute command-line parser for argv."""
    # --experimental and --debug count only before the workload's "--"
    workload_start = argv.index("--") if "--" in argv else len(argv)
    experimental_enabled = "--experimental" in argv[:workload_start]
    debug_enabled = "--debug" in argv[:workload_start]
    parser = argparse.ArgumentParser(
        description=(
            "Command line interface for AMD's GPU profiler, ROCm Compute Profiler"
        ),
        prog="rocprof-compute",
        allow_abbrev=False,
        formatter_class=lambda prog: CliHelpFormatter(prog, max_help_position=30),
        usage="rocprof-compute [mode] [options]",
    )
    _omniarg_parser(
        parser,
        rocprof_compute_home,
        supported_archs,
        rocprof_compute_version,
        experimental_enabled,
        debug_enabled,
    )
    return parser


def _omniarg_parser(
    parser: argparse.ArgumentParser,
    rocprof_compute_home: Path,
    supported_archs: Dict[str, str],
    rocprof_compute_version: Dict[str, Optional[str]],
    experimental_enabled: bool = False,
    debug_enabled: bool = False,
) -> None:
    # -----------------------------------------
    # Parse arguments (dependent on mode)
    # -----------------------------------------

    ## General Command Line Options
    ## ----------------------------
    add_general_group(
        parser,
        supported_archs,
        rocprof_compute_version,
    )
    debug_group = parser.add_argument_group("Debug Options")
    _add_list_metrics_gfx_id(debug_group, supported_archs, debug_enabled)
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
\trocprof-compute profile -n vcopy_sol --speed-of-light -- ./vcopy -n 1048576 -b 256
\trocprof-compute profile -n vcopy_kernel -k vecCopy -- ./vcopy -n 1048576 -b 256
\trocprof-compute profile -n vcopy_iter --kernel-iteration-range 1 -- ./vcopy -n 1048576 -b 256
\trocprof-compute profile -n vcopy_roof --roofline -- ./vcopy -n 1048576 -b 256
\trocprof-compute profile -n my_bench --roofline-bench-only
---------------------------------------------------------------------------------
        """,  # noqa: E501
        prog="rocprof-compute",
        allow_abbrev=False,
        formatter_class=CliHelpFormatter,
    )
    profile_parser._optionals.title = "Help"

    add_general_group(
        profile_parser,
        supported_archs,
        rocprof_compute_version,
    )
    profile_debug_group = profile_parser.add_argument_group("Debug Options")
    _add_list_metrics_gfx_id(profile_debug_group, supported_archs, debug_enabled)
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
        type=int,
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
            "With no value, uses kernel_launch_params.\n"
            "Values: %(choices)s"
        ),
    )

    profile_group.add_argument(
        "--list-available-metrics",
        dest="list_available_metrics",
        action=DeprecatedAction,
        base_action="store_true",
        advice="Use --list-metrics instead.",
        help="List all available metrics for analysis on current arch.",
    )
    profile_group.add_argument(
        "-b",
        "--block",
        dest="filter_blocks",
        metavar="<ids>",
        nargs="+",
        type=block_token_or_alias,
        action="extend",
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
            "Cannot be used with --set or --roofline-bench-only."
        ),
    )
    profile_group.add_argument(
        "--speed-of-light",
        dest="filter_blocks",
        action="append_const",
        const="2",
        help="Profile the System Speed-of-Light block. Same as -b 2.",
    )
    profile_group.add_argument(
        "--memory-chart",
        dest="filter_blocks",
        action="append_const",
        const="3",
        help="Profile the Memory Chart block. Same as -b 3.",
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
            "Cannot be used with -b, --speed-of-light, --memory-chart, --roofline\n"
            "or --roofline-bench-only."
        ),
    )
    profile_group.add_argument(
        "--no-roof",
        required=False,
        default=False,
        action=DeprecatedAction,
        base_action="store_true",
        advice=(
            "To skip roofline, select blocks with -b or --set that do not include "
            "block 4."
        ),
        help="Profile without collecting roofline data.",
    )
    profile_group.add_argument(
        "remaining",
        metavar="-- <workload_cmd>",
        default=None,
        nargs=argparse.REMAINDER,
        help="Provide command for profiling after double dash.",
    )
    profile_group.add_argument(
        "--rocprofiler-sdk-tool-path",
        type=resolve_rocm_library_path,
        dest="rocprofiler_sdk_tool_path",
        metavar="<path>",
        required=False,
        default=resolve_rocm_library_path(
            str(
                Path(os.getenv("ROCM_PATH", "/opt/rocm"))
                / "lib/rocprofiler-sdk/librocprofiler-sdk-tool.so"
            )
        ),
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
        action=DeprecatedAction,
        base_action="store_true",
        help="Retain the large raw rocpd database in workload directory.",
    )

    ## Roofline Command Line Options
    roofline_group.add_argument(
        "--roofline",
        "--roof-only",
        dest="filter_blocks",
        const="4",
        action=DeprecatedAction,
        base_action="append_const",
        deprecated_names=["--roof-only"],
        help=(
            "Profile the roofline block. Same as -b 4.\n"
            "Used alone, only roofline data is profiled."
        ),
    )
    roofline_group.add_argument(
        "--roofline-bench-only",
        "--bench-only",
        dest="bench_only",
        required=False,
        default=False,
        action=DeprecatedAction,
        base_action="store_true",
        deprecated_names=["--bench-only"],
        help=(
            "Run roofline microbenchmark only.\n"
            "No application profiling or counter collection.\n"
            "No application run is required.\n"
            "Cannot be used with -b, --speed-of-light, --memory-chart, --roofline,\n"
            "--set or --no-roof."
        ),
    )
    roofline_group.add_argument(
        "--roofline-device",
        "--device",
        dest="device",
        metavar="<id>",
        required=False,
        default=0,
        type=int,
        action=DeprecatedAction,
        base_action="store",
        deprecated_names=["--device"],
        help="Target GPU device ID per amd-smi for roofline benchmarking (Default: 0).",
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
        metavar="frameworks",
        base_action="store",
        action=ExperimentalAction,
        experimental_enabled=experimental_enabled,
        feature_label="ML API trace",
        help=(
            "Trace ML APIs. User ROCTx regions are always included.\n"
            f"With no arguments, traces {' and '.join(KNOWN_ML_API_BACKENDS)}.\n"
            "Values: %(choices)s"
        ),
    )
    profile_group.add_argument(
        "--membw-analysis",
        dest="membw_analysis",
        required=False,
        default=False,
        base_action="store_true",
        action=ExperimentalAction,
        experimental_enabled=experimental_enabled,
        feature_label="Memory Bandwidth Analysis",
        help="Enable Memory Bandwidth Analysis counters (block 30).",
    )

    profile_group.add_argument(
        "--pc-sampling",
        dest="pc_sampling",
        required=False,
        default=False,
        base_action="store_true",
        action=ExperimentalAction,
        experimental_enabled=experimental_enabled,
        feature_label="PC Sampling",
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
            "Set the PC sampling method. The stochastic method needs MI300 or newer "
            "(Default: stochastic).\n"
            "Values: %(choices)s"
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
\trocprof-compute analyze -p workloads/vcopy/mi200/ --speed-of-light --roofline
-----------------------------------------------------------------------------------
        """,
        prog="rocprof-compute",
        allow_abbrev=False,
        formatter_class=CliHelpFormatter,
    )
    analyze_parser._optionals.title = "Help"

    add_general_group(
        analyze_parser,
        supported_archs,
        rocprof_compute_version,
    )
    analyze_debug_group = analyze_parser.add_argument_group("Debug Options")
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
        action=DeprecatedAction,
        base_action="store_true",
        advice="Use --list-metrics instead.",
        help="List all available metrics for analysis on the workload arch.",
    )
    analyze_group.add_argument(
        "--list-operators",
        dest="list_operators",
        default=None,
        nargs="*",
        choices=OPERATOR_BACKENDS,
        metavar="frameworks",
        base_action="store",
        action=ExperimentalAction,
        experimental_enabled=experimental_enabled,
        feature_label="List operators",
        help=(
            "List collected operators as a call tree.\n"
            "With no arguments, lists every collected operator.\n"
            "Ignores --operator.\n"
            "Values: %(choices)s"
        ),
    )
    analyze_group.add_argument(
        "--operator",
        metavar="patterns",
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
        action="extend",
        help=(
            "Specify metric id(s) or block alias(es) from --list-metrics for filtering."
        ),
    )
    analyze_group.add_argument(
        "--speed-of-light",
        dest="filter_metrics",
        action="append_const",
        const="2",
        help="Show the System Speed-of-Light panel. Same as -b 2.",
    )
    analyze_group.add_argument(
        "--memory-chart",
        dest="filter_metrics",
        action="append_const",
        const="3",
        help="Show the Memory Chart panel. Same as -b 3.",
    )
    analyze_group.add_argument(
        "--roofline",
        dest="filter_metrics",
        action="append_const",
        const="4",
        help="Show the Roofline panel. Same as -b 4.",
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
            "Format of the analysis output (Default: stdout):\n"
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
            "<name> is rocprof_compute_<uuid> unless --output-name is given.\n"
            "Values: %(choices)s"
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
        "Values: %(choices)s",
    )
    analyze_advanced_group.add_argument(
        "-t",
        "--time-unit",
        dest="time_unit",
        metavar="<unit>",
        default="ns",
        choices=["s", "ms", "us", "ns"],
        help="Specify display time unit (Default: ns).\nValues: %(choices)s",
    )
    analyze_advanced_group.add_argument(
        "--decimal",
        type=int,
        metavar="<digits>",
        default=2,
        help="Specify desired decimal precision of analysis results (Default: 2).",
    )
    analyze_debug_group.add_argument(
        "--debug-metric-expression",
        action=DebugAction,
        base_action="store_true",
        debug_enabled=debug_enabled,
        help="Print the counter data, and each metric's expression and inputs.",
    )
    analyze_debug_group.add_argument(
        "--debug-table-view",
        action=DebugAction,
        base_action="store_true",
        debug_enabled=debug_enabled,
        help="Print the memory chart and roofline blocks as plain tables.",
    )
    analyze_debug_group.add_argument(
        "--debug-verify-deps",
        action=DebugAction,
        base_action="store_true",
        debug_enabled=debug_enabled,
        help="Check the Python dependencies analyze mode needs, then exit.",
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


def _add_list_metrics_gfx_id(
    debug_group: argparse._ArgumentGroup,
    supported_archs: Dict[str, str],
    debug_enabled: bool,
) -> None:
    """Add --debug-list-metrics-gfx-id, which analyze doesn't take: it lists the
    metrics of the -p workload's GPU."""
    debug_group.add_argument(
        "--debug-list-metrics-gfx-id",
        metavar="<arch>",
        choices=supported_archs.keys(),
        action=DebugAction,
        base_action="store",
        debug_enabled=debug_enabled,
        help=(
            "With --list-metrics, list the blocks and metrics of GPU <arch>\n"
            "instead of the current GPU.\n"
            "Values: %(choices)s"
        ),
    )


class CustomAction(argparse.Action):
    """Parse an option like the standard argparse action named by base_action.

    Subclasses change the help text or check the option before it is stored.
    """

    def __init__(
        self,
        option_strings: List[str],
        dest: str,
        base_action: str,
        **kwargs: Any,
    ) -> None:
        if base_action not in BASE_ACTIONS:
            raise ValueError(
                f"Unsupported base_action: {base_action}. "
                f"Specify one of: {', '.join(BASE_ACTIONS)}"
            )
        self._base_action = BASE_ACTIONS[base_action](option_strings, dest, **kwargs)
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=self._base_action.nargs,
            const=self._base_action.const,
            default=self._base_action.default,
            type=self._base_action.type,
            choices=self._base_action.choices,
            required=self._base_action.required,
            help=self._base_action.help,
            metavar=self._base_action.metavar,
        )

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: Optional[str] = None,
    ) -> None:
        self._base_action(parser, namespace, values, option_string)


class ExperimentalAction(CustomAction):
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
        dest: str,
        base_action: str,
        experimental_enabled: bool,
        feature_label: str,
        help: str,
        **kwargs: Any,
    ) -> None:
        self.experimental_enabled = experimental_enabled
        self.feature_label = feature_label

        if experimental_enabled:
            leading_whitespace = help[: len(help) - len(help.lstrip())]
            help_content = help.lstrip()
            help = f"{leading_whitespace}EXPERIMENTAL: {help_content}"
        else:
            help = argparse.SUPPRESS

        super().__init__(option_strings, dest, base_action, help=help, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
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

        super().__call__(parser, namespace, values, option_string)


class DebugAction(CustomAction):
    """
    Custom action for debug options.
    - Suppresses help text unless --debug is given
    - Delegates to inner action for proper value storage
    """

    def __init__(
        self,
        option_strings: List[str],
        dest: str,
        base_action: str,
        debug_enabled: bool,
        help: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            option_strings,
            dest,
            base_action,
            help=help if debug_enabled else argparse.SUPPRESS,
            **kwargs,
        )


class DeprecatedAction(CustomAction):
    """
    Custom action for deprecated option names.
    - deprecated_names lists the old names of a renamed option; without it, the
      whole option is deprecated
    - advice says what to use instead; a rename defaults to the new name
    - Adds the deprecation to the help text and warns when a deprecated name is used
    """

    def __init__(
        self,
        option_strings: List[str],
        dest: str,
        base_action: str,
        help: str,
        deprecated_names: Optional[List[str]] = None,
        advice: str = "",
        **kwargs: Any,
    ) -> None:
        self.deprecated_names = deprecated_names or option_strings
        new_names = [
            name for name in option_strings if name not in self.deprecated_names
        ]
        self.advice = advice or (f"Use {new_names[0]} instead." if new_names else "")
        if new_names:
            help = f"{help}\nDeprecated alias: {', '.join(self.deprecated_names)}."
        else:
            first_line = self.advice or "Will be removed in a future release."
            help = f"DEPRECATED: {first_line}\n{help}"
        super().__init__(option_strings, dest, base_action, help=help, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: Optional[str] = None,
    ) -> None:
        if option_string in self.deprecated_names:
            warning = (
                f"{option_string} is deprecated and will be removed in a future "
                "release."
            )
            console_warning(f"{warning} {self.advice}" if self.advice else warning)
        super().__call__(parser, namespace, values, option_string)


class CliHelpFormatter(argparse.RawTextHelpFormatter):
    """Show options as "-b, --block <ids>...", per CONTRIBUTING.md."""

    def __init__(self, prog: str, max_help_position: int = 40) -> None:
        super().__init__(prog, max_help_position=max_help_position)

    def _format_args(self, action: argparse.Action, default_metavar: str) -> str:
        metavar = action.metavar or default_metavar
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
