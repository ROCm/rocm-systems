# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from utils.inject_roctx.constants import KNOWN_ML_API_BACKENDS, OPERATOR_BACKENDS
from utils.logger import console_warning
from utils.utils_common import METRIC_ID_RE, resolve_rocm_library_path

# Deprecated options by argparse dest: the option and what to use instead
DEPRECATED_OPTIONS: Dict[str, Tuple[str, str]] = {
    "list_blocks": ("--list-blocks", "--list-metrics"),
    "list_available_metrics": ("--list-available-metrics", "--list-metrics"),
}

# Standard argparse actions that ExperimentalAction and DebugAction delegate to
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
        help="Increase output verbosity (use multiple times for higher levels)",
        action="count",
        default=0,
    )
    general_group.add_argument(
        "-q", "--quiet", action="store_true", help="Reduce output and run quietly."
    )
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
        "--list-blocks",
        dest="list_blocks",
        metavar="<arch>",
        choices=supported_archs.keys(),
        help=(
            "DEPRECATED: use --list-metrics instead.\n"
            "List available blocks and their aliases for specified GPU <arch>.\n"
            f"Values: {', '.join(supported_archs)}"
        ),
    )
    general_group.add_argument(
        "--config-dir",
        dest="config_dir",
        metavar="",
        help="Specify the directory of customized report section configs.",
        default=rocprof_compute_home / "rocprof_compute_soc/analysis_configs/",
    )
    # Nowhere to load specs from in db mode
    if parser.usage:
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


def omniarg_parser(
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
        rocprof_compute_home,
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
\trocprof-compute profile -n vcopy_kernel -k vecCopy -- ./vcopy -n 1048576 -b 256
\trocprof-compute profile -n vcopy_iter --kernel-iteration-range 1 -- ./vcopy -n 1048576 -b 256
\trocprof-compute profile -n vcopy_roof --roof-only -- ./vcopy -n 1048576 -b 256
\trocprof-compute profile -n my_bench --bench-only
---------------------------------------------------------------------------------
        """,  # noqa: E501
        prog="tool",
        allow_abbrev=False,
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(
            prog, max_help_position=40
        ),
    )
    profile_parser._optionals.title = "Help"

    add_general_group(
        profile_parser,
        rocprof_compute_home,
        supported_archs,
        rocprof_compute_version,
    )
    profile_debug_group = profile_parser.add_argument_group("Debug Options")
    _add_list_metrics_gfx_id(profile_debug_group, supported_archs, debug_enabled)
    profile_group = profile_parser.add_argument_group("Profile Options")
    roofline_group = profile_parser.add_argument_group("Standalone Roofline Options")

    profile_group.add_argument(
        "-n",
        "--name",
        type=str,
        metavar="",
        dest="name",
        help=(
            "\t\t\tAssign a name to workload.\n"
            "\t\t\t--name will be ignored if used together with --output-directory.\n"
            "\t\t\tUse --overwrite to re-profile into an existing directory."
        ),
    )
    profile_group.add_argument(
        "--attach-pid",
        type=str,
        dest="attach_pid",
        metavar="",
        default=None,
        required=False,
        help=(
            "\t\t\tProcess id to be attached for profiling.\n"
            "\t\t\tImplies --no-native-tool"
        ),
    )
    profile_group.add_argument(
        "--attach-duration-msec",
        type=str,
        dest="attach_duration_msec",
        metavar="",
        default=None,
        required=False,
        help=(
            "\t\t\tWhen --attach-pid is used, it specifies the attach duration\n"
            "\t\t\tin milliseconds. If not set, detachment occurs when\n"
            '\t\t\t"Enter" key is pressed.'
        ),
    )
    profile_group.add_argument(
        "-d",
        "--output-directory",
        metavar="",
        type=str,
        dest="output_directory",
        default=str(Path.cwd() / "workloads"),
        required=False,
        help=(
            "\t\t\tSpecify output directory to save workload.\n"
            "\t\t\tOutput directory can also be parameterized with the following keywords:\n"  # noqa: E501
            "\t\t\t   %%hostname%%: Host name\n"
            "\t\t\t   %%gpumodel%%: GPU model\n"
            "\t\t\t   %%rank%%: MPI process rank\n"
            '\t\t\t   %%env{NAME}%%: Environment variable "NAME"\n'
            "\t\t\t(DEFAULT: <current-working-directory>/workloads/<name>/%%gpumodel%%) without MPI,\n"  # noqa: E501
            "\t\t\t <current-working-directory>/workloads/<name>/%%rank%% with MPI.)\n"
            "\t\t\tUse --overwrite to re-profile into an existing directory."
        ),
    )
    profile_group.add_argument(
        "--overwrite",
        dest="overwrite",
        required=False,
        default=False,
        action="store_true",
        help=(
            "\t\t\tOverwrite an existing workload directory.\n"
            "\t\t\tWithout it, profiling into a non-empty directory fails\n"
            "\t\t\tinstead of mixing runs. Use a fresh directory per run;\n"
            "\t\t\tpass this flag only to re-profile in place."
        ),
    )
    profile_group.add_argument(
        "-k",
        "--kernel",
        type=str,
        dest="kernel",
        metavar="",
        required=False,
        nargs="+",
        default=None,
        help="\t\t\tKernel filtering.",
    )
    profile_group.add_argument(
        "--kernel-iteration-range",
        type=str,
        metavar="",
        nargs="+",
        dest="kernel_iteration_range",
        required=False,
        help=(
            "\t\t\tWhich iterations of each kernel to profile \n"
            "\t\t\t(1-based; positive integer or 'start:end'/'start-end' \n"
            "\t\t\trange, e.g. 1 3:5 captures 1st, 3rd, 4th and 5th \n"
            "\t\t\titerations)."
        ),
    )
    profile_group.add_argument(
        "--iteration-multiplexing",
        type=str,
        dest="iteration_multiplexing",
        metavar="",
        required=False,
        nargs="?",
        choices=[
            "kernel",
            "kernel_launch_params",
        ],
        const="kernel_launch_params",
        help=(
            "\t\t\tChoose the iteration multiplexing policy: "
            "(DEFAULT: kernel_launch_params).\n"
            "\t\t\t   kernel (i.e. Round robin counters over kernel calls with "
            "unique kernel names.)\n"
            "\t\t\t   kernel_launch_params (i.e. Round robin counters over "
            "kernel calls with unique kernel and launch parameters)"
        ),
    )

    profile_group.add_argument(
        "--list-available-metrics",
        dest="list_available_metrics",
        help=(
            "\t\t\tDEPRECATED: use --list-metrics instead.\n"
            "\t\t\tList all available metrics for analysis on current arch"
        ),
        action="store_true",
    )
    profile_group.add_argument(
        "-b",
        "--block",
        dest="filter_blocks",
        metavar="",
        nargs="+",
        type=block_token_or_alias,
        required=False,
        default=[],
        help=(
            "\t\t\tSpecify metric id(s) from --list-metrics for filtering "
            "(e.g. 12, 12.1, 12.1.1).\n"
            "\t\t\tAlternatively, specify block id(s) for filtering "
            "(e.g. 12, 13, 14).\n"
            "\t\t\tAlternatively, specify block alias(es) for filtering.\n"
            "\t\t\tAliases are arch-specific; run --list-metrics to see\n"
            "\t\t\tall valid block ids and aliases.\n"
            "\t\t\tCan provide multiple space separated arguments.\n"
            "\t\t\tCannot be used with --set, --roof-only, or --bench-only"
        ),
    )
    profile_group.add_argument(
        "--list-sets",
        action="store_true",
        help="\t\t\tDisplay available metric sets and their descriptions",
    )
    profile_group.add_argument(
        "--set",
        default=None,
        dest="set_selected",
        help=(
            "\t\t\tProfile a set of metrics of topic of interest by collecting "
            "counters in a single pass.\n"
            "\t\t\tFor available sets, see --list-sets\n"
            "\t\t\tCannot be used with --block, --roof-only, or --bench-only"
        ),
    )
    profile_group.add_argument(
        "--no-roof",
        required=False,
        default=False,
        action="store_true",
        help="\t\t\tProfile without collecting roofline data.",
    )
    profile_group.add_argument(
        "remaining",
        metavar="-- [ ...]",
        default=None,
        nargs=argparse.REMAINDER,
        help="\t\t\tProvide command for profiling after double dash.",
    )
    profile_group.add_argument(
        "--rocprofiler-sdk-tool-path",
        type=resolve_rocm_library_path,
        dest="rocprofiler_sdk_tool_path",
        required=False,
        default=resolve_rocm_library_path(
            str(
                Path(os.getenv("ROCM_PATH", "/opt/rocm"))
                / "lib/rocprofiler-sdk/librocprofiler-sdk-tool.so"
            )
        ),
        help="\t\t\tSet the path to rocprofiler-sdk tool.",
    )
    profile_group.add_argument(
        "--no-native-tool",
        required=False,
        default=False,
        action="store_true",
        help=(
            "\t\t\tDo not use the native counter collection tool.\n"
            "\t\t\tNative tool is not used if ROCPROF env. var. is set "
            "and not equal to rocprofiler-sdk.\n"
            "\t\t\tNative tool is not used for ROCm version < 7.x.x.\n"
            "\t\t\tNative tool is not used attach/detach scenario"
        ),
    )
    profile_group.add_argument(
        "--retain-rocpd-output",
        required=False,
        default=False,
        action="store_true",
        help=(
            "\t\t\t(DEPRECATED) Retain the large raw rocpd database "
            "in workload directory.\n"
            "\t\t\t --retain-rocpd-output is deprecated. .db files "
            "will be retained by default in a future release."
        ),
    )

    ## Roofline Command Line Options (profile: microbenchmark only)
    roofline_group.add_argument(
        "--roof-only",
        required=False,
        default=False,
        action="store_true",
        help=(
            "\t\t\tProfile roofline data only.\n"
            "\t\t\tCannot be used with --block, --set, or --bench-only"
        ),
    )
    roofline_group.add_argument(
        "--bench-only",
        required=False,
        default=False,
        action="store_true",
        help=(
            "\t\t\tRun roofline microbenchmark only.\n"
            "\t\t\tNo application profiling or counter collection.\n"
            "\t\t\tNo application run is required.\n"
            "\t\t\tCannot be used with --block, --set, --roof-only, or --no-roof"
        ),
    )
    roofline_group.add_argument(
        "--device",
        metavar="",
        required=False,
        default=0,
        type=int,
        help=(
            "\t\t\tTarget GPU device ID per amd-smi for roofline benchmarking"
            " (Default: 0)"
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
            "\t\t\tTrace ML APIs. User ROCTx regions are always included.\n"
            f"\t\t\tWith no arguments, traces {' and '.join(KNOWN_ML_API_BACKENDS)}.\n"
            f"\t\t\tValues: {', '.join(KNOWN_ML_API_BACKENDS)}"
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
        help="\t\t\tEnable Memory Bandwidth Analysis counters (block 30).",
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
        help="\t\t\tEnable PC sampling (block 21) for profile mode.",
    )
    profile_group.add_argument(
        "--pc-sampling-method",
        required=False,
        metavar="",
        dest="pc_sampling_method",
        default="stochastic",
        choices=["stochastic", "host_trap"],
        base_action="store",
        action=ExperimentalAction,
        experimental_enabled=experimental_enabled,
        feature_label="PC Sampling",
        help=(
            "\t\t\tSet the method of pc sampling, stochastic or host_trap. "
            "Support stochastic only >= MI300"
        ),
    )
    profile_group.add_argument(
        "--pc-sampling-interval",
        required=False,
        metavar="",
        dest="pc_sampling_interval",
        default=None,
        type=int,
        base_action="store",
        action=ExperimentalAction,
        experimental_enabled=experimental_enabled,
        feature_label="PC Sampling",
        help=(
            "\t\t\tSet the interval of pc sampling. The accepted range is "
            "read from the device; see 'rocprofv3-avail info --pc-sampling'. "
            "When the device cannot be queried, 1 to 1048576 is accepted.\n"
            "\t\t\t  For stochastic sampling, the interval is in cycles and "
            "must be a power of 2 (DEFAULT: 1048576).\n"
            "\t\t\t  For host_trap sampling, the interval is in microseconds "
            "(DEFAULT: 512)."
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
        prog="tool",
        allow_abbrev=False,
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(
            prog, max_help_position=40
        ),
    )
    analyze_parser._optionals.title = "Help"

    add_general_group(
        analyze_parser,
        rocprof_compute_home,
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
        metavar="",
        nargs="+",
        action="append",
        help="\t\tSpecify the raw data root dirs or desired results directory.",
    )
    analyze_group.add_argument(
        "--list-stats",
        action="store_true",
        help=(
            "\t\tList all detected kernels and kernel dispatches.\n"
            "\t\tCannot be used with --output-format csv or db.\n"
            "\t\tWith --operator, lists only the kernels and dispatches\n"
            "\t\tof the matching operators."
        ),
    )
    analyze_group.add_argument(
        "--list-available-metrics",
        dest="list_available_metrics",
        help=(
            "\t\tDEPRECATED: use --list-metrics instead.\n"
            "\t\tList all available metrics for analysis on the workload arch"
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
            "\t\tList collected operators as a call tree.\n"
            "\t\tWith no arguments, lists every collected operator.\n"
            "\t\tIgnores --operator.\n"
            f"\t\tValues: {', '.join(OPERATOR_BACKENDS)}"
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
            "\t\tFilter operators with glob patterns.\n"
            "\t\tA pattern without a prefix uses the prefix before it.\n"
            "\t\tBefore any prefix, a pattern matches every traced\n"
            "\t\tframework and user regions.\n"
            "\t\tExample: --operator 'torch:*conv*' '*relu*' 'triton:*matmul*'\n"
            "\t\tExample: --operator 'user:*UserDefined*'\n"
            "\t\tWith no arguments, matches every collected operator.\n"
            f"\t\tPrefixes: {', '.join(OPERATOR_BACKENDS)}"
        ),
    )
    analyze_group.add_argument(
        "-k",
        "--kernel",
        metavar="",
        type=int,
        dest="gpu_kernel",
        nargs="+",
        action="append",
        help=(
            "\t\tSpecify kernel id(s) from --list-stats for filtering.\n"
            "\t\tWith an operator filter, use the kernel ids shown in the\n"
            "\t\toperator tree."
        ),
    )
    analyze_group.add_argument(
        "-d",
        "--dispatch",
        dest="gpu_dispatch_id",
        metavar="",
        nargs="+",
        action="append",
        help="\t\tSpecify dispatch id(s) for filtering (1-based).",
    )
    analyze_group.add_argument(
        "-b",
        "--block",
        dest="filter_metrics",
        metavar="",
        nargs="+",
        type=block_token_or_alias,
        help="\t\tSpecify metric id(s) from --list-metrics for filtering.",
    )
    analyze_group.add_argument(
        "--gpu-id",
        dest="gpu_id",
        metavar="",
        nargs="+",
        help="\t\tSpecify GPU id(s) for filtering.",
    )
    analyze_group.add_argument(
        "--output-format",
        metavar="",
        dest="output_format",
        choices=["stdout", "txt", "csv", "db"],
        default="stdout",
        help=(
            "\t\tFormat of the analysis output. One of: stdout, txt, csv, db.\n"
            "\t\t  stdout - print report to the terminal (no file/folder created).\n"
            "\t\t           Required by --list-operators and --operator.\n"
            "\t\t  txt    - write report to <name>.txt; disables terminal output.\n"
            "\t\t  csv    - write one CSV per analysis view into a folder <name>/.\n"
            "\t\t           Requires profiles collected in rocpd format. "
            "Disables terminal output.\n"
            "\t\t  db     - write a SQLite database <name>.db (see analysis\n"
            "\t\t           database schema in the docs). Requires profiles\n"
            "\t\t           collected in rocpd format.\n"
            "\t\t           Disables terminal output.\n"
            "\t\tDefault <name> is rocprof_compute_<uuid>; override with"
            " --output-name.\n"
            "\t\tDefault format is stdout.\n"
        ),
    )
    analyze_group.add_argument(
        "--output-name",
        metavar="",
        dest="output_name",
        help=(
            "\t\tOverride the default output file name rocprof_compute_<uuid> "
            "with the specified name.\n"
            "\t\tThis is only applicable when --output-format txt/csv/db is used.\n"
        ),
    )
    analyze_group.add_argument(
        "--pc-sampling-rows",
        required=False,
        metavar="",
        dest="pc_sampling_rows",
        default=10,
        type=non_negative_int,
        help="\t\tMaximum number of kernels shown in the PC sampling table; "
        "use 0 to show all kernels (Default: 10).",
    )

    analyze_advanced_group.add_argument(
        "--max-stat-num",
        dest="max_stat_num",
        metavar="",
        type=int,
        default=10,
        help="\t\tSpecify the maximum number of stats shown in "
        '"Top Stats" tables (DEFAULT: 10)',
    )
    analyze_advanced_group.add_argument(
        "-n",
        "--normal-unit",
        dest="normal_unit",
        metavar="",
        default="per_kernel",
        choices=["per_wave", "per_cycle", "per_second", "per_kernel"],
        help="\t\tSpecify the normalization unit: (DEFAULT: per_kernel)\n"
        "\t\t   per_wave\n"
        "\t\t   per_cycle\n"
        "\t\t   per_second\n"
        "\t\t   per_kernel",
    )
    analyze_advanced_group.add_argument(
        "-t",
        "--time-unit",
        dest="time_unit",
        metavar="",
        default="ns",
        choices=["s", "ms", "us", "ns"],
        help="\t\tSpecify display time unit: (DEFAULT: ns)\n"
        "\t\t   s\n"
        "\t\t   ms\n"
        "\t\t   us\n"
        "\t\t   ns",
    )
    analyze_advanced_group.add_argument(
        "--decimal",
        type=int,
        metavar="",
        default=2,
        help="\t\tSpecify desired decimal precision of analysis results. (DEFAULT: 2)",
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
        help="Force plain tables and ignore cli_style from YAML.",
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
        type=str,
        metavar="",
        help="\t\tSpecify the specs to correct. e.g. "
        '--specs-correction="specname1:specvalue1,specname2:specvalue2"',
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
            f"Values: {', '.join(supported_archs)}"
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
