# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import os
from pathlib import Path
from typing import Callable, Optional, Union

from utils.logger import console_warning
from utils.utils_common import METRIC_ID_RE, resolve_rocm_library_path

# Panel ids selected by the --speed-of-light, --memory-chart and --roofline options
PANEL_SHORTCUTS = {
    "speed_of_light": "2",
    "memory_chart": "3",
    "roofline": "4",
}

ROOFLINE_MEM_LEVELS = ["ALL", "HBM", "L2", "vL1D", "L0", "LDS"]
ROOFLINE_DATA_TYPES = [
    "FP4",
    "FP6",
    "FP8",
    "MXFP8",
    "FP16",
    "BF16",
    "FP32",
    "FP64",
    "I8",
    "I32",
    "I64",
]

# Deprecated options and what to do instead. A renamed option keeps its old
# name as an extra option string until the old name is removed.
DEPRECATED_OPTIONS = {
    "--roof-only": "Use --roofline instead.",
    "--bench-only": "Use --roofline-bench-only instead.",
    "--device": "Use --roofline-device instead.",
    "--sort": "Use --roofline-sort instead.",
    "--mem-level": "Use --roofline-mem-level instead.",
    "--roofline-data-type": "Use --roofline-data-types instead.",
    "--retain-rocpd-output": ".db files will be retained automatically.",
}


def warn_deprecated_options(argv: list[str], workload: list[str]) -> None:
    """Warn about each deprecated option in argv, except in the trailing workload."""
    for arg in argv[: len(argv) - len(workload)]:
        option = arg.split("=", 1)[0]
        if option in DEPRECATED_OPTIONS:
            console_warning(
                f"{option} is deprecated and will be removed in a future release. "
                f"{DEPRECATED_OPTIONS[option]}"
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


def apply_panel_shortcuts(args: argparse.Namespace, dest: str) -> None:
    """Add the --speed-of-light, --memory-chart, and --roofline panels to args.dest."""
    selected = [
        panel_id
        for option, panel_id in PANEL_SHORTCUTS.items()
        if getattr(args, option, False)
    ]
    if not selected:
        return
    blocks = list(getattr(args, dest, None) or [])
    blocks.extend(panel_id for panel_id in selected if panel_id not in blocks)
    setattr(args, dest, blocks)


def add_general_group(
    parser: argparse.ArgumentParser,
    rocprof_compute_home: Path,
    supported_archs: dict[str, str],
    rocprof_compute_version: dict[str, Optional[str]],
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
        metavar="<arch>",
        choices=supported_archs.keys(),
        help=(
            "List all available metrics for analysis on specified GPU <arch>.\n"
            f"Values: {arch_values}"
        ),
    )
    general_group.add_argument(
        "--list-blocks",
        dest="list_blocks",
        metavar="<arch>",
        choices=supported_archs.keys(),
        help=(
            "List all available blocks for analysis on specified GPU <arch>.\n"
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
            "   Torch trace (--torch-trace, --list-torch-operators, --torch-operator)\n"
            "   Triton trace (--triton-trace, --list-triton-operators, "
            "--triton-operator)\n"
            "   ML API trace (--ml-api-trace)\n"
            "   Memory Bandwidth Analysis (--membw-analysis)\n"
            "   PC Sampling (--pc-sampling, --pc-sampling-method, "
            "--pc-sampling-interval)\n"
        ),
    )


def omniarg_parser(
    parser: argparse.ArgumentParser,
    rocprof_compute_home: Path,
    supported_archs: dict[str, str],
    rocprof_compute_version: dict[str, Optional[str]],
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
        "--kokkos-trace",
        dest="kokkos_trace",
        required=False,
        default=False,
        action="store_true",
        help=argparse.SUPPRESS,
        # help="Kokkos trace, traces Kokkos API calls.",
    )
    profile_group.add_argument(
        "--torch-trace",
        dest="torch_trace",
        required=False,
        default=False,
        const=True,
        nargs=0,
        base_action="store_true",
        action=ExperimentalAction,
        experimental_enabled=experimental_enabled,
        feature_label="Torch trace",
        help=(
            "Torch Trace, maps PyTorch operators to performance counters.\n"
            "Requires PyTorch 2.13 or 2.14."
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
        action=CommaListAction,
        dest="kernel_iteration_range",
        required=False,
        help=(
            "Which iterations of each kernel to profile\n"
            "(1-based; positive integer or 'start:end'/'start-end'\n"
            "range, e.g. 1,3:5 captures 1st, 3rd, 4th and 5th\n"
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
        help="List all available metrics for analysis on current arch.",
        action="store_true",
    )
    profile_group.add_argument(
        "-b",
        "--block",
        dest="filter_blocks",
        metavar="<ids>",
        action=CommaListAction,
        item_type=block_token_or_alias,
        required=False,
        default=[],
        help=(
            "Specify metric id(s) from --list-metrics for filtering "
            "(e.g. 12,12.1,12.1.1).\n"
            "Alternatively, specify block id(s) for filtering "
            "(e.g. 12,13,14).\n"
            "Alternatively, specify block alias(es) for filtering.\n"
            "Aliases are arch-specific; run --list-blocks <arch> to see\n"
            "all valid block ids and aliases.\n"
            "Cannot be used with --set, --roofline or --roofline-bench-only."
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
            "Cannot be used with --block, --roofline or --roofline-bench-only."
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
        "--roofline",
        "--roof-only",
        dest="roof_only",
        required=False,
        default=False,
        action="store_true",
        help=(
            "Profile roofline data only.\n"
            "Cannot be used with --block, --set or --roofline-bench-only.\n"
            "Deprecated alias: --roof-only."
        ),
    )
    roofline_group.add_argument(
        "--roofline-bench-only",
        "--bench-only",
        dest="bench_only",
        required=False,
        default=False,
        action="store_true",
        help=(
            "Run roofline microbenchmark only.\n"
            "No application profiling or counter collection.\n"
            "No application run is required.\n"
            "Cannot be used with --block, --set, --roofline or --no-roof.\n"
            "Deprecated alias: --bench-only."
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
        help=(
            "Target GPU device ID per amd-smi for roofline benchmarking "
            "(Default: 0).\n"
            "Deprecated alias: --device."
        ),
    )

    ## ----------------------------
    # Experimental Features
    ## ----------------------------

    profile_group.add_argument(
        "--triton-trace",
        dest="triton_trace",
        required=False,
        default=False,
        const=True,
        nargs=0,
        base_action="store_true",
        action=ExperimentalAction,
        experimental_enabled=experimental_enabled,
        feature_label="Triton trace",
        help=(
            "Triton Trace, maps Triton kernels to performance counters.\n"
            "Use when profiling Triton kernels, including those generated\n"
            "by torch.compile / Inductor.\n"
            "Can be combined with --torch-trace."
        ),
    )
    profile_group.add_argument(
        "--ml-api-trace",
        dest="ml_api_trace",
        required=False,
        default=False,
        const=True,
        nargs=0,
        base_action="store_true",
        action=ExperimentalAction,
        experimental_enabled=experimental_enabled,
        feature_label="ML API trace",
        help=(
            "ML API Trace, enables tracing for all supported machine\n"
            "learning framework backends (e.g. PyTorch, Triton)."
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
\trocprof-compute analyze -p workloads/vcopy/mi200/ --list-metrics gfx90a
\trocprof-compute analyze -p workloads/mixbench/mi200/ --dispatch 12,34 --decimal 3
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
        help="List all detected kernels and kernel dispatches.",
    )
    analyze_group.add_argument(
        "--list-available-metrics",
        dest="list_available_metrics",
        help="List all available metrics for analysis on current arch.",
        action="store_true",
    )
    analyze_group.add_argument(
        "--list-torch-operators",
        dest="list_torch_operators",
        default=False,
        const=True,
        nargs=0,
        base_action="store_true",
        action=ExperimentalAction,
        experimental_enabled=experimental_enabled,
        feature_label="List torch operators",
        help=(
            "List PyTorch operators as a unified call tree grouped by "
            "source location with kernel launch stats. "
            "Recreates ml_api_trace output directory."
        ),
    )
    analyze_group.add_argument(
        "--torch-operator",
        metavar="patterns",
        type=str,
        dest="torch_operator",
        nargs="*",
        base_action="store",
        action=ExperimentalAction,
        experimental_enabled=experimental_enabled,
        feature_label="Torch operator filter",
        help=(
            "Filter operators using shell-style glob patterns (fnmatch),\n"
            "select their kernels, and display metrics.\n"
            "With no arguments, matches all operators.\n"
            "Examples (operator hierarchy is /-separated):\n"
            "  *relu               ends with relu\n"
            "  *conv*              contains conv\n"
            "  torch.nn.functional.relu   exact match\n"
            "  */torch.nn.functional.relu two-level match\n"
            "  */*functional*/*    intermediate component match\n"
            "  all  or  '*'        match every operator\n"
            "Multiple patterns (space or comma-separated):\n"
            "  --torch-operator *relu,*conv*,*linear\n"
            "  --torch-operator */*conv2d */*relu\n"
            "Combine with -k to intersect with kernel IDs."
        ),
    )
    analyze_group.add_argument(
        "--list-triton-operators",
        dest="list_triton_operators",
        default=False,
        const=True,
        nargs=0,
        base_action="store_true",
        action=ExperimentalAction,
        experimental_enabled=experimental_enabled,
        feature_label="List triton operators",
        help=(
            "List Triton kernels as a unified call tree grouped by "
            "source location with kernel launch stats. "
            "Recreates ml_api_trace output directory."
        ),
    )
    analyze_group.add_argument(
        "--triton-operator",
        metavar="patterns",
        type=str,
        dest="triton_operator",
        nargs="*",
        base_action="store",
        action=ExperimentalAction,
        experimental_enabled=experimental_enabled,
        feature_label="Triton operator filter",
        help=(
            "Filter Triton kernels using shell-style glob patterns\n"
            "(fnmatch), select their GPU kernels, and display metrics.\n"
            "With no arguments, matches all kernels.\n"
            "Examples:\n"
            "  *matmul*            contains matmul\n"
            "  all  or  '*'        match every kernel\n"
            "Multiple patterns (space or comma-separated):\n"
            "  --triton-operator *matmul*,*softmax*\n"
            "Combine with -k to intersect with kernel IDs."
        ),
    )
    analyze_group.add_argument(
        "-k",
        "--kernel",
        metavar="<ids>",
        dest="gpu_kernel",
        action=CommaListAction,
        item_type=int,
        append=True,
        help="Specify kernel id(s) from --list-stats for filtering.",
    )
    analyze_group.add_argument(
        "-d",
        "--dispatch",
        dest="gpu_dispatch_id",
        metavar="<ids>",
        action=CommaListAction,
        append=True,
        help="Specify dispatch id(s) for filtering (1-based).",
    )
    analyze_group.add_argument(
        "-b",
        "--block",
        dest="filter_metrics",
        metavar="<ids>",
        action=CommaListAction,
        item_type=block_token_or_alias,
        help=(
            "Specify metric id(s) or block alias(es) from --list-metrics for filtering."
        ),
    )
    _add_panel_shortcut(
        analyze_group, "speed_of_light", "Show the Speed of Light panel."
    )
    _add_panel_shortcut(analyze_group, "memory_chart", "Show the Memory Chart panel.")
    _add_panel_shortcut(analyze_group, "roofline", "Show the Roofline panel.")
    analyze_group.add_argument(
        "--gpu-id",
        dest="gpu_id",
        metavar="<ids>",
        action=CommaListAction,
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
        "--pc-sampling-sorting-type",
        required=False,
        metavar="<type>",
        dest="pc_sampling_sorting_type",
        default="count",
        type=str,
        choices=["offset", "count"],
        help="Set the sorting type of pc sampling (Default: count).\n"
        "Values: offset, count",
    )
    analyze_group.add_argument(
        "--pc-sampling-rows",
        required=False,
        metavar="<rows>",
        dest="pc_sampling_rows",
        default=10,
        type=non_negative_int,
        help="Specify the maximum number of rows shown in the PC "
        "sampling table; use 0 to show all rows (Default: 10).",
    )

    ## Roofline Command Line Options (analyze: visualization)
    roofline_group_analyze = analyze_parser.add_argument_group("Roofline Options")
    roofline_group_analyze.add_argument(
        "--roofline-sort",
        "--sort",
        dest="sort",
        required=False,
        metavar="<type>",
        type=str,
        default="kernels",
        choices=["kernels", "dispatches"],
        help=(
            "Overlay top kernels or top dispatches (Default: kernels).\n"
            "Values: kernels, dispatches\n"
            "Deprecated alias: --sort."
        ),
    )
    roofline_group_analyze.add_argument(
        "-m",
        "--roofline-mem-level",
        "--mem-level",
        dest="mem_level",
        required=False,
        metavar="<levels>",
        action=CommaListAction,
        item_choices=ROOFLINE_MEM_LEVELS,
        default=["ALL"],
        help=(
            "Filter by memory level (Default: ALL).\n"
            f"Values: {', '.join(ROOFLINE_MEM_LEVELS)}\n"
            "Deprecated alias: --mem-level."
        ),
    )
    roofline_group_analyze.add_argument(
        "-R",
        "--roofline-data-types",
        "--roofline-data-type",
        dest="roofline_data_type",
        required=False,
        metavar="<types>",
        action=CommaListAction,
        item_choices=ROOFLINE_DATA_TYPES,
        default=["FP32"],
        help=(
            "Choose datatypes to view roofline HTMLs for (Default: FP32).\n"
            f"Values: {', '.join(ROOFLINE_DATA_TYPES)}\n"
            "Deprecated alias: --roofline-data-type."
        ),
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
        "--cols",
        dest="cols",
        metavar="<indices>",
        action=CommaListAction,
        item_type=int,
        help="Specify column indices to display (Default: all columns).",
    )
    analyze_advanced_group.add_argument(
        "--include-cols",
        dest="include_cols",
        metavar="<names>",
        action=CommaListAction,
        help=(
            "Specify which hidden column names should be included in cli output.\n"
            'For example, to show "Description" column which is hidden by '
            "default in cli output,\n"
            "use the option --include-cols Description."
        ),
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
            "(e.g. mem_chart, Roofline charts as tables). "
            "Additional views may be added in future releases\n"
            "(Default: use cli_style from the analysis config).\n"
            "Values: table"
        ),
    )
    analyze_advanced_group.add_argument(
        "--dependency",
        action="store_true",
        help="List the installation dependency.",
    )
    analyze_advanced_group.add_argument(
        "--report-diff", default=0, nargs="?", type=int, help=argparse.SUPPRESS
    )
    analyze_advanced_group.add_argument(
        "--specs-correction",
        metavar="<name:value>",
        action=CommaListAction,
        help=(
            "Correct system specs, given as name:value pairs.\n"
            "For example: --specs-correction num_xcd:4,cu_per_gpu:64.\n"
            "Run -s/--specs to see the spec names."
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


def _add_panel_shortcut(
    group: argparse._ArgumentGroup, dest: str, description: str
) -> None:
    """Add the --speed-of-light, --memory-chart or --roofline option for dest."""
    others = ["-b"] + [
        "--" + other.replace("_", "-") for other in PANEL_SHORTCUTS if other != dest
    ]
    group.add_argument(
        "--" + dest.replace("_", "-"),
        dest=dest,
        default=False,
        action="store_true",
        help=(
            f"{description} Same as -b {PANEL_SHORTCUTS[dest]}.\n"
            f"Can be combined with {', '.join(others[:-1])}, and {others[-1]}."
        ),
    )


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
        option_strings: list[str],
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
    """Show options as "-b, --block <ids>...", per .ai/rules/cli-options.md."""

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


class CommaListAction(argparse.Action):
    """List option that accepts "-R FP16,FP32" as well as "-R FP16 FP32".

    With append=True, each use of the option is stored as its own list.
    """

    def __init__(
        self,
        option_strings: list[str],
        dest: str,
        item_type: Callable[[str], Union[int, str]] = str,
        item_choices: Optional[list[str]] = None,
        append: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=argparse.ONE_OR_MORE,
            **kwargs,
        )
        self.item_type = item_type
        self.item_choices = item_choices
        self.append = append

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values,  # noqa: ANN001
        option_string: Optional[str] = None,
    ) -> None:
        texts = [text.strip() for token in values for text in token.split(",")]
        items = [self._parse_item(parser, text) for text in texts if text]
        if not items:
            parser.error(
                f"argument {self._option_label()}: expected at least one value"
            )

        current = getattr(namespace, self.dest)
        # Replace the default instead of extending it
        if current is self.default:
            current = []
        if self.append:
            current.append(items)
        else:
            current.extend(items)
        setattr(namespace, self.dest, current)

    def _parse_item(
        self, parser: argparse.ArgumentParser, text: str
    ) -> Union[int, str]:
        try:
            item = self.item_type(text)
        except argparse.ArgumentTypeError as error:
            parser.error(f"argument {self._option_label()}: {error}")
        except ValueError:
            type_name = getattr(self.item_type, "__name__", "value")
            parser.error(
                f"argument {self._option_label()}: invalid {type_name} value: {text!r}"
            )
        if self.item_choices is not None and item not in self.item_choices:
            choices = ", ".join(str(choice) for choice in self.item_choices)
            parser.error(
                f"argument {self._option_label()}: invalid choice: {text!r} "
                f"(choose from {choices})"
            )
        return item

    def _option_label(self) -> str:
        return "/".join(self.option_strings)
