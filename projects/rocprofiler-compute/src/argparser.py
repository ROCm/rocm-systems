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
import os
from pathlib import Path
from typing import Optional

from utils.utils import METRIC_ID_RE


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


def print_avail_arch(avail_arch: list[str], args: str) -> str:
    ret_str = f"List all available {args} for analysis on specified arch:"
    for arch in avail_arch:
        ret_str += f"\n   {arch}"
    return ret_str


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
        metavar="",
        choices=supported_archs.keys(),  # ["gfx908", "gfx90a"],
        help=print_avail_arch(list(supported_archs.keys()), "metrics"),
    )
    general_group.add_argument(
        "--list-blocks",
        dest="list_blocks",
        metavar="",
        choices=supported_archs.keys(),  # ["gfx908", "gfx90a"],
        help=print_avail_arch(list(supported_archs.keys()), "blocks"),
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


def omniarg_parser(
    parser: argparse.ArgumentParser,
    rocprof_compute_home: Path,
    supported_archs: dict[str, str],
    rocprof_compute_version: dict[str, Optional[str]],
) -> None:
    # -----------------------------------------
    # Parse arguments (dependent on mode)
    # -----------------------------------------

    ## General Command Line Options
    ## ----------------------------
    add_general_group(
        parser, rocprof_compute_home, supported_archs, rocprof_compute_version
    )
    parser._positionals.title = "Modes"
    parser._optionals.title = "Help"

    subparsers = parser.add_subparsers(
        dest="mode", help="Select mode of interaction with the target application:"
    )

    ## Profile Command Line Options
    ## ----------------------------
    profile_parser = subparsers.add_parser(
        "profile",
        help="Profile the target application",
        usage="""

`rocprof-compute profile --name <workload_name> [profile options] [roofline options] -- <workload_cmd>`

---------------------------------------------------------------------------------
Examples:
\trocprof-compute profile -n vcopy_all -- ./vcopy -n 1048576 -b 256
\trocprof-compute profile -n vcopy_SPI_TCC -b SQ TCC -- ./vcopy -n 1048576 -b 256
\trocprof-compute profile -n vcopy_kernel -k vecCopy -- ./vcopy -n 1048576 -b 256
\trocprof-compute profile -n vcopy_disp -d 0 -- ./vcopy -n 1048576 -b 256
\trocprof-compute profile -n vcopy_roof --roof-only -- ./vcopy -n 1048576 -b 256
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
        profile_parser, rocprof_compute_home, supported_archs, rocprof_compute_version
    )
    profile_group = profile_parser.add_argument_group("Profile Options")
    roofline_group = profile_parser.add_argument_group("Standalone Roofline Options")

    profile_group.add_argument(
        "-n",
        "--name",
        type=str,
        metavar="",
        dest="name",
        help="\t\t\tAssign a name to workload.",
    )
    profile_group.add_argument(
        "--target", type=str, default=None, help=argparse.SUPPRESS
    )
    profile_group.add_argument(
        "--attach-pid",
        type=str,
        dest="attach_pid",
        metavar="",
        default=None,
        required=False,
        help="\t\t\tProcess id to be attached for profiling.",
    )
    profile_group.add_argument(
        "--attach-duration-msec",
        type=str,
        dest="attach_duration_msec",
        metavar="",
        default=None,
        required=False,
        help=(
            "\t\t\tWhen --attach-pid is used, it specifies the attach duration "
            "in milliseconds. If not set, detachment occurs when "
            '"Enter" key is pressed.'
        ),
    )
    profile_group.add_argument(
        "-p",
        "--path",
        metavar="",
        type=str,
        dest="path",
        default=str(Path.cwd() / "workloads"),
        required=False,
        help=(
            f"\t\t\tSpecify path to save workload.\n\t\t\t(DEFAULT: {Path.cwd()}/workloads/<name>)"  # noqa: E501
        ),
    )
    profile_group.add_argument(
        "--subpath",
        metavar="",
        type=str,
        dest="subpath",
        default="gpu_model",
        required=False,
        help=(
            "\t\t\tSpecify the type of subpath to save workload: node_name, gpu_model."
        ),
    )
    profile_group.add_argument(
        "--hip-trace",
        dest="hip_trace",
        required=False,
        default=False,
        action="store_true",
        help=(
            "\t\t\tHIP trace, execturion trace for the entire application at the HIP "
            "level."
        ),
    )
    profile_group.add_argument(
        "--kokkos-trace",
        dest="kokkos_trace",
        required=False,
        default=False,
        action="store_true",
        help=argparse.SUPPRESS,
        # help="\t\t\tKokkos trace, traces Kokkos API calls.",
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
        "-d",
        "--dispatch",
        type=str,
        metavar="",
        nargs="+",
        dest="dispatch",
        required=False,
        help="\t\t\tDispatch ID filtering.",
    )

    profile_group.add_argument(
        "--list-available-metrics",
        dest="list_available_metrics",
        help="\t\t\tList all available metrics for analysis on current arch",
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
            "\t\t\tAlternatively, specify block alias(es) for filtering "
            "(e.g. lds, l1i, sl1d).\n"
            "\t\t\tCan provide multiple space separated arguments.\n"
            "\t\t\tCannot be used with --set or --roof-only"
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
            "\t\t\tCannot be used with --block or --roof-only"
        ),
    )
    profile_group.add_argument(
        "--join-type",
        metavar="",
        required=False,
        choices=["kernel", "grid"],
        default="grid",
        help=(
            "\t\t\tChoose how to join rocprof runs: (DEFAULT: grid)\n"
            "\t\t\t   kernel (i.e. By unique kernel name dispatches)\n"
            "\t\t\t   grid (i.e. By unique kernel name + grid size dispatches)"
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
        "--spatial-multiplexing",
        type=int,
        metavar="",
        nargs="+",
        dest="spatial_multiplexing",
        required=False,
        default=None,
        help="\t\t\tProvide Node ID and GPU number per node.",
    )
    profile_group.add_argument(
        "--format-rocprof-output",
        required=False,
        metavar="",
        dest="format_rocprof_output",
        choices=["csv", "rocpd"],
        default="csv",
        help="\t\t\tSet the format of output file of rocprof.",
    )
    profile_group.add_argument(
        "--pc-sampling-method",
        required=False,
        metavar="",
        dest="pc_sampling_method",
        default="stochastic",
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
        default=1048576,
        help=(
            "\t\t\tSet the interval of pc sampling.\n"
            "\t\t\t  For stochastic sampling, the interval is in cycles.\n"
            "\t\t\t  For host_trap sampling, the interval is in microsecond "
            "(DEFAULT: 1048576)."
        ),
    )
    profile_group.add_argument(
        "--rocprofiler-sdk-library-path",
        type=str,
        dest="rocprofiler_sdk_library_path",
        required=False,
        default=str(
            Path(os.getenv("ROCM_PATH", "/opt/rocm")) / "lib/librocprofiler-sdk.so"
        ),
        help="\t\t\tSet the path to rocprofiler SDK library.",
    )
    profile_group.add_argument(
        "--retain-rocpd-output",
        required=False,
        default=False,
        action="store_true",
        help=(
            "\t\t\tRetain the large raw rocpd database in workload directory.\n"
            "\t\t\tThis option requires --format-rocprof-output rocpd."
        ),
    )

    ## Roofline Command Line Options
    roofline_group.add_argument(
        "--roof-only",
        required=False,
        default=False,
        action="store_true",
        help=(
            "\t\t\tProfile roofline data only.\n"
            "\t\t\tCannot be used with --block or --set"
        ),
    )
    roofline_group.add_argument(
        "--sort",
        required=False,
        metavar="",
        type=str,
        default="kernels",
        choices=["kernels", "dispatches"],
        help=(
            "\t\t\tOverlay top kernels or top dispatches: (DEFAULT: kernels)\n"
            "\t\t\t   kernels\n"
            "\t\t\t   dispatches"
        ),
    )
    roofline_group.add_argument(
        "-m",
        "--mem-level",
        required=False,
        choices=["HBM", "L2", "vL1D", "LDS"],
        metavar="",
        nargs="+",
        type=str,
        default="ALL",
        help=(
            "\t\t\tFilter by memory level: (DEFAULT: ALL)\n"
            "\t\t\t   HBM\n"
            "\t\t\t   L2\n"
            "\t\t\t   vL1D\n"
            "\t\t\t   LDS"
        ),
    )
    roofline_group.add_argument(
        "--device",
        metavar="",
        required=False,
        default=0,
        type=int,
        help="\t\t\tTarget GPU device ID. (DEFAULT: 0)",
    )
    roofline_group.add_argument(
        "--kernel-names",
        required=False,
        default=False,
        action="store_true",
        help="\t\t\tInclude kernel names in roofline plot.",
    )
    roofline_group.add_argument(
        "-R",
        "--roofline-data-type",
        required=False,
        choices=[
            "FP4",
            "FP6",
            "FP8",
            "FP16",
            "BF16",
            "FP32",
            "FP64",
            "I8",
            "I32",
            "I64",
        ],
        metavar="",
        nargs="+",
        type=str,
        default=["FP32"],
        help=(
            "\t\t\tChoose datatypes to view roofline PDFs for: (DEFAULT: FP32)\n"
            "\t\t\t   FP4\n"
            "\t\t\t   FP6\n"
            "\t\t\t   FP8\n"
            "\t\t\t   FP16\n"
            "\t\t\t   BF16\n"
            "\t\t\t   FP32\n"
            "\t\t\t   FP64\n"
            "\t\t\t   I8\n"
            "\t\t\t   I32\n"
            "\t\t\t   I64\n"
            "\t\t\t "
        ),
    )

    # roofline_group.add_argument(
    #     '-w', '--workgroups', required=False, default=-1, type=int,
    #     help="\t\t\tNumber of kernel workgroups (DEFAULT: 1024)"
    # )
    # roofline_group.add_argument(
    #     '--wsize', required=False, default=-1, type=int,
    #     help="\t\t\tWorkgroup size (DEFAULT: 256)"
    # )
    # roofline_group.add_argument(
    #     '--dataset', required=False, default=-1, type=int,
    #     help="\t\t\tDataset size (DEFAULT: 536M)"
    # )
    # roofline_group.add_argument(
    #     '-e', '--experiments', required=False, default=-1, type=int,
    #     help="\t\t\tNumber of experiments (DEFAULT: 100)"
    # )
    # roofline_group.add_argument(
    #     '--iter', required=False, default=-1, type=int,
    #     help="\t\t\tNumber of iterations (DEFAULT: 10)"
    # )

    ## Analyze Command Line Options
    ## ----------------------------
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze existing profiling results at command line",
        usage="""
rocprof-compute analyze --path <workload_path> [analyze options]

-----------------------------------------------------------------------------------
Examples:
\trocprof-compute analyze -p workloads/vcopy/mi200/ --list-metrics gfx90a
\trocprof-compute analyze -p workloads/mixbench/mi200/ --dispatch 12 34 --decimal 3
\trocprof-compute analyze -p workloads/mixbench/mi200/ --gui
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
        analyze_parser, rocprof_compute_home, supported_archs, rocprof_compute_version
    )
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
        help="\t\tList all detected kernels and kernel dispatches.",
    )
    analyze_group.add_argument(
        "--list-available-metrics",
        dest="list_available_metrics",
        help="\t\tList all available metrics for analysis on current arch",
        action="store_true",
    )
    analyze_group.add_argument(
        "-k",
        "--kernel",
        metavar="",
        type=int,
        dest="gpu_kernel",
        nargs="+",
        action="append",
        help="\t\tSpecify kernel id(s) from --list-stats for filtering.",
    )
    analyze_group.add_argument(
        "-d",
        "--dispatch",
        dest="gpu_dispatch_id",
        metavar="",
        nargs="+",
        action="append",
        help="\t\tSpecify dispatch id(s) for filtering.",
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
        "--spatial-multiplexing",
        dest="spatial_multiplexing",
        required=False,
        default=False,
        action="store_true",
        help="\t\tMode of spatial multiplexing.",
    )
    analyze_group.add_argument(
        "--output-format",
        metavar="",
        dest="output_format",
        choices=["stdout", "txt", "csv", "db"],
        default="stdout",
        help=(
            "\t\tSet the format of output file or folder containing analysis data.\n"
            "\t\tBy default, file or folder created will "
            "have the name rocprof_compute_<uuid>.\n"
            "\t\tFile or folder name can be overriden using --output-name.\n"
            "\t\tDefault output format is stdout which will not "
            "generate any file/folder.\n"
        ),
    )
    analyze_group.add_argument(
        "--output-name",
        metavar="",
        dest="output_name",
        help=(
            "\t\tOverride the default output file name rocprof_compue_<uuid> "
            "with the specified name.\n"
            "\t\tThis is only applicable when --output-format txt/csv/db is used.\n"
        ),
    )
    analyze_group.add_argument(
        "--gui",
        type=int,
        nargs="?",
        const=8050,
        help=(
            "\t\tActivate a GUI to interate with rocprofiler-compute metrics.\n"
            "\t\tOptionally, specify port to launch application (DEFAULT: 8050)"
        ),
    )
    analyze_group.add_argument(
        "--tui",
        action="store_true",
        help="\t\tActivate a Textual User Interface (TUI) to "
        "interact with rocprofiler-compute metrics.",
    )
    analyze_group.add_argument(
        "-R",
        "--roofline-data-type",
        required=False,
        choices=[
            "FP4",
            "FP6",
            "FP8",
            "FP16",
            "BF16",
            "FP32",
            "FP64",
            "I8",
            "I32",
            "I64",
        ],
        metavar="",
        nargs="+",
        type=str,
        default=["FP32"],
        help=(
            "\t\tChoose datatypes to view roofline PDFs for: (DEFAULT: FP32)\n"
            "\t\t\t   FP4\n"
            "\t\t\t   FP6\n"
            "\t\t\t   FP8\n"
            "\t\t\t   FP16\n"
            "\t\t\t   BF16\n"
            "\t\t\t   FP32\n"
            "\t\t\t   FP64\n"
            "\t\t\t   I8\n"
            "\t\t\t   I32\n"
            "\t\t\t   I64\n\t\t\t "
        ),
    )
    analyze_group.add_argument(
        "--pc-sampling-sorting-type",
        required=False,
        metavar="",
        dest="pc_sampling_sorting_type",
        default="offset",
        type=str,
        help="\t\tSet the sorting type of pc sampling: "
        "offset or count (DEFAULT: offset).",
    )

    analyze_advanced_group.add_argument(
        "--random-port",
        action="store_true",
        help="\t\tRandomly generate a port to launch GUI application.\n"
        "\t\tRegistered Ports range inclusive (1024-49151).",
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
    analyze_advanced_group.add_argument(
        "--cols",
        type=int,
        dest="cols",
        metavar="",
        nargs="+",
        help=(
            "\t\tSpecify column indices to display.\n"
            "\t\tDefaults to display all columns."
        ),
    )
    analyze_advanced_group.add_argument(
        "--include-cols",
        dest="include_cols",
        metavar="",
        nargs="+",
        help=(
            "\t\tSpecify which hidden column names should be included in cli output.\n"
            '\t\tFor example, to show "Description" column which is hidden by '
            "default in cli output,\n"
            "\t\tuse the option --include-cols Description."
        ),
    )
    analyze_advanced_group.add_argument(
        "-g", dest="debug", action="store_true", help="\t\tDebug single metric."
    )
    analyze_advanced_group.add_argument(
        "--dependency",
        action="store_true",
        help="\t\tList the installation dependency.",
    )
    analyze_advanced_group.add_argument(
        "--kernel-verbose",
        required=False,
        metavar="",
        help="\t\tSpecify Kernel Name verbose level 1-5. "
        "Lower the level, shorter the kernel name. "
        "(DEFAULT: 5) (DISABLE: 5)",
        default=5,
        type=int,
    )
    analyze_advanced_group.add_argument(
        "--report-diff", default=0, nargs="?", type=int, help=argparse.SUPPRESS
    )
    analyze_advanced_group.add_argument(
        "--specs-correction",
        type=str,
        metavar="",
        help="\t\tSpecify the specs to correct. e.g. "
        '--specs-correction="specname1:specvalue1,specname2:specvalue2"',
    )
    analyze_advanced_group.add_argument(
        "--list-nodes",
        action="store_true",
        help="\t\tMulti-node option: list all node names.",
    )
    analyze_advanced_group.add_argument(
        "--nodes",
        metavar="",
        type=str,
        dest="nodes",
        nargs="*",
        help=(
            "\t\tMulti-node option: filter with node names. "
            "Enable it without node names means ALL."
        ),
    )
