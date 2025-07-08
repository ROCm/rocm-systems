##############################################################################bl
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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################el

import argparse
import os
import re
import shutil
from pathlib import Path


def print_avail_arch(avail_arch: list):
    ret_str = "\t\tList all available metrics for analysis on specified arch:"
    for arch in avail_arch:
        ret_str += "\n\t\t\t   {}".format(arch)
    return ret_str


def add_general_group(parser, rocprof_compute_version):
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
    # Nowhere to load specs from in db mode
    if "database" not in parser.usage:
        general_group.add_argument(
            "-s", "--specs", action="store_true", help="Print system specs and exit."
        )


def omniarg_parser(
    parser, rocprof_compute_home, supported_archs, rocprof_compute_version
):
    # -----------------------------------------
    # Parse arguments (dependent on mode)
    # -----------------------------------------

    ## General Command Line Options
    ## ----------------------------
    add_general_group(parser, rocprof_compute_version)
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

rocprof-compute profile --name <workload_name> [profile options] [roofline options] -- <profile_cmd>

---------------------------------------------------------------------------------
Examples:
\trocprof-compute profile -n vcopy_all -- ./vcopy -n 1048576 -b 256
\trocprof-compute profile -n vcopy_SPI_TCC -b SQ TCC -- ./vcopy -n 1048576 -b 256
\trocprof-compute profile -n vcopy_kernel -k vecCopy -- ./vcopy -n 1048576 -b 256
\trocprof-compute profile -n vcopy_disp -d 0 -- ./vcopy -n 1048576 -b 256
\trocprof-compute profile -n vcopy_roof --roof-only -- ./vcopy -n 1048576 -b 256
---------------------------------------------------------------------------------
        """,
        prog="tool",
        allow_abbrev=False,
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(
            prog, max_help_position=40
        ),
    )
    profile_parser._optionals.title = "Help"

    add_general_group(profile_parser, rocprof_compute_version)
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
    profile_group.add_argument("--target", type=str, default=None, help=argparse.SUPPRESS)
    profile_group.add_argument(
        "--pid",
        type=str,
        dest="pid",
        metavar="",
        default=None,
        required=False,
        help="\t\t\tenable attach/detach mode and set its process id.",
    )
    profile_group.add_argument(
        "-p",
        "--path",
        metavar="",
        type=str,
        dest="path",
        default=str(Path(os.getcwd()).joinpath("workloads")),
        required=False,
        help="\t\t\tSpecify path to save workload.\n\t\t\t(DEFAULT: {}/workloads/<name>)".format(
            os.getcwd()
        ),
    )
    profile_group.add_argument(
        "--subpath",
        metavar="",
        type=str,
        dest="subpath",
        default="gpu",
        required=False,
        help="\t\t\tSpecify the type of subpath to save workload: node_name, gpu_model.",
    )
    profile_group.add_argument(
        "--hip-trace",
        dest="hip_trace",
        required=False,
        default=False,
        action="store_true",
        help="\t\t\tHIP trace, execturion trace for the entire application at the HIP level.",
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

    class AggregateDict(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            aggregated_dict = getattr(namespace, self.dest, {})
            if aggregated_dict is None:
                aggregated_dict = {}
            for key, value in values:
                aggregated_dict[key] = value
            setattr(namespace, self.dest, aggregated_dict)

    def validate_block(value):
        # Metric id regex, for example, 10, 4, 4.3, 4.32
        # Dont allow more than two digits after decimal point
        metric_id_pattern = re.compile(r"^\d+$|^\d+\.\d$|^\d+\.\d\d$")
        # Allow only the following hardware blocks
        hardware_block_pattern = re.compile(r"^(SQ|SQC|TA|TD|TCP|TCC|SPI|CPC|CPF)$")
        if metric_id_pattern.match(value):
            return (str(value), "metric_id")
        if hardware_block_pattern.match(value):
            return (str(value), "hardware_block")
        raise argparse.ArgumentTypeError(f"Invalid hardware block or metric id: {value}")

    profile_group.add_argument(
        "-b",
        "--block",
        type=validate_block,
        action=AggregateDict,
        dest="filter_blocks",
        metavar="",
        nargs="+",
        required=False,
        default={},
        help="""\t\t\tSpecify metric id(s) from --list-metrics for filtering (e.g. 10, 4, 4.3).
    \t\t\tCan provide multiple space separated arguments.
    \t\t\tCan also accept Hardware blocks.
    \t\t\tHardware block filtering (to be deprecated soon):
    \t\t\t   SQ
    \t\t\t   SQC
    \t\t\t   TA
    \t\t\t   TD
    \t\t\t   TCP
    \t\t\t   TCC
    \t\t\t   SPI
    \t\t\t   CPC
    \t\t\t   CPF""",
    )
    profile_group.add_argument(
        "--list-metrics",
        metavar="",
        nargs="?",
        const="",
        # Argument to --list-metrics is optional
        choices=[""] + list(supported_archs.keys()),  # ["gfx906", "gfx908", "gfx90a"],
        help=print_avail_arch(supported_archs.keys()),
    )
    profile_group.add_argument(
        "--config-dir",
        dest="config_dir",
        metavar="",
        help="\t\t\tSpecify the directory of customized report section configs.",
        default=rocprof_compute_home.joinpath("rocprof_compute_soc/analysis_configs/"),
    )
    profile_group.add_argument(
        "--join-type",
        metavar="",
        required=False,
        choices=["kernel", "grid"],
        default="grid",
        help="\t\t\tChoose how to join rocprof runs: (DEFAULT: grid)\n\t\t\t   kernel (i.e. By unique kernel name dispatches)\n\t\t\t   grid (i.e. By unique kernel name + grid size dispatches)",
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
        choices=["json", "csv"],
        default="csv",
        help="\t\t\tSet the format of output file of rocprof.",
    )

    profile_group.add_argument(
        "--pc-sampling-method",
        required=False,
        metavar="",
        dest="pc_sampling_method",
        default="stochastic",
        help="\t\t\tSet the method of pc sampling, stochastic or host_trap. Support stochastic only >= MI300",
    )

    profile_group.add_argument(
        "--pc-sampling-interval",
        required=False,
        metavar="",
        dest="pc_sampling_interval",
        default=1048576,
        help="\t\t\tSet the interval of pc sampling.\n\t\t\t   For stochastic sampling, the interval is in cycles.\n\t\t\t   For host_trap sampling, the interval is in microsecond (DEFAULT: 1048576).",
    )

    profile_group.add_argument(
        "--rocprofiler-sdk-library-path",
        type=str,
        dest="rocprofiler_sdk_library_path",
        required=False,
        default="/opt/rocm/lib/librocprofiler-sdk.so",
        help="\t\t\tSet the path to rocprofiler SDK library.",
    )

    ## Roofline Command Line Options
    roofline_group.add_argument(
        "--roof-only",
        required=False,
        default=False,
        action="store_true",
        help="\t\t\tProfile roofline data only.",
    )
    roofline_group.add_argument(
        "--sort",
        required=False,
        metavar="",
        type=str,
        default="kernels",
        choices=["kernels", "dispatches"],
        help="\t\t\tOverlay top kernels or top dispatches: (DEFAULT: kernels)\n\t\t\t   kernels\n\t\t\t   dispatches",
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
        help="\t\t\tFilter by memory level: (DEFAULT: ALL)\n\t\t\t   HBM\n\t\t\t   L2\n\t\t\t   vL1D\n\t\t\t   LDS",
    )
    roofline_group.add_argument(
        "--device",
        metavar="",
        required=False,
        default=-1,
        type=int,
        help="\t\t\tTarget GPU device ID. (DEFAULT: ALL)",
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
        choices=["FP4", "FP6", "FP8", "FP16", "BF16", "FP32", "FP64", "I8", "I32", "I64"],
        metavar="",
        nargs="+",
        type=str,
        default=["FP32"],
        help="\t\t\tChoose datatypes to view roofline PDFs for: (DEFAULT: FP32)\n\t\t\t   FP4\n\t\t\t   FP6\n\t\t\t   FP8\n\t\t\t   FP16\n\t\t\t   BF16\n\t\t\t   FP32\n\t\t\t   FP64\n\t\t\t   I8\n\t\t\t   I32\n\t\t\t   I64\n\t\t\t ",
    )

    # roofline_group.add_argument('-w', '--workgroups', required=False, default=-1, type=int, help="\t\t\tNumber of kernel workgroups (DEFAULT: 1024)")
    # roofline_group.add_argument('--wsize', required=False, default=-1, type=int, help="\t\t\tWorkgroup size (DEFAULT: 256)")
    # roofline_group.add_argument('--dataset', required=False, default = -1, type=int, help="\t\t\tDataset size (DEFAULT: 536M)")
    # roofline_group.add_argument('-e', '--experiments', required=False, default=-1, type=int, help="\t\t\tNumber of experiments (DEFAULT: 100)")
    # roofline_group.add_argument('--iter', required=False, default=-1, type=int, help="\t\t\tNumber of iterations (DEFAULT: 10)")

    ## Database Command Line Options
    ## ----------------------------
    db_parser = subparsers.add_parser(
        "database",
        help="Interact with rocprofiler-compute database",
        usage="""
            \nrocprof-compute database <interaction type> [connection options]

            \n\n-------------------------------------------------------------------------------
            \nExamples:
            \n\trocprof-compute database --import -H pavii1 -u temp -t asw -w workloads/vcopy/mi200/
            \n\trocprof-compute database --remove -H pavii1 -u temp -w rocprofiler-compute_asw_sample_mi200
            \n-------------------------------------------------------------------------------\n
        """,
        prog="tool",
        allow_abbrev=False,
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(
            prog, max_help_position=40
        ),
    )
    db_parser._optionals.title = "Help"

    add_general_group(db_parser, rocprof_compute_version)
    interaction_group = db_parser.add_argument_group("Interaction Type")
    connection_group = db_parser.add_argument_group("Connection Options")

    interaction_group.add_argument(
        "-i",
        "--import",
        required=False,
        dest="upload",
        action="store_true",
        help="\t\t\t\tImport workload to rocprofiler-compute DB",
    )
    interaction_group.add_argument(
        "-r",
        "--remove",
        required=False,
        dest="remove",
        action="store_true",
        help="\t\t\t\tRemove a workload from rocprofiler-compute DB",
    )

    connection_group.add_argument(
        "-H",
        "--host",
        required=True,
        metavar="",
        help="\t\t\t\tName or IP address of the server host.",
    )
    connection_group.add_argument(
        "-P",
        "--port",
        required=False,
        metavar="",
        help="\t\t\t\tTCP/IP Port. (DEFAULT: 27018)",
        default=27018,
    )
    connection_group.add_argument(
        "-u",
        "--username",
        required=True,
        metavar="",
        help="\t\t\t\tUsername for authentication.",
    )
    connection_group.add_argument(
        "-p",
        "--password",
        metavar="",
        help="\t\t\t\tThe user's password. (will be requested later if it's not set)",
        default="",
    )
    connection_group.add_argument(
        "-t", "--team", required=False, metavar="", help="\t\t\t\tSpecify Team prefix."
    )
    connection_group.add_argument(
        "-w",
        "--workload",
        required=True,
        metavar="",
        dest="workload",
        help="\t\t\t\tSpecify name of workload (to remove) or path to workload (to import)",
    )
    connection_group.add_argument(
        "--kernel-verbose",
        required=False,
        metavar="",
        help="\t\tSpecify Kernel Name verbose level 1-5. Lower the level, shorter the kernel name. (DEFAULT: 5) (DISABLE: 5)",
        default=5,
        type=int,
    )

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

    add_general_group(analyze_parser, rocprof_compute_version)
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
        "--list-metrics",
        metavar="",
        choices=supported_archs.keys(),  # ["gfx906", "gfx908", "gfx90a"],
        help=print_avail_arch(supported_archs.keys()),
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
        "-o",
        "--output",
        metavar="",
        dest="output_file",
        help="\t\tSpecify an output file to save analysis results.",
    )
    analyze_group.add_argument(
        "--gui",
        type=int,
        nargs="?",
        const=8050,
        help="\t\tActivate a GUI to interate with rocprofiler-compute metrics.\n\t\tOptionally, specify port to launch application (DEFAULT: 8050)",
    )
    analyze_group.add_argument(
        "--tui",
        action="store_true",
        help="\t\tActivate a Textual User Interface (TUI) to interact with rocprofiler-compute metrics.",
    )
    analyze_group.add_argument(
        "-R",
        "--roofline-data-type",
        required=False,
        choices=["FP4", "FP6", "FP8", "FP16", "BF16", "FP32", "FP64", "I8", "I32", "I64"],
        metavar="",
        nargs="+",
        type=str,
        default=["FP32"],
        help="\t\tChoose datatypes to view roofline PDFs for: (DEFAULT: FP32)\n\t\t\t   FP4\n\t\t\t   FP6\n\t\t\t   FP8\n\t\t\t   FP16\n\t\t\t   BF16\n\t\t\t   FP32\n\t\t\t   FP64\n\t\t\t   I8\n\t\t\t   I32\n\t\t\t   I64\n\t\t\t ",
    )

    analyze_group.add_argument(
        "--pc-sampling-sorting-type",
        required=False,
        metavar="",
        dest="pc_sampling_sorting_type",
        default="offset",
        type=str,
        help="\t\tSet the sorting type of pc sampling: offset or count (DEFAULT: offset).",
    )

    analyze_advanced_group.add_argument(
        "--random-port",
        action="store_true",
        help="\t\tRandomly generate a port to launch GUI application.\n\t\tRegistered Ports range inclusive (1024-49151).",
    )
    analyze_advanced_group.add_argument(
        "--max-stat-num",
        dest="max_stat_num",
        metavar="",
        type=int,
        default=10,
        help='\t\tSpecify the maximum number of stats shown in "Top Stats" tables (DEFAULT: 10)',
    )
    analyze_advanced_group.add_argument(
        "-n",
        "--normal-unit",
        dest="normal_unit",
        metavar="",
        default="per_kernel",
        choices=["per_wave", "per_cycle", "per_second", "per_kernel"],
        help="\t\tSpecify the normalization unit: (DEFAULT: per_kernel)\n\t\t   per_wave\n\t\t   per_cycle\n\t\t   per_second\n\t\t   per_kernel",
    )
    analyze_advanced_group.add_argument(
        "-t",
        "--time-unit",
        dest="time_unit",
        metavar="",
        default="ns",
        choices=["s", "ms", "us", "ns"],
        help="\t\tSpecify display time unit in kernel top stats: (DEFAULT: ns)\n\t\t   s\n\t\t   ms\n\t\t   us\n\t\t   ns",
    )
    analyze_advanced_group.add_argument(
        "--decimal",
        type=int,
        metavar="",
        default=2,
        help="\t\tSpecify desired decimal precision of analysis results. (DEFAULT: 2)",
    )
    analyze_advanced_group.add_argument(
        "--config-dir",
        dest="config_dir",
        metavar="",
        help="\t\tSpecify the directory of customized configs.",
        default=rocprof_compute_home.joinpath("rocprof_compute_soc/analysis_configs/"),
    )
    analyze_advanced_group.add_argument(
        "--save-dfs",
        dest="df_file_dir",
        metavar="",
        help="\t\tSpecify the dirctory to save analysis dataframe csv files.",
    )
    analyze_advanced_group.add_argument(
        "--cols",
        type=int,
        dest="cols",
        metavar="",
        nargs="+",
        help="\t\tSpecify column indices to display.",
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
        help="\t\tSpecify Kernel Name verbose level 1-5. Lower the level, shorter the kernel name. (DEFAULT: 5) (DISABLE: 5)",
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
        help="\t\tSpecify the specs to correct. e.g. --specs-correction='specname1:specvalue1,specname2:specvalue2'",
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
        help="\t\tMulti-node option: filter with node names. Enable it without node names means ALL.",
    )
