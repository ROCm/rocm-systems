##############################################################################bl
# MIT License
#
# Copyright (c) 2021 - 2023 Advanced Micro Devices, Inc. All Rights Reserved.
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
import shutil
import os

def print_avail_arch(avail_arch: list):
    ret_str = "\t\tList all available metrics for analysis on specified arch:"
    for arch in avail_arch:
        ret_str += "\n\t\t   {}".format(arch)
    return ret_str

def omniarg_parser(parser, omniperf_home, supported_archs, omniperf_version):
    # -----------------------------------------
    # Parse arguments (dependent on mode)
    # -----------------------------------------

    ## General Command Line Options
    ## ----------------------------
    general_group = parser.add_argument_group("General Options")
    parser._positionals.title = "Modes"
    parser._optionals.title = "Help"
    general_group.add_argument("-v", "--version", action="version", version=omniperf_version["ver_pretty"])
    general_group.add_argument("-s", "--specs", action="store_true", help="Print system specs.")

    subparsers = parser.add_subparsers(
        dest="mode", help="Select mode of interaction with the target application:"
    )

    ## Profile Command Line Options
    ## ----------------------------
    profile_parser = subparsers.add_parser(
        "profile",
        help="Profile the target application",
        usage="""
            \nomniperf profile --name <workload_name> [profile options] [roofline options] -- <profile_cmd>

            \n\n-------------------------------------------------------------------------------
            \nExamples:
            \n\tomniperf profile -n vcopy_all -- ./vcopy -n 1048576 -b 256
            \n\tomniperf profile -n vcopy_SPI_TCC -b SQ TCC -- ./vcopy -n 1048576 -b 256
            \n\tomniperf profile -n vcopy_kernel -k vecCopy -- ./vcopy -n 1048576 -b 256
            \n\tomniperf profile -n vcopy_disp -d 0 -- ./vcopy -n 1048576 -b 256
            \n\tomniperf profile -n vcopy_roof --roof-only -- ./vcopy -n 1048576 -b 256
            \n-------------------------------------------------------------------------------\n
        """,
        prog="tool",
        allow_abbrev=False,
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(
            prog, max_help_position=40
        ),
    )
    profile_parser._optionals.title = "Help"

    general_group = profile_parser.add_argument_group("General Options")
    profile_group = profile_parser.add_argument_group("Profile Options")
    roofline_group = profile_parser.add_argument_group("Standalone Roofline Options")

    general_group.add_argument("-v", "--version", action="version", version=omniperf_version["ver_pretty"])
    general_group.add_argument(
        "-V", "--verbose", help="Increase output verbosity", action="count", default=0
    )

    profile_group.add_argument(
        "-n",
        "--name",
        type=str,
        metavar="",
        dest="name",
        required=True,
        help="\t\t\tAssign a name to workload.",
    )
    profile_group.add_argument("--target", type=str, default=None, help=argparse.SUPPRESS)
    profile_group.add_argument(
        "-p",
        "--path",
        metavar="",
        type=str,
        dest="path",
        default=os.path.join(os.getcwd(), "workloads"),
        required=False,
        help="\t\t\tSpecify path to save workload.\n\t\t\t(DEFAULT: {}/workloads/<name>)".format(
            os.getcwd()
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
        "-b",
        "--ipblocks",
        type=str,
        dest="ipblocks",
        metavar="",
        nargs="+",
        required=False,
        choices=["SQ", "SQC", "TA", "TD", "TCP", "TCC", "SPI", "CPC", "CPF"],
        help="\t\t\tIP block filtering:\n\t\t\t   SQ\n\t\t\t   SQC\n\t\t\t   TA\n\t\t\t   TD\n\t\t\t   TCP\n\t\t\t   TCC\n\t\t\t   SPI\n\t\t\t   CPC\n\t\t\t   CPF",
    )

    result = shutil.which("rocscope")
    if result:
        profile_group.add_argument(
            "-l",
            "--i-feel-lucky",
            required=False,
            default=False,
            action="store_true",
            dest="lucky",
            help="\t\t\tProfile only the most time consuming kernels.",
        )
        profile_group.add_argument(
            "-r",
            "--use-rocscope",
            required=False,
            default=False,
            action="store_true",
            dest="use_rocscope",
            help="\t\t\tUse rocscope for profiling",
        )
        profile_group.add_argument(
            "-s",
            "--kernel-summaries",
            required=False,
            default=False,
            action="store_true",
            dest="summaries",
            help="\t\t\tCreate kernel summaries.",
        )
    else:
        profile_group.add_argument(
            "--i-feel-lucky", default=False, dest="lucky", help=argparse.SUPPRESS
        )
        profile_group.add_argument(
            "--use-rocscope", default=False, dest="use_rocscope", help=argparse.SUPPRESS
        )
        profile_group.add_argument(
            "--kernel-summaries", default=False, dest="summaries", help=argparse.SUPPRESS
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
        "--kernel-verbose",
        required=False,
        metavar="",
        help="\t\t\tSpecify Kernel Name verbose level 1-5. Lower the level, shorter the kernel name. (DEFAULT: 2) (DISABLE: 5)",
        default=2,
        type=int,
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
        help="\t\t\tGPU device ID. (DEFAULT: ALL)",
    )
    roofline_group.add_argument(
        "--kernel-names",
        required=False,
        default=False,
        action="store_true",
        help="\t\t\tInclude kernel names in roofline plot.",
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
        help="Interact with Omniperf database",
        usage="""
            \nomniperf database <interaction type> [connection options]

            \n\n-------------------------------------------------------------------------------
            \nExamples:
            \n\tomniperf database --import -H pavii1 -u temp -t asw -w workloads/vcopy/mi200/
            \n\tomniperf database --remove -H pavii1 -u temp -w omniperf_asw_sample_mi200
            \n-------------------------------------------------------------------------------\n
        """,
        prog="tool",
        allow_abbrev=False,
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(
            prog, max_help_position=40
        ),
    )
    db_parser._optionals.title = "Help"

    general_group = db_parser.add_argument_group("General Options")
    interaction_group = db_parser.add_argument_group("Interaction Type")
    connection_group = db_parser.add_argument_group("Connection Options")

    general_group.add_argument("-v", "--version", action="version", version=omniperf_version["ver_pretty"])
    general_group.add_argument(
        "-V", "--verbose", help="Increase output verbosity", action="count", default=0
    )
    general_group.add_argument("-s", "--specs", action="store_true", help="Print system specs.")

    interaction_group.add_argument(
        "-i",
        "--import",
        required=False,
        dest="upload",
        action="store_true",
        help="\t\t\t\tImport workload to Omniperf DB",
    )
    interaction_group.add_argument(
        "-r",
        "--remove",
        required=False,
        dest="remove",
        action="store_true",
        help="\t\t\t\tRemove a workload from Omniperf DB",
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

    ## Analyze Command Line Options
    ## ----------------------------
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze existing profiling results at command line",
        usage="""
            \nomniperf analyze --path <workload_path> [analyze options]

            \n\n-------------------------------------------------------------------------------
            \nExamples:
            \n\tomniperf analyze -p workloads/vcopy/mi200/ --list-metrics gfx90a
            \n\tomniperf analyze -p workloads/mixbench/mi200/ --dispatch 12 34 --decimal 3
            \n\tomniperf analyze -p workloads/mixbench/mi200/ --gui
            \n-------------------------------------------------------------------------------\n
        """,
        prog="tool",
        allow_abbrev=False,
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(
            prog, max_help_position=40
        ),
    )
    analyze_parser._optionals.title = "Help"

    general_group = analyze_parser.add_argument_group("General Options")
    analyze_group = analyze_parser.add_argument_group("Analyze Options")
    analyze_advanced_group = analyze_parser.add_argument_group("Advanced Options")

    general_group.add_argument("-v", "--version", action="version", version=omniperf_version["ver_pretty"])
    general_group.add_argument(
        "-V", "--verbose", help="Increase output verbosity", action="count", default=0
    )
    general_group.add_argument("-s", "--specs", action="store_true", help="Print system specs.")

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
        "--list-kernels",
        action="store_true",
        help="\t\tList all detected kernels. Sorted by duration (descending order).",
    )
    analyze_group.add_argument(
        "--list-metrics",
        metavar="",
        choices=supported_archs.keys(),#["gfx906", "gfx908", "gfx90a"],
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
        help="\t\tSpecify kernel id(s) from --list-kernels for filtering.",
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
        "--metric",
        dest="filter_metrics",
        metavar="",
        nargs="+",
        help="\t\tSpecify IP block/metric id(s) from --list-metrics for filtering.",
    )
    analyze_group.add_argument(
        "--gpu-id",
        dest="gpu_id",
        metavar="",
        nargs="+",
        help="\t\tSpecify GPU id(s) for filtering.",
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
        help="\t\tActivate a GUI to interate with Omniperf metrics.\n\t\tOptionally, specify port to launch application (DEFAULT: 8050)",
    )
    analyze_advanced_group.add_argument(
        "--random-port",
        action="store_true",
        help="\t\tRandomly generate a port to launch GUI application.\n\t\tRegistered Ports range inclusive (1024-49151).",
    )
    analyze_advanced_group.add_argument(
        "--max-kernel-num",
        dest="max_kernel_num",
        metavar="",
        type=int,
        default=10,
        help="\t\tSpecify the maximum number of kernels shown in \"Top Stats\" table (DEFAULT: 10)",
    )
    analyze_advanced_group.add_argument(
        "-n",
        "--normal-unit",
        dest="normal_unit",
        metavar="",
        default="per_wave",
        choices=["per_wave", "per_cycle", "per_second", "per_kernel"],
        help="\t\tSpecify the normalization unit: (DEFAULT: per_wave)\n\t\t   per_wave\n\t\t   per_cycle\n\t\t   per_second\n\t\t   per_kernel",
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
        default=omniperf_home.joinpath("omniperf_soc/analysis_configs/"),
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
    analyze_advanced_group.add_argument("-g", dest="debug", action="store_true", help="\t\tDebug single metric.")
    analyze_advanced_group.add_argument(
        "--dependency", action="store_true", help="\t\tList the installation dependency."
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
        help="\t\tSpecify the specs to correct."
    )