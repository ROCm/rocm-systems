################################################################################
# Copyright (c) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
################################################################################

import os
import argparse
import subprocess

from common import (
    OMNIPERF_HOME,
    PROG,
    SOC_LIST,
)  # Import global variables
from common import getVersion, getVersionDisplay


def parse(my_parser):

    # versioning info
    vData = getVersion()
    versionString = getVersionDisplay(vData["version"], vData["sha"], vData["mode"])

    # -----------------------------------------
    # Parse arguments (dependent on mode)
    # -----------------------------------------

    ## General Command Line Options
    ## ----------------------------
    general_group = my_parser.add_argument_group("General Options")
    my_parser._positionals.title = "Modes"
    my_parser._optionals.title = "Help"
    general_group.add_argument("-v", "--version", action="version", version=versionString)

    subparsers = my_parser.add_subparsers(
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
                                        \n\tomniperf profile -n vcopy_all -- ./vcopy 1048576 256
                                        \n\tomniperf profile -n vcopy_SPI_TD -b SQ TCC -- ./vcopy 1048576 256
                                        \n\tomniperf profile -n vcopy_kernel -k vecCopy -- ./vcopy 1048576 256
                                        \n\tomniperf profile -n vcopy_disp -d 0 -- ./vcopy 1048576 256
                                        \n\tomniperf profile -n vcopy_roof --roof-only -- ./vcopy 1048576 256
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

    general_group.add_argument("-v", "--version", action="version", version=versionString)
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
        default=os.getcwd() + "/workloads",
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

    result = subprocess.run(
        ["which", "rocscope"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    if result.returncode == 0:
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
        help="\t\t\tOverlay top kernels or top dispatches: (DEFAULT: kernels)\n\t\t\t   kernels\n\t\t\t   dispatches",
    )
    roofline_group.add_argument(
        "-m",
        "--mem-level",
        required=False,
        choices=["HBM", "L2", "vL1D", "LDS"],
        metavar="",
        type=str,
        default="ALL",
        help="\t\t\tFilter by memory level: (DEFAULT: ALL)\n\t\t\t   HBM\n\t\t\t   L2\n\t\t\t   vL1D\n\t\t\t   LDS",
    )
    roofline_group.add_argument(
        "--axes",
        default=None,
        type=float,
        required=False,
        nargs="+",
        metavar="",
        help="\t\t\tDesired axis values for graph. As follows:\n\t\t\t   xmin xmax ymin ymax",
    )
    roofline_group.add_argument(
        "--device",
        metavar="",
        required=False,
        default=-1,
        type=int,
        help="\t\t\tGPU device ID. (DEFAULT: ALL)",
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

    general_group.add_argument("-v", "--version", action="version", version=versionString)
    general_group.add_argument(
        "-V", "--verbose", help="Increase output verbosity", action="count", default=0
    )

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
    connection_group.add_argument(
        "-k",
        "--kernelVerbose",
        required=False,
        metavar="",
        help="\t\t\t\tSpecify Kernel Name verbose level 1-5. Lower the level, shorter the kernel name. (DEFAULT: 2) (DISABLE: 5)",
        default=2,
        type=int,
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
                                        \n\tomniperf analyze -p workloads/mixbench/mi200/ --filter-dispatch-ids 12 34 --decimal 3
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

    general_group.add_argument("-v", "--version", action="version", version=versionString)
    general_group.add_argument(
        "-V", "--verbose", help="Increase output verbosity", action="count", default=0
    )

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
        "-o",
        "--output",
        metavar="",
        dest="output_file",
        help="\t\tSpecify the output file.",
    )
    analyze_group.add_argument(
        "--list-kernels", action="store_true", help="\t\tList kernels."
    )
    analyze_group.add_argument(
        "--list-metrics",
        metavar="",
        choices=["gfx906", "gfx908", "gfx90a"],
        help="\t\tList metrics can be customized to analyze on specific arch:\n\t\t   gfx906\n\t\t   gfx908\n\t\t   gfx90a",
    )
    analyze_group.add_argument(
        "-b",
        "--filter-metrics",
        metavar="",
        nargs="+",
        help="\t\tSpecify IP block/metric Ids from --list-metrics.",
    )
    analyze_group.add_argument(
        "-k",
        "--filter-kernels",
        metavar="",
        type=int,
        dest="gpu_kernel",
        nargs="+",
        action="append",
        help="\t\tSpecify kernel id from --list-kernels.",
    )
    analyze_group.add_argument(
        "--filter-dispatch-ids",
        dest="gpu_dispatch_id",
        metavar="",
        nargs="+",
        action="append",
        help="\t\tSpecify dispatch IDs.",
    )
    analyze_group.add_argument(
        "--filter-gpu-ids",
        dest="gpu_id",
        metavar="",
        nargs="+",
        help="\t\tSpecify GPU IDs.",
    )
    analyze_group.add_argument(
        "-n",
        "--normal-unit",
        dest="normal_unit",
        metavar="",
        default="per_wave",
        choices=["per_wave", "per_cycle", "per_second", "per_kernel"],
        help="\t\tSpecify the normalization unit: (DEFAULT: per_wave)\n\t\t   per_wave\n\t\t   per_cycle\n\t\t   per_second\n\t\t   per_kernel",
    )
    analyze_group.add_argument(
        "--config-dir",
        dest="config_dir",
        metavar="",
        help="\t\tSpecify the directory of customized configs.",
        default=OMNIPERF_HOME.joinpath("omniperf_analyze/configs"),
    )
    analyze_group.add_argument(
        "-t",
        "--time-unit",
        dest="time_unit",
        metavar="",
        default="ns",
        choices=["s", "ms", "us", "ns"],
        help="\t\tSpecify display time unit in kernel top stats: (DEFAULT: ns)\n\t\t   s\n\t\t   ms\n\t\t   us\n\t\t   ns",
    )
    analyze_group.add_argument(
        "--decimal",
        type=int,
        metavar="",
        default=2,
        help="\t\tSpecify the decimal to display. (DEFAULT: 2)",
    )
    analyze_group.add_argument(
        "--cols",
        type=int,
        dest="cols",
        metavar="",
        nargs="+",
        help="\t\tSpecify column indices to display.",
    )
    analyze_group.add_argument("-g", action="store_true", help="\t\tDebug single metric.")
    analyze_group.add_argument(
        "--dependency", action="store_true", help="\t\tList the installation dependency."
    )
    analyze_group.add_argument(
        "--gui",
        type=int,
        nargs="?",
        const=8050,
        help="\t\tActivate a GUI to interate with Omniperf metrics.\n\t\tOptionally, specify port to launch application (DEFAULT: 8050)",
    )
