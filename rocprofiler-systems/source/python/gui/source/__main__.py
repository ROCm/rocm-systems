#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All Rights Reserved.
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

from __future__ import absolute_import

__author__ = "AMD Research"
__copyright__ = "Copyright 2023-2024, Advanced Micro Devices, Inc."
__license__ = "MIT"
__maintainer__ = "AMD Research"
__status__ = "Development"

import argparse
import os.path
import dash
import dash_bootstrap_components as dbc
import json

from pathlib import Path

from . import gui
from .parser import parse_files, find_causal_files, set_num_stddev
from . import __version__


def causal(args):
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

    workload_path = args.path[:]
    input_files = []

    input_files = find_causal_files(workload_path, args.verbose, args.recursive)

    # unique
    input_files = list(set(input_files))

    set_num_stddev(args.stddev)
    num_speedups = len(args.speedups)

    if num_speedups > 0 and args.min_points > num_speedups:
        args.min_points = num_speedups

    results_df, samples_df, file_names = parse_files(
        input_files,
        args.experiments,
        args.progress_points,
        args.speedups,
        args.min_points,
        args.validate,
        args.verbose,
        args.cli,
    )

    if not args.cli:
        max_points = 9
        sortOptions = ["Alphabetical", "Max Speedup", "Min Speedup", "Impact"]
        input_filters = [
            {
                "Name": "Sort by",
                "values": list(map(str, sortOptions)),
                "default": "Impact",
                "type": "Name",
                "multi": False,
            },
            {
                "Name": "Select Workload",
                "values": file_names,
                "default": file_names,
                "type": "Name",
                "multi": True,
            },
            {"Name": "points", "filter": [], "values": max_points, "type": "int"},
        ]

        gui.build_causal_layout(
            app,
            input_filters,
            args.path,
            results_df,
            samples_df,
            args.verbose,
            args.light,
        )
        app.run_server(
            debug=True if args.verbose >= 3 else False,
            host=args.ip_address,
            port=args.ip_port,
        )


def main():
    settings = {}

    this_dir = Path(__file__).resolve().parent
    if os.path.basename(this_dir) == "source":
        settings_path = os.path.join(f"{this_dir.parent}", "settings.json")
    else:
        settings_path = os.path.join(f"{this_dir}", "settings.json")

    for itr in [
        settings_path,
        os.path.join(os.environ.get("HOME"), ".rocprof-sys-causal-plot.json"),
    ]:
        if os.path.exists(itr):
            with open(itr, "r") as f:
                settings = json.load(f)
            break

    default_settings = {}
    default_settings["path"] = ""
    default_settings["cli"] = False
    default_settings["light"] = False
    default_settings["ip_address"] = "0.0.0.0"
    default_settings["ip_port"] = 8051
    default_settings["experiments"] = ".*"
    default_settings["progress_points"] = ".*"
    default_settings["min_points"] = 5
    default_settings["recursive"] = False
    default_settings["verbose"] = 0
    default_settings["stddev"] = 1.0

    for key, value in default_settings.items():
        if key not in settings:
            settings[key] = value

    my_parser = argparse.ArgumentParser(
        description="AMD's ROCm Systems Profiler Causal Profiling GUI",
        prog="tool",
        allow_abbrev=False,
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(
            prog, max_help_position=40
        ),
        usage="""
        rocprof-sys-causal-plot [ARGS...]

        -------------------------------------------------------------------------------
        Examples:
        \trocprof-sys-causal-plot --path workloads/toy -n 0
        -------------------------------------------------------------------------------
        """,
    )

    my_parser.add_argument(
        "--version",
        action="version",
        version="ROCm Systems Profiler Causal Viewer v{}\n".format(
            f"{__version__}".strip("\n")
        ),
    )

    my_parser.add_argument(
        "-c",
        "--cli",
        action="store_true",
        required=False,
        default=settings["cli"],
        help="Do not launch the GUI, print the causal analysis out to the console only",
    )

    my_parser.add_argument(
        "-l",
        "--light",
        action="store_true",
        required=False,
        default=settings["light"],
        help="light Mode",
    )

    my_parser.add_argument(
        "-w",
        "--workload",
        "--path",
        metavar="FOLDER",
        type=str,
        dest="path",
        default=settings["path"],
        required=False,
        nargs="+",
        help="Specify path to causal profiles",
    )

    my_parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        default=settings["recursive"],
        help="Recursively search for causal profiles in workload directory",
    )

    my_parser.add_argument(
        "-V",
        "--verbose",
        help="Increase output verbosity, if 3 or greater, CLI output will show in terminal in GUI mode",
        default=settings["verbose"],
        type=int,
    )

    my_parser.add_argument(
        "--ip",
        "--ip-address",
        metavar="IP_ADDR",
        type=str,
        dest="ip_address",
        default=settings["ip_address"],
        help="Specify the IP address for the web app.\n(DEFAULT: {})".format(
            settings["ip_address"]
        ),
    )

    my_parser.add_argument(
        "--port",
        "--ip-port",
        metavar="PORT",
        type=int,
        dest="ip_port",
        default=settings["ip_port"],
        help="Specify the port number for the IP address for the web app.\n(DEFAULT: {})".format(
            settings["ip_address"]
        ),
    )

    my_parser.add_argument(
        "-e",
        "--experiments",
        type=str,
        help="Regex for experiments",
        default=settings["experiments"],
    )

    my_parser.add_argument(
        "-p",
        "--progress-points",
        type=str,
        help="Regex for progress points",
        default=settings["progress_points"],
    )

    my_parser.add_argument(
        "-n",
        "--min-points",
        type=int,
        help="Minimum number of data points",
        default=settings["min_points"],
    )

    my_parser.add_argument(
        "-s",
        "--speedups",
        type=int,
        help="List of speedup values to report",
        nargs="*",
        default=[],
    )

    my_parser.add_argument(
        "-d",
        "--stddev",
        type=float,
        help="Number of standard deviations to report",
        default=settings["stddev"],
    )

    my_parser.add_argument(
        "-v",
        "--validate",
        type=str,
        nargs="*",
        help="Validate speedup: {experiment regex} {progress-point regex} {virtual-speedup} {expected-speedup} {tolerance}",
        default=[],
    )

    args = my_parser.parse_args()

    causal(args)


if __name__ == "__main__":
    main()
