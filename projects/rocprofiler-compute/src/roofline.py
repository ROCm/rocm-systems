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

import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html

from utils.logger import console_debug, console_error, console_log, demarcate
from utils.roofline_calc import (
    MFMA_DATATYPES,
    PEAK_OPS_DATATYPES,
    SUPPORTED_DATATYPES,
    calc_ai,
    constuct_roof,
)
from utils.utils import mibench

SYMBOLS = [0, 1, 2, 3, 4, 5, 13, 17, 18, 20]


class Roofline:
    def __init__(self, args, mspec, run_parameters=None):
        self.__args = args
        self.__mspec = mspec
        self.__run_parameters = (
            run_parameters
            if run_parameters
            else {
                "workload_dir": None,  # in some cases (i.e. --specs) path will not be given
                "device_id": 0,
                "sort_type": "kernels",
                "mem_level": "ALL",
                "include_kernel_names": False,
                "is_standalone": False,
            }
        )
        self.__ai_data = None
        self.__ceiling_data = None
        self.__figure = go.Figure()
        # Set roofline run parameters from args
        if hasattr(self.__args, "path") and not run_parameters:
            self.__run_parameters["workload_dir"] = self.__args.path
        if hasattr(self.__args, "roof_only") and self.__args.roof_only == True:
            self.__run_parameters["is_standalone"] = True
        if hasattr(self.__args, "kernel_names") and self.__args.kernel_names == True:
            self.__run_parameters["include_kernel_names"] = True
        if hasattr(self.__args, "mem_level") and self.__args.mem_level != "ALL":
            self.__run_parameters["mem_level"] = self.__args.mem_level
        if hasattr(self.__args, "sort") and self.__args.sort != "ALL":
            self.__run_parameters["sort_type"] = self.__args.sort
        self.__run_parameters["roofline_data_type"] = self.__args.roofline_data_type
        self.validate_parameters()

    def validate_parameters(self):
        if self.__run_parameters["include_kernel_names"] and (
            not self.__run_parameters["is_standalone"]
        ):
            console_error("--roof-only is required for --kernel-names")

    def roof_setup(self):
        # set default workload path if not specified
        if self.__run_parameters["workload_dir"] == str(
            Path(os.getcwd()).joinpath("workloads")
        ):
            self.__run_parameters["workload_dir"] = str(
                Path(self.__run_parameters["workload_dir"]).joinpath(
                    self.__args.name,
                    self.__mspec.gpu_model,
                )
            )
        # create new directory for roofline if it doesn't exist
        if not Path(self.__run_parameters["workload_dir"]).is_dir():
            os.makedirs(self.__run_parameters["workload_dir"])

    @demarcate
    def empirical_roofline(
        self,
        ret_df,
    ):
        """Generate a set of empirical roofline plots given a directory containing required profiling and benchmarking data"""
        if (
            not isinstance(self.__run_parameters["workload_dir"], list)
            and self.__run_parameters["workload_dir"] != None
        ):
            self.roof_setup()

        # Create arithmetic intensity data that will populate the roofline model
        console_debug("roofline", "Path: %s" % self.__run_parameters["workload_dir"])
        self.__ai_data = calc_ai(self.__mspec, self.__run_parameters["sort_type"], ret_df)

        msg = "AI at each mem level:"
        for i in self.__ai_data:
            msg += "\n\t%s -> %s" % (i, self.__ai_data[i])
        console_debug(msg)

        # Generate a roofline figure for the datatypes
        ops_figure = flops_figure = None
        ops_dt_list = flops_dt_list = ""
        for dt in self.__run_parameters["roofline_data_type"]:
            # Do not generate a roofline figure if the datatype is not supported on this gpu_arch
            if not str(dt) in SUPPORTED_DATATYPES[self.__mspec.gpu_arch]:
                console_error(
                    "{} is not a supported datatype for roofline profiling on {}".format(
                        str(dt), self.__mspec.gpu_model
                    ),
                    exit=False,
                )
                continue

            ops_flops = "Ops" if (str(dt[:1]) == "I") else "Flops"

            if ops_flops == "Ops":
                if ops_figure:
                    ops_combo_figure = self.generate_plot(
                        dtype=str(dt),
                        fig=ops_figure,
                    )
                    ops_figure = ops_combo_figure
                else:
                    ops_figure = self.generate_plot(dtype=str(dt))
                ops_dt_list += "_" + str(dt)
            if ops_flops == "Flops":
                if flops_figure:
                    flops_combo_figure = self.generate_plot(
                        dtype=str(dt),
                        fig=flops_figure,
                    )
                    flops_figure = flops_combo_figure
                else:
                    flops_figure = self.generate_plot(dtype=str(dt))
                flops_dt_list += "_" + str(dt)

        # Create a legend and distinct kernel markers. This can be saved, optionally
        self.__figure = go.Figure(
            go.Scatter(
                mode="markers",
                x=[0] * 10,
                y=self.__ai_data["kernelNames"],
                marker_symbol=SYMBOLS,
                marker_size=15,
            )
        )
        self.__figure.update_layout(
            title="Kernel Names and Markers",
            margin=dict(b=0, r=0),
            xaxis_range=[-1, 1],
            xaxis_side="top",
            yaxis_side="right",
            height=400,
            width=1000,
        )
        self.__figure.update_xaxes(dtick=1)
        # Output will be different depending on interaction type:
        # Save PDFs if we're in "standalone roofline" mode, otherwise return HTML to be used in GUI output
        if self.__run_parameters["is_standalone"]:
            dev_id = str(self.__run_parameters["device_id"])

            # Re-save to remove loading MathJax pop up
            for i in range(2):
                if ops_figure:
                    ops_figure.write_image(
                        self.__run_parameters["workload_dir"]
                        + "/empirRoof_gpu-{}{}.pdf".format(dev_id, ops_dt_list)
                    )
                if flops_figure:
                    flops_figure.write_image(
                        self.__run_parameters["workload_dir"]
                        + "/empirRoof_gpu-{}{}.pdf".format(dev_id, flops_dt_list)
                    )

                # only save a legend if kernel_names option is toggled
                if self.__run_parameters["include_kernel_names"]:
                    self.__figure.write_image(
                        self.__run_parameters["workload_dir"] + "/kernelName_legend.pdf"
                    )
                time.sleep(1)
            console_log("roofline", "Empirical Roofline PDFs saved!")
        else:
            if ops_figure:
                ops_graph = html.Div(
                    className="float-child",
                    children=[
                        html.H3(children="Empirical Roofline Analysis (Ops)"),
                        dcc.Graph(figure=ops_figure),
                    ],
                )
            else:
                ops_graph = None
            if flops_figure:
                flops_graph = html.Div(
                    className="float-child",
                    children=[
                        html.H3(children="Empirical Roofline Analysis (Flops)"),
                        dcc.Graph(figure=flops_figure),
                    ],
                )
            else:
                flops_graph = None
            return html.Section(
                id="roofline",
                children=[
                    html.Div(
                        className="float-container",
                        children=[
                            ops_graph,
                            flops_graph,
                        ],
                    )
                ],
            )

    @demarcate
    def generate_plot(self, dtype, fig=None) -> go.Figure():
        """Create graph object from ai_data (coordinate points) and ceiling_data (peak FLOP and BW) data."""
        if fig is None:
            fig = go.Figure()
        plot_mode = "lines+text" if self.__run_parameters["is_standalone"] else "lines"
        self.__ceiling_data = constuct_roof(
            roofline_parameters=self.__run_parameters,
            dtype=dtype,
        )
        console_debug("roofline", "Ceiling data:\n%s" % self.__ceiling_data)

        #######################
        # Plot ceilings
        #######################
        if self.__run_parameters["mem_level"] == "ALL":
            cache_hierarchy = ["HBM", "L2", "L1", "LDS"]
        else:
            cache_hierarchy = self.__run_parameters["mem_level"]

        # Plot peak BW ceiling(s)
        for cache_level in cache_hierarchy:
            fig.add_trace(
                go.Scatter(
                    x=self.__ceiling_data[cache_level.lower()][0],
                    y=self.__ceiling_data[cache_level.lower()][1],
                    name="{}-{}".format(cache_level, dtype),
                    mode=plot_mode,
                    hovertemplate="<b>%{text}</b>",
                    text=[
                        "{} GB/s".format(
                            to_int(self.__ceiling_data[cache_level.lower()][2])
                        ),
                        (
                            None
                            if self.__run_parameters["is_standalone"]
                            else "{} GB/s".format(
                                to_int(self.__ceiling_data[cache_level.lower()][2])
                            )
                        ),
                    ],
                    textposition="top right",
                )
            )

        ops_flops = "OP" if (dtype[:1] == "I") else "FLOP"

        # Plot peak VALU ceiling
        if dtype in PEAK_OPS_DATATYPES:
            fig.add_trace(
                go.Scatter(
                    x=self.__ceiling_data["valu"][0],
                    y=self.__ceiling_data["valu"][1],
                    name="Peak VALU-{}".format(dtype),
                    mode=plot_mode,
                    hovertemplate="<b>%{text}</b>",
                    text=[
                        (
                            None
                            if self.__run_parameters["is_standalone"]
                            else "{} G{}/s".format(
                                to_int(self.__ceiling_data["valu"][2]), ops_flops
                            )
                        ),
                        "{} G{}/s".format(
                            to_int(self.__ceiling_data["valu"][2]), ops_flops
                        ),
                    ],
                    textposition="top left",
                )
            )

        # Plot peak MFMA ceiling
        if dtype in MFMA_DATATYPES:
            fig.add_trace(
                go.Scatter(
                    x=self.__ceiling_data["mfma"][0],
                    y=self.__ceiling_data["mfma"][1],
                    name="Peak MFMA-{}".format(dtype),
                    mode=plot_mode,
                    hovertemplate="<b>%{text}</b>",
                    text=[
                        (
                            None
                            if self.__run_parameters["is_standalone"]
                            else "{} G{}/s".format(
                                to_int(self.__ceiling_data["mfma"][2]), ops_flops
                            )
                        ),
                        "{} G{}/s".format(
                            to_int(self.__ceiling_data["mfma"][2]), ops_flops
                        ),
                    ],
                    textposition="top left",
                )
            )
        #######################
        # Plot Application AI
        #######################
        # Plot the arithmetic intensity points for each cache level

        # Check for F6F4 PC which applies to both FP4 and FP6 MFMA; avoid duplicate plotting
        skipAI = False
        if dtype == "FP4" or dtype == "FP6":
            if (dtype == "FP6") and (
                "FP4" in self.__run_parameters["roofline_data_type"]
            ):
                skipAI = True
            console_debug(
                "roofline",
                "Datatype {} is captured through the F6F4 perfmon event".format(dtype),
            )
            dtype = "F6F4"

        if ops_flops == "FLOP":
            if not skipAI:
                fig.add_trace(
                    go.Scatter(
                        x=self.__ai_data["ai_l1"][0],
                        y=self.__ai_data["ai_l1"][1],
                        name=dtype + "_ai_l1",
                        mode="markers",
                        marker_symbol=(
                            SYMBOLS
                            if self.__run_parameters["include_kernel_names"]
                            else None
                        ),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=self.__ai_data["ai_l2"][0],
                        y=self.__ai_data["ai_l2"][1],
                        name=dtype + "_ai_l2",
                        mode="markers",
                        marker_symbol=(
                            SYMBOLS
                            if self.__run_parameters["include_kernel_names"]
                            else None
                        ),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=self.__ai_data["ai_hbm"][0],
                        y=self.__ai_data["ai_hbm"][1],
                        name=dtype + "_ai_hbm",
                        mode="markers",
                        marker_symbol=(
                            SYMBOLS
                            if self.__run_parameters["include_kernel_names"]
                            else None
                        ),
                    )
                )

                # Set layout
                fig.update_layout(
                    xaxis_title="Arithmetic Intensity (FLOPs/Byte)",
                    yaxis_title="Performance (GFLOP/sec)",
                    hovermode="x unified",
                    margin=dict(l=50, r=50, b=50, t=50, pad=4),
                )
        else:
            # Set layout
            fig.update_layout(
                xaxis_title="Bandwidth (GB/sec)",
                yaxis_title="Performance (GOP/sec)",
                hovermode="x unified",
                margin=dict(l=50, r=50, b=50, t=50, pad=4),
            )

        fig.update_xaxes(type="log", autorange=True)
        fig.update_yaxes(type="log", autorange=True)

        return fig

    @demarcate
    def standalone_roofline(self):
        if (
            not isinstance(self.__run_parameters["workload_dir"], list)
            and self.__run_parameters["workload_dir"] != None
        ):
            self.roof_setup()

        # Change vL1D to a interpretable str, if required
        if "vL1D" in self.__run_parameters["mem_level"]:
            self.__run_parameters["mem_level"].remove("vL1D")
            self.__run_parameters["mem_level"].append("L1")

        app_path = str(
            Path(self.__run_parameters["workload_dir"]).joinpath("pmc_perf.csv")
        )
        roofline_exists = Path(app_path).is_file()
        if not roofline_exists:
            console_error("roofline", "{} does not exist".format(app_path))
        t_df = OrderedDict()
        t_df["pmc_perf"] = pd.read_csv(app_path)
        self.empirical_roofline(ret_df=t_df)

    @abstractmethod
    def profile(self):
        if self.__args.roof_only:
            # check for roofline benchmark
            console_log(
                "roofline", "Checking for roofline.csv in " + str(self.__args.path)
            )
            roof_path = str(Path(self.__args.path).joinpath("roofline.csv"))
            if not Path(roof_path).is_file():
                mibench(self.__args, self.__mspec)

            # check for profiling data
            console_log(
                "roofline", "Checking for pmc_perf.csv in " + str(self.__args.path)
            )
            app_path = str(Path(self.__args.path).joinpath("pmc_perf.csv"))
            if not Path(app_path).is_file():
                console_log("roofline", "pmc_perf.csv not found. Generating...")
                if not self.__args.remaining:
                    console_error(
                        "profiling"
                        "An <app_cmd> is required to run.\rrocprof-compute profile -n test -- <app_cmd>"
                    )
                # TODO: Add an equivelent of characterize_app() to run profiling directly out of this module

        elif self.__args.no_roof:
            console_log("roofline", "Skipping roofline.")
        else:
            mibench(self.__args, self.__mspec)

    # NB: Currently the post_prossesing() method is the only one being used by rocprofiler-compute,
    # we include pre_processing() and profile() methods for those who wish to borrow the roofline module
    @abstractmethod
    def post_processing(self):
        if self.__run_parameters["is_standalone"]:
            self.standalone_roofline()


def to_int(a):
    if str(type(a)) == "<class 'NoneType'>":
        return np.nan
    else:
        return int(a)
