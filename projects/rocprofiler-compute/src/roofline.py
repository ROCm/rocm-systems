##############################################################################bl
# MIT License
#
# Copyright (c) 2021 - 2024 Advanced Micro Devices, Inc. All Rights Reserved.
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

from abc import ABC, abstractmethod
import os
import time
from dash import dcc
from utils.utils import (
    mibench,
    gen_sysinfo,
    demarcate,
    console_error,
    console_log,
    console_debug,
)
from dash import html
import plotly.graph_objects as go
from utils.roofline_calc import calc_ai, constuct_roof
import numpy as np

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

        if (
            not isinstance(self.__run_parameters["workload_dir"], list)
            and self.__run_parameters["workload_dir"] != None
        ):
            self.roof_setup()

        self.validate_parameters()

    def validate_parameters(self):
        if self.__run_parameters["include_kernel_names"] and (
            not self.__run_parameters["is_standalone"]
        ):
            console_error("--roof-only is required for --kernel-names")

    def roof_setup(self):
        # set default workload path if not specified
        if self.__run_parameters["workload_dir"] == os.path.join(
            os.getcwd(), "workloads"
        ):
            self.__run_parameters["workload_dir"] = os.path.join(
                self.__run_parameters["workload_dir"],
                self.__args.name,
                self.__mspec.gpu_model,
            )
        # create new directory for roofline if it doesn't exist
        if not os.path.isdir(self.__run_parameters["workload_dir"]):
            os.makedirs(self.__run_parameters["workload_dir"])

    @demarcate
    def empirical_roofline(
        self,
        ret_df,
    ):
        """Generate a set of empirical roofline plots given a directory containing required profiling and benchmarking data"""
        # Create arithmetic intensity data that will populate the roofline model
        console_debug("roofline", "Path: %s" % self.__run_parameters["workload_dir"])
        self.__ai_data = calc_ai(self.__run_parameters["sort_type"], ret_df)

        msg = "AI at each mem level:"
        for i in self.__ai_data:
            msg += "\n\t%s -> %s" % (i, self.__ai_data[i])
        console_debug(msg)

        # Generate a roofline figure for each data type
        fp32_fig = self.generate_plot(dtype="FP32")
        fp16_fig = self.generate_plot(dtype="FP16")
        ml_combo_fig = self.generate_plot(
            dtype="I8",
            fig=fp16_fig,
        )
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

            fp32_fig.write_image(
                self.__run_parameters["workload_dir"]
                + "/empirRoof_gpu-{}_fp32_fp64.pdf".format(dev_id)
            )
            ml_combo_fig.write_image(
                self.__run_parameters["workload_dir"]
                + "/empirRoof_gpu-{}_int8_fp16.pdf".format(dev_id)
            )
            # only save a legend if kernel_names option is toggled
            if self.__run_parameters["include_kernel_names"]:
                self.__figure.write_image(
                    self.__run_parameters["workload_dir"] + "/kernelName_legend.pdf"
                )
            time.sleep(1)
            # Re-save to remove loading MathJax pop up
            fp32_fig.write_image(
                self.__run_parameters["workload_dir"]
                + "/empirRoof_gpu-{}_fp32_fp64.pdf".format(dev_id)
            )
            ml_combo_fig.write_image(
                self.__run_parameters["workload_dir"]
                + "/empirRoof_gpu-{}_int8_fp16.pdf".format(dev_id)
            )
            if self.__run_parameters["include_kernel_names"]:
                self.__figure.write_image(
                    self.__run_parameters["workload_dir"] + "/kernelName_legend.pdf"
                )
            console_log("roofline", "Empirical Roofline PDFs saved!")
        else:
            return html.Section(
                id="roofline",
                children=[
                    html.Div(
                        className="float-container",
                        children=[
                            html.Div(
                                className="float-child",
                                children=[
                                    html.H3(
                                        children="Empirical Roofline Analysis (FP32/FP64)"
                                    ),
                                    dcc.Graph(figure=fp32_fig),
                                ],
                            ),
                            html.Div(
                                className="float-child",
                                children=[
                                    html.H3(
                                        children="Empirical Roofline Analysis (FP16/INT8)"
                                    ),
                                    dcc.Graph(figure=ml_combo_fig),
                                ],
                            ),
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

        # Plot peak VALU ceiling
        # VALU info I8/FP16 not collected via microbench
        if dtype != "FP16" and dtype != "I8":
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
                            else "{} GFLOP/s".format(
                                to_int(self.__ceiling_data["valu"][2])
                            )
                        ),
                        "{} GFLOP/s".format(to_int(self.__ceiling_data["valu"][2])),
                    ],
                    textposition="top left",
                )
            )

        if dtype == "FP16":
            pos = "bottom left"
        else:
            pos = "top left"
        # Plot peak MFMA ceiling
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
                        else "{} GFLOP/s".format(to_int(self.__ceiling_data["mfma"][2]))
                    ),
                    "{} GFLOP/s".format(to_int(self.__ceiling_data["mfma"][2])),
                ],
                textposition=pos,
            )
        )
        #######################
        # Plot Application AI
        #######################
        if dtype != "I8":
            # Plot the arithmetic intensity points for each cache level
            # Omitting I8 AIs to clean up graph. FP16 tends to be higher.
            fig.add_trace(
                go.Scatter(
                    x=self.__ai_data["ai_l1"][0],
                    y=self.__ai_data["ai_l1"][1],
                    name="ai_l1",
                    mode="markers",
                    marker={"color": "#00CC96"},
                    marker_symbol=(
                        SYMBOLS if self.__run_parameters["include_kernel_names"] else None
                    ),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=self.__ai_data["ai_l2"][0],
                    y=self.__ai_data["ai_l2"][1],
                    name="ai_l2",
                    mode="markers",
                    marker={"color": "#EF553B"},
                    marker_symbol=(
                        SYMBOLS if self.__run_parameters["include_kernel_names"] else None
                    ),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=self.__ai_data["ai_hbm"][0],
                    y=self.__ai_data["ai_hbm"][1],
                    name="ai_hbm",
                    mode="markers",
                    marker={"color": "#636EFA"},
                    marker_symbol=(
                        SYMBOLS if self.__run_parameters["include_kernel_names"] else None
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
        fig.update_xaxes(type="log", autorange=True)
        fig.update_yaxes(type="log", autorange=True)

        return fig

    @demarcate
    def standalone_roofline(self):
        import pandas as pd
        from collections import OrderedDict

        # Change vL1D to a interpretable str, if required
        if "vL1D" in self.__run_parameters["mem_level"]:
            self.__run_parameters["mem_level"].remove("vL1D")
            self.__run_parameters["mem_level"].append("L1")

        app_path = os.path.join(self.__run_parameters["workload_dir"], "pmc_perf.csv")
        roofline_exists = os.path.isfile(app_path)
        if not roofline_exists:
            console_error("roofline", "{} does not exist".format(app_path))
        t_df = OrderedDict()
        t_df["pmc_perf"] = pd.read_csv(app_path)
        self.empirical_roofline(ret_df=t_df)

    # Main methods
    @abstractmethod
    def pre_processing(self):
        if self.__args.roof_only:
            # check for sysinfo
            console_log(
                "roofline", "Checking for sysinfo.csv in " + str(self.__args.path)
            )
            sysinfo_path = os.path.join(self.__args.path, "sysinfo.csv")
            if not os.path.isfile(sysinfo_path):
                console_log("roofline", "sysinfo.csv not found. Generating...")

                class Dummy_SoC:
                    roofline_obj = True

                gen_sysinfo(
                    workload_name=self.__args.name,
                    workload_dir=self.__workload_dir,
                    ip_blocks=self.__args.ipblocks,
                    app_cmd=self.__args.remaining,
                    skip_roof=self.__args.no_roof,
                    roof_only=self.__args.roof_only,
                    mspec=self.__mspec,
                    soc=Dummy_SoC,
                )

    @abstractmethod
    def profile(self):
        if self.__args.roof_only:
            # check for roofline benchmark
            console_log(
                "roofline", "Checking for roofline.csv in " + str(self.__args.path)
            )
            roof_path = os.path.join(self.__args.path, "roofline.csv")
            if not os.path.isfile(roof_path):
                mibench(self.__args, self.__mspec)

            # check for profiling data
            console_log(
                "roofline", "Checking for pmc_perf.csv in " + str(self.__args.path)
            )
            app_path = os.path.join(self.__args.path, "pmc_perf.csv")
            if not os.path.isfile(app_path):
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
