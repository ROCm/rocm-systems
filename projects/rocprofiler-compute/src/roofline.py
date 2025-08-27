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

import os
import textwrap
import time
from abc import abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import plotext as plt
import plotly.graph_objects as go
from dash import dcc, html

from utils import file_io, rocpd_data
from utils.logger import (
    console_debug,
    console_error,
    console_log,
    console_warning,
    demarcate,
)
from utils.roofline_calc import (
    MFMA_DATATYPES,
    PEAK_OPS_DATATYPES,
    SUPPORTED_DATATYPES,
    calc_ai_analyze,
    calc_ai_profile,
    constuct_roof,
)
from utils.utils import mibench

SYMBOLS = [0, 1, 2, 3, 4, 5, 13, 17, 18, 20]


def wrap_text(text, width=92) -> str:
    """
    Wraps text using textwrap and joins lines with <br> for Plotly.
    """
    if not isinstance(text, str):
        text = str(text)
    wrapped_lines = textwrap.wrap(
        text, width=width, break_long_words=True, replace_whitespace=False
    )
    return "<br>".join(wrapped_lines)


def to_int(value) -> Union[float, int]:
    return int(value) if value is not None else np.nan


class Roofline:
    def __init__(self, args, mspec, run_parameters=None):
        self.__args = args
        self.__mspec = mspec
        self.__run_parameters = run_parameters or {
            "workload_dir": None,  # in some cases (i.e. --specs), path is not given
            "device_id": 0,
            "sort_type": "kernels",
            "mem_level": "ALL",
            "include_kernel_names": False,
            "is_standalone": False,
            "roofline_data_type": ["FP32"],  # default to FP32
        }
        self.__ai_data = None
        self.__ceiling_data = None
        self.__figure = go.Figure()

        # Set roofline run parameters from args
        if hasattr(self.__args, "path") and not run_parameters:
            self.__run_parameters["workload_dir"] = self.__args.path
        if hasattr(self.__args, "no_roof") and not self.__args.no_roof:
            self.__run_parameters["is_standalone"] = True
        if hasattr(self.__args, "kernel_names") and self.__args.kernel_names:
            self.__run_parameters["include_kernel_names"] = True
        if hasattr(self.__args, "mem_level") and self.__args.mem_level != "ALL":
            self.__run_parameters["mem_level"] = self.__args.mem_level
        if hasattr(self.__args, "sort") and self.__args.sort != "ALL":
            self.__run_parameters["sort_type"] = self.__args.sort
        self.__run_parameters["roofline_data_type"] = self.__args.roofline_data_type
        self.validate_parameters()

    def validate_parameters(self) -> None:
        if self.__run_parameters["include_kernel_names"] and (
            not self.__run_parameters["is_standalone"]
        ):
            console_error("--kernel-names cannot be used with --no-roof option")

    def roof_setup(self) -> None:
        # Setup the workload directory for roofline profiling.
        workload_dir_val = self.__run_parameters.get("workload_dir")

        if not workload_dir_val:
            console_error(
                "Workload directory is not set. Cannot perform setup.", exit=False
            )
            return

        if isinstance(workload_dir_val, list):
            if not workload_dir_val or not workload_dir_val[0]:
                console_error(
                    "Workload directory list is empty or invalid. "
                    "Cannot perform setup.",
                    exit=False,
                )
                return
            # Handle nested list structure [0][0] or simple list [0]
            base_dir = (
                workload_dir_val[0][0]
                if isinstance(workload_dir_val[0], (list, tuple))
                else workload_dir_val[0]
            )
        else:
            # workload_dir_val is a string
            base_dir = workload_dir_val

        base_path = Path(base_dir)

        if base_path.name == "workloads" and base_path.parent == Path(os.getcwd()):
            app_name = getattr(self.__args, "name", "default_app_name")
            gpu_model_name = getattr(self.__mspec, "gpu_model", "default_gpu_model")

            # Create the new path
            new_path = base_path / app_name / gpu_model_name

            # Update workload_dir with the new path, maintaining original data structure
            if isinstance(workload_dir_val, list):
                # Update the nested list structure
                if isinstance(workload_dir_val[0], (list, tuple)):
                    self.__run_parameters["workload_dir"][0][0] = str(new_path)
                else:
                    self.__run_parameters["workload_dir"][0] = str(new_path)
            else:
                # Update string value
                self.__run_parameters["workload_dir"] = str(new_path)

            final_dir = str(new_path)
        else:
            final_dir = base_dir

        # Create the directory
        Path(final_dir).mkdir(parents=True, exist_ok=True)

    @demarcate
    def empirical_roofline(
        self,
        ret_df,
    ) -> Optional[html.Section]:
        """
        Generate a set of empirical roofline plots given a directory containing
        required profiling and benchmarking data.
        """
        if (
            not isinstance(self.__run_parameters["workload_dir"], list)
            and self.__run_parameters["workload_dir"] is not None
        ):
            self.roof_setup()

        console_debug("roofline", f"Path: {self.__run_parameters.get('workload_dir')}")

        self.__ai_data = calc_ai_profile(
            self.__mspec, self.__run_parameters.get("sort_type"), ret_df
        )

        msg = "AI at each mem level:"
        for key, value in self.__ai_data.items():
            msg += f"\n\t{key} -> {value}"
        console_debug(msg)

        ops_figure = flops_figure = None
        ops_dt_list = flops_dt_list = ""

        # Generate plots for each data type
        for dt in self.__run_parameters.get("roofline_data_type", []):
            gpu_arch = getattr(self.__mspec, "gpu_arch", "unknown_arch")

            if (
                gpu_arch not in SUPPORTED_DATATYPES
                or str(dt) not in SUPPORTED_DATATYPES[gpu_arch]
            ):
                console_error(
                    f"{dt} is not a supported datatype for roofline profiling on "
                    f"{getattr(self.__mspec, 'gpu_model', 'N/A')} (arch: {gpu_arch})",
                    exit=False,
                )
                continue

            ops_flops = "Ops" if (str(dt[:1]) == "I") else "Flops"

            if ops_flops == "Ops":
                ops_figure = self.generate_plot(dtype=str(dt), fig=ops_figure)
                ops_dt_list += f"_{dt}"
            else:
                flops_figure = self.generate_plot(dtype=str(dt), fig=flops_figure)
                flops_dt_list += f"_{dt}"

        if self.__run_parameters.get("include_kernel_names", False):
            kernel_names = (
                self.__ai_data.get("kernelNames", []) if self.__ai_data else []
            )

            self.__figure.data = []
            self.__figure.layout = {}

            if not kernel_names:
                console_log(
                    "roofline",
                    "No kernel names found to generate "
                    "'Kernel Names and Markers' info.",
                )
                self.__figure.add_annotation(
                    text="No kernel names to display.",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                )
                self.__figure.update_layout(
                    title_text="Kernel Names and Markers",
                    title_x=0.5,
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    height=200,
                    width=400,
                )
            else:
                num_kernels = len(kernel_names)
                symbols_list = [SYMBOLS[i % len(SYMBOLS)] for i in range(num_kernels)]

                self.__figure = go.Figure()

                self.__figure.add_trace(
                    go.Scatter(
                        x=[0.1] * num_kernels,
                        y=list(range(num_kernels, 0, -1)),
                        mode="markers",
                        marker=dict(
                            symbol=symbols_list,
                            size=15,
                            color="blue",
                            line=dict(width=1, color="black"),
                        ),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

                # Add kernel name annotations
                for i, kernel_name in enumerate(kernel_names):
                    self.__figure.add_annotation(
                        x=0.25,
                        y=num_kernels - i,
                        text=wrap_text(kernel_name),
                        showarrow=False,
                        xanchor="left",
                        yanchor="middle",
                        align="left",
                        font=dict(size=11, color="black"),
                    )

                # Add headers
                self.__figure.add_annotation(
                    x=0.1,
                    y=num_kernels + 1,
                    text="<b>Symbol</b>",
                    showarrow=False,
                    xanchor="center",
                    yanchor="middle",
                    font=dict(size=12, color="black"),
                )
                self.__figure.add_annotation(
                    x=0.25,
                    y=num_kernels + 1,
                    text="<b>Kernel Name</b>",
                    showarrow=False,
                    xanchor="left",
                    yanchor="middle",
                    font=dict(size=12, color="black"),
                )

                # Add grid lines
                for i in range(num_kernels + 1):
                    self.__figure.add_shape(
                        type="line",
                        x0=0,
                        x1=1,
                        y0=i + 0.5,
                        y1=i + 0.5,
                        line=dict(color="lightgray", width=1),
                    )

                self.__figure.add_shape(
                    type="line",
                    x0=0.2,
                    x1=0.2,
                    y0=0.5,
                    y1=num_kernels + 1.5,
                    line=dict(color="lightgray", width=1),
                )

                self.__figure.update_layout(
                    title="Kernel Names and Corresponding Markers",
                    title_x=0.5,
                    xaxis=dict(visible=False, range=[0, 1]),
                    yaxis=dict(
                        visible=False, range=[0, num_kernels + 2], autorange=False
                    ),
                    height=max(400, num_kernels * 40 + 150),
                    width=1000,
                    margin=dict(l=50, r=50, t=70, b=30),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                )

        # Output will be different depending on interaction type:
        # Save PDFs if we're in "standalone roofline" mode,
        # otherwise return HTML to be used in GUI output
        if self.__run_parameters["is_standalone"]:
            dev_id = str(self.__run_parameters["device_id"])
            workload_dir = self.__run_parameters["workload_dir"]

            # Re-save to remove loading MathJax pop up
            for _ in range(2):
                if ops_figure:
                    ops_figure.write_image(
                        f"{workload_dir}/empirRoof_gpu-{dev_id}{ops_dt_list}.pdf"
                    )
                if flops_figure:
                    flops_figure.write_image(
                        f"{workload_dir}/empirRoof_gpu-{dev_id}{flops_dt_list}.pdf"
                    )
                if self.__run_parameters["include_kernel_names"]:
                    self.__figure.write_image(f"{workload_dir}/kernelName_legend.pdf")
                time.sleep(1)
            console_log("roofline", "Empirical Roofline PDFs saved!")
        else:
            children = []

            if ops_figure:
                children.append(
                    html.Div(
                        className="float-child",
                        children=[
                            html.H3(children="Empirical Roofline Analysis (Ops)"),
                            dcc.Graph(figure=ops_figure),
                        ],
                    )
                )
            if flops_figure:
                children.append(
                    html.Div(
                        className="float-child",
                        children=[
                            html.H3(children="Empirical Roofline Analysis (Flops)"),
                            dcc.Graph(figure=flops_figure),
                        ],
                    )
                )
            return html.Section(
                id="roofline",
                children=[html.Div(className="float-container", children=children)],
            )

    @demarcate
    def generate_plot(self, dtype, fig=None) -> go.Figure():
        """
        Create graph object from ai_data (coordinate points) and ceiling_data
        (peak FLOP and BW) data.
        """
        if fig is None:
            fig = go.Figure()
            skip_ai = False
        else:
            skip_ai = True

        plot_mode = "lines+text" if self.__run_parameters["is_standalone"] else "lines"
        self.__ceiling_data = constuct_roof(
            roofline_parameters=self.__run_parameters,
            dtype=dtype,
        )
        console_debug("roofline", f"Ceiling data:\n{self.__ceiling_data}")
        ops_flops = "OP" if (dtype[:1] == "I") else "FLOP"  # For printing purposes

        #######################
        # Plot Application AI #
        #######################
        # Plot the arithmetic intensity points for each cache level
        if ops_flops == "FLOP" and not skip_ai:
            ai_traces = ["ai_l1", "ai_l2", "ai_hbm"]
            for trace_name in ai_traces:
                if trace_name in self.__ai_data:
                    fig.add_trace(
                        go.Scatter(
                            x=self.__ai_data[trace_name][0],
                            y=self.__ai_data[trace_name][1],
                            name=trace_name,
                            mode="markers",
                            marker_symbol=(
                                SYMBOLS
                                if self.__run_parameters["include_kernel_names"]
                                else None
                            ),
                        )
                    )

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
            console_debug(
                "roofline",
                "Roofline analysis only supports AI for "
                "floating point calculations at this time",
            )

        #################
        # PLOT CEILINGS #
        #################
        mem_level_config = self.__run_parameters.get("mem_level", "ALL")
        cache_hierarchy = (
            ["HBM", "L2", "L1", "LDS"]
            if mem_level_config == "ALL"
            else (
                mem_level_config
                if isinstance(mem_level_config, list)
                else [mem_level_config]
            )
        )

        ###########################
        # Plot peak BW ceiling(s) #
        ###########################
        for cache_level in cache_hierarchy:
            cache_key = cache_level.lower()
            if (
                not self.__ceiling_data
                or cache_key not in self.__ceiling_data
                or not isinstance(self.__ceiling_data[cache_key], (list, tuple))
                or len(self.__ceiling_data[cache_key]) < 3
            ):
                console_error(
                    f"Ceiling data for {cache_level} is missing or malformed for dtype {dtype}.",
                    exit=False,
                )
                continue

            ceiling_data = self.__ceiling_data[cache_key]
            fig.add_trace(
                go.Scatter(
                    x=ceiling_data[0],
                    y=ceiling_data[1],
                    name=f"{cache_level}-{dtype}",
                    mode=plot_mode,
                    hovertemplate="<b>%{text}</b>",
                    text=[
                        f"{to_int(ceiling_data[2])} GB/s",
                        (
                            None
                            if self.__run_parameters.get("is_standalone")
                            else f"{to_int(ceiling_data[2])} GB/s"
                        ),
                    ],
                    textposition="top right",
                )
            )

        ##########################
        # Plot peak VALU ceiling #
        ##########################
        if dtype in PEAK_OPS_DATATYPES:
            valu_data = self.__ceiling_data["valu"]
            fig.add_trace(
                go.Scatter(
                    x=valu_data[0],
                    y=valu_data[1],
                    name=f"Peak VALU-{dtype}",
                    mode=plot_mode,
                    hovertemplate="<b>%{text}</b>",
                    text=[
                        (
                            None
                            if self.__run_parameters["is_standalone"]
                            else f"{to_int(valu_data[2])} G{ops_flops}/s"
                        ),
                        f"{to_int(valu_data[2])} G{ops_flops}/s",
                    ],
                    textposition="top left",
                )
            )

        ##########################
        # Plot peak MFMA ceiling #
        ##########################
        if dtype in MFMA_DATATYPES:
            mfma_data = self.__ceiling_data["mfma"]
            fig.add_trace(
                go.Scatter(
                    x=mfma_data[0],
                    y=mfma_data[1],
                    name=f"Peak MFMA-{dtype}",
                    mode=plot_mode,
                    hovertemplate="<b>%{text}</b>",
                    text=[
                        (
                            None
                            if self.__run_parameters["is_standalone"]
                            else f"{to_int(mfma_data[2])} G{ops_flops}/s"
                        ),
                        f"{to_int(mfma_data[2])} G{ops_flops}/s",
                    ],
                    textposition="top left",
                )
            )

        fig.update_xaxes(type="log", autorange=True)
        fig.update_yaxes(type="log", autorange=True)

        return fig

    def cli_generate_plot(
        self, dtype, workload=None, config=None, arch_config=None
    ) -> Optional[str]:
        """
        Plot CLI mode roofline analysis in terminal using plotext

        :param dtype: The datatype to be profiled
        :type method: str
        :return: Build the current figure using plot.build(),
        or None if datatype is not valid for the architecture
        :rtype: str or None
        """
        console_debug("roofline", "Generating roofline plot for CLI")

        if str(dtype) not in SUPPORTED_DATATYPES[self.__mspec.gpu_arch]:
            console_error(
                f"{dtype} is not a supported datatype for roofline profiling on {self.__mspec.gpu_model}",
                exit=False,
            )
            return None

        # Normalize workload_dir to get the base directory
        workload_dir = self.__run_parameters.get("workload_dir")
        if workload_dir is None:
            console_error(
                "workload_dir is not set",
                exit=False,
            )
            return

        # Extract base directory path regardless of-
        # whether workload_dir is list or string
        if isinstance(workload_dir, list):
            if not workload_dir or not workload_dir[0]:
                console_error(
                    "workload_dir list is empty or contains invalid entries",
                    exit=False,
                )
                return
            # Handle nested list structure [0][0] or simple list [0]
            base_path = Path(
                workload_dir[0][0]
                if isinstance(workload_dir[0], (list, tuple))
                else workload_dir[0]
            )
        else:
            base_path = Path(workload_dir)

        roofline_csv = base_path / "roofline.csv"

        if not roofline_csv.is_file():
            console_log("roofline", f"{roofline_csv} does not exist")
            return None

        # if workload is detected, utilize Roofline yamls.
        # If not, fallback to legacy calc_ai
        if workload is not None:
            self.__ai_data = calc_ai_analyze(
                workload=workload,
                config=config,
                arch_config=arch_config,
            )

        else:
            pmc_perf_csv = base_path / "pmc_perf.csv"
            if not pmc_perf_csv.is_file():
                console_error("roofline", f"{pmc_perf_csv} does not exist")

            t_df = OrderedDict()
            t_df["pmc_perf"] = pd.read_csv(pmc_perf_csv)
            profiling_config = file_io.load_profiling_config(self.__args.path[0][0])

            if profiling_config.get("format_rocprof_output") == "rocpd":
                t_df["pmc_perf"] = rocpd_data.process_rocpd_csv(t_df["pmc_perf"])

            self.__ai_data = calc_ai_profile(
                self.__mspec, self.__run_parameters["sort_type"], t_df
            )

        self.__ceiling_data = constuct_roof(
            roofline_parameters=self.__run_parameters, dtype=dtype
        )
        console_debug(f"AI data: {self.__ai_data}")
        console_debug(f"Kernel names: {self.__ai_data.get('kernelNames', [])}")

        self.roof_setup()

        # Check proper datatype input - takes single str
        if not isinstance(dtype, str):
            console_error("Unsupported datatype input - must be str")

        # Replace vL1D with L1
        if "vL1D" in self.__run_parameters["mem_level"]:
            self.__run_parameters["mem_level"].remove("vL1D")
            self.__run_parameters["mem_level"].append("L1")

        color_scheme = {
            "HBM": "blue+",
            "L2": "green+",
            "L1": "red+",
            "LDS": "orange+",
            "VALU": "white",
            "MFMA": "magenta+",
        }

        kernel_markers = {
            0: "star",
            1: "cross",
            2: "sd",
            3: "shamrock",
            4: "at",
            5: "atom",
        }

        plt.clf()
        plt.plotsize(plt.tw(), plt.th())

        ops_flops = "OP" if (dtype[:1] == "I") else "FLOP"  # For printing purposes

        mem_level = self.__run_parameters["mem_level"]
        cache_hierarchy = (
            ["HBM", "L2", "L1", "LDS"] if mem_level == "ALL" else mem_level
        )

        #################
        # Plot BW Lines #
        #################
        for cache_level in cache_hierarchy:
            ceiling_data = self.__ceiling_data[cache_level.lower()]

            plt.plot(
                ceiling_data[0],
                ceiling_data[1],
                label=f"{cache_level}-{dtype}",
                marker="braille",
                color=color_scheme[cache_level],
            )
            plt.text(
                f"{round(ceiling_data[2])} GB/s",
                x=ceiling_data[0][0],
                y=ceiling_data[1][0],
                background="black",
                color="white",
                alignment="left",
            )
            console_debug(
                "roofline",
                f"{cache_level}: [{ceiling_data[0][0]},{ceiling_data[0][1]}], [{ceiling_data[1][0]},{ceiling_data[1][1]}], {ceiling_data[2]}",
            )

        ###########################
        # Plot VALU and MFMA Peak #
        ###########################
        if dtype in PEAK_OPS_DATATYPES:
            valu_data = self.__ceiling_data["valu"]
            plt.plot(
                valu_data[0],
                [valu_data[1][0] - 0.1, valu_data[1][1] - 0.1],
                label=f"Peak VALU-{dtype}",
                marker="braille",
                color=color_scheme["VALU"],
            )
            plt.text(
                f"{round(valu_data[2])} G{ops_flops}/s",
                x=valu_data[0][1] - 800,
                y=valu_data[1][1],
                background="black",
                color="white",
                alignment="right",
            )
            console_debug(
                "roofline",
                f"VALU: [{valu_data[0][0]},{valu_data[0][1]}], [{valu_data[1][0]},{valu_data[1][1]}], {valu_data[2]}",
            )
        else:
            console_warning(f"No PEAK measurement available for {dtype}")

        if dtype in MFMA_DATATYPES:
            mfma_data = self.__ceiling_data["mfma"]
            plt.plot(
                mfma_data[0],
                [mfma_data[1][0] - 0.1, mfma_data[1][1] - 0.1],
                label=f"Peak MFMA-{dtype}",
                marker="braille",
                color=color_scheme["MFMA"],
            )
            plt.text(
                f"{round(mfma_data[2])} G{ops_flops}/s",
                x=mfma_data[0][1] - 800,
                y=mfma_data[1][1],
                background="black",
                color="white",
                alignment="right",
            )
            console_debug(
                "roofline",
                f"MFMA: [{mfma_data[0][0]},{mfma_data[0][1]}], [{mfma_data[1][0]},{mfma_data[1][1]}], {mfma_data[2]}",
            )
        else:
            console_warning(f"No MFMA measurement available for {dtype}")

        #######################
        # Plot Application AI #
        #######################
        for cache_level in cache_hierarchy:
            key = f"ai_{cache_level.lower()}"
            if key in self.__ai_data:
                kernel_names = self.__ai_data["kernelNames"]
                for i in range(len(kernel_names)):
                    if self.__ai_data[key][0][i] > 0 and self.__ai_data[key][1][i] > 0:
                        plt.plot(
                            [self.__ai_data[key][0][i]],
                            [self.__ai_data[key][1][i]],
                            label=f"AI_{cache_level}_{kernel_names[i]}",
                            color=color_scheme[cache_level],
                            marker=kernel_markers[i % len(kernel_markers)],
                        )
                    console_debug(
                        "roofline",
                        f"AI_{kernel_names[i]}: {self.__ai_data[key][0][i]}, {self.__ai_data[key][1][i]}",
                    )

        plt.xlabel(f"Arithmetic Intensity ({ops_flops})s/Byte)")
        plt.ylabel("Performance (GFLOP/sec)")
        plt.title(f"Roofline ({dtype}) - {base_path}")

        # Canvas config
        plt.theme("pro")
        plt.xscale("log")
        plt.yscale("log")

        # Build figure
        return plt.build()

    @demarcate
    def standalone_roofline(self) -> None:
        if (
            not isinstance(self.__run_parameters["workload_dir"], list)
            and self.__run_parameters["workload_dir"] is not None
        ):
            self.roof_setup()

        # Replace vL1D with L1
        if "vL1D" in self.__run_parameters["mem_level"]:
            self.__run_parameters["mem_level"].remove("vL1D")
            self.__run_parameters["mem_level"].append("L1")

        app_path = Path(self.__run_parameters["workload_dir"]) / "pmc_perf.csv"

        if not app_path.is_file():
            console_error("roofline", f"{app_path} does not exist")

        t_df = OrderedDict()
        t_df["pmc_perf"] = pd.read_csv(app_path)
        profiling_config = file_io.load_profiling_config(self.__args.path)

        if profiling_config.get("format_rocprof_output") == "rocpd":
            t_df["pmc_perf"] = rocpd_data.process_rocpd_csv(t_df["pmc_perf"])

        self.empirical_roofline(ret_df=t_df)

    @abstractmethod
    def profile(self) -> None:
        if self.__args.roof_only:
            # check for roofline benchmark
            console_log("roofline", f"Checking for roofline.csv in {self.__args.path}")
            roof_path = Path(self.__args.path) / "roofline.csv"
            if not roof_path.is_file():
                mibench(self.__args, self.__mspec)

            # check for profiling data
            console_log("roofline", f"Checking for pmc_perf.csv in {self.__args.path}")
            app_path = Path(self.__args.path) / "pmc_perf.csv"
            if not app_path.is_file():
                console_log("roofline", "pmc_perf.csv not found. Generating...")
                if not self.__args.remaining:
                    console_error(
                        "profiling"
                        "An <app_cmd> is required to run.\r"
                        "rocprof-compute profile -n test -- <app_cmd>"
                    )
                # TODO: Add an equivelent of characterize_app() to run profiling
                # directly out of this module

        elif self.__args.no_roof:
            console_log("roofline", "Skipping roofline.")
        else:
            mibench(self.__args, self.__mspec)

    # NB: Currently the post_prossesing() method is the only one being used by
    # rocprofiler-compute, we include pre_processing() and profile() methods for
    # those who wish to borrow the roofline module
    @abstractmethod
    def post_processing(self) -> None:
        if self.__run_parameters["is_standalone"]:
            self.standalone_roofline()

    def get_dtype(self) -> list[str]:
        return self.__run_parameters["roofline_data_type"]
