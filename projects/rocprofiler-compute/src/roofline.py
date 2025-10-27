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
import textwrap
from abc import abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import plotext as plt
import plotly.graph_objects as go
from dash import dcc, html
from plotly.subplots import make_subplots

from utils import file_io, rocpd_data, schema
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
    construct_roof,
)
from utils.specs import MachineSpecs

SYMBOLS = [0, 1, 2, 3, 4, 5, 13, 17, 18, 20]


def wrap_text(text: str, width: int = 100) -> str:
    """
    Wraps text using textwrap and joins lines with <br> for Plotly.
    """
    if not isinstance(text, str):
        text = str(text)
    wrapped_lines = textwrap.wrap(
        text, width=width, break_long_words=True, replace_whitespace=False
    )
    return "<br>".join(wrapped_lines)


def to_int(value: Union[float, None]) -> Union[int, float]:
    if value is None:
        return np.nan
    return int(value)


class Roofline:
    def __init__(
        self,
        args: argparse.Namespace,
        mspec: MachineSpecs,
        run_parameters: Optional[dict[str, Any]] = None,
    ) -> None:
        self.__args = args
        self.__mspec = mspec
        self.__run_parameters = (
            run_parameters
            if run_parameters
            else {
                "workload_dir": None,  # in some cases (i.e. --specs),
                # path will not be given
                "device_id": 0,
                "sort_type": "kernels",
                "mem_level": "ALL",
                "is_standalone": False,
                "roofline_data_type": ["FP32"],  # default to FP32
                "kernel_filter": False,
            }
        )
        self.__ai_data: Optional[dict[str, Any]] = None
        self.__ceiling_data: Optional[dict[str, Any]] = None
        self.__figure = go.Figure()

        # Set roofline run parameters from args
        if hasattr(self.__args, "path") and not run_parameters:
            self.__run_parameters["workload_dir"] = self.__args.path
        if hasattr(self.__args, "no_roof") and not self.__args.no_roof:
            self.__run_parameters["is_standalone"] = True
        if hasattr(self.__args, "mem_level") and self.__args.mem_level != "ALL":
            self.__run_parameters["mem_level"] = self.__args.mem_level
        if hasattr(self.__args, "sort") and self.__args.sort != "ALL":
            self.__run_parameters["sort_type"] = self.__args.sort
        self.__run_parameters["roofline_data_type"] = self.__args.roofline_data_type
        if (hasattr(self.__args, "kernel") and self.__args.kernel) or (
            hasattr(self.__args, "gpu_kernel") and self.__args.gpu_kernel
        ):
            self.__run_parameters["kernel_filter"] = True

    def get_args(self) -> argparse.Namespace:
        return self.__args

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

        if base_path.name == "workloads" and base_path.parent == Path.cwd():
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

    def apply_profile_kernel_filter(
        self, df: dict[str, pd.DataFrame], args: argparse.Namespace
    ) -> dict[str, pd.DataFrame]:
        """Apply kernel filter for profile mode."""
        df_pmc = df["pmc_perf"]
        df_filtered = df_pmc.copy()
        df_list = df_pmc["Kernel_Name"].tolist()

        for idx in range(len(df_list)):
            if df_list[idx].split("(")[0] not in args.kernel:
                df_filtered.drop(index=idx, inplace=True)

        # Verify that final filtered kernel df matches the kernel list requested
        unique_kernels = len(df_filtered.drop_duplicates(subset=["Kernel_Name"]))
        if unique_kernels != len(args.kernel):
            console_debug(f"Profiled kernels: {df_list}\n`--kernel`: {args.kernel}")
            console_error(
                "Roofline cannot profile - kernels requested with `--kernel` missing "
                "from profiling data!\n"
                "\tRe-profile workload in full or specify subset of available kernels "
                "using `--kernel` option.\n"
                "\tComplete profiled kernels list can be found in pmc_perf file.",
                exit=True,
            )

        df["pmc_perf"] = df_filtered
        return df

    def apply_analyze_kernel_filter(
        self,
        df: dict[str, pd.DataFrame],
        path_str: Optional[str],
        args: argparse.Namespace,
    ) -> dict[str, pd.DataFrame]:
        """Apply kernel filter for analyze mode."""
        if not path_str:
            console_error("roofline", "cannot locate pmc_kernel_top.csv")

        top_kernels_csv = Path(path_str) / "pmc_kernel_top.csv"
        if not top_kernels_csv.is_file():
            console_error("roofline", f"{top_kernels_csv} does not exist")

        k_df = pd.read_csv(top_kernels_csv)
        k_df = k_df.loc[args.gpu_kernel[0], "Kernel_Name"]

        df["pmc_perf"] = df["pmc_perf"][df["pmc_perf"]["Kernel_Name"].isin(k_df)]
        return df

    def validate_apply_kernel_filter(
        self, df: dict[str, pd.DataFrame], path_str: Optional[str] = None
    ) -> dict[str, pd.DataFrame]:
        if not self.__run_parameters["kernel_filter"]:
            return df
        args = self.get_args()

        if args.mode == "profile":
            return self.apply_profile_kernel_filter(df, args)
        elif args.mode == "analyze":
            return self.apply_analyze_kernel_filter(df, path_str, args)

        return df

    def _determine_kernel_bound_status(
        self,
        ai_value: float,
        performance: float,
        cache_level: str,
        ceiling_data: dict[str, Any],
    ) -> str:
        """
        Calculate if a kernel point is memory-bound or compute-bound
        based on its own cache level's roofline
        """
        cache_key = cache_level.replace("ai_", "")

        # Get bw for this cache level
        if cache_key not in ceiling_data or not ceiling_data[cache_key]:
            return "Unknown"

        cache_data = ceiling_data[cache_key]
        if not isinstance(cache_data, (list, tuple)) or len(cache_data) < 3:
            return "Unknown"

        bandwidth = cache_data[2]

        # Get min peak performance
        min_peak = float("inf")
        if "valu" in ceiling_data and ceiling_data["valu"]:
            min_peak = min(min_peak, ceiling_data["valu"][2])
        if "mfma" in ceiling_data and ceiling_data["mfma"]:
            min_peak = min(min_peak, ceiling_data["mfma"][2])

        if min_peak == float("inf"):
            return "Unknown"

        x_intersect = min_peak / bandwidth

        if ai_value < x_intersect:
            return "Memory Bound"
        else:
            return "Compute Bound"

    @demarcate
    def empirical_roofline(
        self, ret_df: dict[str, pd.DataFrame]
    ) -> Optional[html.Section]:
        """
        Generate a set of empirical roofline plots given a directory containing
        required profiling and benchmarking data.
        """
        if (
            not isinstance(self.__run_parameters["workload_dir"], list)
            and self.__run_parameters["workload_dir"] != None
        ):
            self.roof_setup()

        console_debug("roofline", f"Path: {self.__run_parameters.get('workload_dir')}")

        # Verify kernels have been profiled and filter the df
        ret_df = self.validate_apply_kernel_filter(
            df=ret_df, path_str=self.__run_parameters.get("workload_dir")
        )

        self.__ai_data = calc_ai_profile(
            self.__mspec, self.__run_parameters.get("sort_type"), ret_df
        )

        msg = "AI at each mem level:"
        for key, value in self.__ai_data.items():
            msg += f"\n\t{key} -> {value}"
        console_debug(msg)

        kernel_names_data = None
        if self.__ai_data and "kernelNames" in self.__ai_data:
            original_kernel_names = self.__ai_data.get("kernelNames", [])
            filtered_kernel_names = [
                name
                for name in original_kernel_names
                if name != "nan" and isinstance(name, str)
            ]
            if len(filtered_kernel_names) > 0:
                kernel_names_data = {
                    "kernel_names": filtered_kernel_names,
                    "num_kernels": len(filtered_kernel_names),
                }

        ops_figure = flops_figure = None
        ops_dt_list = flops_dt_list = kernel_list = ""

        # collect ceiling data for all datatypes to find global minimums
        all_ops_ceiling_data = {}
        all_flops_ceiling_data = {}

        for dt in self.__run_parameters.get("roofline_data_type", []):
            gpu_arch = getattr(self.__mspec, "gpu_arch", "unknown_arch")
            if (
                "SUPPORTED_DATATYPES" not in globals()
                or gpu_arch not in SUPPORTED_DATATYPES
                or str(dt) not in SUPPORTED_DATATYPES[gpu_arch]
            ):
                console_error(
                    f"{dt} is not a supported datatype for roofline profiling on "
                    f"{getattr(self.__mspec, 'gpu_model', 'N/A')} (arch: {gpu_arch})",
                    exit=False,
                )
                continue

            ops_flops = "Ops" if str(dt).startswith("I") else "Flops"

            if ops_flops == "Ops":
                if ops_figure:
                    ops_figure = self.generate_plot(
                        dtype=str(dt),
                        fig=ops_figure,
                    )
                else:
                    ops_figure = self.generate_plot(
                        dtype=str(dt),
                        kernel_names_data=kernel_names_data,
                    )
                ops_dt_list += "_" + str(dt)
                # store ceiling data for this datatype
                all_ops_ceiling_data[str(dt)] = self.__ceiling_data

            if ops_flops == "Flops":
                if flops_figure:
                    flops_figure = self.generate_plot(
                        dtype=str(dt),
                        fig=flops_figure,
                    )
                else:
                    flops_figure = self.generate_plot(
                        dtype=str(dt),
                        kernel_names_data=kernel_names_data,
                    )
                flops_dt_list += "_" + str(dt)
                # Store ceiling data for this datatype
                all_flops_ceiling_data[str(dt)] = self.__ceiling_data

        # Output will be different depending on interaction type:
        # Save PDFs if we're in "standalone roofline" mode,
        # otherwise return HTML to be used in GUI outputif flops_figure:

        if self.__run_parameters["is_standalone"]:
            dev_id = str(self.__run_parameters["device_id"])
            if self.__run_parameters.get("kernel_filter", False):
                for name in sorted(self.__args.kernel):
                    kernel_list += "_" + name

            if ops_figure:
                actual_height = int(ops_figure.layout.height)
                # minimum height of 1000 to avoid cutting off content
                pdf_height = max(actual_height, 1000)

                ops_figure.write_image(
                    f"{self.__run_parameters['workload_dir']}/empirRoof_gpu-{dev_id}{ops_dt_list}{kernel_list}.pdf",
                    width=1000,
                    height=pdf_height,
                )

            if flops_figure:
                actual_height = int(flops_figure.layout.height)
                # minimum height of 1000 to avoid cutting off content
                pdf_height = max(actual_height, 1000)

                flops_figure.write_image(
                    f"{self.__run_parameters['workload_dir']}/empirRoof_gpu-{dev_id}{flops_dt_list}{kernel_list}.pdf",
                    width=1000,
                    height=pdf_height,
                )

            console_log("roofline", "Empirical Roofline PDFs saved!")
        else:
            # Create HTML output for GUI mode.
            ops_graph = (
                html.Div(
                    className="float-child",
                    children=[
                        html.H3(children="Empirical Roofline Analysis (Ops)"),
                        dcc.Graph(figure=ops_figure),
                    ],
                )
                if ops_figure
                else None
            )

            flops_graph = (
                html.Div(
                    className="float-child",
                    children=[
                        html.H3(children="Empirical Roofline Analysis (Flops)"),
                        dcc.Graph(figure=flops_figure),
                    ],
                )
                if flops_figure
                else None
            )

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
    def generate_plot(
        self,
        dtype: str,
        fig: Optional[go.Figure] = None,
        kernel_names_data: Optional[dict] = None,
    ) -> go.Figure:
        """
        Create graph object from ai_data (coordinate points) and ceiling_data
        (peak FLOP and BW) data.
        """
        is_new_figure = fig is None
        has_kernel_names = kernel_names_data is not None and is_new_figure
        skipAI = not is_new_figure

        subplot_row = None
        total_figure_height = 600  # default height

        if is_new_figure:
            if has_kernel_names:
                raw_kernel_names = kernel_names_data.get("kernel_names", [])
                num_kernels = len(raw_kernel_names)

                wrapped_kernel_names = [wrap_text(name) for name in raw_kernel_names]
                lines_per_kernel = [
                    text.count("<br>") + 1 for text in wrapped_kernel_names
                ]
                temp_ceiling_data = construct_roof(
                    roofline_parameters=self.__run_parameters,
                    dtype=dtype,
                    ai_data=self.__ai_data,
                )

                plot_points_data = []
                cache_colors = {
                    "ai_l1": "blue",
                    "ai_l2": "green",
                    "ai_hbm": "red",
                    "ai_lds": "orange",
                }

                for cache_level in ["ai_l1", "ai_l2", "ai_hbm"]:
                    if cache_level in self.__ai_data:
                        x_vals = self.__ai_data[cache_level][0]
                        y_vals = self.__ai_data[cache_level][1]

                        for i in range(min(len(x_vals), num_kernels)):
                            if x_vals[i] > 0 and y_vals[i] > 0:
                                status = self._determine_kernel_bound_status(
                                    ai_value=x_vals[i],
                                    performance=y_vals[i],
                                    cache_level=cache_level,
                                    ceiling_data=temp_ceiling_data,
                                )

                                plot_points_data.append({
                                    "symbol": None,
                                    "color": cache_colors.get(cache_level, "gray"),
                                    "cache_level": cache_level.replace(
                                        "ai_", "", 1
                                    ).upper(),
                                    "ai": f"{x_vals[i]:.2f}",
                                    "performance": f"{y_vals[i]:.2f}",
                                    "status": status,
                                    "kernel_idx": i,
                                })

                ######################################
                # Define Figure Measurement Constants
                ######################################

                ROOFLINE_PLOT_HEIGHT = 500  # Default height of plot itself

                POINTS_ROW_HEIGHT = 25  # Pixel height of each plot point row
                num_plot_points = len(plot_points_data)  # Number of plot points
                PLOT_POINTS_HEIGHT = (
                    num_plot_points + 2
                ) * POINTS_ROW_HEIGHT  # +2 for header and spacing

                BASE_ROW_HEIGHT = 15  # Base pixel height of each kernel name row
                KERNEL_PADDING = 8  # Padding in between each kernel name row
                KERNEL_NAMES_HEIGHT = (
                    sum(lines_per_kernel) * BASE_ROW_HEIGHT
                    + (num_kernels - 1) * KERNEL_PADDING
                    + BASE_ROW_HEIGHT
                )

                total_figure_height = (
                    ROOFLINE_PLOT_HEIGHT + PLOT_POINTS_HEIGHT + KERNEL_NAMES_HEIGHT
                )

                total_content_height = (
                    ROOFLINE_PLOT_HEIGHT + PLOT_POINTS_HEIGHT + KERNEL_NAMES_HEIGHT
                )
                roofline_ratio = ROOFLINE_PLOT_HEIGHT / total_content_height
                plot_points_ratio = PLOT_POINTS_HEIGHT / total_content_height
                kernel_names_ratio = 1 - roofline_ratio - plot_points_ratio
                SUBPLOT_SPACING_PX = 80  # Constant - num of pixels between each subplot
                fig = make_subplots(
                    rows=3,
                    cols=1,
                    row_heights=[roofline_ratio, plot_points_ratio, kernel_names_ratio],
                    subplot_titles=[
                        f"Roofline Analysis ({dtype})",
                        "Plot Points & Values",
                        "Full Kernel Names",
                    ],
                    vertical_spacing=SUBPLOT_SPACING_PX / total_figure_height,
                    specs=[
                        [{"type": "scatter"}],  # Roofline plot
                        [{"type": "scatter"}],  # Plot points table
                        [{"type": "scatter"}],  # Kernel names table
                    ],
                )

                subplot_row = 1
                skipAI = False
        else:
            # Adding to existing figure
            if hasattr(fig, "_grid_ref") and fig._grid_ref is not None:
                subplot_row = 1
                if hasattr(fig, "layout") and hasattr(fig.layout, "height"):
                    total_figure_height = fig.layout.height
            skipAI = True

        self.__ceiling_data = construct_roof(
            roofline_parameters=self.__run_parameters,
            dtype=dtype,
            ai_data=self.__ai_data,
        )
        console_debug("roofline", f"Ceiling data:\n{self.__ceiling_data}")

        ops_flops = "OP" if dtype.startswith("I") else "FLOP"
        subplot_kwargs = {"row": subplot_row, "col": 1} if subplot_row else {}

        #######################
        # Plot Application AI
        #######################
        if ops_flops == "FLOP" and not skipAI:
            kernel_names = self.__ai_data.get("kernelNames", [])
            symbols_list = [SYMBOLS[i % len(SYMBOLS)] for i in range(len(kernel_names))]
            show_in_legend = not self.__run_parameters["is_standalone"]
            if self.__ai_data["ai_l1"][0]:
                fig.add_trace(
                    go.Scatter(
                        x=self.__ai_data["ai_l1"][0],
                        y=self.__ai_data["ai_l1"][1],
                        name="L1",
                        mode="markers",
                        marker=dict(
                            color="blue",
                            size=10,
                            symbol=symbols_list[: len(self.__ai_data["ai_l1"][0])],
                        ),
                        showlegend=show_in_legend,
                    ),
                    **subplot_kwargs,
                )

            if self.__ai_data["ai_l2"][0]:
                fig.add_trace(
                    go.Scatter(
                        x=self.__ai_data["ai_l2"][0],
                        y=self.__ai_data["ai_l2"][1],
                        name="L2",
                        mode="markers",
                        marker=dict(
                            color="green",
                            size=10,
                            symbol=symbols_list[: len(self.__ai_data["ai_l2"][0])],
                        ),
                        showlegend=show_in_legend,
                    ),
                    **subplot_kwargs,
                )

            if self.__ai_data["ai_hbm"][0]:
                fig.add_trace(
                    go.Scatter(
                        x=self.__ai_data["ai_hbm"][0],
                        y=self.__ai_data["ai_hbm"][1],
                        name="HBM",
                        mode="markers",
                        marker=dict(
                            color="red",
                            size=10,
                            symbol=symbols_list[: len(self.__ai_data["ai_hbm"][0])],
                        ),
                        showlegend=show_in_legend,
                    ),
                    **subplot_kwargs,
                )

        #######################
        # Bandwidth Ceilings
        #######################
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

        bandwidth_lines = []
        for level in cache_hierarchy:
            key = level.lower()
            line_data = self.__ceiling_data.get(key)
            if (
                line_data
                and isinstance(line_data, (list, tuple))
                and len(line_data) >= 3
            ):
                bandwidth_lines.append({
                    "key": key,
                    "level": level,
                    "x": line_data[0],
                    "y": line_data[1],
                    "value": line_data[2],
                    "dtype": dtype,
                })

        for bw_line in bandwidth_lines:
            value = to_int(bw_line["value"])
            level = bw_line["level"]

            trace_to_update = None
            for trace in fig.data:
                is_correct_level = trace.name and trace.name.startswith(
                    f"{level.upper()}-"
                )
                has_correct_value = False
                if trace.name and "<br>" in trace.name:
                    try:
                        # Extract value from legend name
                        value_part = trace.name.split("<br>")[1]
                        existing_val = int(value_part.split()[0])
                        if existing_val == value:
                            has_correct_value = True
                    except (ValueError, IndexError):
                        pass

                if is_correct_level and has_correct_value:
                    trace_to_update = trace
                    break

            if trace_to_update:
                try:
                    # Extract existing datatypes from name
                    name_part = trace_to_update.name.split("<br>")[0]
                    existing_dts_str = name_part.split("-", 1)[1]
                    existing_dts = [dt.strip() for dt in existing_dts_str.split(",")]
                except Exception:
                    continue

                all_dts = sorted(list(set(existing_dts + [dtype])))
                all_dts_str = ", ".join(all_dts)
                legend_name = f"{level.upper()}-{all_dts_str}<br>{value} GB/s"

                fig.update_traces(
                    patch={
                        "name": legend_name,
                        "hovertemplate": f"<b>{legend_name}</b><extra></extra>",
                    },
                    selector={"name": trace_to_update.name},
                )
            else:
                # New bandwidth line with value in legend
                legend_name = f"{level.upper()}-{dtype}<br>{value} GB/s"

                fig.add_trace(
                    go.Scatter(
                        x=bw_line["x"],
                        y=bw_line["y"],
                        name=legend_name,
                        mode="lines",
                        hovertemplate=f"<b>{legend_name}</b><extra></extra>",
                    ),
                    **subplot_kwargs,
                )

        #######################
        # Peak Performance
        #######################
        valu_data = (
            self.__ceiling_data.get("valu") if dtype in PEAK_OPS_DATATYPES else None
        )
        mfma_data = self.__ceiling_data.get("mfma") if dtype in MFMA_DATATYPES else None

        if valu_data:
            legend_name = f"Peak VALU-{dtype}<br>{to_int(valu_data[2])} G{ops_flops}/s"
            fig.add_trace(
                go.Scatter(
                    x=valu_data[0],
                    y=valu_data[1],
                    name=legend_name,
                    mode="lines",
                    hovertemplate=f"<b>{legend_name}</b><extra></extra>",
                ),
                **subplot_kwargs,
            )

        if mfma_data:
            legend_name = f"Peak MFMA-{dtype}<br>{to_int(mfma_data[2])} G{ops_flops}/s"
            fig.add_trace(
                go.Scatter(
                    x=mfma_data[0],
                    y=mfma_data[1],
                    name=legend_name,
                    mode="lines",
                    hovertemplate=f"<b>{legend_name}</b><extra></extra>",
                ),
                **subplot_kwargs,
            )

        #######################
        # Plot Points Table
        #######################
        if is_new_figure and has_kernel_names:
            symbols_list = [SYMBOLS[i % len(SYMBOLS)] for i in range(num_kernels)]

            for point in plot_points_data:
                point["symbol"] = symbols_list[point["kernel_idx"]]

            if not plot_points_data or len(plot_points_data) == 0:
                fig.add_annotation(
                    x=0.5,
                    y=1,
                    text="<b>No plot points available</b>",
                    showarrow=False,
                    xanchor="center",
                    yanchor="middle",
                    font=dict(size=12, color="black"),
                    row=2,
                    col=1,
                )

                fig.update_xaxes(visible=False, range=[0, 1], row=2, col=1)
                fig.update_yaxes(visible=False, range=[0, 2], row=2, col=1)

            else:
                header_y = len(plot_points_data) + 1
                header_positions = {
                    "Symbol": 0.020,
                    f"{ops_flops}s/Byte": 0.15,
                    f"G{ops_flops}/s": 0.35,
                    "Status": 0.55,
                    "Cache Level": 0.80,
                }

                for header_text, x_pos in header_positions.items():
                    fig.add_annotation(
                        x=x_pos,
                        y=header_y,
                        text=f"<b>{header_text}</b>",
                        showarrow=False,
                        xanchor="left",
                        yanchor="middle",
                        font=dict(size=11, color="black"),
                        row=2,
                        col=1,
                    )

                # Scatter plot for symbols
                symbol_x = []
                symbol_y = []
                symbol_markers = []
                symbol_colors = []

                for idx, point in enumerate(plot_points_data):
                    symbol_x.append(0.05)
                    symbol_y.append(len(plot_points_data) - idx)
                    symbol_markers.append(point["symbol"])
                    symbol_colors.append(point["color"])

                fig.add_trace(
                    go.Scatter(
                        x=symbol_x,
                        y=symbol_y,
                        mode="markers",
                        marker=dict(
                            symbol=symbol_markers,
                            size=11,
                            color=symbol_colors,
                            line=dict(width=0, color="black"),
                        ),
                        customdata=[
                            [point["kernel_idx"], point["cache_level"]]
                            for point in plot_points_data
                        ],
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=2,
                    col=1,
                )
                # ai, perf, status, cache_level
                data_positions = [0.15, 0.35, 0.55, 0.80]

                for idx, point in enumerate(plot_points_data):
                    y_pos = len(plot_points_data) - idx

                    # Background shading for every other row
                    if idx % 2 == 0:
                        fig.add_shape(
                            type="rect",
                            x0=0,
                            x1=1,
                            y0=y_pos - 1 / 2,
                            y1=y_pos + 1 / 2,
                            fillcolor="rgba(220, 220, 220, 0.3)",
                            line_width=0,
                            layer="below",
                            row=2,
                            col=1,
                        )

                    # Border lines for this row
                    fig.add_shape(
                        type="line",
                        x0=0,
                        x1=1,
                        y0=y_pos - 0.5,
                        y1=y_pos - 0.5,
                        line=dict(color="rgba(150, 150, 150, 0.5)", width=1),
                        row=2,
                        col=1,
                    )

                    fig.add_annotation(
                        x=data_positions[0],
                        y=y_pos,
                        text=point["ai"],
                        showarrow=False,
                        xanchor="left",
                        yanchor="middle",
                        font=dict(size=10, color="black"),
                        row=2,
                        col=1,
                    )
                    fig.add_annotation(
                        x=data_positions[1],
                        y=y_pos,
                        text=point["performance"],
                        showarrow=False,
                        xanchor="left",
                        yanchor="middle",
                        font=dict(size=10, color="black"),
                        row=2,
                        col=1,
                    )

                    status_text = point["status"]

                    if "Compute Bound" in status_text:
                        status_color = "DarkOrange"
                    elif "Memory Bound" in status_text:
                        status_color = "blue"
                    else:
                        status_color = "gray"
                    fig.add_annotation(
                        x=data_positions[2],
                        y=y_pos,
                        text=status_text,
                        showarrow=False,
                        xanchor="left",
                        yanchor="middle",
                        font=dict(size=10, color=status_color),
                        row=2,
                        col=1,
                    )

                    fig.add_annotation(
                        x=data_positions[3],
                        y=y_pos,
                        text=point["cache_level"],
                        showarrow=False,
                        xanchor="left",
                        yanchor="middle",
                        font=dict(size=10, color="black"),
                        row=2,
                        col=1,
                    )

                # Vertical column separators
                column_x_positions = [0.12, 0.32, 0.52, 0.75]
                for x_pos in column_x_positions:
                    fig.add_shape(
                        type="line",
                        x0=x_pos,
                        x1=x_pos,
                        y0=0.5,
                        y1=header_y + 0.5,
                        line=dict(color="rgba(150, 150, 150, 0.5)", width=1),
                        row=2,
                        col=1,
                    )

                # Configure Plot Points subplot axes
                fig.update_xaxes(
                    visible=False, range=[0, 1], fixedrange=True, row=2, col=1
                )
                fig.update_yaxes(
                    visible=False,
                    range=[0, (len(plot_points_data) + 1.5)],
                    fixedrange=True,
                    row=2,
                    col=1,
                )

            #######################
            # Kernel Names Table
            #######################

            y_positions = []
            row_heights = []
            current_y = 0
            KERNEL_PADDING = 0
            for i in range(num_kernels):
                # Height for this kernel is proportional to its number of lines
                kernel_height = lines_per_kernel[i]
                row_heights.append(kernel_height)
                # Position at the center of this kernel's allocated space
                current_y += kernel_height / 2
                y_positions.append(current_y)
                current_y += kernel_height / 2 + KERNEL_PADDING

            # Reverse to display top to bottom
            y_positions = [current_y - y - KERNEL_PADDING / 2 for y in y_positions]
            max_y = current_y
            min_y = 0

            kernel_symbol_x = []
            kernel_symbol_y = []
            kernel_symbol_markers = []

            for i in range(num_kernels):
                kernel_symbol_x.append(0.05)
                kernel_symbol_y.append(y_positions[i])
                kernel_symbol_markers.append(symbols_list[i])

                # Background shading for every other row
                if i % 2 == 0:
                    fig.add_shape(
                        type="rect",
                        x0=0,
                        x1=1,
                        y0=y_positions[i] - row_heights[i] / 2,
                        y1=y_positions[i] + row_heights[i] / 2,
                        fillcolor="rgba(220, 220, 220, 0.3)",
                        line_width=0,
                        layer="below",
                        row=3,
                        col=1,
                    )

                # Border lines for this kernel
                fig.add_shape(
                    type="line",
                    x0=0,
                    x1=1,
                    y0=y_positions[i] - row_heights[i] / 2,
                    y1=y_positions[i] - row_heights[i] / 2,
                    line=dict(color="rgba(150, 150, 150, 0.5)", width=1),
                    row=3,
                    col=1,
                )

                # Kernel name annotation with wrapped text (left aligned)
                fig.add_annotation(
                    x=0.15,
                    y=y_positions[i],
                    text=wrapped_kernel_names[i],
                    showarrow=False,
                    xanchor="left",
                    yanchor="middle",
                    align="left",
                    font=dict(size=10, color="black"),
                    row=3,
                    col=1,
                )

            # Vertical separator between symbol and kernel name
            fig.add_shape(
                type="line",
                x0=0.12,
                x1=0.12,
                y0=min_y,
                y1=max_y,
                line=dict(color="rgba(150, 150, 150, 0.5)", width=1),
                row=3,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=kernel_symbol_x,
                    y=kernel_symbol_y,
                    mode="markers",
                    marker=dict(
                        symbol=kernel_symbol_markers,
                        size=11,
                        color="black",
                        line=dict(width=0, color="black"),
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=3,
                col=1,
            )

            # Configure Kernel Names subplot axes
            fig.update_xaxes(visible=False, range=[0, 1], fixedrange=True, row=3, col=1)
            fig.update_yaxes(
                visible=False, range=[min_y, max_y], fixedrange=True, row=3, col=1
            )

        #######################
        # Layout Configuration
        #######################
        if is_new_figure:
            if subplot_row:
                fig.update_xaxes(
                    type="log",
                    autorange=True,
                    title_text=f"Arithmetic Intensity ({ops_flops}s/Byte)",
                    row=1,
                    col=1,
                )
                fig.update_yaxes(
                    type="log",
                    autorange=True,
                    title_text=f"Performance (G{ops_flops}/sec)",
                    row=1,
                    col=1,
                )
                fig.update_layout(
                    height=int(total_figure_height),
                    width=1000,
                    hovermode="x unified",
                    margin=dict(l=50, r=180, b=50, t=80, pad=7),
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.01,
                        font=dict(size=10),
                    ),
                )
            else:
                # Fallback to simple figure without subplots
                fig.update_layout(
                    xaxis_title=f"Arithmetic Intensity ({ops_flops}s/Byte)",
                    yaxis_title=f"Performance (G{ops_flops}/sec)",
                    xaxis_type="log",
                    yaxis_type="log",
                    xaxis_autorange=True,
                    yaxis_autorange=True,
                    height=int(total_figure_height),
                    hovermode="x unified",
                    margin=dict(l=50, r=50, b=50, t=50, pad=7),
                )

        # Update subplot title for additional datatypes
        if (
            not is_new_figure
            and subplot_row
            and hasattr(fig, "layout")
            and hasattr(fig.layout, "annotations")
        ):
            for annotation in fig.layout.annotations:
                if annotation.text and "Roofline Analysis" in annotation.text:
                    if "(" in annotation.text and ")" in annotation.text:
                        existing_text = annotation.text.split("(")[0]
                        existing_types = annotation.text.split("(")[1].split(")")[0]
                        new_types = f"{existing_types}, {dtype}"
                        annotation.text = f"{existing_text}({new_types})"
                    break

        return fig

    def cli_generate_plot(
        self,
        dtype: str,
        workload: Optional[schema.Workload] = None,
        config: Optional[dict[str, Any]] = None,
        arch_config: Optional[schema.ArchConfig] = None,
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

        if not (str(dtype) in SUPPORTED_DATATYPES[str(self.__mspec.gpu_arch)]):
            console_error(
                f"{dtype} is not a supported datatype for roofline profiling on "
                f"{getattr(self.__mspec, 'gpu_model', 'N/A')} (arch: "
                f"{self.__mspec.gpu_arch})",
                exit=False,
            )
            return

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
            base_dir = (
                workload_dir[0][0]
                if isinstance(workload_dir[0], (list, tuple))
                else workload_dir[0]
            )
        else:
            # workload_dir is a string
            base_dir = workload_dir

        base_path = Path(base_dir)
        roofline_csv = base_path / "roofline.csv"
        if not roofline_csv.is_file():
            console_log("roofline", f"{roofline_csv} does not exist")
            return

        # if workload is detected, utilize Roofline yamls.
        # If not, fallback to legacy calc_ai
        if workload and config and arch_config:
            self.__ai_data = calc_ai_analyze(
                workload=workload,
                mspec=self.__mspec,
                sort_type=str(self.__run_parameters.get("sort_type")),
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

            t_df = self.validate_apply_kernel_filter(df=t_df, path_str=str(base_path))
            self.__ai_data = calc_ai_profile(
                self.__mspec, self.__run_parameters["sort_type"], t_df
            )

        self.__ceiling_data = construct_roof(
            roofline_parameters=self.__run_parameters, dtype=dtype
        )

        console_debug(f"AI data: {self.__ai_data}")
        console_debug(f"Kernel names: {self.__ai_data.get('kernelNames', [])}")

        self.roof_setup()

        # Check proper datatype input - takes single str
        if not isinstance(dtype, str):
            console_error("Unsupported datatype input - must be str")

        # Change vL1D to a interpretable str, if required
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

        ops_flops = "OP" if dtype.startswith("I") else "FLOP"

        # Plot bandwidth lines
        cache_hierarchy = (
            ["HBM", "L2", "L1", "LDS"]
            if self.__run_parameters["mem_level"] == "ALL"
            else self.__run_parameters["mem_level"]
        )

        for cache_level in cache_hierarchy:
            cache_key = cache_level.lower()
            plt.plot(
                self.__ceiling_data[cache_key][0],
                self.__ceiling_data[cache_key][1],
                label=f"{cache_level}-{dtype}",
                marker="braille",
                color=color_scheme[cache_level],
            )
            plt.text(
                f"{round(self.__ceiling_data[cache_key][2])} GB/s",
                x=self.__ceiling_data[cache_key][0][0],
                y=self.__ceiling_data[cache_key][1][0],
                background="black",
                color="white",
                alignment="left",
            )
            console_debug(
                "roofline",
                f"{cache_level}: [{self.__ceiling_data[cache_key][0][0]},"
                f"{self.__ceiling_data[cache_key][0][1]}], "
                f"[{self.__ceiling_data[cache_key][1][0]},"
                f"{self.__ceiling_data[cache_key][1][1]}], "
                f"{self.__ceiling_data[cache_key][2]}",
            )

        # Plot VALU and MFMA Peak
        if dtype in PEAK_OPS_DATATYPES:
            plt.plot(
                self.__ceiling_data["valu"][0],
                [
                    self.__ceiling_data["valu"][1][0] - 0.1,
                    self.__ceiling_data["valu"][1][1] - 0.1,
                ],
                label=f"Peak VALU-{dtype}",
                marker="braille",
                color=color_scheme["VALU"],
            )
            plt.text(
                f"{round(self.__ceiling_data['valu'][2])} G{ops_flops}/s",
                x=self.__ceiling_data["valu"][0][1] - 800,
                y=self.__ceiling_data["valu"][1][1],
                background="black",
                color="white",
                alignment="right",
            )
            console_debug(
                "roofline",
                f"VALU: [{self.__ceiling_data['valu'][0][0]},"
                f"{self.__ceiling_data['valu'][0][1]}], "
                f"[{self.__ceiling_data['valu'][1][0]},"
                f"{self.__ceiling_data['valu'][1][1]}], "
                f"{self.__ceiling_data['valu'][2]}",
            )
        else:
            console_warning(f"No PEAK measurement available for {dtype}")

        if dtype in MFMA_DATATYPES:
            plt.plot(
                self.__ceiling_data["mfma"][0],
                [
                    self.__ceiling_data["mfma"][1][0] - 0.1,
                    self.__ceiling_data["mfma"][1][1] - 0.1,
                ],
                label=f"Peak MFMA-{dtype}",
                marker="braille",
                color=color_scheme["MFMA"],
            )
            plt.text(
                f"{round(self.__ceiling_data['mfma'][2])} G{ops_flops}/s",
                x=self.__ceiling_data["mfma"][0][1] - 800,
                y=self.__ceiling_data["mfma"][1][1],
                background="black",
                color="white",
                alignment="right",
            )
            console_debug(
                "roofline",
                f"MFMA: [{self.__ceiling_data['mfma'][0][0]},"
                f"{self.__ceiling_data['mfma'][0][1]}], "
                f"[{self.__ceiling_data['mfma'][1][0]},"
                f"{self.__ceiling_data['mfma'][1][1]}], "
                f"{self.__ceiling_data['mfma'][2]}",
            )
        else:
            console_warning(f"No MFMA measurement available for {dtype}")

        # Plot Application AI
        for cache_level in cache_hierarchy:
            key = f"ai_{cache_level.lower()}"
            if key not in self.__ai_data:
                continue

            kernel_names = self.__ai_data.get("kernelNames", [])
            for i in range(len(self.__ai_data.get("kernelNames", []))):
                # Zero intensity level means no data reported for this cache level
                if self.__ai_data[key][0][i] > 0 and self.__ai_data[key][1][i] > 0:
                    plt.plot(
                        [self.__ai_data[key][0][i]],
                        [self.__ai_data[key][1][i]],
                        label=f"AI_{cache_level}_{kernel_names[i]}",
                        color=color_scheme[cache_level],
                        marker=kernel_markers[i % len(kernel_markers)],
                    )
                val1 = (
                    self.__ai_data[key][0][i]
                    if i < len(self.__ai_data[key][0])
                    else "N/A"
                )
                val2 = (
                    self.__ai_data[key][1][i]
                    if i < len(self.__ai_data[key][1])
                    else "N/A"
                )
                console_debug("roofline", f"AI_{kernel_names[i]}: {val1}, {val2}")
        plt.xlabel(f"Arithmetic Intensity ({ops_flops}s/Byte)")
        plt.ylabel("Performance (GFLOP/sec)")
        plt.title(f"Roofline ({dtype}) - {base_path}")

        # Canvas config
        plt.theme("pro")
        plt.xscale("log")
        plt.yscale("log")

        # Build figure
        # Print plot using `plt._utility.write(self.cli_generate_plot(dtype))`
        return plt.build()

    @demarcate
    def standalone_roofline(self) -> None:
        if (
            not isinstance(self.__run_parameters["workload_dir"], list)
            and self.__run_parameters["workload_dir"] != None
        ):
            self.roof_setup()

        # Change vL1D to a interpretable str, if required
        if "vL1D" in self.__run_parameters["mem_level"]:
            self.__run_parameters["mem_level"].remove("vL1D")
            self.__run_parameters["mem_level"].append("L1")

        app_path = Path(str(self.__run_parameters["workload_dir"])) / "pmc_perf.csv"
        if not app_path.is_file():
            console_error("roofline", f"{app_path} does not exist")

        t_df = OrderedDict()
        t_df["pmc_perf"] = pd.read_csv(app_path)

        profiling_config = file_io.load_profiling_config(self.__args.path)
        if profiling_config.get("format_rocprof_output") == "rocpd":
            t_df["pmc_perf"] = rocpd_data.process_rocpd_csv(t_df["pmc_perf"])

        self.empirical_roofline(ret_df=t_df)

    # NB: Currently the post_prossesing() method is the only one being used by
    # rocprofiler-compute, we include pre_processing() and profile() methods for
    # those who wish to borrow the roofline module
    @abstractmethod
    def post_processing(self) -> None:
        if self.__run_parameters["is_standalone"]:
            self.standalone_roofline()

    def get_dtype(self) -> list[str]:
        return self.__run_parameters["roofline_data_type"]
