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

from omniperf_analyze.utils import roofline_calc

import time
import numpy as np
from dash import html, dash_table

from dash import dcc
import plotly.graph_objects as go


def to_int(a):
    if str(type(a)) == "<class 'NoneType'>":
        return np.nan
    else:
        return int(a)


def generate_plots(roof_info, ai_data, mem_level, is_standalone, verbose, fig=None):
    if fig is None:
        fig = go.Figure()
    plotMode = "lines+text" if is_standalone else "lines"
    line_data = roofline_calc.empirical_roof(roof_info, mem_level, verbose)
    print("Line data:\n", line_data)

    #######################
    # Plot BW Lines
    #######################
    if mem_level == "ALL":
        cacheHierarchy = ["HBM", "L2", "L1", "LDS"]
    else:
        cacheHierarchy = mem_level

    for cacheLevel in cacheHierarchy:
        fig.add_trace(
            go.Scatter(
                x=line_data[cacheLevel.lower()][0],
                y=line_data[cacheLevel.lower()][1],
                name="{}-{}".format(cacheLevel, roof_info["dtype"]),
                mode=plotMode,
                hovertemplate="<b>%{text}</b>",
                text=[
                    "{} GB/s".format(to_int(line_data[cacheLevel.lower()][2])),
                    None
                    if is_standalone
                    else "{} GB/s".format(to_int(line_data[cacheLevel.lower()][2])),
                ],
                textposition="top right",
            )
        )

    if roof_info["dtype"] != "FP16" and roof_info["dtype"] != "I8":
        fig.add_trace(
            go.Scatter(
                x=line_data["valu"][0],
                y=line_data["valu"][1],
                name="Peak VALU-{}".format(roof_info["dtype"]),
                mode=plotMode,
                hovertemplate="<b>%{text}</b>",
                text=[
                    None
                    if is_standalone
                    else "{} GFLOP/s".format(to_int(line_data["valu"][2])),
                    "{} GFLOP/s".format(to_int(line_data["valu"][2])),
                ],
                textposition="top left",
            )
        )

    if roof_info["dtype"] == "FP16":
        pos = "bottom left"
    else:
        pos = "top left"
    fig.add_trace(
        go.Scatter(
            x=line_data["mfma"][0],
            y=line_data["mfma"][1],
            name="Peak MFMA-{}".format(roof_info["dtype"]),
            mode=plotMode,
            hovertemplate="<b>%{text}</b>",
            text=[
                None
                if is_standalone
                else "{} GFLOP/s".format(to_int(line_data["mfma"][2])),
                "{} GFLOP/s".format(to_int(line_data["mfma"][2])),
            ],
            textposition=pos,
        )
    )
    #######################
    # Plot Application AI
    #######################
    if roof_info["dtype"] != "I8":
        fig.add_trace(
            go.Scatter(
                x=ai_data["ai_l1"][0],
                y=ai_data["ai_l1"][1],
                name="ai_l1",
                mode="markers",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ai_data["ai_l2"][0],
                y=ai_data["ai_l2"][1],
                name="ai_l2",
                mode="markers",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ai_data["ai_hbm"][0],
                y=ai_data["ai_hbm"][1],
                name="ai_hbm",
                mode="markers",
            )
        )

    fig.update_layout(
        xaxis_title="Arithmetic Intensity (FLOPs/Byte)",
        yaxis_title="Performance (GFLOP/sec)",
        hovermode="x unified",
        margin=dict(l=50, r=50, b=50, t=50, pad=4),
    )
    fig.update_xaxes(type="log", autorange=True)
    fig.update_yaxes(type="log", autorange=True)

    return fig


def get_roofline(
    path_to_dir,
    ret_df,
    verbose,
    dev_id=None,
    sort_type="kernels",
    mem_level="ALL",
    is_standalone=False,
):
    # Roofline settings
    fp32_details = {
        "path": path_to_dir,
        "sort": sort_type,
        "device": 0,
        "dtype": "FP32",
    }
    fp16_details = {
        "path": path_to_dir,
        "sort": sort_type,
        "device": 0,
        "dtype": "FP16",
    }
    int8_details = {"path": path_to_dir, "sort": sort_type, "device": 0, "dtype": "I8"}

    # Generate roofline plots
    print("Path: ", path_to_dir)
    ai_data = roofline_calc.plot_application(sort_type, ret_df, verbose)
    if verbose >= 1:
        # print AI data for each mem level
        print("AI at each mem level")
        for i in ai_data:
            print(i, "->", ai_data[i])
        print("\n")

    fp32_fig = generate_plots(fp32_details, ai_data, mem_level, is_standalone, verbose)
    fp16_fig = generate_plots(fp16_details, ai_data, mem_level, is_standalone, verbose)
    ml_combo_fig = generate_plots(
        int8_details, ai_data, mem_level, is_standalone, verbose, fp16_fig
    )

    if is_standalone:
        dev_id = "ALL" if dev_id == -1 else str(dev_id)

        fp32_fig.write_image(path_to_dir + "/empirRoof_gpu-{}_fp32.pdf".format(dev_id))
        ml_combo_fig.write_image(
            path_to_dir + "/empirRoof_gpu-{}_fp8_fp16.pdf".format(dev_id)
        )
        time.sleep(1)
        # Re-save to remove loading MathJax pop up
        fp32_fig.write_image(path_to_dir + "/empirRoof_gpu-{}_fp32.pdf".format(dev_id))
        ml_combo_fig.write_image(
            path_to_dir + "/empirRoof_gpu-{}_fp8_fp16.pdf".format(dev_id)
        )
        print("Empirical Roofline PDFs saved!")
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
