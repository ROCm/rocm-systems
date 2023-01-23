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

from omniperf_analyze.utils import roofline_calc

import numpy as np
from dash import html, dash_table

from dash import dcc
import plotly.graph_objects as go


def to_int(a):
    if str(type(a)) == "<class 'NoneType'>":
        return np.nan
    else:
        return int(a)


def generate_plots(roof_info, ai_data, verbose, fig=None):
    if fig is None:
        fig = go.Figure()
    line_data = roofline_calc.empirical_roof(roof_info, verbose)

    #######################
    # Plot BW Lines
    #######################
    fig.add_trace(
        go.Scatter(
            x=line_data["hbm"][0],
            y=line_data["hbm"][1],
            name="HBM-{}".format(roof_info["dtype"]),
            mode="lines",
            hovertemplate="<b>%{text}</b>",
            text=[
                "{} GB/s".format(to_int(line_data["hbm"][2])),
                "{} GFLOP/s".format(to_int(line_data["hbm"][2])),
            ],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=line_data["l2"][0],
            y=line_data["l2"][1],
            name="L2-{}".format(roof_info["dtype"]),
            mode="lines",
            hovertemplate="<b>%{text}</b>",
            text=[
                "{} GB/s".format(to_int(line_data["l2"][2])),
                "{} GFLOP/s".format(to_int(line_data["l2"][2])),
            ],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=line_data["l1"][0],
            y=line_data["l1"][1],
            name="L1-{}".format(roof_info["dtype"]),
            mode="lines",
            hovertemplate="<b>%{text}</b>",
            text=[
                "{} GB/s".format(to_int(line_data["l1"][2])),
                "{} GFLOP/s".format(to_int(line_data["l1"][2])),
            ],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=line_data["lds"][0],
            y=line_data["lds"][1],
            name="LDS-{}".format(roof_info["dtype"]),
            mode="lines",
            hovertemplate="<b>%{text}</b>",
            text=[
                "{} GB/s".format(to_int(line_data["lds"][2])),
                "{} GFLOP/s".format(to_int(line_data["lds"][2])),
            ],
        )
    )
    if roof_info["dtype"] != "FP16" and roof_info["dtype"] != "I8":
        fig.add_trace(
            go.Scatter(
                x=line_data["valu"][0],
                y=line_data["valu"][1],
                name="Peak VALU-{}".format(roof_info["dtype"]),
                mode="lines",
                hovertemplate="<b>%{text}</b>",
                text=[
                    "{} GFLOP/s".format(to_int(line_data["valu"][2])),
                    "{} GFLOP/s".format(to_int(line_data["valu"][2])),
                ],
            )
        )
    fig.add_trace(
        go.Scatter(
            x=line_data["mfma"][0],
            y=line_data["mfma"][1],
            name="Peak MFMA-{}".format(roof_info["dtype"]),
            mode="lines",
            hovertemplate="<b>%{text}</b>",
            text=[
                "{} GFLOP/s".format(to_int(line_data["mfma"][2])),
                "{} GFLOP/s".format(to_int(line_data["mfma"][2])),
            ],
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


def get_roofline(path_to_dir, ret_df, verbose):
    # Roofline settings
    fp32_details = {
        "path": path_to_dir,
        "sort": "kernels",
        "device": 0,
        "dtype": "FP32",
    }
    fp16_details = {
        "path": path_to_dir,
        "sort": "kernels",
        "device": 0,
        "dtype": "FP16",
    }
    int8_details = {"path": path_to_dir, "sort": "kernels", "device": 0, "dtype": "I8"}

    # Generate roofline plots
    print("Path: ", path_to_dir)
    ai_data = roofline_calc.plot_application("kernels", ret_df, verbose)
    if verbose >= 1:
        # print AI data for each mem level
        for i in ai_data:
            print(i, "->", ai_data[i])
        print("\n")

    fp32_fig = generate_plots(fp32_details, ai_data, verbose)
    fp16_fig = generate_plots(fp16_details, ai_data, verbose)
    ml_combo_fig = generate_plots(int8_details, ai_data, verbose, fp16_fig)

    return html.Section(
        id="roofline",
        children=[
            html.Div(
                className="float-container",
                children=[
                    html.Div(
                        className="float-child",
                        children=[
                            html.H3(children="Empirical Roofline Analysis (FP32/FP64)"),
                            dcc.Graph(figure=fp32_fig),
                        ],
                    ),
                    html.Div(
                        className="float-child",
                        children=[
                            html.H3(children="Empirical Roofline Analysis (FP16/INT8)"),
                            dcc.Graph(figure=ml_combo_fig),
                        ],
                    ),
                ],
            )
        ],
    )
