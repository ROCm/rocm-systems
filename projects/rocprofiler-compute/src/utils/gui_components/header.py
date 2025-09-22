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

import dash_bootstrap_components as dbc
from dash import dcc, html

from utils import schema

avail_normalizations = ["per_wave", "per_cycle", "per_second", "per_kernel"]


# List all the unique column values for desired column in df, 'target_col'
def list_unique(orig_list, is_numeric):
    list_set = set(orig_list)
    unique_list = list(list_set)
    if is_numeric:
        unique_list.sort()
    return unique_list


def create_span(input):
    return {"label": html.Span(str(input), title=str(input)), "value": str(input)}


def get_header(raw_pmc, input_filters, kernel_names):
    kernel_names = list(
        map(
            str,
            raw_pmc[schema.pmc_perf_file_prefix]["Kernel_Name"],
        )
    )
    kernel_names = [x.strip() for x in kernel_names]
    return html.Header(
        id="home",
        children=[
            html.Nav(
                id="nav-wrap",
                children=[
                    html.Ul(
                        id="nav",
                        children=[
                            html.Div(
                                className="nav-left",
                                children=[
                                    dbc.DropdownMenu(
                                        [
                                            dbc.DropdownMenuItem(
                                                "Overview", header=True
                                            ),
                                            dbc.DropdownMenuItem(
                                                "Roofline",
                                                href="#roofline",
                                                external_link=True,
                                            ),
                                            dbc.DropdownMenuItem(
                                                "Top Stats",
                                                href="#top_stats",
                                                external_link=True,
                                            ),
                                            dbc.DropdownMenuItem(
                                                "System Info",
                                                href="#system_info",
                                                external_link=True,
                                            ),
                                            dbc.DropdownMenuItem(
                                                "System Speed-of-Light",
                                                href="#system_speed-of-light",
                                                external_link=True,
                                            ),
                                            dbc.DropdownMenuItem(
                                                "Compute", header=True
                                            ),
                                            dbc.DropdownMenuItem(
                                                "Command Processor (CPF/CPC)",
                                                href="#command_processor_cpccpf",
                                                external_link=True,
                                            ),
                                            dbc.DropdownMenuItem(
                                                "Workgroup Manager (SPI)",
                                                href="#workgroup_manager_spi",
                                                external_link=True,
                                            ),
                                            dbc.DropdownMenuItem(
                                                "Wavefront",
                                                href="#wavefront",
                                                external_link=True,
                                            ),
                                            dbc.DropdownMenuItem(
                                                "Compute Units - Instruction Mix",
                                                href="#compute_units_-_instruction_mix",
                                                external_link=True,
                                            ),
                                            dbc.DropdownMenuItem(
                                                "Compute Units - Compute Pipeline",
                                                href="#compute_units_-_compute_pipeline",
                                                external_link=True,
                                            ),
                                            dbc.DropdownMenuItem("Cache", header=True),
                                            dbc.DropdownMenuItem(
                                                "Local Data Share (LDS)",
                                                href="#local_data_share_lds",
                                                external_link=True,
                                            ),
                                            dbc.DropdownMenuItem(
                                                "Instruction Cache",
                                                href="#instruction_cache",
                                                external_link=True,
                                            ),
                                            dbc.DropdownMenuItem(
                                                "Scalar L1 Data Cache",
                                                href="#scalar_l1_data_cache",
                                                external_link=True,
                                            ),
                                            dbc.DropdownMenuItem(
                                                (
                                                    "Address Processing Unit and "
                                                    "Data Return Path (TA/TD)"
                                                ),
                                                href=(
                                                    "#address_processing_unit_and"
                                                    "_data_return_path_tatd"
                                                ),
                                                external_link=True,
                                            ),
                                            dbc.DropdownMenuItem(
                                                "Vector L1 Data Cache",
                                                href="#vector_l1_data_cache",
                                                external_link=True,
                                            ),
                                            dbc.DropdownMenuItem(
                                                "L2 Cache",
                                                href="#l2_cache",
                                                external_link=True,
                                            ),
                                            dbc.DropdownMenuItem(
                                                "L2 Cache (per channel)",
                                                href="#l2_cache_per_channel",
                                                external_link=True,
                                            ),
                                        ],
                                        label="Menu",
                                        menu_variant="dark",
                                    ),
                                ],
                            ),
                            html.Li(
                                className="filter",
                                children=[
                                    html.Div(
                                        children=[
                                            html.A(
                                                className="smoothscroll",
                                                children=["Normalization:"],
                                            ),
                                            dcc.Dropdown(
                                                avail_normalizations,
                                                id="norm-filt",
                                                value=input_filters["normalization"],
                                                clearable=False,
                                                style={"width": "150px"},
                                            ),
                                        ]
                                    )
                                ],
                            ),
                            html.Li(
                                className="filter",
                                children=[
                                    html.Div(
                                        children=[
                                            html.A(
                                                className="smoothscroll",
                                                children=["GCD:"],
                                            ),
                                            dcc.Dropdown(
                                                list_unique(
                                                    list(
                                                        map(
                                                            str,
                                                            raw_pmc[
                                                                schema.pmc_perf_file_prefix
                                                            ]["GPU_ID"],
                                                        )
                                                    ),
                                                    True,
                                                ),  # list avail gcd ids
                                                id="gcd-filt",
                                                multi=True,
                                                # default to any gpu filters
                                                # passed as args
                                                value=input_filters["gpu"],
                                                placeholder="ALL",
                                                clearable=False,
                                                style={"width": "60px"},
                                            ),
                                        ]
                                    )
                                ],
                            ),
                            html.Li(
                                className="filter",
                                children=[
                                    html.Div(
                                        children=[
                                            html.A(
                                                className="smoothscroll",
                                                children=["Dispatch Filter:"],
                                            ),
                                            dcc.Dropdown(
                                                list(
                                                    map(
                                                        str,
                                                        raw_pmc[
                                                            schema.pmc_perf_file_prefix
                                                        ]["Dispatch_ID"],
                                                    )
                                                ),
                                                id="disp-filt",
                                                multi=True,
                                                # default to any dispatch
                                                # filters passed as args
                                                value=input_filters["dispatch"],
                                                placeholder="ALL",
                                                style={"width": "150px"},
                                            ),
                                        ]
                                    )
                                ],
                            ),
                            html.Li(
                                className="filter",
                                children=[
                                    html.Div(
                                        children=[
                                            html.A(
                                                className="smoothscroll",
                                                children=["Top N:"],
                                            ),
                                            dcc.Dropdown(
                                                [1, 5, 10, 15, 20, 50, 100],
                                                id="top-n-filt",
                                                value=input_filters[
                                                    "top_n"
                                                ],  # default to any dispatch filters
                                                # passed as args
                                                clearable=False,
                                                style={"width": "50px"},
                                            ),
                                        ]
                                    )
                                ],
                            ),
                            html.Li(
                                className="filter",
                                children=[
                                    html.Div(
                                        children=[
                                            html.A(
                                                className="smoothscroll",
                                                children=["Kernels:"],
                                            ),
                                            dcc.Dropdown(
                                                list(
                                                    map(
                                                        create_span,
                                                        list_unique(
                                                            orig_list=kernel_names,
                                                            is_numeric=False,
                                                        ),  # list avail kernel names
                                                    )
                                                ),
                                                id="kernel-filt",
                                                multi=True,
                                                value=input_filters["kernel"],
                                                optionHeight=150,
                                                placeholder="ALL",
                                                style={
                                                    "width": "600px",
                                                    # TODO: Change these widths to
                                                    # % rather than fixed value
                                                },
                                            ),
                                        ]
                                    )
                                ],
                            ),
                            html.Div(
                                className="nav-right",
                                children=[
                                    html.Li(
                                        children=[
                                            # Report bug button
                                            html.A(
                                                href="https://github.com/ROCm/rocprofiler-compute/issues",
                                                children=[
                                                    html.Button(
                                                        className="report",
                                                        children=["Report Bug"],
                                                    )
                                                ],
                                            )
                                        ]
                                    )
                                ],
                            ),
                        ],
                    )
                ],
            ),
            html.Div(
                className="row banner",
                children=[
                    html.H3(
                        children=["Placeholder. Guided Analysis coming soon..."],
                        style={"color": "white"},
                    ),
                ],
            ),
            html.P(
                className="scrolldown",
                children=[
                    html.A(
                        className="smoothscroll",
                        href="#roofline",
                        children=[html.I(className="icon-down-circle")],
                    )
                ],
            ),
        ],
    )
