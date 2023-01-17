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

from dash import html, dcc
import dash_bootstrap_components as dbc

from omniperf_analyze.utils import schema

avail_normalizations = ["per_wave", "per_cycle", "per_second", "per_kernel"]

# List all the unique column values for desired column in df, 'target_col'
def list_unique(orig_list, is_numeric):
    list_set = set(orig_list)
    unique_list = list(list_set)
    if is_numeric:
        unique_list.sort()
    return unique_list


def get_header(raw_pmc, input_filters, kernel_names):
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
                                            dbc.DropdownMenuItem("Overview", header=True),
                                            dbc.DropdownMenuItem(
                                                "Roofline",
                                                href="#roofline",
                                                external_link=True,
                                            ),
                                            dbc.DropdownMenuItem(
                                                "Top Stat",
                                                href="#top_stat",
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
                                            dbc.DropdownMenuItem("Compute", header=True),
                                            dbc.DropdownMenuItem(
                                                "Command Processor (CPF/CPC)",
                                                href="#command_processor_cpccpf",
                                                external_link=True,
                                            ),
                                            dbc.DropdownMenuItem(
                                                "Shader Processor Input (SPI)",
                                                href="#shader_processor_input_spi",
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
                                                href="#local_data_sharelds",
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
                                                "Texture Addresser & Texture Data (TA/TD)",
                                                href="#texture_addresser_and_texture_data_tatd",
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
                                                children=["Kernels:"],
                                            ),
                                            dcc.Dropdown(
                                                list_unique(
                                                    list(
                                                        map(
                                                            str,
                                                            raw_pmc[
                                                                schema.pmc_perf_file_prefix
                                                            ]["KernelName"],
                                                        )
                                                    ),
                                                    False,
                                                ),  # list avail kernel names
                                                id="kernel-filt",
                                                multi=True,
                                                value=kernel_names,
                                                placeholder="ALL",
                                                style={
                                                    "width": "600px",  # TODO: Change these widths to % rather than fixed value
                                                    "height": "34px",
                                                },
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
                                                            ]["gpu-id"],
                                                        )
                                                    ),
                                                    True,
                                                ),  # list avail gcd ids
                                                id="gcd-filt",
                                                multi=True,
                                                value=input_filters[
                                                    "gpu"
                                                ],  # default to any gpu filters passed as args
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
                                                        ]["Index"],
                                                    )
                                                ),
                                                id="disp-filt",
                                                multi=True,
                                                value=input_filters[
                                                    "dispatch"
                                                ],  # default to any dispatch filters passed as args
                                                placeholder="ALL",
                                                style={"width": "150px"},
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
                                                href="",
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
