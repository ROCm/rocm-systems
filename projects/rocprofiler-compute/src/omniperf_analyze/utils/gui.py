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

from selectors import EpollSelector
import sys
import copy
import os.path
import pandas as pd
from dash.dash_table import FormatTemplate
from dash.dash_table.Format import Format, Scheme, Symbol
from dash import html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from dash import dcc
import plotly.express as px

import colorlover

from omniperf_analyze.utils import parser, file_io, schema

from omniperf_analyze.utils.gui_components.header import get_header
from omniperf_analyze.utils.gui_components.roofline import get_roofline
from omniperf_analyze.utils.gui_components.memchart import get_memchart
from omniperf_analyze.omniperf_analyze import initialize_run

pd.set_option(
    "mode.chained_assignment", None
)  # ignore SettingWithCopyWarning pandas warning

HIDDEN_SECTIONS = ["Memory Chart Analysis", "Kernels"]
HIDDEN_COLUMNS = ["Tips", "coll_level"]
IS_DARK = True  # default dark theme

# Define any elements which will have full width
full_width_elmt = {1801}

# Define different types of bar charts
barchart_elements = {
    # Group table ids by chart type
    "instr_mix": [1001, 1002],
    "multi_bar": [1604, 1704],
    "sol": [1101, 1201, 1301, 1401, 1601, 1701],
    "l2_cache_per_chan": [1802, 1803],
}


##################
# HELPER FUNCTIONS
##################
def filter_df(column, df, filt):
    filt_df = df
    if filt != []:
        filt_df = df.loc[df[schema.pmc_perf_file_prefix][column].astype(str).isin(filt)]
    return filt_df


def multi_bar_chart(table_id, display_df):
    if table_id == 1604:
        nested_bar = {"NC": {}, "UC": {}, "RW": {}, "CC": {}}
        for index, row in display_df.iterrows():
            nested_bar[row["Coherency"]][row["Xfer"]] = row["Avg"]
    if table_id == 1704:
        nested_bar = {"Read": {}, "Write": {}}
        for index, row in display_df.iterrows():
            nested_bar[row["Transaction"]][row["Type"]] = row["Avg"]

    return nested_bar


def discrete_background_color_bins(df, n_bins=5, columns="all"):
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    if columns == "all":
        if "id" in df:
            df_numeric_columns = df.select_dtypes("number").drop(["id"], axis=1)
        else:
            df_numeric_columns = df.select_dtypes("number")
    else:
        df_numeric_columns = df[columns]
    df_max = df_numeric_columns.max().max()
    df_min = df_numeric_columns.min().min()
    ranges = [((df_max - df_min) * i) + df_min for i in bounds]
    styles = []
    legend = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        backgroundColor = colorlover.scales[str(n_bins)]["seq"]["Blues"][i - 1]
        color = "white" if i > len(bounds) / 2.0 else "inherit"

        for column in df_numeric_columns:
            styles.append(
                {
                    "if": {
                        "filter_query": (
                            "{{{column}}} >= {min_bound}"
                            + (
                                " && {{{column}}} < {max_bound}"
                                if (i < len(bounds) - 1)
                                else ""
                            )
                        ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                        "column_id": column,
                    },
                    "backgroundColor": backgroundColor,
                    "color": color,
                }
            )
        legend.append(
            html.Div(
                style={"display": "inline-block", "width": "60px"},
                children=[
                    html.Div(
                        style={
                            "backgroundColor": backgroundColor,
                            "borderLeft": "1px rgb(50, 50, 50) solid",
                            "height": "10px",
                        }
                    ),
                    html.Small(round(min_bound, 2), style={"paddingLeft": "2px"}),
                ],
            )
        )

    return (styles, html.Div(legend, style={"padding": "5px 0 5px 0"}))


####################
# GRAPHICAL ELEMENTS
####################
def build_bar_chart(display_df, table_config, norm_filt):
    d_figs = []

    # Insr Mix bar chart
    if table_config["id"] in barchart_elements["instr_mix"]:
        display_df["Avg"] = [
            x.astype(int) if x != "" else int(0) for x in display_df["Avg"]
        ]
        df_unit = display_df["Unit"][0]
        d_figs.append(
            px.bar(
                display_df,
                x="Avg",
                y="Metric",
                color="Avg",
                labels={"Avg": "# of {}".format(df_unit.lower())},
                height=400,
                orientation="h",
            )
        )

    # Multi bar chart
    elif table_config["id"] in barchart_elements["multi_bar"]:
        display_df["Avg"] = [
            x.astype(int) if x != "" else int(0) for x in display_df["Avg"]
        ]
        df_unit = display_df["Unit"][0]
        nested_bar = multi_bar_chart(table_config["id"], display_df)
        # generate chart for each coherency
        for group, metric in nested_bar.items():
            d_figs.append(
                px.bar(
                    title=group,
                    x=metric.values(),
                    y=metric.keys(),
                    labels={"x": df_unit, "y": ""},
                    text=metric.values(),
                    orientation="h",
                    height=200,
                )
                .update_xaxes(showgrid=False, rangemode="nonnegative")
                .update_yaxes(showgrid=False)
                .update_layout(title_x=0.5)
            )
    # L2 Cache per channel
    elif table_config["id"] in barchart_elements["l2_cache_per_chan"]:
        nested_bar = {}
        channels = []
        for colName, colData in display_df.items():
            if colName == "Channel":
                channels = list(colData.values)
            else:
                display_df[colName] = [
                    x.astype(float) if x != "" and x != None else float(0)
                    for x in display_df[colName]
                ]
                nested_bar[colName] = list(display_df[colName])
        for group, metric in nested_bar.items():
            d_figs.append(
                px.bar(
                    title=group[0 : group.rfind("(")],
                    x=channels,
                    y=metric,
                    labels={
                        "x": "Channel",
                        "y": group[group.rfind("(") + 1 : len(group) - 1].replace(
                            "per", norm_filt
                        ),
                    },
                ).update_yaxes(rangemode="nonnegative")
            )

    # Speed-of-light bar chart
    elif table_config["id"] in barchart_elements["sol"]:
        display_df["Value"] = [
            x.astype(float) if x != "" else float(0) for x in display_df["Value"]
        ]
        if table_config["id"] == 1701:
            # special layout for L2 Cache SOL
            d_figs.append(
                px.bar(
                    display_df[display_df["Unit"] == "Pct"],
                    x="Value",
                    y="Metric",
                    color="Value",
                    range_color=[0, 100],
                    labels={"Value": "%"},
                    height=220,
                    orientation="h",
                ).update_xaxes(range=[0, 110], ticks="inside")
            )  # append first % chart
            d_figs.append(
                px.bar(
                    display_df[display_df["Unit"] == "Gb/s"],
                    x="Value",
                    y="Metric",
                    color="Value",
                    range_color=[0, 1638],
                    labels={"Value": "GB/s"},
                    height=220,
                    orientation="h",
                ).update_xaxes(range=[0, 1638])
            )  # append second GB/s chart
        else:
            d_figs.append(
                px.bar(
                    display_df,
                    x="Value",
                    y="Metric",
                    color="Value",
                    range_color=[0, 100],
                    labels={"Value": "%"},
                    height=400,
                    orientation="h",
                ).update_xaxes(range=[0, 110])
            )
    else:
        print(
            "ERROR: Table id {}. Cannot determine barchart type.".format(
                table_config["id"]
            )
        )
        sys.exit(-1)

    # update layout for each of the charts
    for fig in d_figs:
        fig.update_layout(
            margin=dict(l=50, r=50, b=50, t=50, pad=4),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#ffffff"},
        )
    return d_figs


def build_table_chart(
    display_df, table_config, original_df, display_columns, comparable_columns, decimal
):
    d_figs = []

    # build comlumns/header with formatting
    formatted_columns = []
    for col in display_df.columns:
        if (
            str(col).lower() == "pct"
            or str(col).lower() == "pop"
            or str(col).lower() == "percentage"
        ):
            formatted_columns.append(
                dict(
                    id=col,
                    name=col,
                    type="numeric",
                    format={"specifier": ".{}f".format(decimal)},
                )
            )
        elif col in comparable_columns:
            formatted_columns.append(
                dict(
                    id=col,
                    name=col,
                    type="numeric",
                    format={"specifier": ".{}f".format(decimal)},
                )
            )
        else:
            formatted_columns.append(dict(id=col, name=col, type="text"))

    # tooltip shows only on the 1st col for now if 'Tips' available
    table_tooltip = (
        [
            {
                column: {
                    "value": str(row["Tips"])
                    if column == display_columns[0] and row["Tips"]
                    else "",
                    "type": "markdown",
                }
                for column, value in row.items()
            }
            for row in original_df.to_dict("records")
        ]
        if "Tips" in original_df.columns.values.tolist()
        else None
    )

    # build data table with columns, tooltip, df and other properties
    d_t = dash_table.DataTable(
        id=str(table_config["id"]),
        sort_action="native",
        sort_mode="multi",
        columns=formatted_columns,
        tooltip_data=table_tooltip,
        # left-aligning the text of the 1st col
        style_cell_conditional=[
            {"if": {"column_id": display_columns[0]}, "textAlign": "left"}
        ],
        # style cell
        style_cell={"maxWidth": "500px"},
        # display style
        style_header={
            "backgroundColor": "rgb(30, 30, 30)",
            "color": "white",
            "fontWeight": "bold",
        }
        if IS_DARK
        else {},
        style_data={
            "backgroundColor": "rgb(50, 50, 50)",
            "color": "white",
            "whiteSpace": "normal",
            "height": "auto",
        }
        if IS_DARK
        else {},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "rgb(60, 60, 60)"},
            {
                "if": {"column_id": "PoP", "filter_query": "{PoP} > 50"},
                "backgroundColor": "#ffa90a",
                "color": "white",
            },
            {
                "if": {"column_id": "PoP", "filter_query": "{PoP} > 80"},
                "backgroundColor": "#ff120a",
                "color": "white",
            },
            {
                "if": {
                    "column_id": "Avg",
                    "filter_query": "{Unit} = Pct && {Avg} > 50",
                },
                "backgroundColor": "#ffa90a",
                "color": "white",
            },
            {
                "if": {
                    "column_id": "Avg",
                    "filter_query": "{Unit} = Pct && {Avg} > 80",
                },
                "backgroundColor": "#ff120a",
                "color": "white",
            },
        ]
        if IS_DARK
        else [],
        # the df to display
        data=display_df.to_dict("records"),
    )
    # print("DATA: \n", display_df.to_dict('records'))
    d_figs.append(d_t)
    return d_figs
    # print(d_t.columns)


def build_layout(
    app,
    runs,
    archConfigs,
    input_filters,
    decimal,
    time_unit,
    cols,
    path_to_dir,
    debug,
    verbose,
    args,
):
    """
    Build gui layout
    """
    comparable_columns = parser.build_comparable_columns(time_unit)
    base_run, base_data = next(iter(runs.items()))
    app.layout = html.Div(style={"backgroundColor": "rgb(50, 50, 50)" if IS_DARK else ""})

    filt_kernel_names = []
    kernel_top_df = base_data.dfs[1]
    for kernel_id in base_data.filter_kernel_ids:
        filt_kernel_names.append(kernel_top_df.loc[kernel_id, "KernelName"])

    app.layout.children = html.Div(
        children=[
            dbc.Spinner(
                children=[
                    get_header(base_data.raw_pmc, input_filters, filt_kernel_names),
                    html.Div(id="container", children=[]),
                ],
                fullscreen=True,
                color="primary",
                spinner_style={"width": "6rem", "height": "6rem"},
            )
        ]
    )

    @app.callback(
        Output("container", "children"),
        [Input("disp-filt", "value")],
        [Input("kernel-filt", "value")],
        [Input("gcd-filt", "value")],
        [Input("norm-filt", "value")],
        [Input("top-n-filt", "value")],
        [State("container", "children")],
    )
    def generate_from_filter(
        disp_filt, kernel_filter, gcd_filter, norm_filt, top_n_filt, div_children
    ):
        if verbose >= 1:
            print("normalization is ", norm_filt)

        base_data = initialize_run(args, norm_filt)  # Re-initalize everything
        panel_configs = copy.deepcopy(archConfigs.panel_configs)
        # Generate original raw df
        base_data[base_run].raw_pmc = file_io.create_df_pmc(path_to_dir, verbose)
        if verbose >= 1:
            print("disp-filter is ", disp_filt)
            print("kernel-filter is ", kernel_filter)
            print("gpu-filter is ", gcd_filter)
            print("top-n kernel filter is ", top_n_filter, "\n")
        base_data[base_run].filter_kernel_ids = kernel_filter
        base_data[base_run].filter_gpu_ids = gcd_filter
        base_data[base_run].filter_dispatch_ids = disp_filt
        base_data[base_run].filter_top_n = top_n_filt
        # Reload the pmc_kernel_top.csv for Top Stats panel
        file_io.create_df_kernel_top_stats(
            path_to_dir,
            base_data[base_run].filter_gpu_ids,
            base_data[base_run].filter_dispatch_ids,
            time_unit,
            base_data[base_run].filter_top_n,
        )
        is_gui = True
        # Only display basic metrics if no filters are applied
        if not (disp_filt or kernel_filter or gcd_filter):
            temp = {}
            keep = [1, 201, 101, 1901]
            for key in base_data[base_run].dfs:
                if keep.count(key) != 0:
                    temp[key] = base_data[base_run].dfs[key]

            base_data[base_run].dfs = temp
            temp = {}
            keep = [0, 100, 200, 1900]
            for key in panel_configs:
                if keep.count(key) != 0:
                    temp[key] = panel_configs[key]
            panel_configs = temp

        parser.load_table_data(
            base_data[base_run], path_to_dir, True, debug, verbose
        )  # Note: All the filtering happens in this function

        div_children = []
        div_children.append(
            get_memchart(panel_configs[1900]["data source"], base_data[base_run])
        )
        # append roofline section
        has_roofline = os.path.isfile(path_to_dir + "/roofline.csv")
        if has_roofline:
            div_children.append(
                get_roofline(
                    path_to_dir,
                    parser.apply_filters(base_data[base_run], is_gui, debug),
                    verbose,
                )
            )
        # Iterate over each section as defined in panel configs
        for panel_id, panel in panel_configs.items():
            title = str(panel_id // 100) + ". " + panel["title"]
            section_title = (
                panel["title"]
                .replace("(", "")
                .replace(")", "")
                .replace("/", "")
                .replace(" ", "_")
                .lower()
            )
            html_section = []

            if panel["title"] not in HIDDEN_SECTIONS:
                # Iterate over each table per section
                for data_source in panel["data source"]:
                    for t_type, table_config in data_source.items():
                        content = []
                        original_df = base_data[base_run].dfs[table_config["id"]]

                        # The sys info table need to add index back
                        if t_type == "raw_csv_table" and "Info" in original_df.keys():
                            original_df.reset_index(inplace=True)

                        display_columns = original_df.columns.values.tolist().copy()
                        # Remove hidden columns. Better way to do it?
                        for col in HIDDEN_COLUMNS:
                            if col in display_columns:
                                display_columns.remove(col)
                        display_df = original_df[display_columns]

                        # Determine chart type:
                        # a) Barchart
                        if table_config["id"] in [
                            x for i in barchart_elements.values() for x in i
                        ]:
                            d_figs = build_bar_chart(display_df, table_config, norm_filt)
                            # Smaller formatting if barchart yeilds several graphs
                            if (
                                len(d_figs) > 2
                                and not table_config["id"]
                                in barchart_elements["l2_cache_per_chan"]
                            ):
                                temp_obj = []
                                for fig in d_figs:
                                    temp_obj.append(
                                        html.Div(
                                            className="float-child",
                                            children=[
                                                dcc.Graph(
                                                    figure=fig, style={"margin": "2%"}
                                                )
                                            ],
                                        )
                                    )
                                content.append(
                                    html.Div(
                                        className="float-container", children=temp_obj
                                    )
                                )
                            # Normal formatting if < 2 graphs
                            else:
                                for fig in d_figs:
                                    content.append(
                                        dcc.Graph(figure=fig, style={"margin": "2%"})
                                    )
                        # B) Tablechart
                        else:
                            d_figs = build_table_chart(
                                display_df,
                                table_config,
                                original_df,
                                display_columns,
                                comparable_columns,
                                decimal,
                            )
                            for fig in d_figs:
                                content.append(html.Div([fig], style={"margin": "2%"}))

                        # subtitle for each table in a panel if existing
                        if "title" in table_config and table_config["title"]:
                            subtitle = (
                                str(table_config["id"] // 100)
                                + "."
                                + str(table_config["id"] % 100)
                                + " "
                                + table_config["title"]
                                + "\n"
                            )

                            content.insert(
                                0,
                                html.H4(
                                    children=subtitle,
                                    style={"color": "white" if IS_DARK else ""},
                                ),
                            )
                        # Update content for this section
                        if table_config["id"] in full_width_elmt:
                            # Optionally override default (50%) width
                            html_section.append(
                                html.Div(
                                    className="float-child",
                                    children=content,
                                    style={"width": "100%"},
                                )
                            )
                        else:
                            html_section.append(
                                html.Div(className="float-child", children=content)
                            )

                # Append the new section with all of it's contents
                div_children.append(
                    html.Section(
                        id=section_title,
                        children=[
                            html.H3(
                                children=title,
                                style={"color": "white" if IS_DARK else ""},
                            ),
                            html.Div(className="float-container", children=html_section),
                        ],
                    )
                )

        # Display pop-up message if no filters are applied
        if not (disp_filt or kernel_filter or gcd_filter):
            div_children.append(
                html.Section(
                    id="popup",
                    children=[
                        html.Div(
                            children="To dive deeper, use the top drop down menus to isolate particular kernel(s) or dispatch(s). You will then see the web page update with additional low-level metrics specific to the filter you've applied.",
                        ),
                    ],
                )
            )

        return div_children
