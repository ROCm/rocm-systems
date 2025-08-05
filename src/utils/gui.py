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

import colorlover
import pandas as pd
import plotly.express as px
from dash import dash_table, html

from utils import schema
from utils.logger import console_error

pd.set_option("mode.chained_assignment",
              None)  # ignore SettingWithCopyWarning pandas warning

IS_DARK = True  # TODO: Remove hardcoded in favor of class property


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
        nested_bar = {}
        for index, row in display_df.iterrows():
            if not row["Coherency"] in nested_bar:
                nested_bar[row["Coherency"]] = {}
            nested_bar[row["Coherency"]][row["Xfer"]] = row["Avg"]
    if table_id == 1705:  # L2 - Fabric Interface Stalls
        nested_bar = {}
        for index, row in display_df.iterrows():
            if not row["Transaction"] in nested_bar:
                nested_bar[row["Transaction"]] = {}
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
            styles.append({
                "if": {
                    "filter_query":
                    ("{{{column}}} >= {min_bound}" +
                     (" && {{{column}}} < {max_bound}" if
                      (i < len(bounds) - 1) else "")).format(column=column,
                                                             min_bound=min_bound,
                                                             max_bound=max_bound),
                    "column_id":
                    column,
                },
                "backgroundColor": backgroundColor,
                "color": color,
            })
        legend.append(
            html.Div(
                style={
                    "display": "inline-block",
                    "width": "60px"
                },
                children=[
                    html.Div(
                        style={
                            "backgroundColor": backgroundColor,
                            "borderLeft": "1px rgb(50, 50, 50) solid",
                            "height": "10px",
                        }),
                    html.Small(round(min_bound, 2), style={"paddingLeft": "2px"}),
                ],
            ))

    return (styles, html.Div(legend, style={"padding": "5px 0 5px 0"}))


####################
# GRAPHICAL ELEMENTS
####################
def build_bar_chart(display_df, table_config, barchart_elements, norm_filt):
    """
    Read data into a bar chart. ID will determine which subtype of barchart.
    """
    d_figs = []

    # Insr Mix bar chart
    if table_config["id"] in barchart_elements["instr_mix"]:
        display_df["Avg"] = [
            x.astype(int) if x != "" else int(0) for x in display_df["Avg"]
        ]
        df_unit = display_df["Unit"].iloc[0]
        d_figs.append(
            px.bar(
                display_df,
                x="Avg",
                y="Metric",
                color="Avg",
                labels={"Avg": "# of {}".format(df_unit.lower())},
                height=400,
                orientation="h",
            ))

    # Multi bar chart
    elif table_config["id"] in barchart_elements["multi_bar"]:
        display_df["Avg"] = [
            x.astype(int) if x != "" else int(0) for x in display_df["Avg"]
        ]
        df_unit = display_df["Unit"].iloc[0]
        nested_bar = multi_bar_chart(table_config["id"], display_df)
        # generate chart for each coherency
        for group, metric in nested_bar.items():
            d_figs.append(
                px.bar(
                    title=group,
                    x=metric.values(),
                    y=metric.keys(),
                    labels={
                        "x": df_unit,
                        "y": ""
                    },
                    text=metric.values(),
                    orientation="h",
                    height=200,
                ).update_xaxes(showgrid=False, rangemode="nonnegative").update_yaxes(
                    showgrid=False).update_layout(title_x=0.5))
    # L2 Cache per channel
    # elif table_config["id"] in barchart_elements["l2_cache_per_chan"]:
    # nested_bar = {}
    # channels = []
    # for colName, colData in display_df.items():
    #     if colName == "Channel":
    #         channels = list(colData.values)
    #     else:
    #         display_df[colName] = [
    #             x.astype(float) if x != "" and x != None else float(0)
    #             for x in display_df[colName]
    #         ]
    #         nested_bar[colName] = list(display_df[colName])
    # for group, metric in nested_bar.items():
    #     d_figs.append(
    #         px.bar(
    #             title=group[0 : group.rfind("(")],
    #             x=channels,
    #             y=metric,
    #             labels={
    #                 "x": "Channel",
    #                 "y": group[group.rfind("(") + 1 : len(group) - 1].replace(
    #                     "per", norm_filt
    #                 ),
    #             },
    #         ).update_yaxes(rangemode="nonnegative")
    #     )

    # Speed-of-light bar chart
    elif table_config["id"] in barchart_elements["sol"]:
        display_df["Avg"] = [
            float(x) if x != "" else float(0) for x in display_df["Avg"]
        ]
        if table_config["id"] == 1701:
            # special layout for L2 Cache SOL
            d_figs.append(
                px.bar(
                    display_df[display_df["Unit"] == "Pct"],
                    x="Avg",
                    y="Metric",
                    color="Avg",
                    range_color=[0, 100],
                    labels={
                        "Avg": "%"
                    },
                    height=220,
                    orientation="h",
                ).update_xaxes(range=[0, 110], ticks="inside",
                               title="%"))  # append first % chart
            hbm_bw = float(
                display_df[display_df["Metric"] == "HBM Bandwidth"]["Avg"].iloc[0])
            d_figs.append(
                px.bar(
                    display_df[display_df["Unit"] == "Gb/s"],
                    x="Avg",
                    y="Metric",
                    color="Avg",
                    range_color=[0, hbm_bw],
                    labels={
                        "Avg": "GB/s"
                    },
                    height=220,
                    orientation="h",
                ).update_xaxes(range=[0, hbm_bw]))  # append second GB/s chart
        elif table_config["id"] == 1101:
            # Special formatting reference 'Pct of Peak' value
            display_df["Pct of Peak"] = [
                x.astype(float) if x != "" else float(0)
                for x in display_df["Pct of Peak"]
            ]
            d_figs.append(
                px.bar(
                    display_df,
                    x="Pct of Peak",
                    y="Metric",
                    color="Pct of Peak",
                    range_color=[0, 100],
                    labels={
                        "Avg": "%"
                    },
                    height=400,
                    orientation="h",
                ).update_xaxes(range=[0, 110]))
        else:
            d_figs.append(
                px.bar(
                    display_df,
                    x="Avg",
                    y="Metric",
                    color="Avg",
                    range_color=[0, 100],
                    labels={
                        "Avg": "%"
                    },
                    height=400,
                    orientation="h",
                ).update_xaxes(range=[0, 110]))
    else:
        console_error("Table id %s. Cannot determine barchart type." %
                      table_config["id"])

    # update layout for each of the charts
    for fig in d_figs:
        fig.update_layout(
            margin=dict(l=50, r=50, b=50, t=50, pad=4),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#ffffff"},
        )
    return d_figs


def build_table_chart(display_df, table_config, original_df, display_columns,
                      comparable_columns, decimal):
    """
    Read data into a DashTable
    """
    d_figs = []
    # build comlumns/header with formatting
    formatted_columns = []
    for col in display_df.columns:
        if (str(col).lower() == "pct" or str(col).lower() == "pop"
                or str(col).lower() == "percentage"):
            formatted_columns.append(
                dict(
                    id=col,
                    name=col,
                    type="numeric",
                    format={"specifier": ".{}f".format(decimal)},
                ))
        elif col in comparable_columns:
            formatted_columns.append(
                dict(
                    id=col,
                    name=col,
                    type="numeric",
                    format={"specifier": ".{}f".format(decimal)},
                ))
        else:
            formatted_columns.append(dict(id=col, name=col, type="text"))

    # tooltip shows only on the 1st col for now if 'Metric Description' available
    table_tooltip = ([{
        column: {
            "value": (str(row["Description"])
                      if column == display_columns[0] and row["Description"] else ""),
            "type":
            "markdown",
        }
        for column, value in row.items()
    } for row in original_df.to_dict("records")]
                     if "Description" in original_df.columns.values.tolist() else None)

    # build data table with columns, tooltip, df and other properties
    d_t = dash_table.DataTable(
        id=str(table_config["id"]),
        sort_action="native",
        sort_mode="multi",
        columns=formatted_columns,
        tooltip_data=table_tooltip,
        # left-aligning the text of the 1st col
        style_cell_conditional=[{
            "if": {
                "column_id": display_columns[0]
            },
            "textAlign": "left"
        }],
        # style cell
        style_cell={"maxWidth": "500px"},
        # display style
        style_header=({
            "backgroundColor": "rgb(30, 30, 30)",
            "color": "white",
            "fontWeight": "bold",
        } if IS_DARK else {}),
        style_data=({
            "backgroundColor": "rgb(50, 50, 50)",
            "color": "white",
            "whiteSpace": "normal",
            "height": "auto",
        } if IS_DARK else {}),
        style_data_conditional=([
            {
                "if": {
                    "row_index": "odd"
                },
                "backgroundColor": "rgb(60, 60, 60)"
            },
        ] if IS_DARK else []),
        # the df to display
        data=display_df.to_dict("records"),
    )
    # print("DATA: \n", display_df.to_dict('records'))
    d_figs.append(d_t)
    return d_figs
    # print(d_t.columns)
