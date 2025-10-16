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

from typing import Any

import colorlover  # type: ignore
import pandas as pd
import plotly.express as px  # type: ignore
from dash import dash_table, html  # type: ignore

from utils import schema
from utils.logger import console_error

pd.set_option(
    "mode.chained_assignment", None
)  # ignore SettingWithCopyWarning pandas warning

IS_DARK = True  # TODO: Remove hardcoded in favor of class property


##################
# HELPER FUNCTIONS
##################
def filter_df(column: str, df: pd.DataFrame, filt: list[str]) -> pd.DataFrame:
    if not filt:
        return df
    return df.loc[df[schema.PMC_PERF_FILE_PREFIX][column].astype(str).isin(filt)]


def multi_bar_chart(
    table_id: int, display_df: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    nested_bar: dict[str, dict[str, Any]] = {}
    if table_id == 1604:
        for _, row in display_df.iterrows():
            coherency = row["Coherency"]
            if coherency not in nested_bar:
                nested_bar[coherency] = {}
            nested_bar[coherency][row["Xfer"]] = row["Avg"]
    elif table_id == 1705:  # L2 - Fabric Interface Stalls
        for _, row in display_df.iterrows():
            transaction = row["Transaction"]
            if transaction not in nested_bar:
                nested_bar[transaction] = {}
            nested_bar[transaction][row["Type"]] = row["Avg"]

    return nested_bar


def discrete_background_color_bins(
    df: pd.DataFrame, n_bins: int = 5, columns: str | list[str] = "all"
) -> tuple[list[dict[str, Any]], html.Div]:
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]

    if columns == "all":
        df_numeric_columns = (
            df.select_dtypes("number").drop(["id"], axis=1)
            if "id" in df.columns
            else df.select_dtypes("number")
        )
    else:
        df_numeric_columns = df[columns]

    df_max = df_numeric_columns.max().max()
    df_min = df_numeric_columns.min().min()
    ranges = [((df_max - df_min) * i) + df_min for i in bounds]

    styles: list[dict[str, Any]] = []
    legend: list[html.Div] = []

    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        background_color = colorlover.scales[str(n_bins)]["seq"]["Blues"][i - 1]
        color = "white" if i > len(bounds) / 2.0 else "inherit"

        for column in df_numeric_columns.columns:
            filter_query = f"{{{column}}} >= {min_bound}" + (
                f" && {{{column}}} < {max_bound}" if i < len(bounds) - 1 else ""
            )
            styles.append({
                "if": {
                    "filter_query": filter_query,
                    "column_id": column,
                },
                "backgroundColor": background_color,
                "color": color,
            })

        legend.append(
            html.Div(
                style={"display": "inline-block", "width": "60px"},
                children=[
                    html.Div(
                        style={
                            "backgroundColor": background_color,
                            "borderLeft": "1px rgb(50, 50, 50) solid",
                            "height": "10px",
                        }
                    ),
                    html.Small(round(min_bound, 2), style={"paddingLeft": "2px"}),
                ],
            )
        )

    return styles, html.Div(legend, style={"padding": "5px 0 5px 0"})


####################
# GRAPHICAL ELEMENTS
####################
def create_instruction_mix_bar_chart(display_df: pd.DataFrame, df_unit: str) -> px.bar:
    display_df = display_df.copy()
    display_df["Avg"] = display_df["Avg"].apply(lambda x: int(x) if x != "" else 0)

    return px.bar(
        display_df,
        x="Avg",
        y="Metric",
        color="Avg",
        labels={"Avg": f"# of {df_unit.lower()}"},
        height=400,
        orientation="h",
    )


def create_multi_bar_charts(
    display_df: pd.DataFrame, table_id: int, df_unit: str
) -> list[px.bar]:
    display_df = display_df.copy()
    display_df["Avg"] = display_df["Avg"].apply(lambda x: int(x) if x != "" else 0)

    nested_bar = multi_bar_chart(table_id, display_df)
    charts = []

    for group, metric in nested_bar.items():
        chart = px.bar(
            title=group,
            x=list(metric.values()),
            y=list(metric.keys()),
            labels={"x": df_unit, "y": ""},
            text=list(metric.values()),
            orientation="h",
            height=200,
        )
        chart.update_xaxes(showgrid=False, rangemode="nonnegative")
        chart.update_yaxes(showgrid=False)
        chart.update_layout(title_x=0.5)
        charts.append(chart)

    return charts


def create_sol_charts(display_df: pd.DataFrame, table_id: int) -> list[px.bar]:
    display_df = display_df.copy()
    display_df["Avg"] = display_df["Avg"].apply(lambda x: float(x) if x != "" else 0.0)

    charts = []

    if table_id == 1701:
        # Special layout for L2 Cache SOL
        pct_data = display_df[display_df["Unit"] == "Pct"]
        charts.append(
            px.bar(
                pct_data,
                x="Avg",
                y="Metric",
                color="Avg",
                range_color=[0, 100],
                labels={"Avg": "%"},
                height=220,
                orientation="h",
            ).update_xaxes(range=[0, 110], ticks="inside", title="%")
        )

        # HBM Bandwidth chart
        hbm_row = display_df[display_df["Metric"] == "HBM Bandwidth"]
        if not hbm_row.empty:
            hbm_bw = float(hbm_row["Avg"].iloc[0])
            gb_data = display_df[display_df["Unit"] == "Gb/s"]
            charts.append(
                px.bar(
                    gb_data,
                    x="Avg",
                    y="Metric",
                    color="Avg",
                    range_color=[0, hbm_bw],
                    labels={"Avg": "GB/s"},
                    height=220,
                    orientation="h",
                ).update_xaxes(range=[0, hbm_bw])
            )

    elif table_id == 1101:
        # Special formatting reference 'Pct of Peak' value
        display_df["Pct of Peak"] = display_df["Pct of Peak"].apply(
            lambda x: float(x) if x != "" else 0.0
        )
        charts.append(
            px.bar(
                display_df,
                x="Pct of Peak",
                y="Metric",
                color="Pct of Peak",
                range_color=[0, 100],
                labels={"Avg": "%"},
                height=400,
                orientation="h",
            ).update_xaxes(range=[0, 110])
        )
    else:
        charts.append(
            px.bar(
                display_df,
                x="Avg",
                y="Metric",
                color="Avg",
                range_color=[0, 100],
                labels={"Avg": "%"},
                height=400,
                orientation="h",
            ).update_xaxes(range=[0, 110])
        )

    return charts


def build_bar_chart(
    display_df: pd.DataFrame,
    table_config: dict[str, Any],
    barchart_elements: dict[str, Any],
) -> list:
    """
    Read data into a bar chart. ID will determine which subtype of barchart.
    """
    table_id = table_config["id"]
    charts: list[px.bar] = []

    # Get unit from first row if available
    df_unit = display_df["Unit"].iloc[0] if "Unit" in display_df.columns else ""

    # Instruction Mix bar chart
    if table_id in barchart_elements["instr_mix"]:
        charts.append(create_instruction_mix_bar_chart(display_df, df_unit))

    # Multi bar chart
    elif table_id in barchart_elements["multi_bar"]:
        charts.extend(create_multi_bar_charts(display_df, table_id, df_unit))

    # Speed-of-light bar chart
    elif table_id in barchart_elements["sol"]:
        charts.extend(create_sol_charts(display_df, table_id))

    else:
        console_error(
            f"Table id {table_id}. Cannot determine barchart type.", exit=False
        )
        return []

    # Apply consistent styling to all charts
    for fig in charts:
        fig.update_layout(
            margin=dict(l=50, r=50, b=50, t=50, pad=4),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#ffffff"},
        )

    return charts


def get_dark_mode_styles() -> tuple[
    dict[str, Any], dict[str, Any], list[dict[str, Any]]
]:
    if not IS_DARK:
        return {}, {}, []

    style_header = {
        "backgroundColor": "rgb(30, 30, 30)",
        "color": "white",
        "fontWeight": "bold",
    }

    style_data = {
        "backgroundColor": "rgb(50, 50, 50)",
        "color": "white",
        "whiteSpace": "normal",
        "height": "auto",
    }

    style_data_conditional = [
        {"if": {"row_index": "odd"}, "backgroundColor": "rgb(60, 60, 60)"}
    ]

    return style_header, style_data, style_data_conditional


def build_table_chart(
    display_df: pd.DataFrame,
    table_config: dict[str, Any],
    original_df: pd.DataFrame,
    display_columns: list[str],
    comparable_columns: list[str],
    decimal: int,
) -> list[dash_table.DataTable]:
    """
    Read data into a DashTable
    """
    d_figs = []

    # build comlumns/header with formatting
    formatted_columns = []
    for col in display_df.columns:
        col_lower = str(col).lower()
        if col_lower in {"pct", "pop", "percentage"}:
            formatted_columns.append({
                "id": col,
                "name": col,
                "type": "numeric",
                "format": {"specifier": f".{decimal}f"},
            })
        elif col in comparable_columns:
            formatted_columns.append({
                "id": col,
                "name": col,
                "type": "numeric",
                "format": {"specifier": f".{decimal}f"},
            })
        else:
            formatted_columns.append({"id": col, "name": col, "type": "text"})

    # tooltip shows only on the 1st col for now if 'Metric Description' available
    table_tooltip = (
        [
            {
                column: {
                    "value": (
                        str(row["Description"])
                        if column == display_columns[0] and row["Description"]
                        else ""
                    ),
                    "type": "markdown",
                }
                for column in row.keys()
            }
            for row in original_df.to_dict("records")
        ]
        if "Description" in original_df.columns.values.tolist()
        else None
    )

    # Get styling based on dark mode
    style_header, style_data, style_data_conditional = get_dark_mode_styles()

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
        style_header=style_header,
        style_data=style_data,
        style_data_conditional=style_data_conditional,
        # the df to display
        data=display_df.to_dict("records"),
    )
    d_figs.append(d_t)
    return d_figs
