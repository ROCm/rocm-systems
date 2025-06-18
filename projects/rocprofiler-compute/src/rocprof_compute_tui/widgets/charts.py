##############################################################################bl
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

from __future__ import annotations

import math
import sys
import traceback
from io import StringIO

import pandas as pd
import plotext as plt
import plotly.express as px
from textual.widgets import Static
from textual_plotext import PlotextPlot

from utils.mem_chart import plot_mem_chart


def simple_bar(df, title=None):
    """
    Plot data with simple bar chart
    """

    # TODO: handle None properly

    if "Metric" in df.columns and "Avg" in df.columns:
        metric_dict = (
            pd.DataFrame([df["Metric"], df["Avg"]])
            .replace("", 0)
            .replace(float("inf"), -1)  # It should not happen
            .replace(float("-inf"), -1)
            .transpose()
            .set_index("Metric")
            .to_dict()["Avg"]
        )
    else:
        raise NameError(f"simple_bar: No Metric or Avg in df columns: {str(df.columns)}")

    plt.clear_figure()

    # adjust plot size along x axis based on the max value
    w = max(list(metric_dict.values())) - 40
    if w < 20 and w > 1:
        w *= 3
    elif w < 1:
        w *= 100
    plt.simple_bar(list(metric_dict.keys()), list(metric_dict.values()), width=w)
    # plt.show()
    return "\n" + plt.build() + "\n"


def simple_multiple_bar(df, title=None):
    """
    Plot data with simple multiple bar chart
    """

    # TODO: handle Nan and None properly

    plt.clear_figure()
    t_df = (
        df.fillna(0).replace("", 0).replace(float("inf"), -1).replace(float("-inf"), -1)
    )
    sub_labels = t_df.transpose().to_dict("split")["index"]
    sub_labels.pop(0)
    data = t_df.transpose().to_dict("split")["data"]
    labels = data.pop(0)

    # plt.simple_multiple_bar(labels, data, labels = sub_labels) #, width=w)

    # print(data)
    plt.theme("pro")
    # adjust plot size along y axis based on the max value
    h = max(max(y) for y in data)
    # print(h)
    if h < 20 and h > 0.5:
        h *= 10
    elif h < 0.5 or math.isclose(h, 0.5):
        h *= 300

    plt.plot_size(height=h)
    plt.multiple_bar(labels, data, color=["blue", "blue+", 68, 63])

    # plt.show()
    return "\n" + plt.build() + "\n"


def simple_box(df, orientation="v", title=None):
    """
    Plot data with simple box/whisker chart.
    Accept pre-calculated data only for now.
    """

    # Example:
    # labels = ["apple", "bee", "cat", "dog"]
    # datas = [
    #     # max, q3, q2, q1, min
    #     [10, 7, 5, 3, 1.5],
    #     [19, 12.3, 9, 7, 4],
    #     [15, 14, 11, 9, 8],
    #     [13, 12, 11, 10, 6]]

    # plt.box(labels, datas, width=0.1, hint='hint')
    # plt.theme("pro")
    # plt.title("Most Favored Pizzas in the World")
    # plt.show()

    plt.clear_figure()
    labels = []
    data = []

    # TODO:
    # handle Nan and None properly
    # error checking for labels
    # show unit if provided

    labels_length = 0
    t_df = (
        df.fillna(0).replace("", 0).replace(float("inf"), -1).replace(float("-inf"), -1)
    )
    for index, row in t_df.iterrows():
        column_name = row.get("Metric") or row.get("Channel")

        if column_name is None:
            raise KeyError("Neither 'Metric' nor 'Channel' column found")

        labels.append(column_name)
        # TODO: need better fix for horizontal overflow
        labels_length += len(str(column_name)) + 8
        data.append([row["Max"], row["Q3"], row["Median"], row["Q1"], row["Min"]])

    # TODO: need better fix for horizontal overflow
    # labels_length *= 0.80
    # print("~~~~~~~~~~~~~~~~~~~~")
    # print(labels)
    # print(labels_length)
    # print(data)
    # print("~~~~~~~~~~~~~~~~~~~~")
    # print(plt.bar.__doc__)

    if orientation == "v":
        # adjust plot size along x axis based on total labels length
        plt.plot_size(labels_length, 30)

    plt.box(
        labels,
        data,
        width=0.1,
        colors=["blue+", "orange+"],
        orientation=orientation,
    )
    plt.theme("pro")

    # plt.show()
    return "\n" + plt.build() + "\n"


def px_simple_bar(df, title: str = None, id=None, style: dict = None, orientation="h"):
    """
    Plot data with simple bar chart
    """

    # TODO: handle None properly
    if "Metric" in df.columns and ("Count" in df.columns or "Value" in df.columns):
        detected_label = "Count" if "Count" in df.columns else "Value"
        df[detected_label] = [
            x.astype(int) if x != "" else int(0) for x in df[detected_label]
        ]
    else:
        raise NameError("simple_bar: No Metric or Count in df columns!")

    # Assign figure characteristics
    range_color = style.get("range_color", None)
    label_txt = style.get("label_txt", None)
    xrange = style.get("xrange", None)
    if label_txt is not None:
        label_txt = label_txt.strip("()")
        try:
            label_txt = label_txt.replace("+ $normUnit", df["Unit"][0])
        except KeyError:
            print("No units found in df. Auto labeling.")

    # Overrides for figure chatacteristics
    if id == 1701.1:
        label_txt = "%"
        range_color = [0, 100]
        xrange = [0, 110]
    if id == 1701.2:
        label_txt = "Gb/s"
        range_color = [0, 1638]
        xrange = [0, 1638]

    fig = px.bar(
        df,
        title=title,
        x=detected_label,
        y="Metric",
        color=detected_label,
        range_color=range_color,
        labels={detected_label: label_txt},
        orientation=orientation,
    ).update_xaxes(range=xrange)

    return fig


def px_simple_multi_bar(df, title=None, id=None):
    """
    Plot data with simple multiple bar chart
    """

    # TODO: handle Nan and None properly
    if "Metric" in df.columns and "Avg" in df.columns:
        df["Avg"] = [x.astype(int) if x != "" else int(0) for x in df["Avg"]]
    else:
        raise NameError("simple_multi_bar: No Metric or Count in df columns!")

    dfigs = []
    nested_bar = {}
    df_unit = df["Unit"][0]
    if id == 1604:
        nested_bar = {"NC": {}, "UC": {}, "RW": {}, "CC": {}}
        for index, row in df.iterrows():
            nested_bar[row["Coherency"]][row["Xfer"]] = row["Avg"]
    if id == 1704:
        nested_bar = {"Read": {}, "Write": {}}
        for index, row in df.iterrows():
            nested_bar[row["Transaction"]][row["Type"]] = row["Avg"]

    for group, metric in nested_bar.items():
        dfigs.append(
            px.bar(
                title=group,
                x=metric.values(),
                y=metric.keys(),
                labels={"x": df_unit, "y": ""},
                text=metric.values(),
            )
            .update_xaxes(showgrid=False, rangemode="nonnegative")
            .update_yaxes(showgrid=False)
        )
    return dfigs


class RooflinePlot(Static):
    """Roofline Plot visualization widget."""

    DEFAULT_CSS = """
    RooflinePlot {
        border: solid $accent;
        padding: 0;
        width: auto;
        height: auto;
        overflow-y: auto;
        overflow-x: auto;
        background: $surface;
        color: $text;
    }
    """

    def __init__(self, df: pd.DataFrame, **kwargs):
        """Initialize the roofline plot"""
        super().__init__("", classes="roofline", **kwargs)
        self.df = df

        # Disable markup rendering
        self._render_markup = False

        try:
            plot_str = ""
            try:
                result = self.df["roofline"]
                if result:
                    plot_str = str(result)
            except:
                plot_str = "No roofline data generated"

            self.update(plot_str)
        except Exception as e:
            error_message = f"Roofline plot error: {str(e)}\n{traceback.format_exc()}"
            self.update(error_message)


class MemoryChart(Static):
    """Memory chart visualization widget."""

    DEFAULT_CSS = """
    MemoryChart {
        border: solid $accent;
        padding: 0;
        width: auto;
        height: auto;
        overflow-y: auto;
        overflow-x: auto;
        background: $surface;
        color: $text;
    }
    """

    def __init__(self, df: pd.DataFrame, **kwargs):
        """Initialize the memory chart."""
        super().__init__("", classes="mem-chart", **kwargs)
        self.df = df

        # Generate the chart content on initialization
        try:
            # Prepare data
            metric_dict = (
                self.df[["Metric", "Value"]].set_index("Metric").to_dict()["Value"]
            )

            # Capture stdout
            original_stdout = sys.stdout
            string_buffer = StringIO()
            sys.stdout = string_buffer

            try:
                # Generate the chart
                result = plot_mem_chart("", "per_kernel", metric_dict)
                stdout_output = string_buffer.getvalue()

                if stdout_output:
                    plot_str = stdout_output
                elif result:
                    plot_str = str(result)
                else:
                    plot_str = "No chart data generated"
            finally:
                sys.stdout = original_stdout

            self.update(plot_str)

        except Exception as e:
            error_message = f"Memory chart error: {str(e)}\n{traceback.format_exc()}"
            self.update(f"Error: {str(error_message)}")


class SimpleBar(Static):
    """Simple Bar visualization widget."""

    DEFAULT_CSS = """
    SimpleBar {
        padding: 0;
        width: auto;
        height: auto;
        overflow-y: auto;
        overflow-x: auto;
        background: $surface;
        color: $text;
    }
    """

    def __init__(self, df: pd.DataFrame, **kwargs):
        """Initialize the simple bar."""
        super().__init__("", classes="simple-bar", **kwargs)
        self.df = df

        try:
            result = simple_bar(self.df)

            if result:
                plot_str = str(result)
                # Escape markup characters
                escaped_content = plot_str.replace("[", r"\[").replace("]", r"\]")
                self.update(escaped_content)

                # Alternative - wrap in [pre] tags for preformatted text
                # self.update(f"[pre]{plot_str}[/pre]")

            else:
                self.update("No simple bar data generated")

        except Exception as e:
            error_message = f"Simple Bar error: {str(e)}\n{traceback.format_exc()}"
            escaped_error = error_message.replace("[", r"\[").replace("]", r"\]")
            self.update(f"Error: {escaped_error}")


class SimpleBox(Static):
    """Simple Box visualization widget."""

    DEFAULT_CSS = """
    SimpleBox {
        padding: 0;
        width: auto;
        height: auto;
        overflow-y: auto;
        overflow-x: auto;
        background: $surface;
        color: $text;
    }
    """

    def __init__(self, df: pd.DataFrame, **kwargs):
        """Initialize the simple box."""
        super().__init__("", classes="simple-box", **kwargs)
        self.df = df

        try:
            result = simple_box(self.df)

            if result:
                plot_str = str(result)
                # Escape markup characters
                escaped_content = plot_str.replace("[", r"\[").replace("]", r"\]")
                self.update(escaped_content)
            else:
                self.update("No simple box data generated")

        except Exception as e:
            error_message = f"Simple Box error: {str(e)}\n{traceback.format_exc()}"
            escaped_error = error_message.replace("[", r"\[").replace("]", r"\]")
            self.update(f"Error: {escaped_error}")


class SimpleMultiBar(Static):
    """Simple Multiple Bar visualization widget."""

    DEFAULT_CSS = """
    SimpleMultiBar {
        padding: 0;
        width: auto;
        height: auto;
        overflow-y: auto;
        overflow-x: auto;
        background: $surface;
        color: $text;
    }
    """

    def __init__(self, df: pd.DataFrame, **kwargs):
        """Initialize the simple multiple bar."""
        super().__init__("", classes="simple-multi-bar", **kwargs)
        self.df = df

        try:
            result = simple_multiple_bar(self.df)

            if result:
                plot_str = str(result)
                # Escape markup characters
                escaped_content = plot_str.replace("[", r"\[").replace("]", r"\]")
                self.update(escaped_content)
            else:
                self.update("No simple multi bar data generated")

        except Exception as e:
            error_message = (
                f"Simple Multiple Box error: {str(e)}\n{traceback.format_exc()}"
            )
            escaped_error = error_message.replace("[", r"\[").replace("]", r"\]")
            self.update(f"Error: {escaped_error}")
