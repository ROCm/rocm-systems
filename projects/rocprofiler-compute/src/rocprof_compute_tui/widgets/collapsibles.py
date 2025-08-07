##############################################################################
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


from typing import Any, Dict, List

import pandas as pd
import yaml
from textual.widgets import Collapsible, DataTable, Label

from rocprof_compute_tui.widgets.charts import (
    MemoryChart,
    RooflinePlot,
    SimpleBar,
    SimpleBox,
    SimpleMultiBar,
)


def create_table(df: pd.DataFrame) -> DataTable:
    table = DataTable(zebra_stripes=True)
    df = df.reset_index()
    df = df[~df.apply(lambda row: row.astype(str).str.strip().eq("").any(), axis=1)]
    str_columns = [str(col) for col in df.columns]
    table.add_columns(*str_columns)
    table.add_rows([tuple(str(x) for x in row) for row in df.itertuples(index=False)])
    return table


def create_widget_from_data(df: pd.DataFrame, tui_style: str = None, context: str = ""):
    if df is None or df.empty:
        return Label(
            f"Data not available{f' for {context}' if context else ''}", classes="warning"
        )

    match tui_style:
        # TODO: implement tui_style == "roofline"
        # case "roofline":
        #     return Roofline(df)
        case None:
            return create_table(df)
        case "mem_chart":
            return MemoryChart(df)
        case "simple_bar":
            return SimpleBar(df)
        case "simple_box":
            return SimpleBox(df)
        case "simple_multiple_bar":
            return SimpleMultiBar(df)
        case _:
            return Label(f"Unknown display type: {tui_style}")


def load_config(config_path) -> Dict[str, Any]:
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Configuration file {config_path} not found, \nplease populate the analysis_config.yaml file."
        )
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")


def build_section_from_config(
    dfs: Dict[str, Any], section_config: Dict[str, Any]
) -> Collapsible:
    title = section_config["title"]
    collapsed = section_config.get("collapsed", True)

    children = []
    for subsection_config in section_config["subsections"]:
        # Handle arch_config_data
        if subsection_config.get("arch_config_data", False):
            if isinstance(dfs, dict):
                exclude_keys = subsection_config.get("exclude_keys", [])
                for section_name, subsections in dfs.items():
                    if section_name not in exclude_keys and isinstance(subsections, dict):
                        kernel_children = []
                        for subsection_name, data in subsections.items():
                            if isinstance(data, dict) and "df" in data:
                                widget = create_widget_from_data(
                                    data["df"], data.get("tui_style"), subsection_name
                                )
                                kernel_children.append(
                                    Collapsible(
                                        widget, title=subsection_name, collapsed=True
                                    )
                                )

                        if kernel_children:
                            children.append(
                                Collapsible(
                                    *kernel_children, title=section_name, collapsed=True
                                )
                            )
        else:
            # Handle data_path
            subsection_title = subsection_config.get("title", "Untitled")
            subsection_collapsed = subsection_config.get("collapsed", True)
            tui_style = subsection_config.get("tui_style")
            data_path = subsection_config["data_path"]

            df = dfs.get(data_path[0], {}).get(data_path[1], {})
            df = df.get("df") if isinstance(df, dict) else None
            if df is not None and tui_style is None:
                tui_style = df.get("tui_style")

            widgets = [
                create_widget_from_data(df, tui_style, f"path {' -> '.join(data_path)}")
            ]

            children.append(
                Collapsible(
                    *widgets, title=subsection_title, collapsed=subsection_collapsed
                )
            )

    return Collapsible(*children, title=title, collapsed=collapsed)


def build_all_sections(dfs: Dict[str, Any], config_path) -> List[Collapsible]:
    config = load_config(config_path)
    sections = []

    for section_config in config["sections"]:
        section = build_section_from_config(dfs, section_config)
        sections.append(section)

    return sections
