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

from typing import Any, Optional, Union

import pandas as pd
import yaml
from textual.widgets import Collapsible, DataTable, Label, Static

from rocprof_compute_tui.widgets.charts import (
    MemoryChart,
    SimpleBar,
    SimpleBox,
    SimpleMultiBar,
)


def create_table(
    df: pd.DataFrame, hidden_columns: Optional[list[str]] = []
) -> Union[DataTable, Label]:
    table = DataTable(zebra_stripes=True)

    df = df.reset_index().dropna()
    df = df[~df.astype(str).apply(lambda row: row.str.strip().eq("").any(), axis=1)]

    if df.empty:
        return Label("No table data generated")

    table._df = df  # type: ignore[attr-defined]
    table._visible_cols = [col for col in df.columns if col not in hidden_columns]  # type: ignore[attr-defined]

    table.add_columns(*table._visible_cols)
    for _, row in df.iterrows():
        table.add_row(*[str(row[col]) for col in table._visible_cols])

    return table


def create_widget_from_data(
    df: Optional[pd.DataFrame], tui_style: Optional[str] = None, context: str = ""
) -> Union[Label, Static, DataTable]:
    if df is None or df.empty:
        return Label(
            f"Data not available{f' for {context}' if context else ''}",
            classes="warning",
        )

    if tui_style is None:
        return create_table(df, hidden_columns=["Description"])
    elif tui_style == "mem_chart":
        return MemoryChart(df)
    elif tui_style == "simple_bar":
        return SimpleBar(df)
    elif tui_style == "simple_box":
        return SimpleBox(df)
    elif tui_style == "simple_multiple_bar":
        return SimpleMultiBar(df)
    else:
        return Label(f"Unknown display type: {tui_style}")


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path) as file:
        return yaml.safe_load(file)


def build_section_from_config(
    dfs: dict[str, Any], section_config: dict[str, Any]
) -> Collapsible:
    title = section_config["title"]
    collapsed = section_config.get("collapsed", True)
    children: list[Collapsible] = []

    for subsection_config in section_config["subsections"]:
        subsection_title = subsection_config.get("title", "Untitled")
        subsection_collapsed = subsection_config.get("collapsed", True)

        # Handle arch_config_data (dynamic sections from dfs)
        if subsection_config.get("arch_config_data", False):
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

        # Handle data_path (specific data sections)
        elif "data_path" in subsection_config:
            data_path = subsection_config["data_path"]
            tui_style = subsection_config.get("tui_style")

            # Navigate data path
            current = dfs
            for key in data_path:
                current = current.get(key, {}) if isinstance(current, dict) else {}

            df = current.get("df") if isinstance(current, dict) else None
            if df is not None and tui_style is None:
                tui_style = current.get("tui_style")

            widget = create_widget_from_data(
                df, tui_style, f"path {' -> '.join(data_path)}"
            )
            children.append(
                Collapsible(
                    widget, title=subsection_title, collapsed=subsection_collapsed
                )
            )

    return Collapsible(*children, title=title, collapsed=collapsed)


def build_all_sections(dfs: dict[str, Any], config_path: str) -> list[Collapsible]:
    config = load_config(config_path)
    return [
        build_section_from_config(dfs, section_config)
        for section_config in config["sections"]
    ]
