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

from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from textual.containers import VerticalScroll
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

    # Clean the DataFrame - remove NaN and empty cells
    df = df.reset_index()
    df = df.dropna(how="any")
    df = df[~df.apply(lambda row: row.astype(str).str.strip().eq("").any(), axis=1)]

    # Add columns and rows
    str_columns = [str(col) for col in df.columns]
    table.add_columns(*str_columns)
    table.add_rows([tuple(str(x) for x in row) for row in df.itertuples(index=False)])

    return table


def load_config(config_path) -> Dict[str, Any]:
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_path} not found")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")


def get_data_from_path(dfs: Dict[str, Any], path: List[str]) -> Optional[pd.DataFrame]:
    try:
        current = dfs
        for key in path:
            current = current[key]
        return current["df"]
    except (KeyError, TypeError):
        return None


def get_tui_style_from_path(dfs: Dict[str, Any], path: List[str]) -> Optional[str]:
    try:
        current = dfs
        for key in path:
            current = current[key]
        return current.get("tui_style")
    except (KeyError, TypeError):
        return None


def create_widget_from_data(df: pd.DataFrame, tui_style: Optional[str] = None) -> Any:
    if df is not None and not df.empty:
        match tui_style:
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
    else:
        return Label(f"Data not available for display in {tui_style}.")


def build_subsection(
    subsection_config: Dict[str, Any], dfs: Dict[str, Any]
) -> Collapsible:
    title = subsection_config["title"]
    collapsed = subsection_config.get("collapsed", True)
    tui_style = subsection_config.get("tui_style")

    # Handle data-driven widgets
    if "data_path" in subsection_config:
        data_path = subsection_config["data_path"]

        if tui_style is None:
            tui_style = (
                get_tui_style_from_path(dfs, data_path) if dfs is not None else None
            )

        df = get_data_from_path(dfs, data_path)

        if df is None and tui_style is None:
            error_msg = (
                f"{title} data not available: Path {' -> '.join(data_path)} not found"
            )
            return Collapsible(
                Label(error_msg, classes="warning"), title=title, collapsed=collapsed
            )

        # Create main widget
        widget = create_widget_from_data(df, tui_style)

        # Add header label if specified
        widgets = []
        if "header_label" in subsection_config:
            header_class = subsection_config.get("header_class", "")
            widgets.append(Label(subsection_config["header_label"], classes=header_class))

        widgets.append(widget)

        collapsible = Collapsible(*widgets, title=title, collapsed=collapsed)
    elif tui_style == "roofline":
        if dfs["roofline"]:
            widget = RooflinePlot(dfs)
            collapsible = Collapsible(widget, title=title, collapsed=collapsed)
        else:
            return None
    # Fallback for subsections without data or style
    else:
        collapsible = Collapsible(
            Label(f"No data or style configuration for {title}"),
            title=title,
            collapsed=collapsed,
        )

    # Add ID if specified
    if "widget_id" in subsection_config:
        collapsible.id = subsection_config["widget_id"]

    return collapsible


def build_dynamic_kernel_sections(
    dfs: Dict[str, Any], skip_sections: List[str]
) -> List[Collapsible]:
    children = []

    try:
        for section_name, subsections in dfs.items():
            if section_name in skip_sections:
                continue

            kernel_children = []
            for subsection_name, data in subsections.items():
                if isinstance(data, dict) and "df" in data:
                    df = data["df"]
                    tui_style = data.get("tui_style")
                    widget = create_widget_from_data(df, tui_style)
                    kernel_children.append(
                        Collapsible(widget, title=subsection_name, collapsed=True)
                    )

            if kernel_children:
                children.append(
                    Collapsible(*kernel_children, title=section_name, collapsed=True)
                )

    except Exception as e:
        children.append(Label(f"Error in Kernel Section: {str(e)}", classes="error"))

    return children


def build_section_from_config(
    section_config: Dict[str, Any], dfs: Dict[str, Any]
) -> Collapsible:
    title = section_config["title"]
    collapsed = section_config.get("collapsed", True)
    css_class = section_config.get("class")

    # Handle under construction sections
    if section_config.get("under_construction", False):
        construction_label = section_config.get(
            "construction_label", "Under Construction"
        )
        construction_class = section_config.get("construction_class", "")
        children = [Label(construction_label, classes=construction_class)]

    # Handle dynamic sections (like kernel sections)
    elif section_config.get("dynamic_sections", False):
        skip_sections = section_config.get("skip_sections", [])
        children = build_dynamic_kernel_sections(dfs, skip_sections)

    # Handle regular sections with subsections
    elif "subsections" in section_config:
        children = []
        for subsection_config in section_config["subsections"]:
            try:
                subsection = build_subsection(subsection_config, dfs)
                if subsection:
                    children.append(subsection)
            except Exception as e:
                error_msg = f"{subsection_config.get('title', 'Unknown')} error: {str(e)}"
                children.append(Label(error_msg, classes="warning"))

    else:
        children = [Label("No configuration provided for this section")]

    # Create the main collapsible
    collapsible = Collapsible(*children, title=title, collapsed=collapsed)

    # Add CSS class if specified
    if css_class:
        collapsible.add_class(css_class)

    return collapsible


def build_all_sections(dfs: Dict[str, Any], config_path) -> List[Collapsible]:
    config = load_config(config_path)
    sections = []

    for section_config in config["sections"]:
        try:
            section = build_section_from_config(section_config, dfs)
            sections.append(section)
        except Exception as e:
            # Create error section if something goes wrong
            error_title = section_config.get("title", "Unknown Section")
            error_section = Collapsible(
                Label(f"Error building section: {str(e)}", classes="error"),
                title=f"❌ {error_title}",
                collapsed=True,
            )
            sections.append(error_section)

    return sections
