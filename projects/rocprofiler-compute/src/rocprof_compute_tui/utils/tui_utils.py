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
import argparse
import logging
from collections.abc import Hashable
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import pandas as pd
from textual.widgets import TextArea

import config
from utils import schema


class LogLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


class Logger:
    def __init__(self, output_area: Optional[TextArea] = None) -> None:
        self.output_area = output_area
        self._setup_logger()

    def _setup_logger(self) -> None:
        self.logger = logging.getLogger("app")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def set_output_area(self, output_area: TextArea) -> None:
        self.output_area = output_area

    def log(
        self, message: str, log_level: str = "INFO", update_ui: bool = True
    ) -> None:
        level_map = {
            "INFO": logging.INFO,
            "SUCCESS": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
        }
        self.logger.log(level_map[log_level], message)

        if update_ui and self.output_area and hasattr(self.output_area, "text"):
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_msg = f"[{timestamp}] [{log_level}] {message}"
            self.output_area.text = (
                f"{self.output_area.text}\n{formatted_msg}"
                if self.output_area.text
                else formatted_msg
            )
            self.output_area.cursor_location = (999999, 0)

    def info(self, message: str, update_ui: bool = True) -> None:
        self.log(message, "INFO", update_ui)

    def success(self, message: str, update_ui: bool = True) -> None:
        self.log(message, "SUCCESS", update_ui)

    def warning(self, message: str, update_ui: bool = True) -> None:
        self.log(message, "WARNING", update_ui)

    def error(self, message: str, update_ui: bool = True) -> None:
        self.log(message, "ERROR", update_ui)


def get_top_kernels_and_dispatch_ids(
    runs: dict[str, Any],
) -> Optional[list[dict[Hashable, Any]]]:
    if not runs:
        return None

    base_run = next(iter(runs.values()))
    if not hasattr(base_run, "dfs"):
        return None

    top_kernel_df = base_run.dfs.get(1)
    dispatch_id_df = base_run.dfs.get(2)

    if top_kernel_df is None or dispatch_id_df is None:
        return None

    merged_df = pd.merge(
        top_kernel_df, dispatch_id_df, on="Kernel_Name", how="outer"
    ).sort_values("Pct", ascending=False)

    merged_df = merged_df.drop(columns=["Count", "GPU_ID"])
    return merged_df.to_dict("records")


def process_panels_to_dataframes(
    args: argparse.Namespace,
    kernel_df: dict[int, pd.DataFrame],
    arch_configs: schema.ArchConfig,
    roof_plot: Optional[str] = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Process panel data into pandas DataFrames.
    Returns a nested dictionary structure with DataFrames and tui_style information.

    Returns:
        Dict[str, Dict[str, Dict[str, Any]]]: Nested structure {
            "section_name": {
                "subsection_name": {
                    "df": DataFrame,
                    "tui_style": dict or None
                }
            }
        }
    """

    # TODO: add individual kernel roofline logic
    # TODO: implement args logic:
    #       args.filter_metrics
    #       args.cols
    #       args.max_stat_num
    #       dfs file dir

    result_structure = {}
    decimal_precision = getattr(args, "decimal", 2) if args else 2

    for panel_id, panel in arch_configs.panel_configs.items():
        if panel_id in config.HIDDEN_SECTIONS:
            continue

        section_name = f"{panel_id // 100}. {panel['title']}"
        section_data = {}

        for data_source in panel["data source"]:
            for type, table_config in data_source.items():
                table_id = table_config["id"]

                if (
                    table_id not in kernel_df
                    or kernel_df[table_id] is None
                    or kernel_df[table_id].empty
                ):
                    continue

                base_df = kernel_df[table_id]

                df = pd.DataFrame(index=base_df.index)

                for header in list(base_df.columns):
                    if header in config.HIDDEN_COLUMNS_TUI:
                        continue
                    else:
                        df[header] = base_df[header]

                df = apply_rounding_logic(df, decimal_precision)

                subsection_name = (
                    f"{table_config['id'] // 100}.{table_config['id'] % 100}"
                )
                if table_config.get("title"):
                    subsection_name += f" {table_config['title']}"

                section_data[subsection_name] = {
                    "df": df,
                    "tui_style": (
                        table_config.get("tui_style")
                        if type == "metric_table"
                        else None
                    ),
                }

        if section_data:
            result_structure[section_name] = section_data

    return result_structure


def apply_rounding_logic(df: pd.DataFrame, decimal_precision: int) -> pd.DataFrame:
    if df.empty:
        return df

    df_rounded = df.copy()

    float_cols = df_rounded.select_dtypes(include=["float"]).columns
    if len(float_cols) > 0:
        df_rounded[float_cols] = df_rounded[float_cols].round(decimal_precision)

    non_float_cols = df_rounded.select_dtypes(exclude=["float"]).columns
    for col in non_float_cols:
        numeric_series = pd.to_numeric(df_rounded[col], errors="coerce")
        if numeric_series.notna().any():
            df_rounded[col] = numeric_series.round(decimal_precision)

    return df_rounded
