import logging
from collections import defaultdict
from datetime import datetime
from enum import Enum

import pandas as pd

import config


class LogLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


class Logger:

    def __init__(self, output_area=None):
        self.output_area = output_area
        self._setup_logger()

    def _setup_logger(self):
        self.logger = logging.getLogger("app")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                          datefmt="%Y-%m-%d %H:%M:%S")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def set_output_area(self, output_area):
        self.output_area = output_area

    def log(self, message, level=LogLevel.INFO, update_ui=True):
        level_map = {
            LogLevel.INFO: logging.INFO,
            LogLevel.SUCCESS: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
        }

        self.logger.log(level_map[level], message)

        timestamp = datetime.now().strftime("%H:%M:%S")

        if update_ui and self.output_area:
            if level == LogLevel.ERROR:
                formatted_msg = f"[{timestamp}] [ERROR] {message}"
            elif level == LogLevel.WARNING:
                formatted_msg = f"[{timestamp}] [WARNING] {message}"
            elif level == LogLevel.SUCCESS:
                formatted_msg = f"[{timestamp}] [SUCCESS] {message}"
            else:
                formatted_msg = f"[{timestamp}] [INFO] {message}"

            if hasattr(self.output_area, "text"):
                current_text = self.output_area.text
                self.output_area.text = (f"{current_text}\n{formatted_msg}"
                                         if current_text else formatted_msg)
                # HACK: moving curson to end of outpu (Is there a better way to achieve this?)
                self.output_area.cursor_location = (999999, 0)

    def info(self, message, update_ui=True):
        self.log(message, LogLevel.INFO, update_ui)

    def success(self, message, update_ui=True):
        self.log(message, LogLevel.SUCCESS, update_ui)

    def warning(self, message, update_ui=True):
        self.log(message, LogLevel.WARNING, update_ui)

    def error(self, message, update_ui=True):
        self.log(message, LogLevel.ERROR, update_ui)


def get_top_kernels_and_dispatch_ids(runs):
    if not runs:
        return None

    base_run = next(iter(runs.values()))
    if not hasattr(base_run, "dfs"):
        return None

    top_kernel_df = base_run.dfs.get(1)
    dispatch_id_df = base_run.dfs.get(2)

    if top_kernel_df is None or dispatch_id_df is None:
        return None

    merged_df = pd.merge(top_kernel_df, dispatch_id_df, on="Kernel_Name",
                         how="outer").sort_values("Pct", ascending=False)

    return merged_df.to_dict("records")


def process_panels_to_dataframes(args, kernel_df, archConfigs, roof_plot=None):
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
    #       args.df_file_dir

    result_structure = defaultdict(dict)

    decimal_precision = getattr(args, "decimal", 2) if args else 2

    for panel_id, panel in archConfigs.panel_configs.items():
        if panel_id in config.HIDDEN_SECTIONS:
            continue

        section_name = f"{panel_id // 100}. {panel['title']}"

        for data_source in panel["data source"]:
            for type, table_config in data_source.items():
                table_id = table_config["id"]

                if table_id not in kernel_df:
                    continue

                base_df = kernel_df[table_id]

                if base_df is None or base_df.empty:
                    continue

                df = pd.DataFrame(index=base_df.index)

                for header in list(base_df.columns):
                    if header in config.HIDDEN_COLUMNS_TUI:
                        continue
                    else:
                        df[header] = base_df[header]

                df = apply_rounding_logic(df, decimal_precision)

                subsection_name = (str(table_config["id"] // 100) + "." +
                                   str(table_config["id"] % 100))
                if "title" in table_config and table_config["title"]:
                    subsection_name += " " + table_config["title"]

                result_structure[section_name][subsection_name] = {
                    "df": df,
                    "tui_style": None,
                }

                if type == "metric_table" and "tui_style" in table_config:
                    result_structure[section_name][subsection_name]["tui_style"] = (
                        table_config["tui_style"])

    return dict(result_structure)


def apply_rounding_logic(df, decimal_precision):
    df_copy = df.copy()

    for column in df_copy.columns:
        if column in ["Metric", "Tips", "coll_level", "Unit", "Kernel_Name", "Info"]:
            continue

        if df_copy[column].dtype in ["float64", "float32", "int64", "int32"]:
            df_copy[column] = df_copy[column].round(decimal_precision)
        else:
            try:
                numeric_series = pd.to_numeric(df_copy[column], errors="coerce")
                if not numeric_series.isna().all():
                    rounded_series = numeric_series.round(decimal_precision)

                    if df_copy[column].dtype == "object":
                        df_copy[column] = df_copy[column].combine(
                            rounded_series,
                            lambda orig, rounded: (rounded
                                                   if pd.notna(rounded) else orig),
                        )
                    else:
                        df_copy[column] = rounded_series
            except (ValueError, TypeError):
                continue

    return df_copy
