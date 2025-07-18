import copy
import logging
import os
import re
from collections import defaultdict
from datetime import datetime
from enum import Enum
from pathlib import Path

import pandas as pd

from config import HIDDEN_COLUMNS, HIDDEN_SECTIONS

supported_field = [
    "Value",
    "Minimum",
    "Maximum",
    "Average",
    "Median",
    "Min",
    "Max",
    "Avg",
    "Pct of Peak",
    "Peak",
    "Count",
    "Mean",
    "Pct",
    "Std Dev",
    "Q1",
    "Q3",
    "Expression",
    # Special keywords for L2 channel
    "Channel",
    "L2 Cache Hit Rate",
    "Requests",
    "L2 Read",
    "L2 Write",
    "L2 Atomic",
    "L2-Fabric Requests",
    "L2-Fabric Read",
    "L2-Fabric Write and Atomic",
    "L2-Fabric Atomic",
    "L2 Read Req",
    "L2 Write Req",
    "L2 Atomic Req",
    "L2-Fabric Read Req",
    "L2-Fabric Write and Atomic Req",
    "L2-Fabric Atomic Req",
    "L2-Fabric Read Latency",
    "L2-Fabric Write Latency",
    "L2-Fabric Atomic Latency",
    "L2-Fabric Read Stall (PCIe)",
    "L2-Fabric Read Stall (Infinity Fabric™)",
    "L2-Fabric Read Stall (HBM)",
    "L2-Fabric Write Stall (PCIe)",
    "L2-Fabric Write Stall (Infinity Fabric™)",
    "L2-Fabric Write Stall (HBM)",
    "L2-Fabric Write Starve",
]


class LogLevel(str, Enum):
    """Log levels for consistent logging."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"  # Maintained for UI compatibility


class Logger:
    """Centralized logging handler for the application."""

    def __init__(self, output_area=None):
        """
        Initialize the logger.
        """
        self.output_area = output_area
        self._setup_logger()

    def _setup_logger(self):
        """
        Setup the Python logger with proper formatting.
        """
        self.logger = logging.getLogger("app")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def set_output_area(self, output_area):
        """
        Set or update the output area for displaying logs.
        """
        self.output_area = output_area

    def log(self, message, level=LogLevel.INFO, update_ui=True):
        """
        Log a message with the specified level.
        """
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
                self.output_area.text = (
                    f"{current_text}\n{formatted_msg}" if current_text else formatted_msg
                )
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


def split_table_line(line):
    """
    Splits a table row line into a list of cell strings (trimmed). For example:

    │    │ Kernel_Name                              │   Count │ ...
    """

    cells = line.split("│")
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1] == "":
        cells = cells[:-1]
    return [cell.strip() for cell in cells]


def parse_ascii_table(table_lines):
    """
    Given a list of lines belonging to one ASCII table (including border rows),
    return a tuple (header, data_rows) where header is a list of column names and
    data_rows is a list of rows (each a list of cell strings).

    Skips border/separator lines and also checks for continuation
    rows (which have an empty first cell). Continuation rows get merged into the previous row.
    """

    header = None
    data_rows = []

    for line in table_lines:
        if re.match(r"^[╒╞╘├└─]+", line):
            continue
        if "│" not in line:
            continue

        cells = split_table_line(line)

        if header is None:
            header = cells
            continue

        if cells and cells[0] == "":
            if data_rows:  # There should be at least one row already.
                for i, cell in enumerate(cells):
                    if cell:
                        data_rows[-1][i] += " " + cell
            else:
                continue
        else:
            data_rows.append(cells)
    return header, data_rows


def parse_file(filename):
    """
    Returns nested structure:
    {
        "0. Top Stats": {
            "0.1 Top Kernels": {header: [...], data: [...]},
            "0.2 Dispatch List": {header: [...], data: [...]}
        },
        "1. System Info": {
            "1.1 System Information": {header: [...], data: [...]}
        },
        ...
    }
    """
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sections = {}
    current_section = None
    current_subsection = None
    table_lines = []
    in_table = False

    for line in lines:
        line = line.rstrip("\n")

        # Skip separator lines
        if line.startswith(
            "--------------------------------------------------------------------------------"
        ):
            continue

        # Check for section header (e.g., "0. Top Stats")
        section_match = re.match(r"^\s*(\d+\. .+)$", line)
        if section_match:
            current_section = section_match.group(1).strip()
            sections[current_section] = {}
            continue

        # Check for subsection header (e.g., "0.1 Top Kernels")
        # FIXME: 1. System Info is an exception, no subsection
        subsection_match = re.match(r"^\s*(\d+\.\d+ .+)$", line)
        if subsection_match:
            current_subsection = subsection_match.group(1).strip()
            if current_section is None:
                current_section = "Uncategorized"
                sections[current_section] = {}
            continue

        # Table parsing logic
        if line.startswith("╒"):
            in_table = True
            table_lines = [line]
            continue

        if in_table:
            table_lines.append(line)
            if line.startswith("╘"):
                if current_section and current_subsection:
                    header, data = parse_ascii_table(table_lines)
                    sections[current_section][current_subsection] = {
                        "header": header,
                        "data": data,
                    }
                in_table = False
                table_lines = []

    return sections


def get_table_dfs():
    filename = str(Path(os.getcwd()).joinpath("analyze_output.csv"))
    sections_info = parse_file(filename)

    # Convert to DataFrames while maintaining nested structure
    section_dfs = {}
    for section_name, subsections in sections_info.items():
        section_dfs[section_name] = {}
        for subsection_name, table_data in subsections.items():
            if table_data and table_data["data"]:
                try:
                    df = pd.DataFrame(table_data["data"], columns=table_data["header"])
                    section_dfs[section_name][subsection_name] = df
                except Exception as e:
                    print(f"Error creating DataFrame for {subsection_name}: {e}")
                    continue

    return section_dfs


def process_panels_to_dataframes(
    args, runs, archConfigs, profiling_config, roof_plot=None
):
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

    comparable_columns = build_comparable_columns(args.time_unit)
    filter_panel_ids = profiling_config.get("filter_blocks", [])
    if isinstance(filter_panel_ids, dict):
        # For backward compatibility
        filter_panel_ids = [
            name for name, type in filter_panel_ids.items() if type == "metric_id"
        ]
    filter_panel_ids = [
        int(convert_metric_id_to_panel_info(metric_id)[0])
        for metric_id in filter_panel_ids
    ]

    # Initialize the result structure
    result_structure = defaultdict(dict)

    for panel_id, panel in archConfigs.panel_configs.items():
        # Skip panels that don't support baseline comparison
        if panel_id in HIDDEN_SECTIONS:
            continue

        # Get section name (e.g., "0. Top Stats")
        section_name = f"{panel_id // 100}. {panel['title']}"

        for data_source in panel["data source"]:
            for type, table_config in data_source.items():
                # Check for filtering conditions
                if (
                    not args.filter_metrics
                    and filter_panel_ids
                    and table_config["id"] not in filter_panel_ids
                    and panel_id not in filter_panel_ids
                    and panel_id > 100
                ):
                    table_id_str = (
                        str(table_config["id"] // 100)
                        + "."
                        + str(table_config["id"] % 100)
                    )
                    continue

                # Process the data
                base_run, base_data = next(iter(runs.items()))
                base_df = base_data.dfs[table_config["id"]]

                df = pd.DataFrame(index=base_df.index)

                # Process columns
                for header in list(base_df.keys()):
                    if should_process_column(header, args, type):
                        if header in HIDDEN_COLUMNS:
                            pass
                        elif header not in comparable_columns:
                            df = process_non_comparable_column(
                                df, header, base_df, type, table_config, runs
                            )
                        else:
                            df = process_comparable_column(
                                df,
                                header,
                                base_df,
                                table_config,
                                runs,
                                base_run,
                                type,
                                args,
                                HIDDEN_COLUMNS,
                            )

                if not df.empty:
                    # Check for empty columns
                    is_empty_columns_exist = check_empty_columns(df)

                    if not is_empty_columns_exist:
                        # Get subsection name
                        table_id_str = (
                            str(table_config["id"] // 100)
                            + "."
                            + str(table_config["id"] % 100)
                        )
                        subsection_name = table_id_str
                        if "title" in table_config and table_config["title"]:
                            subsection_name += " " + table_config["title"]

                        # Handle special cases for top stats
                        if type == "raw_csv_table" and (
                            table_config["source"] == "pmc_kernel_top.csv"
                            or table_config["source"] == "pmc_dispatch_info.csv"
                        ):
                            df = df.head(args.max_stat_num)

                        # Check for transpose requirement
                        transpose = (
                            type != "raw_csv_table"
                            and "columnwise" in table_config
                            and table_config.get("columnwise") == True
                        )

                        if transpose:
                            df = df.T

                        # Store the DataFrame with tui_style as separate keys
                        result_structure[section_name][subsection_name] = {
                            "df": df,
                            "tui_style": None,
                        }

                        # Set tui_style if available
                        if type == "metric_table" and "tui_style" in table_config:
                            result_structure[section_name][subsection_name][
                                "tui_style"
                            ] = table_config["tui_style"]

                        # Save to CSV if requested
                        if args.df_file_dir:
                            save_dataframe_to_csv(df, table_id_str, table_config, args)
    result_structure["4. Roofline"] = roof_plot
    return dict(result_structure)


def should_process_column(header, args, type):
    """Check if a column should be processed based on arguments."""
    return (
        (not args.cols)
        or (
            args.cols and header in args.cols
        )  # Assuming args.cols is now a list of column names
        or (type == "raw_csv_table")
    )


def process_non_comparable_column(df, header, base_df, type, table_config, runs):
    """Process columns that are not comparable across runs."""
    if (
        type == "raw_csv_table"
        and (
            table_config["source"] == "pmc_kernel_top.csv"
            or table_config["source"] == "pmc_dispatch_info.csv"
        )
        and header == "Kernel_Name"
    ):
        # Adjust kernel name width based on source
        if table_config["source"] == "pmc_kernel_top.csv":
            adjusted_name = base_df["Kernel_Name"].apply(
                lambda x: string_multiple_lines(x, 40, 3)
            )
        else:
            adjusted_name = base_df["Kernel_Name"].apply(
                lambda x: string_multiple_lines(x, 80, 4)
            )
        df = pd.concat([df, adjusted_name], axis=1)
    elif type == "raw_csv_table" and header == "Info":
        for run, data in runs.items():
            cur_df = data.dfs[table_config["id"]]
            df = pd.concat([df, cur_df[header]], axis=1)
    else:
        df = pd.concat([df, base_df[header]], axis=1)

    return df


def process_comparable_column(
    df, header, base_df, table_config, runs, base_run, type, args, hidden_columns
):
    """Process columns that can be compared across runs."""
    for run, data in runs.items():
        cur_df = data.dfs[table_config["id"]]
        if (type == "raw_csv_table") or (
            type == "metric_table" and (header not in hidden_columns)
        ):
            if run != base_run:
                # Calculate percentage over the baseline
                base_values = [float(x) if x != "" else float(0) for x in base_df[header]]
                cur_values = [float(x) if x != "" else float(0) for x in cur_df[header]]

                base_df[header] = base_values
                cur_df[header] = cur_values

                t_df = pd.concat(
                    [base_df[header], cur_df[header]],
                    axis=1,
                )
                absolute_diff = (t_df.iloc[:, 1] - t_df.iloc[:, 0]).round(args.decimal)
                t_df = absolute_diff / t_df.iloc[:, 0].replace(0, 1)

                t_df_pretty = t_df.astype(float).mul(100).round(args.decimal)

                # Show value + percentage
                t_df = (
                    cur_df[header].astype(float).round(args.decimal).map(str).astype(str)
                    + " ("
                    + t_df_pretty.map(str)
                    + "%)"
                )
                df = pd.concat([df, t_df], axis=1)

                # Check for threshold violations
                if (
                    header in ["Value", "Count", "Avg"]
                    and t_df_pretty.abs().gt(args.report_diff).any()
                ):
                    df["Abs Diff"] = absolute_diff
                    if args.report_diff:
                        violation_idx = t_df_pretty.index[
                            t_df_pretty.abs() > args.report_diff
                        ]
            else:
                cur_df_copy = copy.deepcopy(cur_df)
                cur_df_copy[header] = [
                    (round(float(x), args.decimal) if x != "" else x)
                    for x in base_df[header]
                ]
                df = pd.concat([df, cur_df_copy[header]], axis=1)

    return df


def check_empty_columns(df):
    """Check if any column in the DataFrame is empty."""
    return any(
        [
            df.columns[col_idx]
            for col_idx in range(len(df.columns))
            if df.replace("", None).iloc[:, col_idx].isnull().all()
        ]
    )


def save_dataframe_to_csv(df, table_id_str, table_config, args):
    """Save DataFrame to CSV file if directory is specified."""
    p = Path(args.df_file_dir)
    if not p.exists():
        p.mkdir()
    if p.is_dir():
        filename = table_id_str
        if "title" in table_config and table_config["title"]:
            filename += "_" + table_config["title"]
        df.to_csv(
            p.joinpath(filename.replace(" ", "_") + ".csv"),
            index=False,
        )


def string_multiple_lines(source, width, max_rows):
    """
    Adjust string with multiple lines by inserting '\n'
    """
    idx = 0
    lines = []
    while idx < len(source) and len(lines) < max_rows:
        lines.append(source[idx : idx + width])
        idx += width

    if idx < len(source):
        last = lines[-1]
        lines[-1] = last[0:-3] + "..."
    return "\n".join(lines)


def convert_metric_id_to_panel_info(metric_id):
    """
    Convert metric id into panel information.
    Output is a tuples of the form (file_id, panel_id, metric_id).

    For example:

    Input: "2"
    Output: ("0200", None, None)

    Input: "11"
    Output: ("1100", None, None)

    Input: "11.1"
    Output: ("1100", 1101, None)

    Input: "11.1.1"
    Output: ("1100", 1101, 1)

    Raises exception for invalid metric id.
    """
    tokens = metric_id.split(".")
    if 0 < len(tokens) < 4:
        # File id
        file_id = str(int(tokens[0]))
        # 4 -> 04
        if len(file_id) < 2:
            file_id = f"0{file_id}"
        # Multiply integer by 100
        file_id = f"{file_id}00"
        # Panel id
        if len(tokens) > 1:
            panel_id = int(tokens[0]) * 100
            panel_id += int(tokens[1])
        else:
            panel_id = None
        # Metric id
        if len(tokens) > 2:
            metric_id = int(tokens[2])
        else:
            metric_id = None
        return (file_id, panel_id, metric_id)
    else:
        raise Exception(f"Invalid metric id: {metric_id}")


def build_comparable_columns(time_unit):
    """
    Build comparable columns/headers for display
    """
    comparable_columns = supported_field
    top_stat_base = ["Count", "Sum", "Mean", "Median", "Standard Deviation"]

    for h in top_stat_base:
        comparable_columns.append(h + "(" + time_unit + ")")

    return comparable_columns
