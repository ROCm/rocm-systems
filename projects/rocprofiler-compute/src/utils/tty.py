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

import argparse
import copy
import textwrap
from pathlib import Path
from typing import Any, Optional, TextIO

import pandas as pd
from tabulate import tabulate

import config
from utils import mem_chart, parser, schema
from utils.kernel_name_shortener import kernel_name_shortener
from utils.logger import console_error, console_log, console_warning
from utils.utils import (
    METRIC_ID_RE,
    convert_metric_id_to_panel_info,
    get_panel_alias,
    get_uuid,
)


def string_multiple_lines(source: str, width: int, max_rows: int) -> str:
    """
    Adjust string with multiple lines by inserting '\n'
    """
    lines: list[str] = []
    for i in range(0, len(source), width):
        if len(lines) >= max_rows:
            break
        lines.append(source[i : i + width])

    if len(lines) == max_rows and len(source) > max_rows * width:
        lines[-1] = lines[-1][:-3] + "..."

    return "\n".join(lines)


def get_table_string(
    df: pd.DataFrame, transpose: bool = False, decimal: int = 2
) -> str:
    """
    Convert DataFrame to a formatted table string, wrapping specified columns.
    """
    df_to_show = df.transpose() if transpose else df

    wrap_columns = ["Description"]
    wrap_width = 40
    for col in wrap_columns:
        if col in df_to_show.columns:
            df_to_show[col] = (
                df_to_show[col]
                .astype(str)
                .apply(lambda x: textwrap.fill(x, width=wrap_width))
            )
    df_with_index = df_to_show.reset_index()
    return tabulate(
        df_with_index.values,
        headers=list(df_with_index.columns),
        tablefmt="fancy_grid",
        floatfmt=f".{decimal}f",
    )


def convert_time_columns(df: pd.DataFrame, time_unit: str) -> pd.DataFrame:
    """
    Convert time column values based on the specified time unit.
    Uses the Unit column to identify which columns contain time data.
    """

    if time_unit not in config.TIME_UNITS or "Unit" not in df.columns:
        return df

    # Avoid modifying the original
    df_copy = df.copy()
    time_rows = df_copy["Unit"].str.lower().str.contains("ns", na=False)
    time_value_columns = ["Avg", "Min", "Max"]

    for col in time_value_columns:
        if col in df_copy.columns and time_rows.any():
            try:
                numeric_values = pd.to_numeric(
                    df_copy.loc[time_rows, col], errors="coerce"
                )
                df_copy.loc[time_rows, col] = (
                    numeric_values / config.TIME_UNITS[time_unit]
                )
            except Exception:
                pass

    # Update the Unit column
    if time_rows.any():
        df_copy.loc[time_rows, "Unit"] = time_unit

    return df_copy


def has_time_data(df: pd.DataFrame) -> bool:
    """
    Check if the dataframe contains time data by looking at the Unit column.
    """

    if "Unit" not in df.columns:
        return False
    # NOTE: "ns" / "NS" / "nS" / "Ns" are reserved for Nanosec time unit
    return bool(df["Unit"].str.lower().str.contains("ns", na=False).any())


def is_roofline_shown(
    args: argparse.Namespace,
    runs: dict[str, Any],
    output: Optional[TextIO],
    panel: dict[str, Any],
    roof_plot: Optional[str],
    hidden_cols: list[str],
) -> bool:
    has_roofline_style = any(
        data_source.get(table_type, {}).get("cli_style") == "Roofline"
        for data_source in panel["data source"]
        for table_type in data_source
    )

    if not has_roofline_style or (
        args.filter_metrics
        and "4" not in args.filter_metrics
        and "roof" not in args.filter_metrics
    ):
        return False

    print(f"\n{'=' * 80}", file=output)
    print("4. Roofline", file=output)
    print("=" * 80, file=output)

    # Display roofline metrics for each run
    for run_path, workload in runs.items():
        if hasattr(workload, "roofline_metrics") and workload.roofline_metrics:
            print(
                "\n(4.1) Per-Kernel Roofline Metrics and (4.2) AI Plot Points",
                file=output,
            )
            print("-" * 80, file=output)

            kernel_top_df = workload.dfs.get(1, pd.DataFrame())
            if not kernel_top_df.empty:
                kernel_name_shortener(kernel_top_df, args.kernel_verbose)

            # Display roofline metrics
            for kernel_id, metrics in workload.roofline_metrics.items():
                if not kernel_top_df.empty and kernel_id in kernel_top_df.index:
                    kernel_name = kernel_top_df.loc[kernel_id, "Kernel_Name"]
                    kernel_pct = (
                        kernel_top_df.loc[kernel_id, "Pct"]
                        if "Pct" in kernel_top_df.columns
                        else 0
                    )
                else:
                    kernel_name = metrics.get("name", f"Kernel {kernel_id}")
                    kernel_pct = 0

                display_name = (
                    kernel_name[:80] + "..." if len(kernel_name) > 80 else kernel_name
                )
                print(
                    f"\nKernel {kernel_id}: {display_name} ({kernel_pct:.1f}%)",
                    file=output,
                )

                base_indent = "  "
                table_indent_prefix = f"{base_indent}|   "
                print(f"{base_indent}|", file=output)

                tables = {
                    401: (
                        "4.1 Roofline Rate Metrics:",
                        metrics.get("ai_table", pd.DataFrame()),
                    ),
                    402: (
                        "4.2 Roofline AI Plot Points:",
                        metrics.get("calc_table", pd.DataFrame()),
                    ),
                }

                for table_id, (table_name, df) in tables.items():
                    if df.empty:
                        continue

                    print(f"{base_indent}├─ {table_name}", file=output)

                    # Remove hidden columns
                    display_df = df.copy()
                    for col in hidden_cols:
                        if col in display_df.columns:
                            display_df = display_df.drop(columns=[col])

                    table_string = get_table_string(
                        display_df, transpose=False, decimal=args.decimal
                    )
                    indented_table = textwrap.indent(table_string, table_indent_prefix)
                    print(indented_table, file=output)

        else:
            print("\nNo per-kernel metrics available", file=output)

    # Show the roofline plot
    if roof_plot:
        show_roof_plot(roof_plot)
    return True


def process_table_data(
    args: argparse.Namespace,
    runs: dict[str, Any],
    table_config: dict[str, Any],
    table_type: str,
    comparable_columns: list[str],
    hidden_cols: list[str],
) -> pd.DataFrame:
    # take the 1st run as baseline
    base_run, base_data = next(iter(runs.items()))
    base_df = base_data.dfs[table_config["id"]]

    if args.time_unit and has_time_data(base_df):
        base_df = convert_time_columns(base_df, args.time_unit)

    result_df = pd.DataFrame(index=base_df.index)

    for header in base_df.columns:
        # Skip filtered columns
        if (
            table_type != "raw_csv_table"
            and args.cols
            and base_df.columns.get_loc(header) not in args.cols
        ):
            continue

        if header in hidden_cols:
            continue

        if header not in comparable_columns:
            # Process columns that are not comparable across runs.
            if (
                table_type == "raw_csv_table"
                and table_config["source"]
                in ["pmc_kernel_top.csv", "pmc_dispatch_info.csv"]
                and header == "Kernel_Name"
            ):
                # NB: the width of kernel name might depend
                # on the header of the table.
                width = 40 if table_config["source"] == "pmc_kernel_top.csv" else 80
                max_rows = 3 if table_config["source"] == "pmc_kernel_top.csv" else 4

                adjusted_names = base_df["Kernel_Name"].apply(
                    lambda x: string_multiple_lines(x, width, max_rows)
                )
                result_df = pd.concat([result_df, adjusted_names], axis=1)

            elif table_type == "raw_csv_table" and header == "Info":
                for run_data in runs.values():
                    cur_df = run_data.dfs[table_config["id"]]
                    result_df = pd.concat([result_df, cur_df[header]], axis=1)
            else:
                result_df = pd.concat([result_df, base_df[header]], axis=1)
        else:
            # Process columns that can be compared across runs.
            for run_name, run_data in runs.items():
                cur_df = run_data.dfs[table_config["id"]]

                if args.time_unit and has_time_data(base_df):
                    cur_df = convert_time_columns(cur_df, args.time_unit)

                if (table_type == "raw_csv_table") or (
                    table_type == "metric_table" and header not in hidden_cols
                ):
                    if run_name != base_run:
                        # Calculate percentage difference between current and
                        # base dataframe.
                        base_series = pd.to_numeric(
                            base_df[header], errors="coerce"
                        ).fillna(0.0)
                        cur_series = pd.to_numeric(
                            cur_df[header], errors="coerce"
                        ).fillna(0.0)

                        # Calculate absolute and percentage differences
                        absolute_diff = (cur_series - base_series).round(args.decimal)
                        percentage_diff = (
                            absolute_diff / base_series.replace(0, 1) * 100
                        ).round(args.decimal)

                        if args.verbose >= 2:
                            console_log("---------", header, percentage_diff)

                        # Format as "value (percentage%)"
                        formatted_diff = (
                            cur_series.round(args.decimal).astype(str)
                            + " ("
                            + percentage_diff.astype(str)
                            + "%)"
                        )

                        result_df = pd.concat([result_df, formatted_diff], axis=1)

                        # DEBUG: When in a CI setting and flag is set,
                        #       then verify metrics meet threshold
                        #       requirement
                        if (
                            header in ["Value", "Count", "Avg"]
                            and percentage_diff.abs().gt(args.report_diff).any()
                        ):
                            result_df["Abs Diff"] = absolute_diff

                            if args.report_diff:
                                violation_idx = percentage_diff.index[
                                    percentage_diff.abs() > args.report_diff
                                ]
                                console_warning(
                                    f"Dataframe diff exceeds {args.report_diff}% "
                                    "threshold requirement\n"
                                    f"See metric {violation_idx.to_numpy()}"
                                )
                                console_warning(result_df)
                    else:
                        # Base run - just add the rounded values
                        cur_df_copy = copy.deepcopy(cur_df)
                        cur_df_copy[header] = [
                            (round(float(x), args.decimal) if x != "" else x)
                            for x in base_df[header]
                        ]
                        result_df = pd.concat([result_df, cur_df_copy[header]], axis=1)

    return result_df


def format_table_output(
    args: argparse.Namespace,
    table_config: dict[str, Any],
    df: pd.DataFrame,
    table_type: str,
    runs: dict[str, Any],
    csv_dir: Optional[Path] = None,
) -> str:
    """Format table for output, handling special cases and saving to files if needed."""

    table_id_str = f"{table_config['id'] // 100}.{table_config['id'] % 100}"
    content = ""

    # Check if any column in df is empty
    is_empty_columns_exist = any(
        df.replace("", None).iloc[:, col_idx].isnull().all()
        for col_idx in range(len(df.columns))
    )

    # Do not print the table if any column is empty
    if is_empty_columns_exist:
        title = table_config.get("title", "")
        console_log(f"Not showing table with empty column(s): {table_id_str} {title}")
        return content

    if "title" in table_config and table_config["title"]:
        content += f"{table_id_str} {table_config['title']}\n"

    if args.output_format == "csv" and csv_dir and csv_dir.is_dir():
        if "title" in table_config and table_config["title"]:
            table_id_str += f"_{table_config['title']}"

        csv_filename = csv_dir / f"{table_id_str.replace(' ', '_')}.csv"
        df.to_csv(csv_filename, index=False)
        console_warning(f"Created file: {csv_filename}")

    # Only show top N kernels (as specified in --max-kernel-num)
    # in "Top Stats" section
    if table_type == "raw_csv_table" and table_config["source"] in [
        "pmc_kernel_top.csv",
        "pmc_dispatch_info.csv",
    ]:
        df = df.head(args.max_stat_num)
    # NB:
    # "columnwise: True" is a special attr of a table/df
    # For raw_csv_table, such as system_info, we transpose the
    # df when load it, because we need those items in column.
    # For metric_table, we only need to show the data in column
    # fash for now.
    transpose = table_type != "raw_csv_table" and table_config.get("columnwise", False)

    # enable mem_chart only with single run
    if (
        table_config.get("cli_style") == "mem_chart"
        and len(runs) == 1
        and "Metric" in df.columns
        and "Value" in df.columns
    ):
        mem_data = (
            pd.DataFrame([df["Metric"], df["Value"]])
            .transpose()
            .set_index("Metric")
            .to_dict()["Value"]
        )
        content += mem_chart.plot_mem_chart("", args.normal_unit, mem_data) + "\n"
    else:
        content += (
            get_table_string(df, transpose=transpose, decimal=args.decimal) + "\n"
        )

    return content


def show_all(
    args: argparse.Namespace,
    runs: dict[str, Any],
    arch_configs: schema.ArchConfig,
    output: Optional[TextIO],
    profiling_config: dict[str, Any],
    roof_plot: Optional[str] = None,
) -> None:
    """
    Show all panels with their data in plain text mode.
    """
    comparable_columns = parser.build_comparable_columns(args.time_unit)
    raw_filter_panel_ids = profiling_config.get("filter_blocks", [])
    csv_dir = None

    if isinstance(raw_filter_panel_ids, dict):
        # For backward compatibility
        raw_filter_panel_ids = [
            name
            for name, table_type in raw_filter_panel_ids.items()
            if table_type == "metric_id"
        ]

    panel_alias = get_panel_alias()  # alias -> panel_id (string or int)

    filter_panel_ids = set()
    for bid in raw_filter_panel_ids:
        bid_s = str(bid)

        # If it's not already an ID, resolve alias -> ID
        if not METRIC_ID_RE.match(bid_s):
            try:
                bid_s = str(panel_alias[bid_s])
            except KeyError as e:
                raise KeyError(f"Unknown panel alias: {bid_s!r}") from e

        file_id, _, _ = convert_metric_id_to_panel_info(bid_s)
        if file_id is not None:
            filter_panel_ids.add(int(file_id))

    if args.include_cols:
        hidden_cols = list(set(config.HIDDEN_COLUMNS_CLI) - set(args.include_cols))
    else:
        hidden_cols = config.HIDDEN_COLUMNS_CLI

    if args.output_format == "csv":
        if args.output_name:
            csv_dir = Path(f"{args.output_name}")
        else:
            csv_dir = Path(f"rocprof_compute_{get_uuid()}")
        if not csv_dir.exists():
            csv_dir.mkdir()

    for panel_id, panel in arch_configs.panel_configs.items():
        # Skip panels that don't support baseline comparison
        if len(args.path) > 1 and panel_id in config.HIDDEN_SECTIONS:
            continue

        if panel_id == 400 and not is_roofline_shown(
            args, runs, output, panel, roof_plot, hidden_cols
        ):
            continue

        panel_content = ""  # store content of all data_source from one panel

        for data_source in panel["data source"]:
            for table_type, table_config in data_source.items():
                # Block-filter logic:
                # - If analysis used --filter-metrics, ignore profiling block filters
                # - If profiling had block filters, only show selected tables/panels
                # - Always show panels with id <= 100
                if (
                    not args.filter_metrics
                    and filter_panel_ids
                    and table_config["id"] not in filter_panel_ids
                    and panel_id not in filter_panel_ids
                    and panel_id > 100
                ):
                    table_id_str = (
                        f"{table_config['id'] // 100}.{table_config['id'] % 100}"
                    )

                    console_log(
                        f"Not showing table not selected during profiling: "
                        f"{table_id_str} {table_config['title']}"
                    )
                    continue

                # Metrics baseline comparison mode: only show common metrics across runs
                # We cannot guarantee that all runs have the same metrics.
                if (
                    table_type == "metric_table"
                    and "Metric" in table_config["header"].values()
                    and len(runs) > 1
                ):
                    # Find common metrics across all runs
                    common_metrics: set[str] = set()
                    for run_data in runs.values():
                        run_metrics = set(run_data.dfs[table_config["id"]]["Metric"])
                        common_metrics = (
                            run_metrics
                            if not common_metrics
                            else common_metrics & run_metrics
                        )

                    # Apply common metrics across all runs
                    # Reindex all runs based on first run
                    initial_index = None
                    for run_data in runs.values():
                        run_data.dfs[table_config["id"]] = run_data.dfs[
                            table_config["id"]
                        ].loc[lambda df: df["Metric"].isin(common_metrics)]
                        if initial_index is None:
                            initial_index = run_data.dfs[table_config["id"]].index
                        else:
                            run_data.dfs[table_config["id"]].index = initial_index

                processed_df = process_table_data(
                    args,
                    runs,
                    table_config,
                    table_type,
                    comparable_columns,
                    hidden_cols,
                )

                if not processed_df.empty:
                    panel_content += format_table_output(
                        args, table_config, processed_df, table_type, runs, csv_dir
                    )

        if panel_content:
            print(f"\n{'-' * 80}", file=output)
            print(f"{panel_id // 100}. {panel['title']}", file=output)
            print(panel_content, file=output)


def show_roof_plot(roof_plot: str) -> None:
    # TODO: short term solution to display roofline plot
    print(f"\n{'-' * 80}")
    print("4. Roofline")
    print("4.3 Roofline Plot")

    if roof_plot:
        print(roof_plot)
    else:
        console_error(
            "Cannot create roofline plot for CLI with incomplete/missing "
            "roofline profiling data.",
            exit=False,
        )


def show_kernel_stats(
    args: argparse.Namespace,
    runs: dict[str, Any],
    arch_configs: schema.ArchConfig,
    output: Optional[TextIO],
) -> None:
    """
    Show the kernels and dispatches from "Top Stats" section.
    """

    for panel_id, panel in arch_configs.panel_configs.items():
        for data_source in panel["data source"]:
            for table_type, table_config in data_source.items():
                for run, data in runs.items():
                    single_df = data.dfs[table_config["id"]]
                    # NB:
                    #   For pmc_kernel_top.csv, have to sort here if not
                    #   sorted when load_table_data.
                    if table_config["id"] == 1:
                        print(f"\n{'-' * 80}", file=output)
                        print(
                            "Detected Kernels (sorted descending by duration)",
                            file=output,
                        )
                        display_df = pd.DataFrame()
                        display_df = pd.concat(
                            [display_df, single_df["Kernel_Name"]], axis=1
                        )
                        print(
                            get_table_string(
                                display_df, transpose=False, decimal=args.decimal
                            ),
                            file=output,
                        )

                    if table_config["id"] == 2:
                        print(f"\n{'-' * 80}", file=output)
                        print("Dispatch list", file=output)
                        print(
                            get_table_string(
                                single_df, transpose=False, decimal=args.decimal
                            ),
                            file=output,
                        )
