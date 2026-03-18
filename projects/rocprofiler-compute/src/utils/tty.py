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
import shutil
import textwrap
from pathlib import Path
from typing import Any, Optional, TextIO

import pandas as pd
from tabulate import tabulate

import config
from utils import mem_chart, parser, schema
from utils.kernel_name_shortener import (
    kernel_name_shortener,
)
from utils.logger import console_error, console_log, console_warning
from utils.utils import (
    METRIC_ID_RE,
    NS_TO_MS,
    CallTreeNode,
    convert_metric_id_to_panel_info,
    get_panel_alias,
    get_uuid,
    simplify_kernel_name,
)

KERNEL_NAME_WRAP_WIDTH = 40


def wrap_kernel_name(name: str) -> str:
    """Wrap a kernel name at KERNEL_NAME_WRAP_WIDTH for table display."""
    return textwrap.fill(str(name), width=KERNEL_NAME_WRAP_WIDTH)


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
    df_to_show = df.transpose().copy() if transpose else df.copy()

    if "Description" in df_to_show.columns:
        df_to_show["Description"] = (
            df_to_show["Description"]
            .astype(str)
            .apply(lambda x: textwrap.fill(x, width=40))
        )
    if "Kernel_Name" in df_to_show.columns:
        df_to_show["Kernel_Name"] = (
            df_to_show["Kernel_Name"].astype(str).apply(wrap_kernel_name)
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
    time_rows = (
        df_copy["Unit"].str.lower().str.contains(r"\bns\b", na=False, regex=True)
    )
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
    return bool(
        df["Unit"].str.lower().str.contains(r"\bns\b", na=False, regex=True).any()
    )


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

    # Check if any run has valid roofline data (already validated in analysis_base.py)
    # This check determines whether to display roofline section in the output report
    if not any(
        hasattr(workload, "roofline_peaks") and not workload.roofline_peaks.empty
        for workload in runs.values()
    ):
        # roofline_peaks is empty, meaning CSV validation failed earlier.
        # Skip displaying this section entirely (error already logged).
        return False

    print(f"\n{'-' * 80}", file=output)
    print("4. Roofline", file=output)

    # Display roofline metrics for each run
    for run_path, workload in runs.items():
        if hasattr(workload, "roofline_metrics") and workload.roofline_metrics:
            print(
                "\n(4.1) Per-Kernel Roofline Metrics and (4.2) AI Plot Points",
                file=output,
            )

            kernel_top_df = workload.dfs.get(1, pd.DataFrame())
            if not kernel_top_df.empty:
                kernel_name_shortener(kernel_top_df, args.kernel_verbose)

            # Display roofline metrics
            for kernel_id, metrics in workload.roofline_metrics.items():
                if not kernel_top_df.empty and kernel_id in kernel_top_df.index:
                    kernel_name = kernel_top_df.loc[kernel_id, "Kernel_Name"]
                    kernel_pct = (
                        kernel_top_df.loc[kernel_id, "Percent"]
                        if "Percent" in kernel_top_df.columns
                        else 0
                    )
                else:
                    kernel_name = metrics.get("name", f"Kernel {kernel_id}")
                    kernel_pct = 0

                print(
                    f"\nKernel {kernel_id}: "
                    f"{wrap_kernel_name(kernel_name)}"
                    f" ({kernel_pct:.1f}%)",
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


def show_torch_operator_table(operator_name: str, df: pd.DataFrame) -> None:
    """Display torch operator data in a properly formatted table."""
    if df is None or df.empty:
        console_log(f"No data available for operator: {operator_name}")
        return

    console_log(f"\n{operator_name}")
    console_log("=" * len(operator_name))

    # Create a copy for display formatting
    display_df = df.copy()

    # Max display width per column type; Kernel_Name is skipped (show wrapped).
    column_widths = {
        "Operator_Name": 40,
        "Context": 35,
        "default": 20,  # fallback for all other string columns
    }

    # Truncate string columns; wrap Kernel_Name (full, no truncation)
    for col in display_df.columns:
        if display_df[col].dtype != "object":
            continue  # skip numeric columns
        if col == "Kernel_Name":
            display_df[col] = (
                display_df[col].astype(str).apply(wrap_kernel_name)
            )  # wrap full name instead of truncating
            continue
        max_width = column_widths.get(
            col, column_widths["default"]
        )  # use column-specific width or fallback
        display_df[col] = (
            display_df[col]
            .astype(str)  # ensure type is string before continuing text operations
            .apply(
                lambda x: (
                    string_multiple_lines(
                        x, max_width, 2
                    )  # split into at most 2 lines, add "..." if still too long
                    if len(x) > max_width
                    else x  # leave short values as-is
                )
            )
        )

    # Reset index for row numbering
    display_df = display_df.reset_index(drop=True)

    # Use tabulate for consistent formatting (no maxcolwidths: natural column width)
    table_str = tabulate(
        display_df,
        headers=display_df.columns,
        tablefmt="fancy_grid",
        showindex=True,
        floatfmt=".2f",
    )

    console_log(table_str)


def list_torch_operators(
    workload_path: str,
    call_trees: dict[str, CallTreeNode],
) -> None:
    """Display PyTorch operators as a unified call tree grouped by source location."""
    if not call_trees:
        console_warning(
            "No PyTorch operator data found. "
            "Please ensure profiling was done with --torch-trace option."
        )
        return

    print(f"\n{'=' * 80}")
    print(f"PyTorch Operator Call Tree: {workload_path}")
    print("Grouped by source location, sorted by total GPU kernel duration.")
    print(f"{'=' * 80}")
    show_call_tree(call_trees)
    print(f"\n{'=' * 80}")


def format_stats(launches: int, duration_ms: float) -> str:
    """Format launch count and duration as an inline parenthesized string."""
    if duration_ms < 0.01:
        formatted_duration = f"{duration_ms * 1000:.2f} us"
    else:
        formatted_duration = f"{duration_ms:.2f} ms"
    return f"(kernel_launches: {launches}, total_duration: {formatted_duration})"


def get_tree_wrap_width(min_width: int = 72, max_width: int = 120) -> int:
    """Pick wrap width based on terminal size to avoid terminal hard-wrap artifacts."""
    terminal_cols = shutil.get_terminal_size((max_width, 20)).columns
    safe_width = max(terminal_cols - 2, min_width)
    return min(safe_width, max_width)


def print_wrapped_tree_line(
    prefix: str,
    body: str,
    width: Optional[int] = None,
    break_long_words: bool = False,
) -> None:
    """Print a tree line and wrap continuation lines to preserve indentation."""
    effective_width = get_tree_wrap_width() if width is None else width
    print(
        textwrap.fill(
            body,
            width=effective_width,
            initial_indent=prefix,
            subsequent_indent=" " * len(prefix),
            break_long_words=break_long_words,
            break_on_hyphens=False,
        )
    )


def print_wrapped_kernel_line(
    prefix: str,
    kernel_name: str,
    suffix: str,
    width: Optional[int] = None,
    continuation_prefix: str = "",
) -> None:
    """Wrap long kernel names while keeping suffix attached to final name chunk."""
    effective_width = get_tree_wrap_width() if width is None else width
    content_width = max(effective_width - len(prefix), 20)

    inline = f"{kernel_name} {suffix}"
    if len(inline) <= content_width:
        print(f"{prefix}{inline}")
        return

    # Reserve room so suffix is never detached on its own line.
    name_width = max(content_width - len(suffix) - 1, 8)
    wrapped_name = textwrap.wrap(
        kernel_name,
        width=name_width,
        break_long_words=True,
        break_on_hyphens=False,
    )
    if not wrapped_name:
        print(f"{prefix}{suffix}")
        return

    # Build continuation with vertical pipes from parent levels
    # Preserve parent pipes but replace branch character with spaces
    if len(continuation_prefix) > 0:
        # continuation_prefix has pipes, add spaces for branch chars
        spaces_needed = len(prefix) - len(continuation_prefix)
        continuation = continuation_prefix + " " * spaces_needed
    else:
        # No parent pipes, just use spaces matching the prefix
        continuation = " " * len(prefix)

    for i, chunk in enumerate(wrapped_name):
        if i == 0:
            if len(wrapped_name) == 1:
                print(f"{prefix}{chunk} {suffix}")
            else:
                print(f"{prefix}{chunk}")
        elif i == len(wrapped_name) - 1:
            print(f"{continuation}{chunk} {suffix}")
        else:
            print(f"{continuation}{chunk}")


def show_call_tree(call_trees: dict[str, CallTreeNode]) -> None:
    """Print the unified call tree grouped by source location."""
    sorted_locations = sorted(
        call_trees.items(), key=lambda kv: kv[1].total_duration_ms, reverse=True
    )
    for i, (location, root) in enumerate(sorted_locations):
        if i > 0:
            print(f"\n{'- ' * 40}")
        stats = format_stats(root.kernel_launches, root.total_duration_ms)
        print(f"\n{location} {stats}")
        for child in sorted(
            root.children.values(),
            key=lambda c: c.total_duration_ms,
            reverse=True,
        ):
            print_operator_node(child)


def print_operator_node(
    node: CallTreeNode, is_last: bool = True, parent_pipes: str = ""
) -> None:
    # Build indent with vertical pipes for parent levels
    indent = parent_pipes
    is_branching = len(node.children) + len(node.kernels) > 1

    # Use ├─ for non-last items, └─ for last items
    branch_char = "└─ " if is_last else "├─ "
    node_prefix = f"{indent}{branch_char}"

    if is_branching:
        stats = format_stats(node.kernel_launches, node.total_duration_ms)
        print_wrapped_tree_line(node_prefix, f"{node.name} {stats}")
    else:
        print_wrapped_tree_line(node_prefix, node.name)

    # Build new parent_pipes for children
    if is_last:
        new_parent_pipes = parent_pipes + "   "  # 3 spaces
    else:
        new_parent_pipes = parent_pipes + "|  "  # pipe + 2 spaces

    # Process child nodes
    children = sorted(
        node.children.values(), key=lambda c: c.total_duration_ms, reverse=True
    )
    for i, child in enumerate(children):
        # A child is last if it's the final child AND there are no kernels after it
        child_is_last = (i == len(children) - 1) and (len(node.kernels) == 0)
        print_operator_node(child, is_last=child_is_last, parent_pipes=new_parent_pipes)

    # Process kernels
    for i, (kernel_name, kernel_stats) in enumerate(
        sorted(
            node.kernels.items(),
            key=lambda kv: kv[1].total_duration_ns,
            reverse=True,
        )
    ):
        launches = kernel_stats.launches
        duration_ns = kernel_stats.total_duration_ns
        kernel_id = kernel_stats.kernel_id
        id_suffix = f" (id {kernel_id})" if kernel_id is not None else ""
        display_name = simplify_kernel_name(kernel_name)
        total_ms = duration_ns * NS_TO_MS
        stats = format_stats(launches, total_ms)

        # Last kernel gets └─, others get ├─
        kernel_is_last = i == len(node.kernels) - 1
        kernel_branch_char = "└─ " if kernel_is_last else "├─ "
        kernel_prefix = f"{new_parent_pipes}{kernel_branch_char}"

        print_wrapped_kernel_line(
            kernel_prefix,
            display_name,
            f"{id_suffix} {stats}".strip(),
            continuation_prefix=new_parent_pipes,
        )


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
                result_df = pd.concat([result_df, base_df["Kernel_Name"]], axis=1)
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
                        # Calculate percent difference between current and
                        # base dataframe.
                        base_series = pd.to_numeric(
                            base_df[header], errors="coerce"
                        ).fillna(0.0)
                        cur_series = pd.to_numeric(
                            cur_df[header], errors="coerce"
                        ).fillna(0.0)

                        # Calculate absolute and percent differences
                        absolute_diff = (cur_series - base_series).round(args.decimal)
                        percent_diff = (
                            absolute_diff / base_series.replace(0, 1) * 100
                        ).round(args.decimal)

                        if args.verbose >= 2:
                            console_log("---------", header, percent_diff)

                        # Format as "value (percent%)"
                        formatted_diff = (
                            cur_series.round(args.decimal).astype(str)
                            + " ("
                            + percent_diff.astype(str)
                            + "%)"
                        )

                        result_df = pd.concat([result_df, formatted_diff], axis=1)

                        # DEBUG: When in a CI setting and flag is set,
                        #       then verify metrics meet threshold
                        #       requirement
                        if (
                            header in ["Value", "Count", "Avg"]
                            and percent_diff.abs().gt(args.report_diff).any()
                        ):
                            result_df["Abs Diff"] = absolute_diff

                            if args.report_diff:
                                violation_idx = percent_diff.index[
                                    percent_diff.abs() > args.report_diff
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
                            (round(float(x), args.decimal) if x != "N/A" else x)
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
        df.replace(["", "N/A"], None).iloc[:, col_idx].isnull().all()
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
            pd
            .DataFrame([df["Metric"], df["Value"]])
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

    # Check for valid roofline data once (used to skip roofline tables in the loop)
    has_valid_roofline = any(
        hasattr(workload, "roofline_peaks") and not workload.roofline_peaks.empty
        for workload in runs.values()
    )
    roofline_warning_shown = False

    # True if roofline (block 4) is in the active filter
    # or no filter is applied
    roofline_in_filter = (
        any(str(m).split(".")[0] == "4" for m in args.filter_metrics)
        if args.filter_metrics
        else (not filter_panel_ids or 400 in filter_panel_ids)
    )

    for panel_id, panel in arch_configs.panel_configs.items():
        # NOTE: Experimental Feature Toggle
        # HARD GATE: Block 30 (panel 3000) requires membw_analysis flag
        if panel_id == 3000 and not args.membw_analysis:
            continue

        # Skip panels that don't support baseline comparison
        if len(args.path) > 1 and panel_id in config.HIDDEN_SECTIONS:
            continue

        # Handle roofline panel (400) with custom display logic
        if panel_id == 400:
            _ = is_roofline_shown(args, runs, output, panel, roof_plot, hidden_cols)

        panel_content = ""  # store content of all data_source from one panel

        for data_source in panel["data source"]:
            for table_type, table_config in data_source.items():
                # Emit warnings for roofline tables (401, 402)
                # if roofline data is invalid
                if table_config["id"] in [401, 402] and not has_valid_roofline:
                    if not roofline_warning_shown and roofline_in_filter:
                        console_warning(
                            "Roofline",
                            "Not showing roofline table due to invalid roofline data",
                        )
                        roofline_warning_shown = True

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

        # Roofline printing is handled separately above in is_roofline_shown
        if panel_content and table_config["id"] not in [401, 402]:
            print(f"\n{'-' * 80}", file=output)
            print(f"{panel_id // 100}. {panel['title']}", file=output)
            print(panel_content, file=output)


def show_roof_plot(roof_plot: str) -> None:
    # TODO: short term solution to display roofline plot
    print("4.3 Roofline Plot:")

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
