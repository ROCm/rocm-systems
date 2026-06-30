# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import copy
import math
import shutil
import textwrap
from typing import Any, Optional, TextIO

import pandas as pd
from tabulate import tabulate

import config
from utils import mem_chart_gfx9, mem_chart_gfx11, parser, schema
from utils.kernel_name_shortener import (
    kernel_name_shortener,
)
from utils.logger import console_error, console_log, console_warning
from utils.metrics.aggregation import calc_pct_of_peak
from utils.utils_analysis import (
    NS_TO_MS,
    CallTreeNode,
    build_operator_summary,
    get_bw_scale_and_unit,
    simplify_kernel_name,
)
from utils.utils_common import convert_filter_blocks_to_panel_ids


def _tty_view_is_table(args: argparse.Namespace) -> bool:
    """True when ``--view table`` was given (plain tables; ignore cli_style)."""
    return getattr(args, "view", None) == "table"


KERNEL_NAME_WRAP_WIDTH = 40


def wrap_kernel_name(name: str) -> str:
    """Wrap a kernel name at KERNEL_NAME_WRAP_WIDTH for table display."""
    return textwrap.fill(str(name), width=KERNEL_NAME_WRAP_WIDTH)


def scale_bw_columns(
    df: pd.DataFrame, value_columns: list[str], decimal: int = 2
) -> pd.DataFrame:
    """Scale Bytes/s rows to human-readable units; recalculate Percent of Peak."""
    if "Unit" not in df.columns:
        return df

    df_copy = df.copy()

    bw_rows = df_copy["Unit"].str.lower().str.contains("bytes/s", na=False)
    if not bw_rows.any():
        return df_copy

    pct_cols = ["Percent of Peak"]
    value_col = "Value" if "Value" in df_copy.columns else "Avg"
    peak_col = "Peak (Empirical)" if "Peak (Empirical)" in df_copy.columns else "Peak"

    for idx in df_copy.index[bw_rows]:
        # Determine scale from the primary value column
        primary_value = None
        for col in ["Value", "Avg"]:
            if col in df_copy.columns:
                try:
                    val = df_copy.loc[idx, col]
                    if pd.notna(val) and val != "N/A":
                        primary_value = float(val)
                        break
                except (ValueError, TypeError):
                    continue

        if primary_value is None or primary_value == 0:
            continue

        divisor, unit = get_bw_scale_and_unit(primary_value)

        # Scale all numeric bandwidth columns with the same divisor
        for col in value_columns:
            if col not in df_copy.columns:
                continue
            try:
                val = df_copy.loc[idx, col]
                if pd.notna(val) and val != "N/A":
                    df_copy.loc[idx, col] = round(float(val) / divisor, decimal)
            except (ValueError, TypeError):
                pass

        for pct_col in pct_cols:
            if pct_col in df_copy.columns:
                pct = calc_pct_of_peak(
                    df_copy.loc[idx, value_col], df_copy.loc[idx, peak_col]
                )
                if pct is not None:
                    df_copy.loc[idx, pct_col] = round(pct, decimal)

        df_copy.loc[idx, "Unit"] = unit

    return df_copy


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


def list_ml_operators(
    workload_path: str,
    call_trees: dict[str, CallTreeNode],
    framework_label: str = "PyTorch",
) -> None:
    """Display operators as a unified call tree grouped by source location.

    ``framework_label`` sets the heading text (for example "PyTorch" or
    "Triton").
    """
    if not call_trees:
        print(f"\n{framework_label} Operators in: {workload_path}")
        print("Total: 0 operators")
        return

    print(f"\n{'=' * 80}")
    print(f"{framework_label} Operator Call Tree: {workload_path}")
    print("Grouped by source location, sorted by total GPU kernel duration.")
    print(f"{'=' * 80}")
    show_call_tree(call_trees)
    show_operator_summary(build_operator_summary(call_trees))
    print(f"\n{'=' * 80}")


def format_duration(duration_ms: Optional[float]) -> str:
    """Format a duration in ms; switch to us below 0.01 ms; None/NaN render as N/A."""
    if duration_ms is None:
        return "N/A"
    if isinstance(duration_ms, float) and math.isnan(duration_ms):
        return "N/A"
    if duration_ms < 0.01:
        return f"{duration_ms * 1000:.2f} us"
    return f"{duration_ms:.2f} ms"


def format_node_stats(node: CallTreeNode) -> str:
    """Format operator-node stats (calls, dispatches, total, dispatch_mean/min/max).

    dispatch_mean / dispatch_min / dispatch_max are per kernel dispatch.
    The "calls:" segment is omitted when invocation_ids is empty (location
    roots and frames recorded without Context_Id).
    """
    mean_ms = (
        node.mean_dispatch_ns * NS_TO_MS if node.mean_dispatch_ns is not None else None
    )
    min_ms = (
        node.min_dispatch_ns * NS_TO_MS if node.min_dispatch_ns is not None else None
    )
    max_ms = (
        node.max_dispatch_ns * NS_TO_MS if node.max_dispatch_ns is not None else None
    )
    calls_prefix = f"calls: {node.call_count}, " if len(node.invocation_ids) > 0 else ""
    return (
        f"({calls_prefix}"
        f"dispatches: {node.kernel_launches}, "
        f"total: {format_duration(node.total_duration_ms)}, "
        f"dispatch_mean: {format_duration(mean_ms)}, "
        f"dispatch_min: {format_duration(min_ms)}, "
        f"dispatch_max: {format_duration(max_ms)})"
    )


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
        stats = format_node_stats(root)
        print(f"\n{location} {stats}")
        for child in sorted(
            root.children.values(),
            key=lambda c: c.total_duration_ms,
            reverse=True,
        ):
            print_operator_node(child)


def show_operator_summary(summary_df: pd.DataFrame) -> None:
    """Print a flat per-operator summary table alongside the call tree.

    - Rendered as a fancy_grid bordered table via tabulate, matching the
      rest of the analyze CLI.

    - Operator column wraps long paths; numeric columns size to content.

    - A header line above the table explains the aggregation so column
      names can stay short.

    - Time cells are formatted per-cell via format_duration (auto-switching
      between ms and us). NaN renders as "N/A".
    """
    if summary_df is None or summary_df.empty:
        print("\nOperator summary: (no operators with recorded dispatches)")
        return

    operator_name_wrap_width = 72

    # (DataFrame column, display label) pairs. DataFrame names match the
    # schema produced by build_operator_summary; display labels are short
    # because the header line below explains the semantics and time cells
    # self-label their unit.
    column_map = [
        ("Operator", "Operator"),
        ("Calls", "Calls"),
        ("Dispatches", "Dispatches"),
        ("Total_GPU", "Total"),
        ("Pct_Total_GPU", "% Total"),
        ("Mean_Per_Call", "Mean/Call"),
        ("Mean_Per_Dispatch", "Mean"),
        ("Min_Dispatch", "Min"),
        ("Max_Dispatch", "Max"),
    ]
    source_cols = [c for c, _ in column_map]
    headers = [h for _, h in column_map]
    time_cols = (
        "Total_GPU",
        "Mean_Per_Call",
        "Mean_Per_Dispatch",
        "Min_Dispatch",
        "Max_Dispatch",
    )

    display_df = summary_df[source_cols].copy()
    display_df["Operator"] = (
        display_df["Operator"]
        .astype(str)
        .apply(lambda s: textwrap.fill(s, width=operator_name_wrap_width))
    )
    for col in time_cols:
        display_df[col] = display_df[col].apply(format_duration)

    # Time columns are pre-formatted strings; only % Total still needs floatfmt.
    floatfmt = ("", "", "", "", ".2f", "", "", "", "")
    colalign = ("left",) + ("right",) * (len(headers) - 1)

    print(
        "\nOperator summary (Min/Max/Mean are per-dispatch over the subtree; "
        "sorted by Total):"
    )
    print(
        tabulate(
            display_df.values,
            headers=headers,
            tablefmt="fancy_grid",
            floatfmt=floatfmt,
            colalign=colalign,
            missingval="N/A",
        )
    )


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
        print_wrapped_tree_line(node_prefix, f"{node.name} {format_node_stats(node)}")
    else:
        if len(node.invocation_ids) > 0:
            suffix = f" (calls: {node.call_count})"
        else:
            suffix = ""
        print_wrapped_tree_line(node_prefix, f"{node.name}{suffix}")

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
        stats = f"(dispatches: {launches}, total: {format_duration(total_ms)})"

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


def _safe_round_value(
    value: object,
    decimal: int,
) -> object:
    """Round *value* to *decimal* places, returning ``"N/A"`` on failure."""
    if value == "N/A":
        return value
    try:
        return round(float(value), decimal)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return "N/A"


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

                if header not in cur_df.columns:
                    continue

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
                            _safe_round_value(x, args.decimal) for x in base_df[header]
                        ]
                        result_df = pd.concat([result_df, cur_df_copy[header]], axis=1)

    return result_df


def _gfx115_mem_chart_heading(panel: Optional[dict[str, Any]], normal_unit: str) -> str:
    """Section number from ``panel id // 100`` (panel 300 → ``3. Memory Chart``)."""
    panel_id = int((panel or {}).get("id", 300))
    return mem_chart_gfx11.format_mem_chart_heading(normal_unit, panel_id=panel_id)


def _panel_is_mem_chart_only(panel: dict[str, Any]) -> bool:
    """True when every table uses ``cli_style: mem_chart`` (one merged chart)."""
    sources = panel.get("data source") or []
    return bool(sources) and all(
        tcfg.get("cli_style") == "mem_chart" for ds in sources for tcfg in ds.values()
    )


def format_table_output(
    args: argparse.Namespace,
    table_config: dict[str, Any],
    df: pd.DataFrame,
    table_type: str,
    runs: dict[str, Any],
    gpu_arch: Optional[str] = None,
    mem_data_override: Optional[dict[str, Any]] = None,
) -> str:
    """Format table for output, handling special cases and saving to files if needed."""

    table_id_str = f"{table_config['id'] // 100}.{table_config['id'] % 100}"
    content = ""

    # Check if any column in df is empty
    is_empty_columns_exist = any(
        df.replace(["", "N/A"], None).iloc[:, col_idx].isnull().all()
        for col_idx in range(len(df.columns))
    )

    # Do not print the table if any column is empty. PC sampling table 21.1 is
    # exempt: its source column is all N/A when the workload lacks debug info.
    if is_empty_columns_exist and table_id_str != "21.1":
        title = table_config.get("title", "")
        console_log(f"Not showing table with empty column(s): {table_id_str} {title}")
        return content

    # mem_chart diagram mode: one merged chart, no per-table titles (3.1, 3.2, …).
    # With --view table, keep titles so tabular output stays navigable.
    skip_mem_chart_title = table_config.get(
        "cli_style"
    ) == "mem_chart" and not _tty_view_is_table(args)
    if "title" in table_config and table_config["title"] and not skip_mem_chart_title:
        content += f"{table_id_str} {table_config['title']}\n"

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

    # For single run and gfx115x, format BW metrics (Bytes/s) to human-readable
    # For multiple runs (baseline comparison), keep Bytes for accurate comparison
    is_single_run = len(runs) == 1
    is_gfx115x = gpu_arch and gpu_arch.startswith("gfx115")

    if is_single_run and is_gfx115x and "Unit" in df.columns:
        # Identify value columns to format
        value_cols = ["Value", "Avg", "Min", "Max", "Peak", "Peak (Empirical)"]
        df = scale_bw_columns(df, value_cols, args.decimal)

    # When --view table is set, force table output and ignore cli_style from config
    use_mem_chart = (
        not _tty_view_is_table(args)
        and table_config.get("cli_style") == "mem_chart"
        and len(runs) == 1
        and "Metric" in df.columns
        and "Value" in df.columns
    )

    if use_mem_chart:
        if mem_data_override is not None:
            mem_data = mem_data_override
        else:
            mem_data = (
                pd
                .DataFrame([df["Metric"], df["Value"]])
                .transpose()
                .set_index("Metric")
                .to_dict()["Value"]
            )

        if gpu_arch and gpu_arch.startswith("gfx115"):
            content += (
                mem_chart_gfx11.plot_mem_chart(
                    args.normal_unit,
                    mem_data,
                    chart_title=_gfx115_mem_chart_heading(None, args.normal_unit),
                )
                + "\n"
            )
        else:
            content += mem_chart_gfx9.plot_mem_chart(args.normal_unit, mem_data) + "\n"
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

    first_run = next(iter(runs.values()))
    gpu_arch = (
        first_run.sys_info.iloc[0]["gpu_arch"]
        if hasattr(first_run, "sys_info") and not first_run.sys_info.empty
        else None
    )
    filter_panel_ids = convert_filter_blocks_to_panel_ids(
        profiling_config.get("filter_blocks", []), gpu_arch
    )

    if args.include_cols:
        hidden_cols = list(set(config.HIDDEN_COLUMNS_CLI) - set(args.include_cols))
    else:
        hidden_cols = config.HIDDEN_COLUMNS_CLI

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
        # Skip if --view table is set; tables 401/402 will be rendered as normal tables
        if panel_id == 400 and not _tty_view_is_table(args):
            _ = is_roofline_shown(args, runs, output, panel, roof_plot, hidden_cols)

        panel_content = ""  # store content of all data_source from one panel
        mem_chart_data: dict[str, Any] = {}  # merged mem_chart metrics (gfx115x)

        for data_source in panel["data source"]:
            for table_type, table_config in data_source.items():
                # Skip tables that were filtered out at build_dfs time
                # (e.g. analyze-mode -b dropped this block). In baseline mode
                # require the table in every run so per-run dfs[id] lookups
                # downstream stay safe.
                if not all(table_config["id"] in run.dfs for run in runs.values()):
                    continue

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

                if processed_df.empty:
                    continue

                # For gfx115x mem_chart panels, collect all tables and merge
                # into a single chart on the first table; skip subsequent ones.
                is_mem_chart = table_config.get(
                    "cli_style"
                ) == "mem_chart" and not _tty_view_is_table(args)
                is_gfx115x = gpu_arch and gpu_arch.startswith("gfx115")

                if is_mem_chart and is_gfx115x and len(runs) == 1:
                    has_cols = (
                        "Metric" in processed_df.columns
                        and "Value" in processed_df.columns
                    )
                    if has_cols:
                        table_mem = dict(
                            zip(processed_df["Metric"], processed_df["Value"])
                        )
                        mem_chart_data.update(table_mem)
                    # Skip individual table output; merged chart emitted below
                    continue

                panel_content += format_table_output(
                    args,
                    table_config,
                    processed_df,
                    table_type,
                    runs,
                    gpu_arch,
                )

        # Emit merged gfx115x mem_chart for the panel
        if mem_chart_data and not _tty_view_is_table(args):
            heading = _gfx115_mem_chart_heading(panel, args.normal_unit)
            panel_content += (
                mem_chart_gfx11.plot_mem_chart(
                    args.normal_unit,
                    mem_chart_data,
                    chart_title=heading,
                )
                + "\n"
            )

        # Roofline printing is handled separately above in is_roofline_shown.
        # With --view table, roofline tables (401/402) render as normal tables.
        if panel_content and (
            table_config["id"] not in [401, 402] or _tty_view_is_table(args)
        ):
            if _panel_is_mem_chart_only(panel) and not _tty_view_is_table(args):
                print(panel_content, file=output)
            else:
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
                    if table_config["id"] not in data.dfs:
                        continue
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
