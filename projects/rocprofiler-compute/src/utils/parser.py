# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from pc_sampling.code_object_analysis import InstructionPipelines
from pc_sampling.pc_sampling_analysis import (
    SOURCE_LINE_MISSING,
)
from utils import schema
from utils.logger import console_error, demarcate
from utils.pattern_matching import fnmatch_glob_matches
from utils.specs import MachineSpecs
from utils.utils_common import (
    MEMBW_ANALYSIS_PANEL_ID,
    SUPPORTED_FIELD,
    convert_filter_blocks_to_panel_ids,
    convert_metric_id_to_panel_info,
    expand_placeholder_ranges,
)
from utils.utils_counter_defs import metric_expressions, metric_value_keys

# ------------------------------------------------------------------------------
# Internal global definitions

# NB:
# Ammolite is unique gemstone from the Rocky Mountains.
# "ammolite__" is a special internal prefix used by the shared metrics
# evaluation code to mark build-in global variables calculated or parsed from
# raw data sources. Other generic prefixes, like "buildin__", might be used by
# the editor. Whenever this prefix changes, update all shared metric helpers.

# 001 is ID of the Top Kernels table (built by create_df_kernel_top_stats)
PMC_KERNEL_TOP_TABLE_ID: int = 1
# 002 is ID of the Dispatch List table (built by create_df_kernel_top_stats)
PMC_DISPATCH_INFO_TABLE_ID: int = 2


@demarcate
def build_dfs(
    arch_configs: schema.ArchConfig,
    sys_info: pd.Series,
    profiling_config: dict[str, Any],
    arch: Optional[str] = None,
) -> None:
    """Build a dataframe template for each table in each panel. Tables outside
    the profile-mode filter_blocks are omitted from arch_configs.dfs. Alias
    tokens (e.g. "lds", "roofline") are resolved against arch's panel aliases.
    """

    simple_box = {
        "Min": ["MIN(", ")"],
        "Q1": ["QUANTILE(", ", 0.25)"],
        "Median": ["MEDIAN(", ")"],
        "Q3": ["QUANTILE(", ", 0.75)"],
        "Max": ["MAX(", ")"],
    }

    dfs: dict[int, pd.DataFrame] = {}
    dfs_type: dict[int, str] = {}
    dfs_expressions: dict[int, list[str]] = {}

    profile_panel_filter = convert_filter_blocks_to_panel_ids(
        profiling_config.get("filter_blocks", []), arch
    )

    arch_configs.panel_configs = expand_placeholder_ranges(
        arch_configs.panel_configs, sys_info
    )

    for panel_id, panel in arch_configs.panel_configs.items():
        # Profile only collects block 30 counters with --membw-analysis.
        if panel_id == MEMBW_ANALYSIS_PANEL_ID and not profiling_config.get(
            "membw_analysis", False
        ):
            continue

        for data_source in panel["data source"]:
            for table_type, data_config in data_source.items():
                table_id = data_config["id"]
                file_data_source_idx = str(table_id // 100)

                if table_type == "metric_table":
                    df, expressions = _build_metric_table_df(
                        panel=panel,
                        data_config=data_config,
                        simple_box=simple_box,
                        panel_id=panel_id,
                        profile_panel_filter=profile_panel_filter,
                    )
                    # Filter excluded every metric in this panel; skip the empty table.
                    if data_config["metric"] and df.empty:
                        continue
                    dfs_expressions[table_id] = expressions

                elif table_type == "raw_csv_table":
                    if not _metric_passes_filter(
                        metric_id=file_data_source_idx,
                        panel_id=panel_id,
                        data_source_idx=file_data_source_idx,
                        profile_panel_filter=profile_panel_filter,
                    ):
                        continue
                    if data_config.get("columnwise"):
                        df = pd.DataFrame(
                            [data_config["source"]],
                            columns=["from_csv_columnwise"],
                        )
                    else:
                        df = pd.DataFrame([data_config["source"]], columns=["from_csv"])

                elif table_type == "pc_sampling_table":
                    if not _metric_passes_filter(
                        metric_id=file_data_source_idx,
                        panel_id=panel_id,
                        data_source_idx=file_data_source_idx,
                        profile_panel_filter=profile_panel_filter,
                    ):
                        continue
                    # Only the column name is read downstream; the value is a
                    # placeholder to keep the marker frame non-empty.
                    df = pd.DataFrame([True], columns=["from_pc_sampling"])

                else:
                    df = pd.DataFrame()

                dfs[table_id] = df
                dfs_type[table_id] = table_type

    arch_configs.dfs = dfs
    arch_configs.dfs_type = dfs_type
    arch_configs.dfs_expressions = dfs_expressions


def _metric_passes_filter(
    metric_id: str,
    panel_id: int,
    data_source_idx: str,
    profile_panel_filter: set[int],
) -> bool:
    """Return True if a metric or table identified by metric_id passes the
    profile block filter. metric_id is the file-level id for raw_csv /
    pc_sampling tables, or the per-metric id (e.g. "2.1.0") for metric_table rows.
    """
    if panel_id <= 100 or data_source_idx == "0" or not profile_panel_filter:
        return True
    file_id, _, _ = convert_metric_id_to_panel_info(metric_id)
    return int(file_id) in profile_panel_filter


def _build_metric_table_df(
    panel: dict[str, Any],
    data_config: dict[str, Any],
    simple_box: dict[str, list[str]],
    panel_id: int,
    profile_panel_filter: set[int],
) -> tuple[pd.DataFrame, list[str]]:
    """Build the metric_table dataframe and its list of formula strings for
    data_config, dropping rows the active filter excludes.
    """
    table_id = data_config["id"]
    table_data_source_idx = f"{table_id // 100}.{table_id % 100}"
    is_simple_box = data_config.get("cli_style") == "simple_box"

    headers: list[str] = ["Metric_ID", data_config["header"]["metric"]]
    value_keys = metric_value_keys(data_config)
    if is_simple_box:
        headers.extend(simple_box)
        for key, tile in data_config["header"].items():
            if key != "metric" and key != "expr":
                headers.append(tile)
    else:
        for key, tile in data_config["header"].items():
            if key != "metric":
                headers.append(tile)
    if "metrics_description" in panel:
        headers.append("Description")

    rows: list[list[Any]] = []
    expressions: list[str] = []
    metric_entries = data_config["metric"]
    for i, (key, entries) in enumerate(metric_entries.items()):
        metric_idx = f"{table_data_source_idx}.{i}"

        if not _metric_passes_filter(
            metric_id=metric_idx,
            panel_id=panel_id,
            data_source_idx=table_data_source_idx,
            profile_panel_filter=profile_panel_filter,
        ):
            continue

        values: list[Any] = [metric_idx, key]
        for k, v in entries.items():
            if is_simple_box and k == "expr":
                values.extend(
                    prefix + v + suffix for prefix, suffix in simple_box.values()
                )
            elif k in value_keys:
                values.append(v)
        expressions.extend(metric_expressions(data_config, entries))

        if "metrics_description" in panel:
            values.append(panel["metrics_description"].get(key, ""))

        rows.append(values)

    df = pd.DataFrame(rows, columns=headers)
    df.set_index("Metric_ID", inplace=True)
    return df, expressions


def format_pc_sampling_display_frame(
    df: pd.DataFrame,
    method: str,
    sorting_type: str,
    num_rows: Optional[int] = None,
    gpu_arch: Optional[str] = None,
) -> pd.DataFrame:
    """Return the sampling rows in their requested display layout."""
    # Project stall_reason as a descending list[(reason, count)].
    df["stall_reason"] = df["stall_reason"].apply(_stall_reason_dict_to_list)
    df["source_line"] = df["source_line"].apply(_trim_source_line)
    df["instruction_type"] = df["instruction"].apply(
        lambda instruction: InstructionPipelines.lookup(instruction, gpu_arch)
    )

    # Sort on the numeric offset (lexicographic hex order is wrong), then
    # format offset as hex for display. Leading with pid keeps each process's
    # rows contiguous.
    if sorting_type == "offset":
        df_sorted = df.sort_values(
            by=["pid", "code_object_id", "Kernel_Name", "offset"]
        )
    elif sorting_type == "count":
        df_sorted = df.sort_values(by=["count"], ascending=False)
    else:
        console_error(
            'Error: pc sampling sorting_type must be either "offset" or "count".'
        )
        return pd.DataFrame()

    # num_rows of 0 or None (or a negative passed programmatically) shows all.
    if num_rows and num_rows > 0:
        df_sorted = df_sorted.head(num_rows)

    df_sorted["offset"] = df_sorted["offset"].apply(hex)

    # Stochastic adds issue/stall detail on top of the host_trap columns.
    shared_columns = [
        "pid",
        "source_line",
        "instruction",
        "instruction_type",
        "code_object_id",
        "offset",
        "count",
        "active_thread_percent",
    ]
    # wave_occupancy_percent is stochastic-only: a host_trap record has no
    # wave count to derive it from.
    stochastic_only_columns = [
        "issue_count",
        "stall_count",
        "wave_occupancy_percent",
        "stall_reason",
    ]
    columns_to_return = shared_columns + (
        stochastic_only_columns if method == "stochastic" else []
    )
    columns_to_return.append("Kernel_Name")
    # Renumber after sorting so the displayed index reads 0..n-1 rather than the
    # pre-sort positions the row happened to land on.
    return df_sorted[columns_to_return].reset_index(drop=True)


def _stall_reason_dict_to_list(
    stall_reason: Optional[dict[str, int]],
) -> list[tuple[str, int]]:
    """Convert a {reason: count} dict to a descending list[(reason, count)]."""
    if not stall_reason:
        return []
    return sorted(stall_reason.items(), key=lambda item: item[1], reverse=True)


def _trim_source_line(source_line: str) -> str:
    """Show only the trailing path component of a real source line."""
    if source_line == SOURCE_LINE_MISSING:
        return source_line
    return f".../{Path(source_line).name}"


def torch_operator_pattern_matches(pattern: str, operator_name: str) -> bool:
    """Return True if *pattern* glob-matches *operator_name* hierarchy path."""
    return fnmatch_glob_matches(pattern, operator_name)


def build_comparable_columns(time_unit: str) -> list[str]:
    """
    Build comparable columns/headers for display
    """
    comparable_columns = list(SUPPORTED_FIELD)  # Copy to avoid mutating the original
    top_stat_base = [
        "Count",
        "Sum",
        "Mean",
        "Median",
        "Standard Deviation",
        "Description",
    ]

    for h in top_stat_base:
        comparable_columns.append(f"{h}({time_unit})")

    return comparable_columns


def correct_sys_info(mspec: MachineSpecs, specs_correction: str) -> pd.DataFrame:
    """
    Correct system spec items manually based on user-provided corrections.
    """
    # Parse key:value pairs
    pairs: dict[str, str] = {}
    for pair in specs_correction.split(","):
        if ":" in pair:
            key, value = pair.split(":", 1)
            pairs[key.strip()] = value.strip()

    # Apply corrections
    for key, value in pairs.items():
        if hasattr(mspec, key):
            setattr(mspec, key, value)
        else:
            console_error(
                "analyze", f'Invalid spec "{key}". Use --specs to see valid options'
            )
    # Convert dict to DataFrame for downstream pandas-based processing
    return pd.DataFrame(mspec.get_class_members(), index=[0])
