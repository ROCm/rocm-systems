# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import bisect
import json
import os
import re
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from utils import schema
from utils.logger import console_debug, console_error, console_warning, demarcate
from utils.metrics.evaluation_pipeline import eval_metric
from utils.metrics.expression import gen_counter_list
from utils.pattern_matching import fnmatch_glob_matches
from utils.specs import MachineSpecs
from utils.utils_common import (
    METRIC_ID_RE,
    SUPPORTED_FIELD,
    convert_filter_blocks_to_panel_ids,
    convert_metric_id_to_panel_info,
    expand_placeholder_ranges,
    normalize_filter_to_str_list,
)

# ------------------------------------------------------------------------------
# Internal global definitions

# NB:
# Ammolite is unique gemstone from the Rocky Mountains.
# "ammolite__" is a special internal prefix used by the shared metrics
# evaluation code to mark build-in global variables calculated or parsed from
# raw data sources. Other generic prefixes, like "buildin__", might be used by
# the editor. Whenever this prefix changes, update all shared metric helpers.

# 001 is ID of pmc_kernel_top.csv table
PMC_KERNEL_TOP_TABLE_ID: int = 1
# 002 is ID of pmc_dispatch_info.csv table
PMC_DISPATCH_INFO_TABLE_ID: int = 2

PC_SAMPLING_NOT_ISSUE_PREFIX = "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_"


@demarcate
def build_dfs(
    arch_configs: schema.ArchConfig,
    filter_metrics: Optional[list[str]],
    sys_info: pd.Series,
    profiling_config: dict[str, Any],
    arch: Optional[str] = None,
) -> None:
    """Build a dataframe template for each table in each panel. Analyze-mode
    filter_metrics overrides profile-mode filter_blocks; tables that fail the
    active filter are omitted from arch_configs.dfs. Alias tokens (e.g. "lds",
    "roofline") in either filter are resolved against arch's panel aliases.
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
    metric_counters: dict[str, list[str]] = {}

    if filter_metrics:
        numeric_tokens = [t for t in filter_metrics if METRIC_ID_RE.match(str(t))]
        alias_tokens = [t for t in filter_metrics if not METRIC_ID_RE.match(str(t))]
        user_metric_filter: Optional[list[str]] = numeric_tokens or None
        profile_panel_filter: set[int] = convert_filter_blocks_to_panel_ids(
            alias_tokens, arch
        )
    else:
        user_metric_filter = None
        profile_panel_filter = convert_filter_blocks_to_panel_ids(
            profiling_config.get("filter_blocks", []), arch
        )

    arch_configs.panel_configs = expand_placeholder_ranges(
        arch_configs.panel_configs, sys_info
    )

    for panel_id, panel in arch_configs.panel_configs.items():
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
                        user_metric_filter=user_metric_filter,
                        profile_panel_filter=profile_panel_filter,
                        metric_counters=metric_counters,
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
                        user_metric_filter=user_metric_filter,
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
                        user_metric_filter=user_metric_filter,
                        profile_panel_filter=profile_panel_filter,
                    ):
                        continue
                    df = pd.DataFrame(
                        [data_config["source"]], columns=["from_pc_sampling"]
                    )

                else:
                    df = pd.DataFrame()

                dfs[table_id] = df
                dfs_type[table_id] = table_type

    arch_configs.dfs = dfs
    arch_configs.dfs_type = dfs_type
    arch_configs.dfs_expressions = dfs_expressions
    arch_configs.metric_counters = metric_counters


def _metric_passes_filter(
    metric_id: str,
    panel_id: int,
    data_source_idx: str,
    user_metric_filter: Optional[list[str]],
    profile_panel_filter: set[int],
) -> bool:
    """Return True if a metric or table identified by metric_id passes the
    active filter. metric_id is the file-level id for raw_csv / pc_sampling
    tables, or the per-metric id (e.g. "2.1.0") for metric_table rows.
    """
    if panel_id <= 100 or data_source_idx == "0":
        return True
    if user_metric_filter is None and not profile_panel_filter:
        return True
    if user_metric_filter and (
        metric_id in user_metric_filter
        or data_source_idx in user_metric_filter
        or str(panel_id // 100) in user_metric_filter
    ):
        return True
    if profile_panel_filter:
        file_id, _, _ = convert_metric_id_to_panel_info(metric_id)
        return int(file_id) in profile_panel_filter
    return False


def _build_metric_table_df(
    panel: dict[str, Any],
    data_config: dict[str, Any],
    simple_box: dict[str, list[str]],
    panel_id: int,
    user_metric_filter: Optional[list[str]],
    profile_panel_filter: set[int],
    metric_counters: dict[str, list[str]],
) -> tuple[pd.DataFrame, list[str]]:
    """Build the metric_table dataframe and its list of formula strings for
    data_config, dropping rows the active filter excludes. Updates
    metric_counters in place.
    """
    table_id = data_config["id"]
    table_data_source_idx = f"{table_id // 100}.{table_id % 100}"
    is_simple_box = data_config.get("cli_style") == "simple_box"

    headers: list[str] = ["Metric_ID", data_config["header"]["metric"]]
    header_keys: set[str] = set(data_config["header"]) - {"metric", "expr"}
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
            user_metric_filter=user_metric_filter,
            profile_panel_filter=profile_panel_filter,
        ):
            continue

        values: list[Any] = [metric_idx, key]
        eqn_content: list[Any] = []
        if is_simple_box:
            for k, v in entries.items():
                if k == "expr":
                    for bv in simple_box.values():
                        values.append(bv[0] + v + bv[1])
                    eqn_content.append(v)
                elif k not in {"coll_level", "alias"} and k in header_keys:
                    values.append(v)
        else:
            for k, v in entries.items():
                if k not in {"coll_level", "alias"} and k in header_keys:
                    values.append(v)
                    eqn_content.append(v)
        expressions.extend(
            v for v in eqn_content if isinstance(v, str) and v and v != "None"
        )

        if "alias" in entries:
            values.append(entries["alias"])
        if "metrics_description" in panel:
            values.append(panel["metrics_description"].get(key, ""))

        rows.append(values)

        filtered_counters: dict[str, None] = {}
        formula_visited = False
        for formula in eqn_content:
            if formula is None or formula == "None":
                continue
            visited, counters = gen_counter_list(formula)
            if visited:
                formula_visited = True
            for counter in counters:
                filtered_counters[counter] = None

        if filtered_counters or formula_visited:
            metric_counters[key] = list(filtered_counters)

    df = pd.DataFrame(rows, columns=headers)
    df.set_index("Metric_ID", inplace=True)
    return df, expressions


@demarcate
def apply_filters(
    workload: schema.Workload, dir_path: str, is_gui: bool, debug: bool
) -> pd.DataFrame:
    """
    Apply user's filters to the raw_pmc df.
    """

    # TODO: error out properly if filters out of bound
    filtered_df = workload.raw_pmc

    # Apply node filter
    if workload.filter_nodes:
        filtered_df = filtered_df.loc[
            filtered_df["Node"]
            .astype(str)
            .isin(normalize_filter_to_str_list(workload.filter_nodes))
        ]
        if filtered_df.empty:
            console_error("analysis", f"{workload.filter_nodes} is invalid")

    # Apply GPU ID filter
    if workload.filter_gpu_ids:
        filtered_df = filtered_df.loc[
            filtered_df["GPU_ID"]
            .astype(str)
            .isin(normalize_filter_to_str_list(workload.filter_gpu_ids))
        ]
        if filtered_df.empty:
            console_error("analysis", f"{workload.filter_gpu_ids} is an invalid gpu-id")

    # Apply kernel filter
    # NB:
    # Kernel id is unique!
    # We pick up kernel names from kerne ids first.
    # Then filter valid entries with kernel names.
    if workload.filter_kernel_ids:
        filtered_df = apply_kernel_filter(filtered_df, workload)

    # Apply dispatch filter
    if workload.filter_dispatch_ids:
        filtered_df = apply_dispatch_filter(filtered_df, workload)

    if debug:
        print("~" * 40, "\nraw pmc df info:\n")
        print(workload.raw_pmc.info())
        print("~" * 40, "\nfiltered pmc df info:")
        print(filtered_df.info())

    return filtered_df


def apply_kernel_filter(df: pd.DataFrame, workload: schema.Workload) -> pd.DataFrame:
    """Apply kernel ID or name filters."""
    if all(isinstance(kernel_id, int) for kernel_id in workload.filter_kernel_ids):
        # Handle integer kernel IDs
        kernel_top_dataframe = workload.dfs.get(PMC_KERNEL_TOP_TABLE_ID)
        if kernel_top_dataframe is None:
            console_error(
                "Kernel top stats table not loaded. "
                "Ensure create_df_kernel_top_stats() "
                "is called before applying kernel filters."
            )

        # Validate kernel IDs
        for kernel_id in workload.filter_kernel_ids:
            if kernel_id >= len(kernel_top_dataframe["Kernel_Name"]):
                console_error(
                    f"{kernel_id} is an invalid kernel id. "
                    "Please enter an id between 0-"
                    f"{len(kernel_top_dataframe['Kernel_Name']) - 1}"
                )

        # Extract kernel names and mark selected kernels with "*"
        # TODO: fix it for unaligned comparison
        selected_kernels = []
        kernel_top_dataframe["Selected"] = ""

        for kernel_id in workload.filter_kernel_ids:
            selected_kernels.append(kernel_top_dataframe.loc[kernel_id, "Kernel_Name"])
            kernel_top_dataframe.loc[kernel_id, "Selected"] = "*"

        if selected_kernels:
            df = df.loc[df["Kernel_Name"].isin(selected_kernels)]

    elif all(isinstance(kernel_id, str) for kernel_id in workload.filter_kernel_ids):
        # Handle string kernel names
        cleaned_dataframe = df["Kernel_Name"].apply(
            lambda kernel_name: (
                kernel_name.strip() if isinstance(kernel_name, str) else kernel_name
            )
        )
        df = df.loc[cleaned_dataframe.isin(workload.filter_kernel_ids)]
    else:
        console_error(
            "analyze",
            "Mixing kernel indices and string filters is not currently supported",
        )

    return df


def apply_dispatch_filter(df: pd.DataFrame, workload: schema.Workload) -> pd.DataFrame:
    """Apply dispatch ID filters."""
    # NB: support ignoring the 1st n dispatched execution by '> n'
    #     The better way may be parsing python slice string
    for dispatch_id in workload.filter_dispatch_ids:
        if isinstance(dispatch_id, str) and ">" in dispatch_id:
            dispatch_id = re.match(r"\>\s*(\d+)", dispatch_id).group(1)
        if int(dispatch_id) >= len(df):  # subtract 2 bc of the two header rows
            console_error("analysis", f"{dispatch_id} is an invalid dispatch id.")

    if (
        isinstance(workload.filter_dispatch_ids[0], str)
        and ">" in workload.filter_dispatch_ids[0]
    ):
        dispatch_match = re.match(r"\>\s*(\d+)", workload.filter_dispatch_ids[0])
        df = df[df["Dispatch_ID"] > int(dispatch_match.group(1))]
    else:
        selected_dispatches = [
            int(dispatch_str) for dispatch_str in workload.filter_dispatch_ids
        ]
        df = df.loc[selected_dispatches]

    return df


def find_key_recursively(
    data: Union[dict, list], search_key: str
) -> Union[list, dict, None]:
    """
    Recursively search for the search_key in the given data
    (which can be a dict or list).
    If the key is found, returns the value as a DataFrame.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key == search_key:
                return value
            elif isinstance(value, (dict, list)):
                result = find_key_recursively(value, search_key)
                if result:
                    return result
    elif isinstance(data, list):
        for item in data:
            result = find_key_recursively(item, search_key)
            if result:
                return result
    return None  # Return None if the key was not found


def search_key_in_json(file_path: Path, search_key: str) -> Union[list, dict, None]:
    # FIXME:
    #   Load the entire JSON into memory.
    #   Should not use for large file.
    with open(file_path, encoding="utf-8") as file:
        data = json.load(file)
        found = find_key_recursively(data, search_key)
        if found is None:
            console_error(f'Key "{search_key}" not found in the JSON file.')
        return found


def search_pc_sampling_record(
    records: Union[list[dict], dict],
) -> Optional[list[tuple]]:
    """
    Search PC sampling records.

    Group by (code_object_id, code_object_offset, inst_index), and aggregate
    counts, stall reasons, and dispatch IDs.

    Returns:
        A sorted list of tuples:
        (
            code_object_id,
            code_object_offset,
            inst_index,
            total_count,
            count_issued,
            count_stalled,
            sorted_stall_reasons,
            sorted_dispatch_ids,
        )
    """

    if not records:
        console_warning("PC sampling: no pc sampling record found!")
        return None

    # records should always be a list of dict
    if isinstance(records, dict):
        records = [records]

    rocp_inst_not_issued_prefix_len = len(PC_SAMPLING_NOT_ISSUE_PREFIX)

    stall_reason_keys = {
        "NONE": 0,
        # No instruction available in the instruction cache.
        "NO_INSTRUCTION_AVAILABLE": 0,
        "ALU_DEPENDENCY": 0,  # ALU dependency not resolved.
        "WAITCNT": 0,
        "INTERNAL_INSTRUCTION": 0,  # Wave executes an internal instruction.
        "BARRIER_WAIT": 0,
        "ARBITER_NOT_WIN": 0,  # The instruction did not win the arbiter.
        "ARBITER_WIN_EX_STALL": 0,
        # Arbiter issued an instruction, but the execution pipe
        # pushed it back from execution.
        "OTHER_WAIT": 0,
        # Other types of wait (e.g., wait for XNACK acknowledgment).
        "SLEEP_WAIT": 0,
        "LAST": 0,
    }

    grouped_data: dict[tuple, list] = {}

    for item in records:
        record = item.get("record", {})
        pc_info = record.get("pc", {})

        code_object_id = pc_info.get("code_object_id")
        code_object_offset = pc_info.get("code_object_offset")
        inst_index = item.get("inst_index")
        dispatch_id = record.get("dispatch_id")

        if None in (code_object_id, code_object_offset, inst_index):
            continue

        key = (code_object_id, code_object_offset, inst_index)

        snapshot = record.get("snapshot", {})
        issued = record.get("wave_issued", False)

        if key not in grouped_data:
            grouped_data[key] = [0, 0, 0, {}, set()]

        entry = grouped_data[key]

        # Update counts
        entry[0] += 1  # total_count
        if issued:
            entry[1] += 1  # count_issued
        else:
            entry[2] += 1  # count_stalled
            stall_reason = snapshot.get("stall_reason")
            if stall_reason and len(stall_reason) > rocp_inst_not_issued_prefix_len:
                reason_key = stall_reason[rocp_inst_not_issued_prefix_len:]
                if reason_key in stall_reason_keys:
                    entry[3][reason_key] = entry[3].get(reason_key, 0) + 1

        # Add dispatch_id if valid
        if dispatch_id is not None:
            entry[4].add(dispatch_id)

    if not grouped_data:
        console_warning("PC sampling: no pc sampling record found!")
        return None

    # Convert to sorted list of tuples:
    sorted_counts = sorted(
        [
            (
                code_object_id,
                code_object_offset,
                inst_index,
                info[0],  # total_count
                info[1],  # count_issued
                info[2],  # count_stalled
                sorted(
                    ((k, v) for k, v in info[3].items() if v > 0),
                    key=lambda item: item[1],
                    reverse=True,
                ),  # sorted stall reasons
                sorted(info[4]),  # sorted dispatch_ids list
            )
            for (
                code_object_id,
                code_object_offset,
                inst_index,
            ), info in grouped_data.items()
        ],
        key=lambda x: (x[0], x[1], x[2]),
    )

    return sorted_counts


@demarcate
def load_pc_sampling_data_per_kernel(
    method: str,
    file_name: Path,
    csv_file_name: Path,
    kernel_name: str,
    sorting_type: str,
) -> pd.DataFrame:
    """
    Load PC sampling raw data from json file with given method and kernel name,
    count pc sampling and sort it in the order of compiled asm and associate with
    kernel source code if available,
    then return df.

    :param method: "host_trap" or "stochastic".
    :type method: str
    :param file_name: The pc sampling json file.
    :type file_name: Path
    :param kernel_name: The kernel name to be filtered out.
    :type kernel_name: str
    :param sorting_type: "offset" or "count".
    :type sorting_type: str
    :return: The counted and reordering pc sampling info.
    :rtype: pd.DataFrame:
    """
    # Load kernel trace CSV with kernel info
    kernel_trace_df = pd.read_csv(
        csv_file_name, usecols=["Dispatch_Id", "Kernel_Id", "Kernel_Name"]
    )
    console_debug(
        f"PC sampling: loaded kernel trace with {len(kernel_trace_df)} entries"
    )

    # Filter kernels matching requested kernel_name
    matching_kernels = kernel_trace_df[kernel_trace_df["Kernel_Name"] == kernel_name]
    if matching_kernels.empty:
        console_warning(f"PC sampling: cannot find kernel '{kernel_name}' in CSV")
        return pd.DataFrame()

    # Extract raw PC sampling records from JSON
    pc_sample_key_loc = (
        search_key_in_json(file_name, "pc_sample_host_trap")
        if method == "host_trap"
        else search_key_in_json(file_name, "pc_sample_stochastic")
    )

    if not pc_sample_key_loc:
        console_warning("PC sampling: can not find pc sample.")
        return pd.DataFrame()

    # Get processed sampling data grouped by (code_object_id, offset, inst_index)
    records = search_pc_sampling_record(pc_sample_key_loc)
    if not records:
        console_warning("PC sampling: no records found in PC sampling data.")
        return pd.DataFrame()

    # Flatten records by dispatch_id to create one row per dispatch ID
    rows = []
    for (
        code_object_id,
        offset,
        inst_index,
        count,
        count_issued,
        count_stalled,
        stall_reasons,
        dispatch_ids,
    ) in records:
        for dispatch_id in dispatch_ids:
            rows.append({
                "dispatch_id": dispatch_id,
                "code_object_id": code_object_id,
                "offset": offset,
                "inst_index": inst_index,
                "count": count,
                "count_issued": count_issued,
                "count_stalled": count_stalled,
                "stall_reason": stall_reasons,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        console_warning("PC sampling: no records found after flattening dispatch IDs.")
        return df

    # Map dispatch_id to kernel info (Kernel_Id and Kernel_Name)
    dispatch_to_kernel = kernel_trace_df.set_index("Dispatch_Id")[
        ["Kernel_Id", "Kernel_Name"]
    ]

    # Map dispatch_id to kernel info (Kernel_Id and Kernel_Name)
    df["kernel_id"] = df["dispatch_id"].map(dispatch_to_kernel["Kernel_Id"])
    df["kernel_name"] = df["dispatch_id"].map(dispatch_to_kernel["Kernel_Name"])

    # Drop dispatch_id
    df.drop(columns=["dispatch_id"], inplace=True)

    def merge_stall_reasons(
        stall_reason_series: list[Optional[list[tuple[str, int]]]],
    ) -> list[tuple[str, int]]:
        """
        Function to merge stall_reason lists (list of dicts -> merged & sorted dict)
        """
        merged_counts = {}

        for entry in stall_reason_series:
            if not entry:
                continue
            # Each entry is a list of (key, count) tuples
            for k, v in entry:
                if v > 0:
                    merged_counts[k] = merged_counts.get(k, 0) + v

        # Return sorted list of tuples by descending count
        return sorted(merged_counts.items(), key=lambda item: item[1], reverse=True)

    # Group and aggregate
    df = df.groupby(["code_object_id", "offset", "kernel_id"], as_index=False).agg({
        "inst_index": "first",
        "count": "sum",
        "count_issued": "sum",
        "count_stalled": "sum",
        "stall_reason": merge_stall_reasons,
        "kernel_name": "first",
    })

    # Filter DataFrame to only include rows matching the requested kernel_name
    df = df[df["kernel_name"] == kernel_name]

    # Convert offset column to hex string for display, keep original numeric for sorting
    df["offset"] = df["offset"].apply(lambda x: hex(x))

    # Load PC sampling instructions from JSON (if available)
    pc_sample_instructions = search_key_in_json(file_name, "pc_sample_instructions")
    df["instruction"] = (
        df["inst_index"].apply(
            lambda x: (
                pc_sample_instructions[x] if x < len(pc_sample_instructions) else None
            )
        )
        if pc_sample_instructions
        else None
    )

    # Load source code comments (if available)
    pc_sample_comments = search_key_in_json(file_name, "pc_sample_comments")
    workload_dir = Path(file_name).parent
    _source_cache: dict[str, Optional[str]] = {}

    def _display_source(index: int) -> object:
        if index >= len(pc_sample_comments):
            return None
        return display_source_comment(
            pc_sample_comments[index], workload_dir, _source_cache
        )

    df["source_line"] = (
        df["inst_index"].apply(_display_source) if pc_sample_comments else None
    )

    # Sorting and returning relevant columns depending on method and sorting_type
    if sorting_type == "offset":
        df_sorted = df.sort_values(by=["code_object_id", "offset"])
    elif sorting_type == "count":
        df_sorted = df.sort_values(by=["count"], ascending=False)
    else:
        console_error(
            'Error: pc sampling sorting_type must be either "offset" or "count".'
        )
        return pd.DataFrame()

    columns_to_return = (
        [
            "source_line",
            "instruction",
            "code_object_id",
            "offset",
            "count",
        ]
        if method == "host_trap"
        else [
            "source_line",
            "instruction",
            "code_object_id",
            "offset",
            "count",
            "count_issued",
            "count_stalled",
            "stall_reason",
        ]
    )

    return df_sorted[columns_to_return]
    # might support sort by stall reason in the future


@demarcate
def load_pc_sampling_data(
    workload: schema.Workload, dir_path: str, file_prefix: str, sorting_type: str
) -> pd.DataFrame:
    """
    Load PC sampling raw data, filter and sort it by specified conditions,
    then return df.
    """

    if not file_prefix or file_prefix.lower() == "none":
        return pd.DataFrame()

    pc_sampling_method = None

    # NB:
    #  - The default file name is subject to changes from rocprofv3
    #  - Prioritize stochastic
    #  - Alternatively, we could check pc_sampling_method in json
    stochastic_path = Path(dir_path) / f"{file_prefix}_pc_sampling_stochastic.csv"
    host_trap_path = Path(dir_path) / f"{file_prefix}_pc_sampling_host_trap.csv"
    json_file_path = Path(dir_path) / f"{file_prefix}_results.json"
    csv_kernel_trace_file_path = Path(dir_path) / f"{file_prefix}_kernel_trace.csv"

    if not csv_kernel_trace_file_path.exists():
        console_warning(f"PC sampling: can not read {csv_kernel_trace_file_path}")
        return pd.DataFrame()

    if stochastic_path.exists():
        pc_sampling_method = "stochastic"
        csv_file_path = stochastic_path
    elif host_trap_path.exists():
        pc_sampling_method = "host_trap"
        csv_file_path = host_trap_path
    else:
        console_warning(
            f"PC sampling: can not detect pc sampling method for {file_prefix}"
        )
        return pd.DataFrame()

    # No kernel filter, return grouped and sorted csv dir_pathectly
    if not workload.filter_kernel_ids:
        # Load instruction CSV
        df = pd.read_csv(csv_file_path)

        # Load kernel trace CSV
        kernel_trace_df = pd.read_csv(csv_kernel_trace_file_path)

        # Merge on Correlation_Id (instruction CSV) and Dispatch_Id (kernel trace CSV)
        merged_df = df.merge(
            kernel_trace_df[["Dispatch_Id", "Kernel_Name", "Kernel_Id"]],
            how="left",
            left_on="Correlation_Id",
            right_on="Dispatch_Id",
        )

        # Group by Instruction_Comment and aggregate
        grouped_counts = (
            merged_df
            .groupby("Instruction_Comment")
            .agg(
                count=("Instruction_Comment", "count"),
                instruction=("Instruction", "first"),
                Kernel_Id=("Kernel_Id", "first"),
                Kernel_Name=("Kernel_Name", "first"),
            )
            .reset_index()
            .rename(columns={"Instruction_Comment": "source_line"})
        )
        grouped_counts = grouped_counts[
            [
                "source_line",
                "Kernel_Name",
                "instruction",
                "count",
            ]
        ]
        workload_dir = Path(dir_path)
        _source_cache: dict[str, Optional[str]] = {}

        def _display_source(comment: object) -> object:
            if not isinstance(comment, str) or not comment:
                return comment
            return display_source_comment(comment, workload_dir, _source_cache)

        grouped_counts["source_line"] = grouped_counts["source_line"].apply(
            _display_source
        )

        return grouped_counts.sort_values(by="count", ascending=False)

    elif len(workload.filter_kernel_ids) > 1:
        console_error(
            "PC sampling supports single kernel only! Please specify -k with "
            "single kernel.",
            exit=False,
        )
        return pd.DataFrame()

    elif len(workload.filter_kernel_ids) == 1:
        if not json_file_path.exists():
            console_warning(f"PC sampling: can not read {json_file_path}")
            return pd.DataFrame()
        else:
            kernel_top_df = workload.dfs[PMC_KERNEL_TOP_TABLE_ID]
            kernel_index = workload.filter_kernel_ids[0]

            if kernel_index >= len(kernel_top_df):
                console_warning(
                    f"Kernel index {kernel_index} is out of bounds. "
                    f"kernel_top table has only {len(kernel_top_df)} rows."
                )
                return pd.DataFrame()

            kernel_name = kernel_top_df.iloc[kernel_index]["Kernel_Name"]

            return load_pc_sampling_data_per_kernel(
                pc_sampling_method,
                json_file_path,
                csv_kernel_trace_file_path,
                kernel_name,
                sorting_type,
            )
    else:
        console_warning("PC sampling: No data")
        return pd.DataFrame()


def split_instruction_comment(
    comment: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Split an ISA comment at the last ':' into (source_file, line).

    Returns (None, None) when comment is empty/None, contains no ':', or either
    side of the last ':' is empty. ``line`` is returned as a string (not int)
    because a comment may carry a non-numeric or multi-source tail; the raw
    comment is retained separately for those cases.

    Examples:
        "/a/b.cpp:42" -> ("/a/b.cpp", "42")
        "C:/x.cpp:5"  -> ("C:/x.cpp", "5")  # splits on the last ':'
    """
    if not comment:
        return None, None

    source_file, sep, line = comment.rpartition(":")
    if not sep or not source_file or not line:
        return None, None

    return source_file, line


def resolve_snapshot_source_path(
    comment: Optional[str], workload_dir: Path
) -> Optional[str]:
    """Resolve an ISA ``path:line`` comment to a snapshot copy when needed.

    Source paths baked into the disassembly are absolute on the capture host and
    may not exist when the workload is analyzed elsewhere. PC sampling collection
    snapshots the referenced files under ``<workload>/code_obj_sources/``. When
    the original path is missing but a snapshot copy exists, return
    ``"<snapshot>:line"`` so the displayed reference points at a file that
    actually exists.

    Returns None when there is no usable path, when the original file is still
    present (keep the default display), or when no snapshot copy is found.
    """
    source_file, line = split_instruction_comment(comment)
    resolved = resolve_source_file(source_file, workload_dir)
    if resolved is None or resolved == source_file:
        return None
    return f"{resolved}:{line}"


def display_source_comment(
    comment: str,
    workload_dir: Path,
    cache: Optional[dict[str, Optional[str]]] = None,
) -> str:
    """Render an ISA ``path:line`` comment for display (``comment`` is a str).

    Returns the snapshot-redirected reference when the capture-host file is gone
    but a copy exists under ``code_obj_sources/``; otherwise a ``.../<name>``
    abbreviation. Callers apply their own empty/non-str guard before calling, so
    the two display paths keep their existing handling of blank comments.
    ``cache``, when supplied, memoizes the result per comment so the same file is
    stat'd at most once across a full column ``apply``.
    """
    if cache is not None and comment in cache:
        return cache[comment]
    resolved = resolve_snapshot_source_path(comment, workload_dir)
    display = resolved if resolved is not None else f".../{Path(comment).name}"
    if cache is not None:
        cache[comment] = display
    return display


def resolve_source_file(
    source_file: Optional[str],
    workload_dir: Path,
    cache: Optional[dict[str, Optional[str]]] = None,
) -> Optional[str]:
    """Return the best on-disk path for ``source_file`` (a bare path, no line).

    Keeps the original when it still exists on the analyze host; otherwise
    redirects to the ``code_obj_sources/`` snapshot copied at collection so
    off-host analysis resolves to a file that exists. Falls back to the original
    when no snapshot is found. ``cache`` memoizes the stat lookups per path.
    """
    if source_file is None:
        return None
    if cache is not None and source_file in cache:
        return cache[source_file]

    def _exists(path: Path) -> bool:
        try:
            return path.exists()
        except OSError:
            return False

    resolved = source_file
    if not _exists(Path(source_file)):
        snapshot = resolve_snapshot_path(source_file, workload_dir)
        if snapshot is not None and _exists(snapshot):
            resolved = str(snapshot)
    if cache is not None:
        cache[source_file] = resolved
    return resolved


def resolve_snapshot_path(source_file: str, workload_dir: Path) -> Optional[Path]:
    """Map a capture-host source path to its snapshot under the workload dir.

    Mirrors ``copy_source_files`` (the leading ``/`` is dropped and ``..``
    segments are normalized away). Returns None when the normalized path would
    escape ``<workload>/code_obj_sources/`` so a crafted comment cannot point the
    lookup outside the snapshot tree.
    """
    root = workload_dir / "code_obj_sources"
    candidate = os.path.normpath(str(root / source_file.lstrip("/")))
    root_str = os.path.normpath(str(root))
    if candidate != root_str and not candidate.startswith(root_str + os.sep):
        return None
    return Path(candidate)


@demarcate
def load_code_obj_info(
    workload_path: Path,
) -> Optional[dict[int, list[dict[str, Any]]]]:
    """Discover and load native code-object disassembly for a workload.

    Globs ``workload_path`` for ``*_code_obj_info.json`` files. Multi-process
    captures emit one file per PID; all matches are merged and keyed by code
    object id. Returns None when no native files are found.

    Each per-id list contains instruction dicts with keys ``code_obj_offset``,
    ``size``, ``name``, and ``comment``, sorted ascending by ``code_obj_offset``
    so callers can bisect.
    """
    native_files = sorted(workload_path.glob("*_code_obj_info.json"))
    if not native_files:
        return None

    merged: dict[int, list[dict[str, Any]]] = {}

    for native_file in native_files:
        try:
            with native_file.open() as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as err:
            # A truncated or unreadable file (e.g. an interrupted capture) must
            # not abort the whole analyze run; skip it and fall back to the
            # rocprofiler-sdk strings if nothing usable parses.
            console_warning(
                f"PC sampling: skipping unreadable native file {native_file}: {err}"
            )
            continue

        for code_object in data.get("code_objects", []):
            code_object_id = code_object.get("id")
            if code_object_id is None:
                continue

            instructions = merged.setdefault(code_object_id, [])
            for symbol_index, symbol in enumerate(code_object.get("symbols", [])):
                for inst in symbol.get("instructions", []):
                    offset = inst.get("code_obj_offset")
                    size = inst.get("size")
                    if offset is None or size is None:
                        # Skip malformed entries so one bad record does not break
                        # the sort / lookup for the whole workload.
                        continue
                    instructions.append({
                        "code_obj_offset": offset,
                        "size": size,
                        "name": inst.get("name"),
                        "comment": inst.get("comment"),
                        # Per-symbol (per-kernel) marker so waitcnt dependency
                        # analysis can reset its outstanding queues at kernel
                        # boundaries instead of leaking producers across kernels.
                        "symbol_index": symbol_index,
                    })

    if not merged:
        # Every native file was unreadable or empty; signal absence so callers
        # fall back to the rocprofiler-sdk strings instead of attributing
        # against empty disassembly.
        console_warning(
            "PC sampling: no usable native code-object disassembly; "
            "falling back to rocprofiler-sdk strings."
        )
        return None

    for instructions in merged.values():
        instructions.sort(key=lambda inst: inst["code_obj_offset"])

    console_debug(f"PC sampling: merged {len(native_files)} native code-object file(s)")

    return merged


def match_instruction_for_offset(
    intervals: list[dict[str, Any]],
    offset: int,
    offsets: Optional[list[int]] = None,
) -> Optional[dict[str, Any]]:
    """Return the instruction whose range contains ``offset``, else None.

    ``intervals`` must be sorted ascending by ``code_obj_offset`` (as produced by
    ``load_code_obj_info``). An instruction matches when
    ``code_obj_offset <= offset < code_obj_offset + size``. An empty list or an
    unmatched offset returns None. Never raises for unknown ids because callers
    pass ``intervals.get(code_object_id, [])``.

    ``offsets`` is the precomputed ascending ``code_obj_offset`` key list for
    ``intervals``; pass it when matching many offsets against the same intervals
    to avoid rebuilding the key list on every call.
    """
    if not intervals:
        return None

    if offsets is None:
        offsets = [inst["code_obj_offset"] for inst in intervals]
    idx = bisect.bisect_right(offsets, offset) - 1
    if idx < 0:
        return None

    candidate = intervals[idx]
    start = candidate["code_obj_offset"]
    size = candidate.get("size")
    if size is not None and start <= offset < start + size:
        return candidate

    return None


# Mnemonic prefixes that INCREMENT (produce) each hardware wait counter. A
# producer instruction issued before an ``s_waitcnt`` keeps the counter above
# zero until it retires, so the waitcnt depends on it. Membership is tunable
# here without changing the public contract of parse_waitcnt_dependencies.
_WAITCNT_PRODUCER_PREFIXES: dict[str, tuple[str, ...]] = {
    # Vector-memory ops bump vmcnt: flat/buffer/global/scratch loads & stores
    # plus image loads.
    "vmcnt": ("flat_", "buffer_", "global_", "scratch_", "image_"),
    # LDS/GDS and scalar-memory ops bump lgkmcnt.
    "lgkmcnt": ("ds_", "gds_", "s_load", "s_buffer_load", "s_scratch_load"),
    # Export / parameter writes bump expcnt.
    "expcnt": ("exp",),
}

# Regex pulling explicit counter targets from a combined waitcnt operand text,
# e.g. "s_waitcnt vmcnt(0) lgkmcnt(0)".
_WAITCNT_TARGET_RE = re.compile(r"(vmcnt|lgkmcnt|expcnt|vscnt)\((\d+)\)")
# Regex pulling a bare integer operand from a class-specific waitcnt mnemonic,
# e.g. "s_waitcnt_vmcnt 0".
_WAITCNT_BARE_INT_RE = re.compile(r"\b(\d+)\b")

# Map class-specific waitcnt mnemonic stems to the counter class they drain.
# ``vscnt`` shares the vmcnt outstanding queue here (vector-store counter).
_WAITCNT_CLASS_MNEMONICS: dict[str, str] = {
    "s_waitcnt_vmcnt": "vmcnt",
    "s_waitcnt_vscnt": "vmcnt",
    "s_waitcnt_lgkmcnt": "lgkmcnt",
    "s_waitcnt_expcnt": "expcnt",
}


def _classify_waitcnt(name: str) -> Optional[dict[str, int]]:
    """Return the per-class target counts an s_waitcnt-family op waits FOR.

    Returns e.g. ``{"vmcnt": 0}`` or ``{"vmcnt": 0, "lgkmcnt": 0}``; None when
    ``name`` is not an s_waitcnt instruction. A class with no parseable target
    drains fully (target 0).
    """
    lowered = name.lower().strip()
    mnemonic = lowered.split()[0] if lowered else ""

    # Class-specific mnemonics: target is the bare integer operand, default 0.
    if mnemonic in _WAITCNT_CLASS_MNEMONICS:
        klass = _WAITCNT_CLASS_MNEMONICS[mnemonic]
        match = _WAITCNT_BARE_INT_RE.search(lowered[len(mnemonic) :])
        target = int(match.group(1)) if match else 0
        return {klass: target}

    # Bare/combined "s_waitcnt vmcnt(N) lgkmcnt(N) expcnt(N)".
    if mnemonic == "s_waitcnt":
        targets: dict[str, int] = {}
        for klass, value in _WAITCNT_TARGET_RE.findall(lowered):
            # vscnt is a vmcnt-family counter for our outstanding queue. When a
            # single waitcnt carries both vmcnt(N) and vscnt(M) they collide on
            # this shared key, so keep the stricter (lower) target rather than
            # letting operand order decide.
            key = "vmcnt" if klass == "vscnt" else klass
            target = int(value)
            targets[key] = min(targets.get(key, target), target)
        if not targets:
            # No explicit operands: drain every class fully.
            return {"vmcnt": 0, "lgkmcnt": 0, "expcnt": 0}
        return targets

    return None


def _waitcnt_producer_classes(name: str) -> tuple[str, ...]:
    """Return the counter classes a producer instruction increments (maybe none)."""
    lowered = name.lower()
    classes = tuple(
        klass
        for klass, prefixes in _WAITCNT_PRODUCER_PREFIXES.items()
        if lowered.startswith(prefixes)
    )
    return classes


def parse_waitcnt_dependencies(
    instructions: list[dict[str, Any]],
) -> dict[int, list[int]]:
    """Map each s_waitcnt instruction's code_obj_offset to the sorted list of
    producer instruction offsets it waits on, derived from s_waitcnt-family
    instructions.

    ``instructions`` is one code object's list, ascending by ``code_obj_offset``
    (as produced by ``load_code_obj_info``). A code object holds multiple symbols
    (kernels); each instruction carries a ``symbol_index`` so the outstanding
    producer queues reset at every symbol boundary -- a kernel's hardware wait
    counters start empty, so a waitcnt never depends on a producer from a
    different kernel. Instructions with no ``symbol_index`` (e.g. flat synthetic
    lists) are treated as a single symbol. Returns ``{}`` when no waitcnt has
    outstanding producers. Offsets with no dependencies are simply absent
    (callers default to ``[]``). Never raises; malformed entries are skipped.
    """
    outstanding: dict[str, list[int]] = {"vmcnt": [], "lgkmcnt": [], "expcnt": []}
    deps: dict[int, list[int]] = {}
    prev_symbol_index: Any = None

    for inst in instructions:
        # Reset the outstanding producers at each kernel boundary so dependencies
        # never cross symbols within a merged code object.
        symbol_index = inst.get("symbol_index")
        if symbol_index != prev_symbol_index:
            outstanding = {"vmcnt": [], "lgkmcnt": [], "expcnt": []}
            prev_symbol_index = symbol_index

        name = inst.get("name")
        if not name:
            # No mnemonic: cannot classify; leave queues untouched.
            continue

        offset = inst.get("code_obj_offset")
        targets = _classify_waitcnt(name)

        if targets is not None:
            if offset is None:
                continue
            satisfied: set[int] = set()
            for klass, target in targets.items():
                queue = outstanding.get(klass)
                if not queue:
                    continue
                keep_from = max(len(queue) - max(target, 0), 0)
                satisfied.update(queue[:keep_from])
                # Pop the now-complete producers off the front of the queue.
                outstanding[klass] = queue[keep_from:]
            if satisfied:
                deps[offset] = sorted(satisfied)
            continue

        # Producer bookkeeping: record this instruction against each counter
        # class it increments so a later waitcnt can depend on it.
        if offset is None:
            continue
        for klass in _waitcnt_producer_classes(name):
            outstanding[klass].append(offset)

    return deps


def nullify_unevaluated_metric_values(
    workload: schema.Workload,
) -> None:
    """Replace unevaluated formula strings with "N/A" in all metric tables.

    In PC-sampling-only mode ``eval_metric`` is never called, so metric
    table cells still contain raw formula strings produced by
    ``build_metric_value_string``.  This helper walks every
    ``metric_table`` in *workload* and sets each ``SUPPORTED_FIELD``
    column to ``"N/A"`` so that downstream display code (``tty``,
    ``webui``, ``tui``) can safely format the values.
    """
    for df_id, df_type in workload.dfs_type.items():
        if df_type != "metric_table":
            continue
        df = workload.dfs.get(df_id)
        if df is None or df.empty:
            continue
        for col in df.columns:
            if col in SUPPORTED_FIELD and col.lower() != "alias":
                df[col] = "N/A"


@demarcate
def load_non_mertrics_table(
    workload: schema.Workload, dir_path: str, args: argparse.Namespace
) -> None:
    # NB:
    #   - Do pmc_kernel_top.csv loading before eval_metric because we need the
    #     kernel names.
    #   - There might be a better way/timing to load raw_csv_table.

    # NB:
    #   "from_csv", "from_csv_columnwise", and "from_pc_sampling"
    #   are 3 internal symbols converted in build_dfs() for non-metrics table.
    #   There might be better way to store these info without the orginal entry.
    tmp = {}
    for df_id, df in workload.dfs.items():
        if "from_csv" in df.columns:
            csv_file = Path(dir_path) / str(df.loc[0, "from_csv"])
            if csv_file.exists():
                tmp[df_id] = pd.read_csv(csv_file)
            else:
                console_warning(
                    f"Couldn't load {csv_file.name}. "
                    "This may result in missing analysis data."
                )
        # NB: Special case for sysinfo. Probably room for improvement in this whole
        # function design
        elif "from_csv_columnwise" in df.columns and id == 101:
            tmp[df_id] = workload.sys_info.transpose()
            # All transposed columns should be marked with a general header
            tmp[df_id].columns = ["Info"]
        elif "from_csv_columnwise" in df.columns:
            # NB:
            #   Another way might be doing transpose in tty like metric_table.
            #   But we need to figure out headers and comparison properly.
            csv_file = Path(dir_path) / str(df.loc[0, "from_csv_columnwise"])
            if csv_file.exists():
                tmp[df_id] = pd.read_csv(csv_file).transpose()
                # NB:
                #   All transposed columns should be marked with a general header,
                #   so tty could detect them and show them correctly in comparison.
                tmp[df_id].columns = ["Info"]
            else:
                console_warning(
                    f"Couldn't load {csv_file.name}. "
                    "This may result in missing analysis data."
                )
        elif "from_pc_sampling" in df.columns:
            tmp[df_id] = load_pc_sampling_data(
                workload,
                dir_path,
                df.loc[0, "from_pc_sampling"],
                args.pc_sampling_sorting_type,
            )

    workload.dfs.update(tmp)


def torch_operator_pattern_matches(pattern: str, operator_name: str) -> bool:
    """Return True if *pattern* glob-matches *operator_name* hierarchy path."""
    return fnmatch_glob_matches(pattern, operator_name)


@demarcate
def load_table_data(
    workload: schema.Workload,
    dir_path: str,
    is_gui: bool,
    args: argparse.Namespace,
    dfs_expressions: dict[int, list[str]],
    skip_kernel_top: bool = False,
) -> None:
    """
    - Load data for all "raw_csv_table"
    - Load data for "pc_sampling_table"
    - Calculate mertric value for all "metric_table"
    """
    if not skip_kernel_top:
        load_non_mertrics_table(workload, dir_path, args)

    eval_metric(
        workload.dfs,
        workload.dfs_type,
        dfs_expressions,
        workload.sys_info.iloc[0],
        workload.roofline_peaks,
        apply_filters(workload, dir_path, is_gui, args.debug),
        args.debug,
    )


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
