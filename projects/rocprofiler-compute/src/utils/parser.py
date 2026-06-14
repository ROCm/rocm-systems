# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from utils import schema
from utils.logger import console_error, console_warning, demarcate
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
    tool_data: dict[str, Any],
    kernel_name: str,
    sorting_type: str,
) -> pd.DataFrame:
    """
    Count and sort PC sampling data for a single kernel from a parsed
    ``rocprofiler-sdk-tool`` record, ordering by compiled asm and
    associating with kernel source code if available, then return df.

    :param method: "host_trap" or "stochastic".
    :type method: str
    :param tool_data: The parsed ``rocprofiler-sdk-tool[0]`` dict.
    :type tool_data: dict
    :param kernel_name: The kernel name to be filtered out.
    :type kernel_name: str
    :param sorting_type: "offset" or "count".
    :type sorting_type: str
    :return: The counted and reordering pc sampling info.
    :rtype: pd.DataFrame:
    """
    buffer_records = tool_data["buffer_records"]

    # Map dispatch_id -> kernel_id (kernel dispatch records) and
    # kernel_id -> kernel name (kernel symbols). A single code object can
    # hold many kernels, so dispatch_id is the reliable kernel attribution.
    kernel_id_to_name = {
        symbol["kernel_id"]: symbol["formatted_kernel_name"]
        for symbol in tool_data["kernel_symbols"]
    }
    dispatch_to_kernel_id = {
        dispatch["dispatch_info"]["dispatch_id"]: dispatch["dispatch_info"]["kernel_id"]
        for dispatch in buffer_records["kernel_dispatch"]
    }

    if kernel_name not in kernel_id_to_name.values():
        console_warning(f"PC sampling: cannot find kernel '{kernel_name}'")
        return pd.DataFrame()

    pc_samples = buffer_records[
        "pc_sample_host_trap" if method == "host_trap" else "pc_sample_stochastic"
    ]
    if not pc_samples:
        console_error("PC sampling: can not find pc sample.")

    # Get processed sampling data grouped by (code_object_id, offset, inst_index)
    records = search_pc_sampling_record(pc_samples)
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

    # Map dispatch_id to kernel info (kernel_id and kernel_name)
    df["kernel_id"] = df["dispatch_id"].map(dispatch_to_kernel_id)
    df["kernel_name"] = df["kernel_id"].map(kernel_id_to_name)

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

    # Instruction disassembly and source-line comments are indexed by inst_index.
    instructions = tool_data["strings"]["pc_sample_instructions"]
    comments = tool_data["strings"]["pc_sample_comments"]
    if not instructions or not comments:
        console_error("PC sampling: instruction or comment string table is empty.")

    df["instruction"] = df["inst_index"].apply(
        lambda x: instructions[x] if x < len(instructions) else None
    )
    df["source_line"] = df["inst_index"].apply(
        lambda x: f".../{Path(comments[x]).name}" if x < len(comments) else None
    )

    # Sort on the numeric offset (lexicographic hex order is wrong), then
    # format offset as hex for display.
    if sorting_type == "offset":
        df_sorted = df.sort_values(by=["code_object_id", "offset"])
    elif sorting_type == "count":
        df_sorted = df.sort_values(by=["count"], ascending=False)
    else:
        console_error(
            'Error: pc sampling sorting_type must be either "offset" or "count".'
        )
        return pd.DataFrame()

    df_sorted["offset"] = df_sorted["offset"].apply(hex)

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
    workload: schema.Workload,
    dir_path: str,
    file_prefix: str,
    sorting_type: str,
    tool_data: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Load PC sampling raw data, filter and sort it by specified conditions,
    then return df.

    *tool_data* is a parsed ``rocprofiler-sdk-tool[0]`` dict. When omitted
    the (potentially multi-GB) results json is parsed from *dir_path*;
    callers that already hold the parsed data should pass it to avoid
    re-reading the file.
    """

    if not file_prefix or file_prefix.lower() == "none":
        return pd.DataFrame()

    if tool_data is None:
        json_file_path = Path(dir_path) / f"{file_prefix}_results.json"
        if not json_file_path.exists():
            console_warning(f"PC sampling: can not read {json_file_path}")
            return pd.DataFrame()
        with json_file_path.open(encoding="utf-8") as json_file:
            tool_data = json.load(json_file)["rocprofiler-sdk-tool"][0]

    # No kernel filter: aggregate samples across all kernels by source line.
    if not workload.filter_kernel_ids:
        return _load_pc_sampling_no_filter_data(tool_data)

    if len(workload.filter_kernel_ids) > 1:
        console_error(
            "PC sampling supports single kernel only! Please specify -k with "
            "single kernel.",
            exit=False,
        )
        return pd.DataFrame()

    # Exactly one kernel filter.
    kernel_top_df = workload.dfs[PMC_KERNEL_TOP_TABLE_ID]
    kernel_index = workload.filter_kernel_ids[0]
    if kernel_index >= len(kernel_top_df):
        console_warning(
            f"Kernel index {kernel_index} is out of bounds. "
            f"kernel_top table has only {len(kernel_top_df)} rows."
        )
        return pd.DataFrame()

    pc_sampling_method = _detect_pc_sampling_method(tool_data)
    if pc_sampling_method is None:
        console_warning(
            f"PC sampling: can not detect pc sampling method for {file_prefix}"
        )
        return pd.DataFrame()

    kernel_name = kernel_top_df.iloc[kernel_index]["Kernel_Name"]
    return load_pc_sampling_data_per_kernel(
        pc_sampling_method,
        tool_data,
        kernel_name,
        sorting_type,
    )


def _detect_pc_sampling_method(tool_data: dict[str, Any]) -> Optional[str]:
    """Detect the PC sampling method from the populated buffer record array.

    Prioritizes stochastic over host_trap. Returns None when neither
    array holds any samples.
    """
    buffer_records = tool_data["buffer_records"]
    if buffer_records["pc_sample_stochastic"]:
        return "stochastic"
    if buffer_records["pc_sample_host_trap"]:
        return "host_trap"
    return None


def _load_pc_sampling_no_filter_data(tool_data: dict[str, Any]) -> pd.DataFrame:
    """Aggregate all-kernel PC samples by source line.

    Kernel name is resolved per sample via dispatch_id (dispatch -> kernel_id
    -> name), so kernels that share a code object are attributed correctly.
    """
    buffer_records = tool_data["buffer_records"]
    samples = (
        buffer_records["pc_sample_stochastic"] + buffer_records["pc_sample_host_trap"]
    )
    instructions = tool_data["strings"]["pc_sample_instructions"]
    comments = tool_data["strings"]["pc_sample_comments"]
    if not instructions or not comments:
        console_error("PC sampling: instruction or comment string table is empty.")
    kernel_id_to_name = {
        symbol["kernel_id"]: symbol["formatted_kernel_name"]
        for symbol in tool_data["kernel_symbols"]
    }
    dispatch_to_kernel_id = {
        dispatch["dispatch_info"]["dispatch_id"]: dispatch["dispatch_info"]["kernel_id"]
        for dispatch in buffer_records["kernel_dispatch"]
    }

    # Correlate each sample to its kernel through the dispatch it belongs to:
    # record.dispatch_id -> kernel_id -> formatted name. code_object_id is not
    # usable here because one code object can hold several kernels.
    rows = [
        {
            "source_line": (
                comments[sample["inst_index"]]
                if sample["inst_index"] < len(comments)
                else None
            ),
            "instruction": (
                instructions[sample["inst_index"]]
                if sample["inst_index"] < len(instructions)
                else None
            ),
            "Kernel_Name": kernel_id_to_name.get(
                dispatch_to_kernel_id.get(sample["record"]["dispatch_id"])
            ),
        }
        for sample in samples
    ]

    # Aggregate per source line: count is the number of samples on that line.
    # instruction and Kernel_Name take a representative (first) value, since one
    # source line can map to several instructions.
    df = pd.DataFrame(rows, columns=["source_line", "instruction", "Kernel_Name"])
    grouped_counts = (
        df
        .groupby("source_line")
        .agg(
            count=("source_line", "count"),
            instruction=("instruction", "first"),
            Kernel_Name=("Kernel_Name", "first"),
        )
        .reset_index()
    )
    grouped_counts = grouped_counts[
        ["source_line", "Kernel_Name", "instruction", "count"]
    ]
    grouped_counts["source_line"] = grouped_counts["source_line"].apply(
        lambda x: f".../{Path(x).name}" if isinstance(x, str) and x else x
    )

    return grouped_counts.sort_values(by="count", ascending=False)


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
    workload: schema.Workload,
    dir_path: str,
    args: argparse.Namespace,
    pc_sampling_tool_data: Optional[dict[str, Any]] = None,
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
                tool_data=pc_sampling_tool_data,
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
