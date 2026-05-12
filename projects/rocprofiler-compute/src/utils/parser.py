# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import json
import re
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from utils import schema
from utils.logger import console_debug, console_error, console_warning, demarcate
from utils.metrics.evaluation_pipeline import eval_metric
from utils.metrics.expression import gen_counter_list
from utils.pattern_matching import PatternMatcherEngine
from utils.specs import MachineSpecs
from utils.utils_common import (
    SUPPORTED_FIELD,
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
) -> None:
    """
    Build dataframe for each type of data source within each panel.

    Each dataframe will be used as a template to load data with each run later.
    For now, support "metric_table" and "raw_csv_table". Otherwise, put an empty df.
    """

    # TODO: more error checking for filter_metrics!!
    simple_box = {
        "Min": ["MIN(", ")"],
        "Q1": ["QUANTILE(", ", 0.25)"],
        "Median": ["MEDIAN(", ")"],
        "Q3": ["QUANTILE(", ", 0.75)"],
        "Max": ["MAX(", ")"],
    }

    dfs = {}
    dfs_type = {}
    metric_counters = {}

    arch_configs.panel_configs = expand_placeholder_ranges(
        arch_configs.panel_configs, sys_info
    )

    for panel_id, panel in arch_configs.panel_configs.items():
        for data_source in panel["data source"]:
            for type, data_config in data_source.items():
                if type == "metric_table":
                    headers = ["Metric_ID"]
                    data_source_idx = str(data_config["id"] // 100)

                    if (
                        "cli_style" in data_config
                        and data_config["cli_style"] == "simple_box"
                    ):
                        headers.append(data_config["header"]["metric"])
                        for k in simple_box.keys():
                            headers.append(k)

                        for key, tile in data_config["header"].items():
                            if key != "metric" and key != "expr":
                                headers.append(tile)
                    else:
                        headers.append(data_config["header"]["metric"])
                        for key, tile in data_config["header"].items():
                            if key != "metric":
                                headers.append(tile)

                    # Only add Metrics Description column if it is defined in the panel
                    if "metrics_description" in panel:
                        headers.append("Description")

                    df = pd.DataFrame(columns=headers)

                    for i, (key, entries) in enumerate(data_config["metric"].items()):
                        data_source_idx = (
                            f"{data_config['id'] // 100}.{data_config['id'] % 100}"
                        )
                        metric_idx = f"{data_source_idx}.{i}"
                        eqn_content = []

                        if (
                            (not filter_metrics)
                            or (
                                metric_idx in filter_metrics
                            )  # no filter  # metric in filter
                            or
                            # the whole table in filter
                            (data_source_idx in filter_metrics)
                            or
                            # the whole IP block in filter
                            (str(panel_id // 100) in filter_metrics)
                        ):
                            values = [metric_idx, key]

                            if (
                                "cli_style" in data_config
                                and data_config["cli_style"] == "simple_box"
                            ):
                                for k, v in entries.items():
                                    if k == "expr":
                                        for bv in simple_box.values():
                                            values.append(bv[0] + v + bv[1])
                                    else:
                                        if k != "alias":
                                            values.append(v)
                            else:
                                for k, v in entries.items():
                                    if k != "alias":
                                        values.append(v)
                                        eqn_content.append(v)

                            if "alias" in entries.keys():
                                values.append(entries["alias"])

                            if "metrics_description" in panel:
                                values.append(panel["metrics_description"].get(key, ""))

                            df_new_row = pd.DataFrame([values], columns=headers)
                            df = pd.concat([df, df_new_row])

                        # generate mapping of counters and metrics
                        filtered_counters = {}
                        formula_visited = False

                        for formula in eqn_content:
                            if formula is not None and formula != "None":
                                visited, counters = gen_counter_list(formula)
                                if visited:
                                    formula_visited = True
                                for counter in counters:
                                    filtered_counters[counter] = None

                        if filtered_counters or formula_visited:
                            metric_counters[key] = list(filtered_counters)

                    df.set_index("Metric_ID", inplace=True)
                elif type == "raw_csv_table":
                    data_source_idx = str(data_config["id"] // 100)
                    if (
                        (not filter_metrics)
                        or (data_source_idx == "0")  # no filter
                        or (data_source_idx in filter_metrics)
                    ):
                        if "columnwise" in data_config and data_config["columnwise"]:
                            df = pd.DataFrame(
                                [data_config["source"]], columns=["from_csv_columnwise"]
                            )
                        else:
                            df = pd.DataFrame(
                                [data_config["source"]], columns=["from_csv"]
                            )
                    else:
                        df = pd.DataFrame()
                elif type == "pc_sampling_table":
                    data_source_idx = str(data_config["id"] // 100)
                    df = pd.DataFrame(
                        [data_config["source"]], columns=["from_pc_sampling"]
                    )
                else:
                    df = pd.DataFrame()

                dfs[data_config["id"]] = df
                dfs_type[data_config["id"]] = type

    setattr(arch_configs, "dfs", dfs)
    setattr(arch_configs, "dfs_type", dfs_type)
    setattr(arch_configs, "metric_counters", metric_counters)


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
    with open(file_path) as file:
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
    df["source_line"] = (
        df["inst_index"].apply(
            lambda x: (
                f".../{Path(pc_sample_comments[x]).name}"
                if x < len(pc_sample_comments)
                else None
            )
        )
        if pc_sample_comments
        else None
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
        grouped_counts["source_line"] = grouped_counts["source_line"].apply(
            lambda x: f".../{Path(x).name}" if isinstance(x, str) and x else x
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


torch_operator_matcher = PatternMatcherEngine(mode="glob-hierarchy")


def torch_operator_pattern_matches(pattern: str, operator_name: str) -> bool:
    """Return True if *pattern* glob-matches *operator_name* hierarchy path."""
    return torch_operator_matcher.matches(pattern, operator_name)


@demarcate
def load_table_data(
    workload: schema.Workload,
    dir_path: str,
    is_gui: bool,
    args: argparse.Namespace,
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
