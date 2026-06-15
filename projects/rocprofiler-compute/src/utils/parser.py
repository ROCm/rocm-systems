# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from utils import schema
from utils.logger import console_error, console_warning, demarcate
from utils.metrics.evaluation_pipeline import eval_metric
from utils.metrics.expression import gen_counter_list
from utils.pattern_matching import fnmatch_glob_matches
from utils.pc_sampling_analysis import (
    SOURCE_LINE_MISSING,
    aggregate_pc_sample_records,
    detect_pc_sampling_method,
    enrich_with_metadata,
    load_pc_sample_records,
)
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


@demarcate
def load_pc_sampling_data_per_kernel(
    method: str,
    tool_data: dict[str, Any],
    sorting_type: str,
    kernel_name: Optional[str] = None,
) -> pd.DataFrame:
    """Build the detailed per-instruction PC sampling table from *tool_data*.

    Filtered to *kernel_name* when given, otherwise every kernel's rows.

    :param method: "host_trap" or "stochastic".
    :param tool_data: The parsed ``rocprofiler-sdk-tool[0]`` dict.
    :param sorting_type: "offset" or "count".
    :param kernel_name: Kernel to filter to, or None for all kernels.
    """
    pc_samples = tool_data["buffer_records"][
        "pc_sample_host_trap" if method == "host_trap" else "pc_sample_stochastic"
    ]
    if not pc_samples:
        console_error("PC sampling: can not find pc sample.")

    instructions = tool_data["strings"]["pc_sample_instructions"]
    comments = tool_data["strings"]["pc_sample_comments"]
    if not instructions or not comments:
        console_error("PC sampling: instruction or comment string table is empty.")

    records_df = load_pc_sample_records(tool_data)
    aggregated_df = aggregate_pc_sample_records(
        records_df,
        group_by=["code_object_id", "code_object_offset", "kernel_id"],
    )
    df = enrich_with_metadata(
        aggregated_df,
        tool_data,
        attach={"instruction", "source_line", "kernel_name"},
    )
    if df.empty:
        console_warning("PC sampling: no records found in PC sampling data.")
        return df

    df = df.rename(
        columns={"code_object_offset": "offset", "kernel_name": "Kernel_Name"}
    )

    if kernel_name is not None:
        df = df[df["Kernel_Name"] == kernel_name]
        if df.empty:
            console_warning(f"PC sampling: cannot find kernel '{kernel_name}'")
            return df

    # Project stall_reason as a descending list[(reason, count)].
    df["stall_reason"] = df["stall_reason"].apply(_stall_reason_dict_to_list)
    df["source_line"] = df["source_line"].apply(_trim_source_line)

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

    # Stochastic adds issue/stall detail on top of the host_trap columns.
    shared_columns = ["source_line", "instruction", "code_object_id", "offset", "count"]
    stochastic_only_columns = ["count_issued", "count_stalled", "stall_reason"]
    columns_to_return = shared_columns + (
        stochastic_only_columns if method == "stochastic" else []
    )
    columns_to_return.append("Kernel_Name")
    return df_sorted[columns_to_return]


@demarcate
def load_pc_sampling_data(
    workload: schema.Workload,
    file_prefix: str,
    sorting_type: str,
    tool_data: Optional[dict[str, Any]],
) -> pd.DataFrame:
    """Return the detailed per-instruction table for a single kernel or all.

    Thin dispatcher over :func:`load_pc_sampling_data_per_kernel`: detects the
    method, then builds the table for all kernels (no ``-k``) or a single
    kernel (one ``-k``). The output schema is identical either way. Callers
    pass the already-parsed *tool_data*.
    """
    if not file_prefix or file_prefix.lower() == "none" or tool_data is None:
        return pd.DataFrame()

    pc_sampling_method = detect_pc_sampling_method(tool_data)
    if pc_sampling_method is None:
        console_warning(
            f"PC sampling: can not detect pc sampling method for {file_prefix}"
        )
        return pd.DataFrame()

    # No kernel filter: return every kernel's rows.
    if not workload.filter_kernel_ids:
        return load_pc_sampling_data_per_kernel(
            pc_sampling_method,
            tool_data,
            sorting_type,
        )

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

    kernel_name = kernel_top_df.iloc[kernel_index]["Kernel_Name"]
    return load_pc_sampling_data_per_kernel(
        pc_sampling_method,
        tool_data,
        sorting_type,
        kernel_name,
    )


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
                df.loc[0, "from_pc_sampling"],
                args.pc_sampling_sorting_type,
                pc_sampling_tool_data,
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
