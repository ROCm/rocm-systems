# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

import config
from utils import csv_compression, schema, utils_analysis
from utils.logger import (
    console_debug,
    console_error,
    console_log,
    console_warning,
    demarcate,
)
from utils.utils_common import (
    canonical_config_arch,
    normalize_filter_to_str_list,
)

# TODO: use pandas chunksize or dask to read really large csv file
# from dask import dataframe as dd


def load_panel_configs(
    dirs: list[str],
) -> OrderedDict[int, dict[str, Any]]:
    """
    Load all panel configs from yaml file.
    """
    configs: dict[int, dict[str, Any]] = {}
    for dir_path in dirs:
        for yaml_file in Path(dir_path).glob("*.yaml"):
            with open(yaml_file, encoding="utf-8") as file:
                config_yml = yaml.safe_load(file)
                # metric key can be None due to some metric-
                # tables not having any metrics
                # metric key should be empty dict instead of None
                panel_config = config_yml["Panel Config"]
                for data_source in panel_config["data source"]:
                    metric_table = data_source.get("metric_table")
                    if metric_table and metric_table["metric"] is None:
                        metric_table["metric"] = {}
                configs[panel_config["id"]] = panel_config

    # TODO: sort metrics as the header order in case they-
    # are not defined in the same order
    return OrderedDict(sorted(configs.items()))


def load_profiling_config(config_dir: str) -> dict[str, Any]:
    """
    Load profiling config from yaml file.
    """
    config_path = Path(config_dir) / "profiling_config.yaml"
    try:
        with open(config_path, encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
    except FileNotFoundError:
        console_log(f"Could not find profiling_config.yaml in {config_dir}")
    return {}


def rank_kernels_by_total_duration(dispatch_frame: pd.DataFrame) -> list[str]:
    """Return kernel names ordered by total dispatch duration, longest first.

    A kernel's position in this list is the id that ``-k`` selects.
    """
    durations = dispatch_frame["End_Timestamp"] - dispatch_frame["Start_Timestamp"]
    return (
        durations
        .groupby(dispatch_frame["Kernel_Name"])
        .sum()
        .sort_values(ascending=False)
        .index.to_list()
    )


def validate_kernel_filter_ids(
    filter_kernel_ids: list[int],
    kernel_count: int,
) -> None:
    """Exit with a readable message when a ``-k`` id names no kernel."""
    if kernel_count == 0:
        console_error("analysis", "No kernels found in this workload.")

    for kernel_id in filter_kernel_ids:
        if not 0 <= kernel_id < kernel_count:
            console_error(
                "analysis",
                f"{kernel_id} is an invalid kernel id. "
                f"Please enter an id between 0-{kernel_count - 1}",
            )


@demarcate
def create_df_kernel_top_stats(
    df_in: pd.DataFrame,
    raw_data_dir: str,
    filter_gpu_ids: Optional[list[str]],
    filter_dispatch_ids: Optional[list[str]],
    time_unit: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create top stats info by grouping kernels with user's filters.

    Returns:
        A tuple of (kernel_top_df, dispatch_info_df).
    """

    df = df_in.copy()

    # The logic below for filters are the same as in parser.apply_filters(),
    # which can be merged together if need it.

    if filter_gpu_ids:
        df = df.loc[
            df["GPU_ID"].astype(str).isin(normalize_filter_to_str_list(filter_gpu_ids))
        ]

    if filter_dispatch_ids:
        # NB: support ignoring the 1st n dispatched execution by '> n'
        #     The better way may be parsing python slice string
        first_filter = filter_dispatch_ids[0]

        if isinstance(first_filter, str) and first_filter.startswith(">"):
            match = re.match(r">\s*(\d+)", str(first_filter))
            if match:
                threshold = int(match.group(1))
                df = df[df["Dispatch_ID"] > threshold]
        else:
            filter_strings = [str(f) for f in filter_dispatch_ids]
            df = df.loc[df["Dispatch_ID"].astype(str).isin(filter_strings)]

    # First, create a dispatches file used to populate global vars
    dispatch_columns = ["Dispatch_ID", "Kernel_Name", "GPU_ID"]
    if "PID" in df.columns:
        dispatch_columns.insert(1, "PID")

    dispatch_info = df[dispatch_columns]
    dispatch_output_path = Path(raw_data_dir) / "pmc_dispatch_info.csv"
    dispatch_info.to_csv(dispatch_output_path, index=False)

    # Calculate execution times
    execution_times = df["End_Timestamp"] - df["Start_Timestamp"]
    time_stats = pd.DataFrame({
        "Kernel_Name": df["Kernel_Name"],
        "ExeTime": execution_times,
    })

    grouped = time_stats.groupby("Kernel_Name")["ExeTime"].agg([
        "count",
        "sum",
        "mean",
        "median",
    ])

    # Rename columns with time unit
    time_unit_suffix = f"({time_unit})"
    column_mapping = {
        "count": "Count",
        "sum": f"Sum{time_unit_suffix}",
        "mean": f"Mean{time_unit_suffix}",
        "median": f"Median{time_unit_suffix}",
    }
    grouped = grouped.rename(columns=column_mapping)

    # Convert time units
    time_divisor = config.TIME_UNITS[time_unit]
    for col in [
        f"Sum{time_unit_suffix}",
        f"Mean{time_unit_suffix}",
        f"Median{time_unit_suffix}",
    ]:
        grouped[col] = grouped[col] / time_divisor

    grouped = grouped.reset_index()

    # Calculate percent
    sum_column = f"Sum{time_unit_suffix}"
    grouped["Percent"] = grouped[sum_column] / grouped[sum_column].sum() * 100

    kernel_order = rank_kernels_by_total_duration(df)
    grouped = grouped.set_index("Kernel_Name").loc[kernel_order].reset_index()
    grouped.to_csv(str(Path(raw_data_dir) / "pmc_kernel_top.csv"), index=False)

    return grouped.reset_index(drop=True), dispatch_info.reset_index(drop=True)


def build_agent_to_gpu_map_from_json(
    agents: list[dict[str, Any]],
) -> dict[int, int]:
    """
    Map agent ``id.handle`` values to 0-indexed GPU IDs.

    GPU agents are identified by the rocprofiler-sdk agent ``type`` enum
    value 2 in the ``agents`` array of ``<pid>_ps_file_results.json``.  They
    are sorted by ``node_id`` so that the first GPU agent maps to GPU 0,
    the second to GPU 1, etc.
    """
    rocprofiler_agent_type_gpu = 2
    gpu_agents = sorted(
        (agent for agent in agents if agent.get("type") == rocprofiler_agent_type_gpu),
        key=lambda agent: agent["node_id"],
    )
    return {agent["id"]["handle"]: index for index, agent in enumerate(gpu_agents)}


@demarcate
def load_pc_sampling_results(workload_path: str) -> list[dict[str, Any]]:
    """Load valid PC sampling tool records for a workload.

    ``<pid>_ps_file_results.json`` records are returned in numeric PID order.
    Malformed files are skipped with a warning.

    Result files can be multiple GB, so parse each once here and share the records
    with every PC sampling consumer instead of re-reading the files.
    """
    tool_records = []
    for result_file in _find_pid_prefixed_pc_sampling_result_files(Path(workload_path)):
        tool_record = _parse_pc_sampling_result_file(result_file)
        if tool_record is None:
            continue
        tool_records.append(tool_record)
    _validate_pc_sampling_process_ids(tool_records)
    return tool_records


def process_pc_sampling_kernel_traces(
    tool_data_records: list[dict[str, Any]],
) -> pd.DataFrame:
    """Build one dispatch trace containing every PC-sampling tool record."""
    if not tool_data_records:
        return process_pc_sampling_kernel_trace(None)

    combined_trace = pd.concat(
        [
            process_pc_sampling_kernel_trace(tool_data)
            for tool_data in tool_data_records
        ],
        ignore_index=True,
    )
    return _renumber_dispatch_ids_across_processes(combined_trace)


def process_pc_sampling_kernel_trace(
    tool_data: Optional[dict[str, Any]],
) -> pd.DataFrame:
    """
    Build kernel and dispatch info from the kernel dispatch records.

    Used for PC-sampling-only runs where ``pmc_perf`` data is not
    available.  Consumes a parsed ``rocprofiler-sdk-tool[0]`` dict
    (see ``load_pc_sampling_results``): kernel dispatch buffer records for
    timestamps and dispatch info, ``kernel_symbols`` for kernel names, and
    ``agents`` for the GPU ID mapping.  Returns an empty frame when
    *tool_data* is ``None`` (results json absent).
    """
    columns = [
        "Dispatch_Id",
        "PID",
        "Kernel_Name",
        "Start_Timestamp",
        "End_Timestamp",
        "GPU_ID",
    ]
    if tool_data is None:
        console_warning("PC sampling results not found. Cannot build dispatch data.")
        return pd.DataFrame(columns=columns)

    process_id = int(tool_data["metadata"]["pid"])
    dispatches = tool_data["buffer_records"]["kernel_dispatch"]
    kernel_id_to_name = {
        symbol["kernel_id"]: symbol["formatted_kernel_name"]
        for symbol in tool_data["kernel_symbols"]
    }
    agent_to_gpu = build_agent_to_gpu_map_from_json(tool_data["agents"])

    rows = [
        {
            "Dispatch_Id": dispatch["dispatch_info"]["dispatch_id"],
            "PID": process_id,
            "Kernel_Name": kernel_id_to_name.get(
                dispatch["dispatch_info"]["kernel_id"]
            ),
            "Start_Timestamp": dispatch["start_timestamp"],
            "End_Timestamp": dispatch["end_timestamp"],
            "GPU_ID": agent_to_gpu.get(
                dispatch["dispatch_info"]["agent_id"]["handle"], 0
            ),
        }
        for dispatch in dispatches
    ]

    return pd.DataFrame(rows, columns=columns)


@demarcate
def create_df_pmc(
    raw_data_dir: str,
    verbose: int,
) -> pd.DataFrame:
    """
    Load all raw pmc counters and join into one df.
    """
    pmc_perf_path = csv_compression.compressed_name(
        Path(raw_data_dir) / f"{schema.PMC_PERF_FILE_PREFIX}.csv"
    )
    if not pmc_perf_path.is_file():
        return pd.DataFrame()

    df = pd.read_csv(pmc_perf_path)

    # The rocpd counter CSV is long: one row per counter per dispatch. Anything
    # else was written by a removed backend and is no longer supported.
    if not {"Counter_Name", "Counter_Value"}.issubset(df.columns):
        console_error(
            "analysis",
            f"{pmc_perf_path} is not in the supported rocpd format. "
            "Please re-profile this workload with a current release.",
        )
    df = utils_analysis.process_rocpd_csv(df)

    utils_analysis.add_unit_counter(df)

    if verbose >= 2:
        console_debug(f"pmc_raw_data final_single_df {df.info}")
    return df


def collect_wave_occu_per_cu(in_dir: str, out_dir: str, num_se: int) -> None:
    """
    Collect wave occupancy info from in_dir csv files
    and consolidate into out_dir/wave_occu_per_cu.csv.
    It depends highly on wave_occu_se*.csv format.
    """
    in_path = Path(in_dir)
    all_data = pd.DataFrame()

    for i in range(num_se):
        file_path = in_path / f"wave_occu_se{i}.csv"
        if not file_path.exists():
            continue

        tmp_df = pd.read_csv(file_path)
        if tmp_df.empty:
            continue

        se_idx = f"SE{tmp_df.loc[0, 'SE']}"
        tmp_df.rename(
            columns={
                "Dispatch": "Dispatch",
                "SE": "SE",
                "CU": "CU",
                "Occupancy": se_idx,
            }
        )

        # TODO: join instead of concat!
        if i == 0:
            all_data = tmp_df[{"CU", se_idx}]
            all_data.sort_index(axis=1, inplace=True)
        else:
            all_data = pd.concat([all_data, tmp_df[se_idx]], axis=1, copy=False)

    if not all_data.empty:
        all_data.to_csv(Path(out_dir) / "wave_occu_per_cu.csv", index=False)


def is_single_panel_config(
    root_dir: str, supported_archs: dict[str, str]
) -> Optional[bool]:
    """
    Check the root configs dir structure to decide using one config set for all
    archs, or one for each arch.
    """
    # If not single config, verify all supported archs have defined configs
    arch_names = {
        canonical_config_arch(arch) or arch for arch in supported_archs.keys()
    }
    root_path = Path(root_dir)
    arch_count = sum(1 for arch in arch_names if (root_path / arch).exists())

    if arch_count == 0:
        return True
    elif arch_count == len(arch_names):
        return False
    else:
        console_warning(
            "Found multiple panel config sets but incomplete for all archs."
        )


def _renumber_dispatch_ids_across_processes(
    combined_trace: pd.DataFrame,
) -> pd.DataFrame:
    """Replace process-local dispatch ids with ids unique across processes.

    ``dispatch_info.dispatch_id`` restarts in every process, so a multi-process
    workload repeats the same id once per process. Counter profiling already
    folds ``PID`` into ``Dispatch_ID`` via ``utils_profile``'s
    ``GroupIdAssigner``, so both analyze paths agree on what a dispatch id means.
    """
    if combined_trace.empty:
        return combined_trace

    renumbered_trace = combined_trace.copy()
    renumbered_trace["Dispatch_Id"] = range(len(renumbered_trace))
    return renumbered_trace


def _find_pid_prefixed_pc_sampling_result_files(
    workload_path: Path,
) -> tuple[Path, ...]:
    """Return the workload's ``<pid>_ps_file_results.json`` in numeric PID order."""
    if not workload_path.is_dir():
        return ()

    results_filename_suffix = "_ps_file_results.json"
    pid_result_candidates: list[Path] = []

    for candidate_path in workload_path.iterdir():
        if not candidate_path.is_file():
            continue
        if not candidate_path.name.endswith(results_filename_suffix):
            continue

        process_identifier_prefix = candidate_path.name[: -len(results_filename_suffix)]
        if re.fullmatch(r"[0-9]+", process_identifier_prefix) is None:
            continue

        pid_result_candidates.append(candidate_path)

    # The PID prefix alone orders the files: it is unique among siblings.
    return tuple(
        sorted(
            pid_result_candidates,
            key=lambda candidate_path: int(
                candidate_path.name[: -len(results_filename_suffix)]
            ),
        )
    )


def _validate_pc_sampling_process_ids(
    tool_data_records: list[dict[str, Any]],
) -> None:
    """Require a concrete, unique process ID for every tool record.

    This is the precondition that lets every downstream consumer index
    ``tool_data["metadata"]["pid"]`` without a guard. ``console_error`` exits by
    default, so a record that reaches those consumers is known to have a pid.
    """
    if not tool_data_records:
        return

    process_ids = [
        tool_data.get("metadata", {}).get("pid") for tool_data in tool_data_records
    ]
    if any(process_id is None for process_id in process_ids):
        console_error("PC sampling: every result record requires metadata.pid.")

    if len(set(process_ids)) != len(process_ids):
        console_error(
            "PC sampling: multiple result records require unique metadata.pid values."
        )


def _parse_pc_sampling_result_file(json_path: Path) -> Optional[dict[str, Any]]:
    """Extract the sole ``rocprofiler-sdk-tool`` record at index 0.

    Each ``<pid>_ps_file_results.json`` output contains exactly one tool record,
    so index 0 is the complete record for that process.

    Log a warning and return ``None`` when the result file is malformed.
    """
    try:
        with json_path.open(encoding="utf-8") as json_file:
            return json.load(json_file)["rocprofiler-sdk-tool"][0]
    except (json.JSONDecodeError, KeyError, IndexError) as error:
        console_warning(f"PC sampling: failed to parse {json_path}: {error}")
        return None
