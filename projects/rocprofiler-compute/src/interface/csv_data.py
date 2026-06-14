# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""CSV profile artifact readers and writers."""

from __future__ import annotations

import re
import shutil
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import utils.utils_profile_csv as csv_ops
from interface.pmc_frame import prepare_pmc_frame
from interface.profile_artifacts import ArtifactReaderOptions, ProfilePassContext
from utils.logger import (
    console_debug,
    console_error,
    console_log,
    console_warning,
    demarcate,
)

if TYPE_CHECKING:
    import pandas as pd

CSV_RESULT_PATTERNS = ["results_pmc_perf_*.csv", "SQ_*.csv", "SQC_*.csv"]
DUPLICATE_COLUMN_PREFIXES = [
    "GPU_ID_",
    "Grid_Size_",
    "Workgroup_Size_",
    "LDS_Per_Workgroup_",
    "Scratch_Per_Workitem_",
    "vgpr_",
    "Arch_VGPR_",
    "Accum_VGPR_",
    "SGPR_",
    "Dispatch_ID_",
    "Kernel_ID_",
    "Queue_ID",
    "Queue_Index",
    "PID",
    "TID",
    "SIG",
    "OBJ",
    "Correlation_ID_",
    "Wave_Size_",
    "dispatch_",
    "sig",
    "queue-id",
    "queue-index",
    "pid",
    "tid",
    "fbar",
]
TIMESTAMP_COLUMN_PATTERNS = ["DispatchNs", "CompleteNs", "HostDuration"]
CSV_OUTPUT_HEADERS = {
    "KernelName": "Kernel_Name",
    "Index": "Dispatch_ID",
    "grd": "Grid_Size",
    "gpu-id": "GPU_ID",
    "wgr": "Workgroup_Size",
    "lds": "LDS_Per_Workgroup",
    "scr": "Scratch_Per_Workitem",
    "sgpr": "SGPR",
    "arch_vgpr": "Arch_VGPR",
    "accum_vgpr": "Accum_VGPR",
    "BeginNs": "Start_Timestamp",
    "EndNs": "End_Timestamp",
    "GRD": "Grid_Size",
    "WGR": "Workgroup_Size",
    "LDS": "LDS_Per_Workgroup",
    "SCR": "Scratch_Per_Workitem",
    "ACCUM_VGPR": "Accum_VGPR",
}


def pmc_perf_path(workload_dir: Path) -> Path:
    """Return the pmc_perf.csv path for a workload directory."""
    return workload_dir / "pmc_perf.csv"


def find_csv_result_files(workload_dir: Path, kokkos_trace: bool) -> list[Path]:
    """Find CSV result files for profile counter artifacts."""
    files = [
        file for pattern in CSV_RESULT_PATTERNS for file in workload_dir.glob(pattern)
    ]
    if kokkos_trace:
        return [
            file
            for file in files
            if not file.name.endswith("_marker_api_trace.csv")
        ]
    return files


def read_csv_as_dicts(csv_file: Path) -> tuple[list[dict], list[str]]:
    """Read a CSV file as row dictionaries."""
    return csv_ops.read_csv_as_dicts(str(csv_file))


def write_csv_from_dicts(csv_file: Path, rows: list[dict]) -> None:
    """Write row dictionaries to a CSV file."""
    csv_ops.write_csv_from_dicts(str(csv_file), rows)


def concat_csv_files(input_files: list[str]) -> list[dict]:
    """Concatenate CSV files into row dictionaries."""
    return csv_ops.concat_csv_files(input_files)


def join_csv_prof_files(
    workload_dir: Path,
    *,
    join_type: str,
    kokkos_trace: bool,
) -> Optional[pd.DataFrame]:
    """Join CSV profiler outputs into the pmc_perf dataframe."""
    joined_frame = _merge_csv_result_files(
        find_csv_result_files(workload_dir, kokkos_trace),
        join_type,
    )
    if joined_frame is None or joined_frame.empty:
        console_warning("join_prof", "No data available after processing all files")
        return None

    _warn_on_mismatched_duplicate_columns(joined_frame)
    joined_frame = _drop_duplicate_columns(joined_frame)
    joined_frame = _drop_timestamp_columns(joined_frame)
    joined_frame = _drop_duplicate_kernel_name_columns(joined_frame)
    return _replace_replay_timestamps(joined_frame)


def v3_counter_csv_to_v2_csv(
    counter_file: str,
    agent_info_filepath: str,
    converted_csv_file: str,
) -> None:
    """Convert rocprofv3 counter collection CSV rows to v2-compatible rows."""
    counter_collections, _ = csv_ops.read_csv_as_dicts(counter_file)
    agent_info, _ = csv_ops.read_csv_as_dicts(agent_info_filepath)

    if counter_collections and "Accum_VGPR_Count" not in counter_collections[0]:
        csv_ops.add_column_to_rows(
            counter_collections,
            "Accum_VGPR_Count",
            [0] * len(counter_collections),
        )

    result = csv_ops.pivot_table(
        counter_collections,
        index_columns=[
            "Correlation_Id",
            "Dispatch_Id",
            "Agent_Id",
            "Queue_Id",
            "Process_Id",
            "Thread_Id",
            "Grid_Size",
            "Kernel_Id",
            "Kernel_Name",
            "Workgroup_Size",
            "LDS_Block_Size",
            "Scratch_Size",
            "VGPR_Count",
            "Accum_VGPR_Count",
            "SGPR_Count",
            "Start_Timestamp",
            "End_Timestamp",
        ],
        pivot_column="Counter_Name",
        value_column="Counter_Value",
    )

    _normalize_agent_ids(result)
    result = _merge_wave_front_size(result, agent_info)
    _map_gpu_ids(result, agent_info)
    csv_ops.drop_column_from_rows(result, "Node_Id")
    csv_ops.rename_columns(result, _rocprofv3_to_pmc_column_mapping())
    ordered_fieldnames = _ordered_counter_fieldnames(result)
    ordered_fieldnames = _rename_accumulator_columns(result, ordered_fieldnames)
    csv_ops.write_csv_from_dicts(
        converted_csv_file,
        result,
        fieldnames=ordered_fieldnames,
    )


def convert_native_counter_collection_csv(workload_dir: str) -> None:
    """Convert native counter collection CSV files to rocprofiler-sdk format."""
    for native_path in (Path(workload_dir) / "out" / "pmc_1").glob(
        "*_native_counter_collection.csv"
    ):
        _convert_native_counter_file(workload_dir, native_path)


def process_rocprofv3_output(workload_dir: str, using_native_tool: bool) -> list[str]:
    """Process rocprofv3 counter output into converted CSV files."""
    if using_native_tool:
        try:
            convert_native_counter_collection_csv(workload_dir)
        except Exception:
            console_error(
                "Error converting native counter collection csv.\n"
                f"Stacktrace:\n{traceback.format_exc()}"
            )

    counter_files = [
        str(path)
        for path in (Path(workload_dir) / "out" / "pmc_1").glob(
            "*/*_counter_collection.csv"
        )
        if path.is_file()
    ]
    if not counter_files:
        return []

    for counter_file in counter_files:
        if not _convert_rocprofv3_counter_file(counter_file):
            return []

    return [
        str(path)
        for path in (Path(workload_dir) / "out" / "pmc_1").glob("*/*_converted.csv")
    ]


@demarcate
def save_torch_trace_inputs(
    workload_dir: str,
    fbase: str,
    output_format: str = "rocpd",
) -> None:
    """Move torch trace inputs to the workload directory for analyze mode."""
    src_dir = Path(workload_dir) / "out" / "pmc_1"
    if output_format == "rocpd":
        _save_rocpd_torch_trace_inputs(workload_dir, fbase, src_dir)
        return

    if output_format == "csv":
        _save_csv_torch_trace_inputs(workload_dir, fbase, src_dir)
        return

    console_warning(
        "torch trace",
        f"Unknown output_format: {output_format} in save_torch_trace_inputs",
    )


@demarcate
def process_kokkos_trace_output(workload_dir: str, fbase: str) -> None:
    """Copy combined Kokkos marker trace output to the workload directory."""
    marker_files = [
        str(path)
        for path in (Path(workload_dir) / "out" / "pmc_1").glob(
            "*/*_marker_api_trace.csv"
        )
        if path.is_file()
    ]
    combined_results = csv_ops.concat_csv_files(marker_files)
    marker_path = Path(workload_dir) / "out" / "pmc_1" / (
        f"results_{fbase}_marker_api_trace.csv"
    )
    csv_ops.write_csv_from_dicts(str(marker_path), combined_results)

    output_dir = Path(workload_dir) / "out"
    if output_dir.exists():
        shutil.copyfile(
            marker_path,
            Path(workload_dir) / f"{fbase}_marker_api_trace.csv",
        )


def _pandas() -> Any:  # noqa: ANN401
    import pandas as pd

    return pd


def _merge_csv_result_files(
    result_files: list[Path],
    join_type: str,
) -> Optional[pd.DataFrame]:
    pd = _pandas()
    joined_frame: Optional[pd.DataFrame] = None
    for index, result_file in enumerate(result_files):
        current_frame = pd.read_csv(result_file)
        if current_frame.empty:
            console_warning("join_prof", f"Empty dataframe from {result_file}")
            continue

        current_frame = _rename_accumulator_column(current_frame, result_file)
        current_frame["key"] = _build_join_key(current_frame, join_type)
        if joined_frame is None:
            joined_frame = current_frame
            continue
        joined_frame = pd.merge(
            joined_frame,
            current_frame,
            how="inner",
            on="key",
            suffixes=("", f"_{index}"),
        )
    return joined_frame


def _rename_accumulator_column(
    current_frame: pd.DataFrame,
    result_file: Path,
) -> pd.DataFrame:
    if not _has_accumulator_alias(current_frame, result_file):
        return current_frame
    target = result_file.stem[len("results_pmc_perf_") :]
    return current_frame.rename(columns={"SQ_ACCUM_PREV_HIRES": target})


def _has_accumulator_alias(current_frame: pd.DataFrame, result_file: Path) -> bool:
    return (
        result_file.name.startswith("results_pmc_perf_")
        and result_file.stem.endswith("_ACCUM")
        and "SQ_ACCUM_PREV_HIRES" in current_frame.columns
    )


def _build_join_key(current_frame: pd.DataFrame, join_type: str) -> pd.Series:
    if join_type == "kernel":
        replay_index = current_frame.groupby("Kernel_Name").cumcount()
        return current_frame.Kernel_Name + " - " + replay_index.astype(str)
    if join_type == "grid":
        replay_index = current_frame.groupby(["Kernel_Name", "Grid_Size"]).cumcount()
        return (
            current_frame["Kernel_Name"].astype(str)
            + " - "
            + current_frame["Grid_Size"].astype(str)
            + " - "
            + replay_index.astype(str)
        )
    console_error("join_prof", f"{join_type} is an unrecognized option for --join-type")
    return _pandas().Series(dtype=str)


def _warn_on_mismatched_duplicate_columns(joined_frame: pd.DataFrame) -> None:
    for key, columns in _duplicate_columns(joined_frame).items():
        if not columns:
            continue
        current_frame = joined_frame[columns]
        if _all_columns_equal(current_frame):
            console_debug("join_prof", f"Successfully joined {key} in pmc_perf.csv")
            continue
        console_warning(
            "join_prof",
            f"Detected differing {key} values while joining pmc_perf.csv",
        )


def _duplicate_columns(joined_frame: pd.DataFrame) -> dict[str, list[str]]:
    duplicate_columns = {
        "GPU_ID": [col for col in joined_frame.columns if col.startswith("GPU_ID")],
        "Grid_Size": [
            col for col in joined_frame.columns if col.startswith("Grid_Size")
        ],
        "Workgroup_Size": [
            col for col in joined_frame.columns if col.startswith("Workgroup_Size")
        ],
        "LDS_Per_Workgroup": [
            col for col in joined_frame.columns if col.startswith("LDS_Per_Workgroup")
        ],
        "Scratch_Per_Workitem": [
            col
            for col in joined_frame.columns
            if col.startswith("Scratch_Per_Workitem")
        ],
        "SGPR": [col for col in joined_frame.columns if col.startswith("SGPR")],
    }
    if "vgpr" in joined_frame.columns:
        duplicate_columns["vgpr"] = [
            col for col in joined_frame.columns if col.startswith("vgpr")
        ]
        return duplicate_columns

    duplicate_columns["Arch_VGPR"] = [
        col for col in joined_frame.columns if col.startswith("Arch_VGPR")
    ]
    duplicate_columns["Accum_VGPR"] = [
        col for col in joined_frame.columns if col.startswith("Accum_VGPR")
    ]
    return duplicate_columns


def _all_columns_equal(frame: pd.DataFrame) -> bool:
    return frame.eq(frame.iloc[:, 0], axis=0).all(1).all()


def _drop_duplicate_columns(joined_frame: pd.DataFrame) -> pd.DataFrame:
    return joined_frame[
        [
            col
            for col in joined_frame.columns
            if not any(col.startswith(prefix) for prefix in DUPLICATE_COLUMN_PREFIXES)
        ]
    ]


def _drop_timestamp_columns(joined_frame: pd.DataFrame) -> pd.DataFrame:
    return joined_frame[
        [
            col
            for col in joined_frame.columns
            if not any(pattern in col for pattern in TIMESTAMP_COLUMN_PATTERNS)
        ]
    ]


def _drop_duplicate_kernel_name_columns(
    joined_frame: pd.DataFrame,
) -> pd.DataFrame:
    name_columns = [col for col in joined_frame.columns if "Kernel_Name" in col]
    if not name_columns:
        return joined_frame

    for name_column in name_columns[1:]:
        assert (joined_frame[name_columns[0]] == joined_frame[name_column]).all()
    return joined_frame.drop(columns=name_columns[1:])


def _replace_replay_timestamps(joined_frame: pd.DataFrame) -> pd.DataFrame:
    start_columns = [col for col in joined_frame.columns if "Start_Timestamp" in col]
    end_columns = [col for col in joined_frame.columns if "End_Timestamp" in col]
    if not start_columns or not end_columns:
        return _drop_join_key(joined_frame)

    joined_frame = joined_frame.copy()
    mean_start = joined_frame[start_columns].mean(axis=1)
    mean_end = joined_frame[end_columns].mean(axis=1)
    joined_frame = joined_frame.drop(columns=start_columns + end_columns)
    joined_frame["Start_Timestamp"] = mean_start
    joined_frame["End_Timestamp"] = mean_end
    return _drop_join_key(joined_frame)


def _drop_join_key(joined_frame: pd.DataFrame) -> pd.DataFrame:
    if "key" not in joined_frame.columns:
        return joined_frame
    return joined_frame.drop(columns=["key"])


def _normalize_agent_ids(result: list[dict]) -> None:
    if not result or not isinstance(result[0].get("Agent_Id"), str):
        console_debug("Agent ID is already numeric type")
        return

    console_debug("Agent ID is string type, converting to int")
    try:
        for row in result:
            agent_id = row.get("Agent_Id", "")
            if not isinstance(agent_id, str) or "Agent " not in agent_id:
                continue
            match = re.search(r"Agent (\d+)", agent_id)
            if match:
                row["Agent_Id"] = match.group(1)
    except Exception as error:
        console_error(
            "v3_counter_csv_to_v2_csv",
            f'Error getting "Agent_Id": {error}',
        )


def _merge_wave_front_size(result: list[dict], agent_info: list[dict]) -> list[dict]:
    agent_info_subset = [
        {"Node_Id": row.get("Node_Id"), "Wave_Front_Size": row.get("Wave_Front_Size")}
        for row in agent_info
    ]
    return csv_ops.merge_rows(
        result,
        agent_info_subset,
        left_on="Agent_Id",
        right_on="Node_Id",
        how="left",
    )


def _map_gpu_ids(result: list[dict], agent_info: list[dict]) -> None:
    gpu_agents = [row for row in agent_info if row.get("Agent_Type") == "GPU"]
    gpu_id_map = {row.get("Node_Id"): index for index, row in enumerate(gpu_agents)}
    for row in result:
        agent_id = row.get("Agent_Id")
        if agent_id in gpu_id_map:
            row["Agent_Id"] = gpu_id_map[agent_id]


def _rocprofv3_to_pmc_column_mapping() -> dict[str, str]:
    return {
        "Dispatch_Id": "Dispatch_ID",
        "Agent_Id": "GPU_ID",
        "Queue_Id": "Queue_ID",
        "Process_Id": "PID",
        "Thread_Id": "TID",
        "Grid_Size": "Grid_Size",
        "Workgroup_Size": "Workgroup_Size",
        "LDS_Block_Size": "LDS_Per_Workgroup",
        "Scratch_Size": "Scratch_Per_Workitem",
        "VGPR_Count": "Arch_VGPR",
        "Accum_VGPR_Count": "Accum_VGPR",
        "SGPR_Count": "SGPR",
        "Wave_Front_Size": "Wave_Size",
        "Kernel_Name": "Kernel_Name",
        "Start_Timestamp": "Start_Timestamp",
        "End_Timestamp": "End_Timestamp",
        "Correlation_Id": "Correlation_ID",
        "Kernel_Id": "Kernel_ID",
    }


def _ordered_counter_fieldnames(result: list[dict]) -> list[str]:
    preferred_order = [
        "Dispatch_ID",
        "GPU_ID",
        "Queue_ID",
        "PID",
        "TID",
        "Grid_Size",
        "Workgroup_Size",
        "LDS_Per_Workgroup",
        "Scratch_Per_Workitem",
        "Arch_VGPR",
        "Accum_VGPR",
        "SGPR",
        "Wave_Size",
        "Kernel_Name",
        "Start_Timestamp",
        "End_Timestamp",
        "Correlation_ID",
        "Kernel_ID",
    ]
    if not result:
        return preferred_order

    all_columns = list(result[0].keys())
    remaining_columns = [col for col in all_columns if col not in preferred_order]
    return preferred_order + remaining_columns


def _rename_accumulator_columns(
    result: list[dict],
    ordered_fieldnames: list[str],
) -> list[str]:
    if not result:
        return ordered_fieldnames

    accum_mapping = {
        col: "SQ_ACCUM_PREV_HIRES"
        for col in result[0]
        if col.endswith("_ACCUM")
    }
    if not accum_mapping:
        return ordered_fieldnames

    csv_ops.rename_columns(result, accum_mapping)
    return [accum_mapping.get(col, col) for col in ordered_fieldnames]


def _convert_native_counter_file(workload_dir: str, native_path: Path) -> None:
    counter_data, _ = csv_ops.read_csv_as_dicts(str(native_path))
    groupby_cols = ["dispatch_id", "counter_name"]
    agg_dict = _native_counter_aggregation(counter_data, groupby_cols)
    agg_dict["counter_value"] = "sum"
    counter_data = csv_ops.groupby_aggregate(counter_data, groupby_cols, agg_dict)

    pid = native_path.stem.split("_")[0]
    kernel_data_filename = str(
        next((Path(workload_dir) / "out" / "pmc_1").glob(f"*/{pid}_kernel_trace.csv"))
    )
    kernel_data, _ = csv_ops.read_csv_as_dicts(kernel_data_filename)
    merged_data = csv_ops.merge_rows(
        counter_data,
        kernel_data,
        left_on="dispatch_id",
        right_on="Dispatch_Id",
        how="inner",
    )
    rocprofv3_counter_data = [
        _native_counter_row_to_rocprofv3_row(row) for row in merged_data
    ]
    csv_ops.write_csv_from_dicts(
        kernel_data_filename.replace("kernel_trace", "counter_collection"),
        rocprofv3_counter_data,
    )


def _native_counter_aggregation(
    counter_data: list[dict],
    groupby_cols: list[str],
) -> dict[str, str]:
    if not counter_data:
        return {}
    return {col: "first" for col in counter_data[0] if col not in groupby_cols}


def _native_counter_row_to_rocprofv3_row(row: dict) -> dict:
    return {
        "Correlation_Id": row.get("Correlation_Id"),
        "Dispatch_Id": row.get("dispatch_id"),
        "Agent_Id": row.get("Agent_Id"),
        "Queue_Id": row.get("Queue_Id"),
        "Process_Id": row.get("Thread_Id"),
        "Thread_Id": row.get("Thread_Id"),
        "Grid_Size": (
            int(row.get("Grid_Size_X", 1))
            * int(row.get("Grid_Size_Y", 1))
            * int(row.get("Grid_Size_Z", 1))
        ),
        "Kernel_Id": row.get("Kernel_Id"),
        "Kernel_Name": row.get("Kernel_Name"),
        "Workgroup_Size": (
            int(row.get("Workgroup_Size_X", 1))
            * int(row.get("Workgroup_Size_Y", 1))
            * int(row.get("Workgroup_Size_Z", 1))
        ),
        "LDS_Block_Size": row.get("LDS_Block_Size"),
        "Scratch_Size": row.get("Scratch_Size"),
        "VGPR_Count": row.get("VGPR_Count"),
        "Accum_VGPR_Count": row.get("Accum_VGPR_Count"),
        "SGPR_Count": row.get("SGPR_Count"),
        "Counter_Name": row.get("counter_name"),
        "Counter_Value": row.get("counter_value"),
        "Start_Timestamp": row.get("Start_Timestamp"),
        "End_Timestamp": row.get("End_Timestamp"),
    }


def _convert_rocprofv3_counter_file(counter_file: str) -> bool:
    counter_path = Path(counter_file)
    current_dir = counter_path.parent
    agent_info_filepath = current_dir / counter_path.name.replace(
        "_counter_collection",
        "_agent_info",
    )
    if not agent_info_filepath.is_file():
        raise ValueError(f'{counter_file} has no corresponding "agent info" file')

    converted_csv_file = current_dir / counter_path.name.replace(
        "_counter_collection",
        "_converted",
    )
    try:
        v3_counter_csv_to_v2_csv(
            counter_file,
            str(agent_info_filepath),
            str(converted_csv_file),
        )
    except Exception as error:
        console_warning(
            f"Error converting {counter_file} from v3 to v2 csv: {error}"
        )
        return False
    return True


def _save_rocpd_torch_trace_inputs(
    workload_dir: str,
    fbase: str,
    src_dir: Path,
) -> None:
    src_counter = src_dir / f"{fbase}_counter_collection.csv"
    src_marker = src_dir / f"{fbase}_marker_api_trace.csv"
    dst_counter = Path(workload_dir) / f"torch_trace_{fbase}_counter_collection.csv"
    dst_marker = Path(workload_dir) / f"torch_trace_{fbase}_marker_api_trace.csv"
    shutil.copyfile(src_counter, dst_counter)
    shutil.copyfile(src_marker, dst_marker)
    console_log(
        "torch trace",
        "Moved counter collection and marker trace files "
        "to workload dir for PyTorch trace creation.",
    )
    console_log("Counter Collection: ", str(dst_counter))
    console_log("Marker API Trace: ", str(dst_marker))


def _save_csv_torch_trace_inputs(
    workload_dir: str,
    fbase: str,
    src_dir: Path,
) -> None:
    counter_files = list(src_dir.glob("*/*_counter_collection.csv"))
    marker_files = list(src_dir.glob("*/*_marker_api_trace.csv"))
    (Path(workload_dir) / fbase).mkdir(parents=True, exist_ok=True)
    for src_counter in counter_files:
        dst_counter = (
            Path(workload_dir) / fbase / ("torch_trace_" + src_counter.name)
        )
        shutil.copyfile(src_counter, dst_counter)
        console_log("torch trace", f"Copied Counter Collection: {dst_counter}")
    for src_marker in marker_files:
        dst_marker = Path(workload_dir) / fbase / ("torch_trace_" + src_marker.name)
        shutil.copyfile(src_marker, dst_marker)
        console_log("torch trace", f"Copied Marker API Trace: {dst_marker}")


class CsvAnalysisData:
    """Read and materialize CSV artifacts for analyze mode."""

    def __init__(self, join_type: str = "grid", kokkos_trace: bool = False) -> None:
        self._join_type = join_type
        self._kokkos_trace = kokkos_trace

    def has_artifacts(self, workload_dir: Path) -> bool:
        return pmc_perf_path(workload_dir).exists() or bool(
            find_csv_result_files(workload_dir, self._kokkos_trace)
        )

    def materialize_pmc_perf(self, workload_dir: Path, output_path: Path) -> Path:
        if output_path.exists():
            console_debug(f"Using existing {output_path}")
            return output_path

        joined_frame = join_csv_prof_files(
            workload_dir,
            join_type=self._join_type,
            kokkos_trace=self._kokkos_trace,
        )
        if joined_frame is None:
            return output_path

        joined_frame.to_csv(output_path, index=False)
        console_debug(f"Created file: {output_path}")
        return output_path

    def read_pmc_frame(
        self,
        workload_dir: Path,
        *,
        kernel_verbose: int,
        verbose: int,
        node_name: Optional[str] = None,
    ) -> pd.DataFrame:
        pd = _pandas()
        pmc_path = pmc_perf_path(workload_dir)
        if not pmc_path.is_file():
            return pd.DataFrame()

        frame = pd.read_csv(pmc_path)
        return prepare_pmc_frame(
            frame,
            kernel_verbose=kernel_verbose,
            verbose=verbose,
            node_name=node_name,
        )


class CsvProfileData:
    """Write and normalize CSV artifacts for profile mode."""

    def normalize_counter_rows(self, combined_results: list[dict]) -> None:
        csv_ops.add_column_to_rows(
            combined_results,
            "Dispatch_ID",
            list(range(0, len(combined_results))),
        )
        csv_ops.assign_group_ids(
            combined_results,
            ["Kernel_Name", "Grid_Size", "Workgroup_Size", "LDS_Per_Workgroup"],
            "Kernel_ID",
        )

    def write_counter_results(
        self,
        combined_results: list[dict],
        workload_dir: Path,
        fbase: str,
    ) -> None:
        csv_ops.write_csv_from_dicts(
            str(workload_dir / "out" / "pmc_1" / f"results_{fbase}.csv"),
            combined_results,
        )
        if (workload_dir / "out").exists():
            shutil.copyfile(
                str(workload_dir / "out" / "pmc_1" / f"results_{fbase}.csv"),
                str(workload_dir / f"results_{fbase}.csv"),
            )
            shutil.rmtree(str(workload_dir / "out"))

    def standardize_headers(self, workload_dir: Path, fbase: str) -> None:
        csv_path = workload_dir / f"results_{fbase}.csv"
        rows, _ = csv_ops.read_csv_as_dicts(str(csv_path))
        csv_ops.rename_columns(rows, CSV_OUTPUT_HEADERS)
        csv_ops.write_csv_from_dicts(str(csv_path), rows)


class CsvProfileArtifactReader:
    """Read current CSV profile artifacts."""

    def __init__(self, options: ArtifactReaderOptions) -> None:
        self._options = options
        self._csv_data = CsvAnalysisData(
            join_type=options.join_type,
            kokkos_trace=options.kokkos_trace,
        )

    def has_artifacts(self, workload_dir: Path) -> bool:
        return self._csv_data.has_artifacts(workload_dir)

    def materialize_pmc_perf(self, workload_dir: Path, output_path: Path) -> Path:
        return self._csv_data.materialize_pmc_perf(workload_dir, output_path)

    def read_pmc_frame(self, workload_dir: Path) -> pd.DataFrame:
        return self._csv_data.read_pmc_frame(
            workload_dir,
            kernel_verbose=self._options.kernel_verbose,
            verbose=self._options.verbose,
        )


class CsvProfileArtifactWriter:
    """Finalize current CSV profile artifacts."""

    def finalize_pass(self, context: ProfilePassContext) -> None:
        result_files = self._process_csv_outputs(context)
        if context.torch_trace_enabled:
            save_torch_trace_inputs(
                str(context.workload_dir),
                context.fbase,
                "csv",
            )
        if not result_files:
            console_warning(
                f"Cannot write results for {context.fbase}.csv due to no counter "
                "csv files generated."
            )
            return

        csv_profile_data = CsvProfileData()
        combined_results = concat_csv_files(result_files)
        csv_profile_data.normalize_counter_rows(combined_results)
        csv_profile_data.write_counter_results(
            combined_results,
            context.workload_dir,
            context.fbase,
        )
        csv_profile_data.standardize_headers(context.workload_dir, context.fbase)

    def _process_csv_outputs(self, context: ProfilePassContext) -> list[str]:
        if context.profiler_command == "rocprofiler-sdk":
            return process_rocprofv3_output(
                str(context.workload_dir),
                using_native_tool=context.using_native_tool,
            )

        result_files = process_rocprofv3_output(
            str(context.workload_dir),
            using_native_tool=False,
        )
        if context.kokkos_trace_enabled:
            process_kokkos_trace_output(str(context.workload_dir), context.fbase)
        return result_files
