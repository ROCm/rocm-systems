# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""CSV data interface for profile and analysis artifacts."""

import shutil
from pathlib import Path
from typing import Optional

import pandas as pd

import utils.utils_profile_csv as csv_ops
from utils.logger import console_debug, console_error, console_warning

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

        joined_df = join_csv_prof_files(
            workload_dir,
            join_type=self._join_type,
            kokkos_trace=self._kokkos_trace,
        )
        if joined_df is None:
            return output_path

        joined_df.to_csv(output_path, index=False)
        console_debug(f"Created file: {output_path}")
        return output_path


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
    joined_df = _merge_csv_result_files(
        find_csv_result_files(workload_dir, kokkos_trace),
        join_type,
    )
    if joined_df is None or joined_df.empty:
        console_warning("join_prof", "No data available after processing all files")
        return None

    _warn_on_mismatched_duplicate_columns(joined_df)
    joined_df = _drop_duplicate_columns(joined_df)
    joined_df = _drop_timestamp_columns(joined_df)
    joined_df = _drop_duplicate_kernel_name_columns(joined_df)
    return _replace_replay_timestamps(joined_df)


def _merge_csv_result_files(
    result_files: list[Path],
    join_type: str,
) -> Optional[pd.DataFrame]:
    joined_df: Optional[pd.DataFrame] = None
    for index, result_file in enumerate(result_files):
        current_df = pd.read_csv(result_file)
        if current_df.empty:
            console_warning("join_prof", f"Empty dataframe from {result_file}")
            continue

        current_df = _rename_accumulator_column(current_df, result_file)
        current_df["key"] = _build_join_key(current_df, join_type)
        if joined_df is None:
            joined_df = current_df
            continue
        joined_df = pd.merge(
            joined_df,
            current_df,
            how="inner",
            on="key",
            suffixes=("", f"_{index}"),
        )
    return joined_df


def _rename_accumulator_column(
    current_df: pd.DataFrame,
    result_file: Path,
) -> pd.DataFrame:
    if not _has_accumulator_alias(current_df, result_file):
        return current_df
    target = result_file.stem[len("results_pmc_perf_") :]
    return current_df.rename(columns={"SQ_ACCUM_PREV_HIRES": target})


def _has_accumulator_alias(current_df: pd.DataFrame, result_file: Path) -> bool:
    return (
        result_file.name.startswith("results_pmc_perf_")
        and result_file.stem.endswith("_ACCUM")
        and "SQ_ACCUM_PREV_HIRES" in current_df.columns
    )


def _build_join_key(current_df: pd.DataFrame, join_type: str) -> pd.Series:
    if join_type == "kernel":
        replay_index = current_df.groupby("Kernel_Name").cumcount()
        return current_df.Kernel_Name + " - " + replay_index.astype(str)
    if join_type == "grid":
        replay_index = current_df.groupby(["Kernel_Name", "Grid_Size"]).cumcount()
        return (
            current_df["Kernel_Name"].astype(str)
            + " - "
            + current_df["Grid_Size"].astype(str)
            + " - "
            + replay_index.astype(str)
        )
    console_error("join_prof", f"{join_type} is an unrecognized option for --join-type")
    return pd.Series(dtype=str)


def _warn_on_mismatched_duplicate_columns(joined_df: pd.DataFrame) -> None:
    for key, columns in _duplicate_columns(joined_df).items():
        if not columns:
            continue
        current_df = joined_df[columns]
        if _all_columns_equal(current_df):
            console_debug("join_prof", f"Successfully joined {key} in pmc_perf.csv")
            continue
        console_warning(
            "join_prof",
            f"Detected differing {key} values while joining pmc_perf.csv",
        )


def _duplicate_columns(joined_df: pd.DataFrame) -> dict[str, list[str]]:
    duplicate_columns = {
        "GPU_ID": [col for col in joined_df.columns if col.startswith("GPU_ID")],
        "Grid_Size": [col for col in joined_df.columns if col.startswith("Grid_Size")],
        "Workgroup_Size": [
            col for col in joined_df.columns if col.startswith("Workgroup_Size")
        ],
        "LDS_Per_Workgroup": [
            col for col in joined_df.columns if col.startswith("LDS_Per_Workgroup")
        ],
        "Scratch_Per_Workitem": [
            col for col in joined_df.columns if col.startswith("Scratch_Per_Workitem")
        ],
        "SGPR": [col for col in joined_df.columns if col.startswith("SGPR")],
    }
    if "vgpr" in joined_df.columns:
        duplicate_columns["vgpr"] = [
            col for col in joined_df.columns if col.startswith("vgpr")
        ]
        return duplicate_columns

    duplicate_columns["Arch_VGPR"] = [
        col for col in joined_df.columns if col.startswith("Arch_VGPR")
    ]
    duplicate_columns["Accum_VGPR"] = [
        col for col in joined_df.columns if col.startswith("Accum_VGPR")
    ]
    return duplicate_columns


def _all_columns_equal(df: pd.DataFrame) -> bool:
    return df.eq(df.iloc[:, 0], axis=0).all(1).all()


def _drop_duplicate_columns(joined_df: pd.DataFrame) -> pd.DataFrame:
    return joined_df[
        [
            col
            for col in joined_df.columns
            if not any(col.startswith(prefix) for prefix in DUPLICATE_COLUMN_PREFIXES)
        ]
    ]


def _drop_timestamp_columns(joined_df: pd.DataFrame) -> pd.DataFrame:
    return joined_df[
        [
            col
            for col in joined_df.columns
            if not any(pattern in col for pattern in TIMESTAMP_COLUMN_PATTERNS)
        ]
    ]


def _drop_duplicate_kernel_name_columns(joined_df: pd.DataFrame) -> pd.DataFrame:
    name_columns = [col for col in joined_df.columns if "Kernel_Name" in col]
    if not name_columns:
        return joined_df

    for name_column in name_columns[1:]:
        assert (joined_df[name_columns[0]] == joined_df[name_column]).all()
    return joined_df.drop(columns=name_columns[1:])


def _replace_replay_timestamps(joined_df: pd.DataFrame) -> pd.DataFrame:
    start_columns = [col for col in joined_df.columns if "Start_Timestamp" in col]
    end_columns = [col for col in joined_df.columns if "End_Timestamp" in col]
    if not start_columns or not end_columns:
        return _drop_join_key(joined_df)

    joined_df = joined_df.copy()
    mean_start = joined_df[start_columns].mean(axis=1)
    mean_end = joined_df[end_columns].mean(axis=1)
    joined_df = joined_df.drop(columns=start_columns + end_columns)
    joined_df["Start_Timestamp"] = mean_start
    joined_df["End_Timestamp"] = mean_end
    return _drop_join_key(joined_df)


def _drop_join_key(joined_df: pd.DataFrame) -> pd.DataFrame:
    if "key" not in joined_df.columns:
        return joined_df
    return joined_df.drop(columns=["key"])
