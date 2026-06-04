# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""CSV profile artifact reader."""

from pathlib import Path
from typing import Optional

import pandas as pd

from utils.logger import console_debug, console_error, console_warning
from utils.profile_artifacts.interfaces import ArtifactReaderOptions, ProfilePassContext
from utils.profile_artifacts.pmc_frame import load_pmc_frame_from_csv

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


class CsvProfileArtifactReader:
    """Read current CSV profile artifacts."""

    def __init__(self, options: ArtifactReaderOptions) -> None:
        self._options = options

    def has_artifacts(self, workload_dir: Path) -> bool:
        return _pmc_perf_path(workload_dir).exists() or bool(
            _find_csv_result_files(workload_dir, self._options.kokkos_trace)
        )

    def materialize_pmc_perf(self, workload_dir: Path, output_path: Path) -> Path:
        if output_path.exists():
            console_debug(f"Using existing {output_path}")
            return output_path

        joined_df = join_csv_prof_files(
            workload_dir,
            join_type=self._options.join_type,
            kokkos_trace=self._options.kokkos_trace,
        )
        if joined_df is None:
            return output_path

        joined_df.to_csv(output_path, index=False)
        console_debug(f"Created file: {output_path}")
        return output_path

    def read_pmc_frame(self, workload_dir: Path) -> pd.DataFrame:
        return load_pmc_frame_from_csv(
            workload_dir,
            is_rocpd=False,
            kernel_verbose=self._options.kernel_verbose,
            verbose=self._options.verbose,
        )


class CsvProfileArtifactWriter:
    """Finalize current CSV profile artifacts."""

    def finalize_pass(self, context: ProfilePassContext) -> None:
        from utils import utils_profile as profile_ops

        result_files = _process_csv_outputs(context)
        if context.torch_trace_enabled:
            profile_ops.save_torch_trace_inputs(
                str(context.workload_dir),
                context.fbase,
                "csv",
            )
        if not result_files:
            profile_ops.console_warning(
                f"Cannot write results for {context.fbase}.csv due to no counter "
                "csv files generated."
            )
            return

        combined_results = profile_ops.csv_ops.concat_csv_files(result_files)
        _normalize_csv_counter_rows(combined_results)
        _write_csv_counter_results(combined_results, context)
        _standardize_csv_headers(context)


def join_csv_prof_files(
    workload_dir: Path,
    *,
    join_type: str,
    kokkos_trace: bool,
) -> Optional[pd.DataFrame]:
    """Join CSV profiler outputs into the pmc_perf dataframe."""
    joined_df = _merge_csv_result_files(
        _find_csv_result_files(workload_dir, kokkos_trace),
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


def _pmc_perf_path(workload_dir: Path) -> Path:
    return workload_dir / "pmc_perf.csv"


def _find_csv_result_files(workload_dir: Path, kokkos_trace: bool) -> list[Path]:
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


def _process_csv_outputs(context: ProfilePassContext) -> list[str]:
    from utils import utils_profile as profile_ops

    if context.profiler_command == "rocprofiler-sdk":
        return profile_ops.process_rocprofv3_output(
            str(context.workload_dir),
            using_native_tool=context.using_native_tool,
        )

    result_files = profile_ops.process_rocprofv3_output(
        str(context.workload_dir),
        using_native_tool=False,
    )
    if context.kokkos_trace_enabled:
        profile_ops.process_kokkos_trace_output(
            str(context.workload_dir),
            context.fbase,
        )
    return result_files


def _normalize_csv_counter_rows(combined_results: list[dict]) -> None:
    from utils import utils_profile as profile_ops

    profile_ops.csv_ops.add_column_to_rows(
        combined_results,
        "Dispatch_ID",
        list(range(0, len(combined_results))),
    )
    profile_ops.csv_ops.assign_group_ids(
        combined_results,
        ["Kernel_Name", "Grid_Size", "Workgroup_Size", "LDS_Per_Workgroup"],
        "Kernel_ID",
    )


def _write_csv_counter_results(
    combined_results: list[dict],
    context: ProfilePassContext,
) -> None:
    from utils import utils_profile as profile_ops

    profile_ops.csv_ops.write_csv_from_dicts(
        str(context.workload_dir / "out" / "pmc_1" / f"results_{context.fbase}.csv"),
        combined_results,
    )
    if (context.workload_dir / "out").exists():
        profile_ops.shutil.copyfile(
            str(
                context.workload_dir
                / "out"
                / "pmc_1"
                / f"results_{context.fbase}.csv"
            ),
            str(context.workload_dir / f"results_{context.fbase}.csv"),
        )
        profile_ops.shutil.rmtree(str(context.workload_dir / "out"))


def _standardize_csv_headers(context: ProfilePassContext) -> None:
    from utils import utils_profile as profile_ops

    csv_path = context.workload_dir / f"results_{context.fbase}.csv"
    rows, _ = profile_ops.csv_ops.read_csv_as_dicts(str(csv_path))
    profile_ops.csv_ops.rename_columns(rows, _csv_output_headers())
    profile_ops.csv_ops.write_csv_from_dicts(str(csv_path), rows)


def _csv_output_headers() -> dict[str, str]:
    return {
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
