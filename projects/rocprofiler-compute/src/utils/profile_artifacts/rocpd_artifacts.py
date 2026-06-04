# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ROCPD profile artifact reader."""

import csv
from pathlib import Path

import pandas as pd

from utils.logger import console_debug, console_warning
from utils.profile_artifacts.interfaces import ArtifactReaderOptions, ProfilePassContext
from utils.profile_artifacts.pmc_frame import load_pmc_frame_from_csv


class RocpdProfileArtifactReader:
    """Read current ROCPD profile artifacts."""

    def __init__(self, options: ArtifactReaderOptions) -> None:
        self._options = options

    def has_artifacts(self, workload_dir: Path) -> bool:
        return _pmc_perf_path(workload_dir).exists() or bool(
            _find_result_files(workload_dir)
        )

    def materialize_pmc_perf(self, workload_dir: Path, output_path: Path) -> Path:
        if output_path.exists():
            console_debug(f"Using existing {output_path}")
            return output_path

        console_warning(
            "Reading intermediate results_*.csv files is deprecated and "
            "will be removed in a future release."
        )
        _concat_results_to_pmc_perf(_find_result_files(workload_dir), output_path)
        console_debug(f"Created file: {output_path}")
        return output_path

    def read_pmc_frame(self, workload_dir: Path) -> pd.DataFrame:
        return load_pmc_frame_from_csv(
            workload_dir,
            is_rocpd=True,
            kernel_verbose=self._options.kernel_verbose,
            verbose=self._options.verbose,
        )


class RocpdProfileArtifactWriter:
    """Finalize current ROCPD profile artifacts."""

    def finalize_pass(self, context: ProfilePassContext) -> None:
        from utils import utils_profile as profile_ops

        db_paths = list((context.workload_dir / "out" / "pmc_1").glob("*/*.db"))
        if context.using_native_tool:
            _update_rocpd_native_counter_events(db_paths, context)

        profile_ops.rocpd_data.convert_dbs_to_csv(
            [str(path) for path in db_paths],
            str(_rocpd_counter_collection_path(context)),
            str(_rocpd_marker_trace_path(context)),
        )
        combined_rows = _read_rocpd_counter_rows(context)
        if not combined_rows:
            profile_ops.console_warning(
                "No GPU kernel data collected. "
                "The workload may not have dispatched any GPU kernels."
            )
            profile_ops.shutil.rmtree(
                str(context.workload_dir / "out"),
                ignore_errors=True,
            )
            return

        _normalize_rocpd_counter_rows(combined_rows)
        profile_ops.csv_ops.write_csv_from_dicts(
            str(_rocpd_counter_collection_path(context)),
            combined_rows,
        )
        profile_ops.csv_ops.write_csv_from_dicts(
            str(context.workload_dir / f"results_{context.fbase}.csv"),
            combined_rows,
        )
        profile_ops.console_warning(
            "Intermediate results_*.csv generation from rocpd databases is "
            "deprecated and will be replaced with automatic .db file "
            "retention in a future release."
        )
        if context.torch_trace_enabled:
            profile_ops.save_torch_trace_inputs(
                str(context.workload_dir),
                context.fbase,
                "rocpd",
            )
        if context.retain_rocpd_output:
            _retain_rocpd_databases(db_paths, context)
        profile_ops.shutil.rmtree(str(context.workload_dir / "out"))


def _pmc_perf_path(workload_dir: Path) -> Path:
    return workload_dir / "pmc_perf.csv"


def _find_result_files(workload_dir: Path) -> list[Path]:
    return list(workload_dir.glob("results_*.csv"))


def _concat_results_to_pmc_perf(result_files: list[Path], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = None
        for result_file in result_files:
            with result_file.open(newline="", encoding="utf-8") as infile:
                reader = csv.reader(infile)
                header = next(reader)
                if writer is None:
                    writer = csv.writer(outfile)
                    writer.writerow(header)
                for row in reader:
                    writer.writerow(row)


def _update_rocpd_native_counter_events(
    db_paths: list[Path],
    context: ProfilePassContext,
) -> None:
    from utils import utils_profile as profile_ops

    for db_name in db_paths:
        pid = db_name.stem.split("_")[0]
        counter_csv = (
            context.workload_dir
            / "out"
            / "pmc_1"
            / f"{pid}_native_counter_collection.csv"
        )
        if not counter_csv.is_file():
            profile_ops.console_debug(
                f"No native counter CSV for pid {pid}; "
                f"skipping rocpd update for {db_name}."
            )
            continue
        counter_rows, _ = profile_ops.csv_ops.read_csv_as_dicts(str(counter_csv))
        profile_ops.rocpd_data.update_rocpd_pmc_events(counter_rows, str(db_name))
        profile_ops.console_debug(
            f"Updated rocpd db {db_name} with native tool counters."
        )


def _rocpd_counter_collection_path(context: ProfilePassContext) -> Path:
    return context.workload_dir / "out" / "pmc_1" / (
        f"{context.fbase}_counter_collection.csv"
    )


def _rocpd_marker_trace_path(context: ProfilePassContext) -> Path:
    return context.workload_dir / "out" / "pmc_1" / (
        f"{context.fbase}_marker_api_trace.csv"
    )


def _read_rocpd_counter_rows(context: ProfilePassContext) -> list[dict]:
    from utils import utils_profile as profile_ops

    try:
        combined_rows, _ = profile_ops.csv_ops.read_csv_as_dicts(
            str(_rocpd_counter_collection_path(context))
        )
        return combined_rows
    except (FileNotFoundError, ValueError):
        return []


def _normalize_rocpd_counter_rows(combined_rows: list[dict]) -> None:
    from utils import utils_profile as profile_ops

    profile_ops.csv_ops.assign_group_ids(
        combined_rows,
        [
            "PID",
            "Kernel_Name",
            "Grid_Size",
            "Workgroup_Size",
            "LDS_Per_Workgroup",
            "Start_Timestamp",
            "End_Timestamp",
        ],
        "Dispatch_ID",
    )
    profile_ops.csv_ops.assign_group_ids(
        combined_rows,
        ["Kernel_Name", "Grid_Size", "Workgroup_Size", "LDS_Per_Workgroup"],
        "Kernel_ID",
    )
    profile_ops.csv_ops.drop_column_from_rows(combined_rows, "PID")


def _retain_rocpd_databases(
    db_paths: list[Path],
    context: ProfilePassContext,
) -> None:
    from utils import utils_profile as profile_ops

    profile_ops.console_warning(
        "--retain-rocpd-output is deprecated and will be removed in "
        "a future release. .db files will be retained automatically."
    )
    for db_path in db_paths:
        pid = db_path.stem.split("_")[0]
        retained_path = context.workload_dir / f"{context.fbase}_{pid}.db"
        profile_ops.shutil.copyfile(db_path, retained_path)
        profile_ops.console_warning(
            f"Retaining large raw rocpd database: {retained_path}"
        )
