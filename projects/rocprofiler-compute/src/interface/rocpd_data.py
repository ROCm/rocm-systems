# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ROCPD profile data readers and writers."""

from __future__ import annotations

import csv
import shutil
import sqlite3
from contextlib import ExitStack, closing
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from interface import csv_data
from interface.pmc_frame import prepare_pmc_frame, to_canonical_pmc_frame
from interface.profile_data import ProfileDataReaderOptions, ProfilePassContext
from utils.logger import console_debug, console_error, console_warning

if TYPE_CHECKING:
    import pandas as pd

COUNTERS_COLLECTION_QUERY = """
SELECT
    agent_id as GPU_ID,
    guid as GUID,
    stack_id as Correlation_Id,
    dispatch_id as Dispatch_ID,
    pid as PID,
    grid_size as Grid_Size,
    workgroup_size as Workgroup_Size,
    lds_block_size as LDS_Per_Workgroup,
    scratch_size as Scratch_Per_Workitem,
    vgpr_count as Arch_VGPR,
    accum_vgpr_count as Accum_VGPR,
    sgpr_count as SGPR,
    kernel_name as Kernel_Name,
    start as Start_Timestamp,
    end as End_Timestamp,
    kernel_id as Kernel_ID,
    counter_name as Counter_Name,
    value as Counter_Value
FROM counters_collection
"""
MARKER_API_TRACE_QUERY = """
SELECT
    category AS Domain,
    json_extract(extdata, '$.message') AS Function,
    pid AS Process_Id,
    tid AS Thread_Id,
    stack_id AS Correlation_Id,
    guid AS GUID,
    start AS Start_Timestamp,
    end AS End_Timestamp
FROM regions
ORDER BY start
"""
KERNEL_DISPATCH_QUERY = """
SELECT dispatch_id, event_id, guid
FROM rocpd_kernel_dispatch
WHERE guid = ?
"""
ROCPD_PMC_EVENT_TABLE_NAME_PREFIX = "rocpd_pmc_event_"
TABLE_NAME_PREFIX_QUERY = (
    "SELECT name FROM sqlite_master WHERE type='table' "
    "AND name LIKE '{table_name_prefix}%'"
)
INSERT_QUERY = "INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"


def convert_dbs_to_csv(
    db_paths: list[str],
    counter_collection_csv_path: str,
    marker_trace_csv_path: str,
) -> None:
    """Extract ROCPD views from databases into CSV files."""
    queries = {
        counter_collection_csv_path: COUNTERS_COLLECTION_QUERY,
        marker_trace_csv_path: MARKER_API_TRACE_QUERY,
    }
    header_written = {path: False for path in queries}

    with ExitStack() as stack:
        writers = {
            path: csv.writer(
                stack.enter_context(
                    Path(path).open("w", newline="", encoding="utf-8")
                )
            )
            for path in queries
        }
        for db_path in db_paths:
            _write_query_rows_from_db(db_path, queries, writers, header_written)


def update_rocpd_pmc_events(counter_info: list[dict], rocpd_db_path: str) -> None:
    """Updates pmc_event table in the given rocpd database path."""
    try:
        with closing(sqlite3.connect(rocpd_db_path)) as conn:
            _insert_pmc_events(conn, counter_info)
    except OSError as error:
        console_error(f"Database error while updating pmc_event table: {error}")
    except Exception as error:
        console_error(f"Unexpected error updating pmc_event table: {error}")


def _pandas() -> Any:  # noqa: ANN401
    import pandas as pd

    return pd


def _write_result_file_rows(
    result_file: Path,
    outfile: Any,  # noqa: ANN401
    writer: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    with result_file.open(newline="", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        header = next(reader)
        if writer is None:
            writer = csv.writer(outfile)
            writer.writerow(header)
        for row in reader:
            writer.writerow(row)
    return writer


def _write_query_rows_from_db(
    db_path: str,
    queries: dict[str, str],
    writers: dict[str, csv.writer],
    header_written: dict[str, bool],
) -> None:
    with closing(sqlite3.connect(db_path)) as conn:
        for file_path, query in queries.items():
            _write_query_rows(
                conn,
                query,
                writers[file_path],
                file_path,
                header_written,
            )


def _write_query_rows(
    conn: sqlite3.Connection,
    query: str,
    writer: csv.writer,
    file_path: str,
    header_written: dict[str, bool],
) -> None:
    try:
        with closing(conn.execute(query)) as cursor:
            if cursor.description is None:
                return
            if not header_written[file_path]:
                writer.writerow([desc[0] for desc in cursor.description])
                header_written[file_path] = True
            writer.writerows(cursor)
    except OSError as error:
        console_error(f"Database error while extracting {file_path}: {error}")
    except Exception as error:
        console_error(f"Unexpected error while extracting {file_path}: {error}")


def _insert_pmc_events(
    conn: sqlite3.Connection,
    counter_info: list[dict],
) -> None:
    table_name = _get_pmc_event_table_name(conn)
    if table_name is None:
        console_error("No pmc_event table found in the rocpd database")
    guid = table_name[len(ROCPD_PMC_EVENT_TABLE_NAME_PREFIX) :].replace("_", "-")
    dispatch_to_event = _get_dispatch_to_event_map(conn, guid)
    if not dispatch_to_event:
        console_error("No kernel dispatch data found.")
        return

    for row in counter_info:
        dispatch_id = row.get("dispatch_id")
        row["event_id"] = dispatch_to_event.get(dispatch_id)

    columns = ("guid", "event_id", "pmc_id", "value")
    values = [
        (
            guid,
            row.get("event_id"),
            row.get("counter_id"),
            row.get("counter_value"),
        )
        for row in counter_info
    ]
    with conn:
        placeholders = ", ".join(["?"] * len(columns))
        conn.executemany(
            INSERT_QUERY.format(
                table_name=table_name,
                columns=", ".join(columns),
                placeholders=placeholders,
            ),
            values,
        )


def _get_pmc_event_table_name(conn: sqlite3.Connection) -> Optional[str]:
    with closing(
        conn.execute(
            TABLE_NAME_PREFIX_QUERY.format(
                table_name_prefix=ROCPD_PMC_EVENT_TABLE_NAME_PREFIX
            )
        )
    ) as cursor:
        table_name = cursor.fetchone()
    return table_name[0] if table_name is not None else None


def _get_dispatch_to_event_map(
    conn: sqlite3.Connection,
    guid: str,
) -> dict[str, str]:
    with closing(conn.execute(KERNEL_DISPATCH_QUERY, (guid,))) as cursor:
        db_rows = cursor.fetchall()
    return {str(dispatch_id): str(event_id) for dispatch_id, event_id, _ in db_rows}


def _rocpd_counter_collection_path(context: ProfilePassContext) -> Path:
    return context.workload_dir / "out" / "pmc_1" / (
        f"{context.fbase}_counter_collection.csv"
    )


def _rocpd_marker_trace_path(context: ProfilePassContext) -> Path:
    return context.workload_dir / "out" / "pmc_1" / (
        f"{context.fbase}_marker_api_trace.csv"
    )


class RocpdProfileData:
    """Write and normalize ROCPD profile data for profile mode."""

    def update_native_counter_events(
        self,
        db_paths: list[Path],
        workload_dir: Path,
    ) -> None:
        for db_name in db_paths:
            self._update_native_counter_events_for_db(db_name, workload_dir)

    def convert_dbs_to_csv(
        self,
        db_paths: list[Path],
        counter_collection_csv_path: Path,
        marker_trace_csv_path: Path,
    ) -> None:
        convert_dbs_to_csv(
            [str(path) for path in db_paths],
            str(counter_collection_csv_path),
            str(marker_trace_csv_path),
        )

    def read_counter_rows(self, counter_collection_path: Path) -> list[dict]:
        try:
            rows, _ = csv_data.read_csv_as_dicts(counter_collection_path)
            return rows
        except (FileNotFoundError, ValueError):
            return []

    def normalize_counter_rows(self, combined_rows: list[dict]) -> None:
        csv_data.csv_ops.assign_group_ids(
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
        csv_data.csv_ops.assign_group_ids(
            combined_rows,
            ["Kernel_Name", "Grid_Size", "Workgroup_Size", "LDS_Per_Workgroup"],
            "Kernel_ID",
        )
        csv_data.csv_ops.drop_column_from_rows(combined_rows, "PID")

    def write_counter_rows(
        self,
        counter_collection_path: Path,
        workload_results_path: Path,
        combined_rows: list[dict],
    ) -> None:
        csv_data.write_csv_from_dicts(counter_collection_path, combined_rows)
        csv_data.write_csv_from_dicts(workload_results_path, combined_rows)

    def retain_databases(
        self,
        db_paths: list[Path],
        workload_dir: Path,
        fbase: str,
    ) -> None:
        console_warning(
            "--retain-rocpd-output is deprecated and will be removed in "
            "a future release. .db files will be retained automatically."
        )
        for db_path in db_paths:
            pid = db_path.stem.split("_")[0]
            retained_path = workload_dir / f"{fbase}_{pid}.db"
            shutil.copyfile(db_path, retained_path)
            console_warning(f"Retaining large raw rocpd database: {retained_path}")

    def _update_native_counter_events_for_db(
        self,
        db_name: Path,
        workload_dir: Path,
    ) -> None:
        pid = db_name.stem.split("_")[0]
        counter_csv = (
            workload_dir / "out" / "pmc_1" / f"{pid}_native_counter_collection.csv"
        )
        if not counter_csv.is_file():
            console_debug(
                f"No native counter CSV for pid {pid}; "
                f"skipping rocpd update for {db_name}."
            )
            return
        counter_rows, _ = csv_data.read_csv_as_dicts(counter_csv)
        update_rocpd_pmc_events(counter_rows, str(db_name))
        console_debug(f"Updated rocpd db {db_name} with native tool counters.")


class RocpdAnalysisData:
    """Read and materialize ROCPD profile data for analyze mode."""

    def has_profile_data(self, workload_dir: Path) -> bool:
        return csv_data.pmc_perf_path(workload_dir).exists() or bool(
            self.find_result_files(workload_dir)
        )

    def materialize_pmc_perf(self, workload_dir: Path, output_path: Path) -> Path:
        if output_path.exists():
            console_debug(f"Using existing {output_path}")
            return output_path

        console_warning(
            "Reading intermediate results_*.csv files is deprecated and "
            "will be removed in a future release."
        )
        self.concat_results_to_pmc_perf(
            self.find_result_files(workload_dir),
            output_path,
        )
        console_debug(f"Created file: {output_path}")
        return output_path

    def find_result_files(self, workload_dir: Path) -> list[Path]:
        return list(workload_dir.glob("results_*.csv"))

    def concat_results_to_pmc_perf(
        self,
        result_files: list[Path],
        output_path: Path,
    ) -> None:
        with output_path.open("w", newline="", encoding="utf-8") as outfile:
            writer = None
            for result_file in result_files:
                writer = _write_result_file_rows(result_file, outfile, writer)

    def read_pmc_frame(
        self,
        workload_dir: Path,
        *,
        kernel_verbose: int,
        verbose: int,
        node_name: Optional[str] = None,
    ) -> pd.DataFrame:
        pd = _pandas()
        pmc_path = csv_data.pmc_perf_path(workload_dir)
        if not pmc_path.is_file():
            return pd.DataFrame()

        frame = to_canonical_pmc_frame(pd.read_csv(pmc_path))
        return prepare_pmc_frame(
            frame,
            kernel_verbose=kernel_verbose,
            verbose=verbose,
            node_name=node_name,
        )


class RocpdProfileDataReader:
    """Read current ROCPD profile data."""

    def __init__(self, options: ProfileDataReaderOptions) -> None:
        self._options = options
        self._rocpd_data = RocpdAnalysisData()

    def has_profile_data(self, workload_dir: Path) -> bool:
        return self._rocpd_data.has_profile_data(workload_dir)

    def materialize_pmc_perf(self, workload_dir: Path, output_path: Path) -> Path:
        return self._rocpd_data.materialize_pmc_perf(workload_dir, output_path)

    def read_pmc_frame(self, workload_dir: Path) -> pd.DataFrame:
        return self._rocpd_data.read_pmc_frame(
            workload_dir,
            kernel_verbose=self._options.kernel_verbose,
            verbose=self._options.verbose,
        )


class RocpdProfileDataWriter:
    """Finalize current ROCPD profile data."""

    def finalize_pass(self, context: ProfilePassContext) -> None:
        rocpd_profile_data = RocpdProfileData()
        db_paths = list((context.workload_dir / "out" / "pmc_1").glob("*/*.db"))
        if context.using_native_tool:
            rocpd_profile_data.update_native_counter_events(
                db_paths,
                context.workload_dir,
            )

        rocpd_profile_data.convert_dbs_to_csv(
            db_paths,
            _rocpd_counter_collection_path(context),
            _rocpd_marker_trace_path(context),
        )
        combined_rows = rocpd_profile_data.read_counter_rows(
            _rocpd_counter_collection_path(context)
        )
        if not combined_rows:
            console_warning(
                "No GPU kernel data collected. "
                "The workload may not have dispatched any GPU kernels."
            )
            shutil.rmtree(str(context.workload_dir / "out"), ignore_errors=True)
            return

        rocpd_profile_data.normalize_counter_rows(combined_rows)
        rocpd_profile_data.write_counter_rows(
            _rocpd_counter_collection_path(context),
            context.workload_dir / f"results_{context.fbase}.csv",
            combined_rows,
        )
        console_warning(
            "Intermediate results_*.csv generation from rocpd databases is "
            "deprecated and will be replaced with automatic .db file "
            "retention in a future release."
        )
        if context.torch_trace_enabled:
            csv_data.save_torch_trace_inputs(
                str(context.workload_dir),
                context.fbase,
                "rocpd",
            )
        if context.retain_rocpd_output:
            rocpd_profile_data.retain_databases(
                db_paths,
                context.workload_dir,
                context.fbase,
            )
        shutil.rmtree(str(context.workload_dir / "out"))
