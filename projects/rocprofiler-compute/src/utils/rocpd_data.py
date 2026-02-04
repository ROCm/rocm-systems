##############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################

import csv
import sqlite3
from contextlib import ExitStack, closing
from typing import Any

import pandas as pd

from utils.logger import console_error

# From schema definition in source/share/rocprofiler-sdk-rocpd/data_views.sql
# in rocprofiler-sdk repository
COUNTERS_COLLECTION_QUERY = """
SELECT
    agent_id as GPU_ID,
    guid as GUID,
    correlation_id as Correlation_Id,
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
    corr_id AS Correlation_Id,
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
    queries = {
        counter_collection_csv_path: COUNTERS_COLLECTION_QUERY,
        marker_trace_csv_path: MARKER_API_TRACE_QUERY,
    }
    header_written = {path: False for path in queries}

    with ExitStack() as stack:
        writers = {
            path: csv.writer(stack.enter_context(open(path, "w", newline="")))
            for path in queries
        }
        for db_path in db_paths:
            with closing(sqlite3.connect(db_path)) as conn:
                for file_path, query in queries.items():
                    try:
                        with closing(conn.execute(query)) as cursor:
                            if cursor.description is None:
                                continue
                            if not header_written[file_path]:
                                writers[file_path].writerow([
                                    desc[0] for desc in cursor.description
                                ])
                                header_written[file_path] = True
                            writers[file_path].writerows(cursor)
                    except OSError as e:
                        console_error(
                            f"Database error while extracting {file_path} "
                            f"from {db_path}: {e}"
                        )
                    except Exception as e:
                        console_error(
                            f"Unexpected error while extracting {file_path} "
                            f"from {db_path}: {e}"
                        )


def process_rocpd_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge counters across unique dispatches from the
    input dataframe and return processed dataframe.
    """
    if df.empty:
        return df

    data: list[dict[str, Any]] = []

    # Group by unique kernel and merge into a single row
    for _, group_df in df.groupby([
        "Dispatch_ID",
        "Kernel_Name",
        "Grid_Size",
        "Workgroup_Size",
        "LDS_Per_Workgroup",
    ]):
        row = {
            "GPU_ID": group_df["GPU_ID"].iloc[0],
            "Grid_Size": group_df["Grid_Size"].iloc[0],
            "Workgroup_Size": group_df["Workgroup_Size"].iloc[0],
            "LDS_Per_Workgroup": group_df["LDS_Per_Workgroup"].iloc[0],
            "Scratch_Per_Workitem": group_df["Scratch_Per_Workitem"].iloc[0],
            "Arch_VGPR": group_df["Arch_VGPR"].iloc[0],
            "Accum_VGPR": group_df["Accum_VGPR"].iloc[0],
            "SGPR": group_df["SGPR"].iloc[0],
            "Kernel_Name": group_df["Kernel_Name"].iloc[0],
            "Kernel_ID": group_df["Kernel_ID"].iloc[0],
            "Start_Timestamp": group_df["Start_Timestamp"].iloc[0],
            "End_Timestamp": group_df["End_Timestamp"].iloc[0],
        }
        # Each counter will become its own column
        row.update(dict(zip(group_df["Counter_Name"], group_df["Counter_Value"])))
        data.append(row)
    df = pd.DataFrame(data)
    # Rank GPU IDs, map lowest number to 0, next to 1, etc.
    df["GPU_ID"] = df["GPU_ID"].rank(method="dense").astype(int) - 1
    # Reset dispatch IDs
    df["Dispatch_ID"] = range(len(df))
    return df


def update_rocpd_pmc_events(counter_info: pd.DataFrame, rocpd_db_path: str) -> None:
    """Updates pmc_event table in the given rocpd database path."""
    try:
        with closing(sqlite3.connect(rocpd_db_path)) as conn:
            # Get pmc_event table name
            with closing(
                conn.execute(
                    TABLE_NAME_PREFIX_QUERY.format(
                        table_name_prefix=ROCPD_PMC_EVENT_TABLE_NAME_PREFIX
                    )
                )
            ) as cursor:
                table_name = cursor.fetchone()
            if table_name is None:
                console_error("No pmc_event table found in the rocpd database")
            table_name = table_name[0]

            # get pmc_event table data
            guid = table_name[len(ROCPD_PMC_EVENT_TABLE_NAME_PREFIX) :].replace(
                "_", "-"
            )
            # Map dispatch_id to event_id from rocpd_kernel_dispatch
            # Native counter collection CSV has dispatch_id, but schema needs event_id
            # event_id may differ from dispatch_id when marker API tracing is enabled
            with closing(conn.execute(KERNEL_DISPATCH_QUERY, (guid,))) as cursor:
                rows = cursor.fetchall()
            if not rows:
                console_error("No kernel dispatch data found.")
                return
            dispatch_to_event = {
                dispatch_id: event_id for dispatch_id, event_id, _ in rows
            }
            counter_info["event_id"] = counter_info["dispatch_id"].map(
                dispatch_to_event
            )
            columns = ("guid", "event_id", "pmc_id", "value")
            values = list(
                zip(
                    # guid
                    [guid] * len(counter_info),
                    # event_id
                    counter_info["event_id"],
                    # pmc_id
                    counter_info["counter_id"],
                    # value
                    counter_info["counter_value"],
                )
            )

            # insert into pmc_event table
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
    except OSError as e:
        console_error(f"Database error while updating pmc_event table: {e}")
    except Exception as e:
        console_error(f"Unexpected error updating pmc_event table: {e}")
