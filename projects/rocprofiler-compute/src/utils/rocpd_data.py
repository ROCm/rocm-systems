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
from contextlib import closing
from typing import Any

import pandas as pd

from utils.logger import console_error

# From schema definition in source/share/rocprofiler-sdk-rocpd/data_views.sql
# in rocprofiler-sdk repository
COUNTERS_COLLECTION_QUERY = """
SELECT
    agent_id as GPU_ID,
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
ROCPD_PMC_EVENT_TABLE_NAME_PREFIX = "rocpd_pmc_event_"
TABLE_NAME_PREFIX_QUERY = (
    "SELECT name FROM sqlite_master WHERE type='table' "
    "AND name LIKE '{table_name_prefix}%'"
)
INSERT_QUERY = "INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"


def convert_dbs_to_csv(
    db_paths: list[str],
    csv_file_path: str,
) -> None:
    """
    Read rocpd databases and write to CSV file
    """
    # Read counters_collection view from the databases and write to CSV
    try:
        with open(csv_file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            header_written = False
            for db_path in db_paths:
                with closing(sqlite3.connect(db_path)) as conn:
                    with closing(conn.execute(COUNTERS_COLLECTION_QUERY)) as cursor:
                        if not header_written:
                            writer.writerow([
                                description[0] for description in cursor.description
                            ])
                            header_written = True
                        for row in cursor:
                            writer.writerow(row)
    except OSError as e:
        console_error(f"Database error while converting to CSV: {e}")
    except Exception as e:
        console_error(f"Unexpected error converting database to CSV: {e}")


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
    """Update pmc_event table in the given rocpd database path"""
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
            columns = ("guid", "event_id", "pmc_id", "value")
            values = list(
                zip(
                    # guid
                    [guid] * len(counter_info),
                    # event_id
                    counter_info["dispatch_id"],
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
