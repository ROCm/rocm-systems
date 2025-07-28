import csv
import sqlite3
from contextlib import closing

from utils.logger import console_error

# From schema definition in source/share/rocprofiler-sdk-rocpd/data_views.sql in rocprofiler-sdk repository
COUNTERS_COLLECTION_QUERY = """
SELECT
    agent_id as GPU_ID,
    dispatch_id as Dispatch_ID,
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


def convert_db_to_csv(
    db_path: str,
    csv_file_path: str,
) -> None:
    """
    Read rocpd database and write to CSV file
    """
    # Read counters_collection view from the database and write to CSV
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            with closing(conn.execute(COUNTERS_COLLECTION_QUERY)) as cursor:
                with open(csv_file_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(
                        [description[0] for description in cursor.description]
                    )
                    for row in cursor:
                        writer.writerow(row)
    except (sqlite3.DatabaseError, IOError) as e:
        console_error(f"Error converting database to CSV: {e}")


def process_rocpd_csv(df):
    """
    Merge counters across unique dispatches from the input dataframe and return processed dataframe.
    """
    # Only import pandas if needed
    import pandas as pd

    data = list()
    # Group by unique kernel and merge into a single row
    for _, group_df in df.groupby(
        [
            "Dispatch_ID",
            "Kernel_Name",
            "Grid_Size",
            "Workgroup_Size",
            "LDS_Per_Workgroup",
        ]
    ):
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
        }
        # Each counter will become its own column
        row.update(dict(zip(group_df["Counter_Name"], group_df["Counter_Value"])))
        # Replace end timestamp with median of durations of group, start timestamp is set to 0
        row["End_Timestamp"] = (
            group_df["End_Timestamp"] - group_df["Start_Timestamp"]
        ).median()
        row["Start_Timestamp"] = 0.0
        data.append(row)
    df = pd.DataFrame(data)
    # Rank GPU IDs, map lowest number to 0, next to 1, etc.
    df["GPU_ID"] = df["GPU_ID"].rank(method="dense").astype(int) - 1
    # Reset dispatch IDs
    df["Dispatch_ID"] = range(len(df))
    return df
