# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Canonical PMC DataFrame transformations."""

from pathlib import Path
from typing import Optional

import pandas as pd

from utils import schema
from utils.kernel_name_shortener import kernel_name_shortener
from utils.logger import console_debug

ROCPD_COUNTER_NAME_COLUMN = "Counter_Name"
ROCPD_COUNTER_VALUE_COLUMN = "Counter_Value"
ROCPD_GROUP_COLUMNS = [
    "Dispatch_ID",
    "Kernel_Name",
    "Grid_Size",
    "Workgroup_Size",
    "LDS_Per_Workgroup",
]
ROCPD_METADATA_COLUMNS = [
    "GPU_ID",
    "Grid_Size",
    "Workgroup_Size",
    "LDS_Per_Workgroup",
    "Scratch_Per_Workitem",
    "Arch_VGPR",
    "Accum_VGPR",
    "SGPR",
    "Kernel_Name",
    "Kernel_ID",
    "Start_Timestamp",
    "End_Timestamp",
]


def is_long_counter_frame(df: pd.DataFrame) -> bool:
    """Return True when a frame stores one counter value per row."""
    counter_columns = {ROCPD_COUNTER_NAME_COLUMN, ROCPD_COUNTER_VALUE_COLUMN}
    return counter_columns.issubset(df.columns)


def to_canonical_pmc_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return the canonical wide PMC frame used by analysis."""
    if df.empty:
        return df

    if is_long_counter_frame(df):
        return pivot_counter_rows(df)

    return df


def process_rocpd_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize rocpd counter rows to the canonical PMC frame."""
    return to_canonical_pmc_frame(df)


def pivot_counter_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-form counter rows into one row per dispatch."""
    if df.empty:
        return df

    data: list[dict] = []

    for _, group_df in df.groupby(ROCPD_GROUP_COLUMNS):
        row = _build_rocpd_metadata_row(group_df)
        row.update(
            dict(
                zip(
                    group_df[ROCPD_COUNTER_NAME_COLUMN],
                    group_df[ROCPD_COUNTER_VALUE_COLUMN],
                )
            )
        )
        data.append(row)

    processed_df = pd.DataFrame(data)
    processed_df["GPU_ID"] = processed_df["GPU_ID"].rank(method="dense").astype(int) - 1
    processed_df["Dispatch_ID"] = range(len(processed_df))
    return processed_df


def load_pmc_frame_from_csv(
    raw_data_dir: Path,
    *,
    is_rocpd: bool,
    kernel_verbose: int,
    verbose: int,
    node_name: Optional[str] = None,
) -> pd.DataFrame:
    """Load a pmc_perf.csv file and return the canonical PMC DataFrame."""
    pmc_perf_path = raw_data_dir / f"{schema.PMC_PERF_FILE_PREFIX}.csv"
    if not pmc_perf_path.is_file():
        return pd.DataFrame()

    df = pd.read_csv(pmc_perf_path)
    if is_rocpd:
        df = to_canonical_pmc_frame(df)

    return prepare_pmc_frame(
        df,
        kernel_verbose=kernel_verbose,
        verbose=verbose,
        node_name=node_name,
    )


def prepare_pmc_frame(
    df: pd.DataFrame,
    *,
    kernel_verbose: int,
    verbose: int,
    node_name: Optional[str] = None,
) -> pd.DataFrame:
    """Apply common post-load formatting to a PMC DataFrame."""
    if df.empty:
        return df

    if kernel_verbose >= 0:
        kernel_name_shortener(df, kernel_verbose)

    if node_name is not None:
        df.insert(0, "Node", node_name)

    if verbose >= 2:
        console_debug(f"pmc_raw_data final_single_df {df.info}")
    return df


def _build_rocpd_metadata_row(group_df: pd.DataFrame) -> dict:
    return {
        column: group_df[column].iloc[0]
        for column in ROCPD_METADATA_COLUMNS
        if column in group_df.columns
    }
