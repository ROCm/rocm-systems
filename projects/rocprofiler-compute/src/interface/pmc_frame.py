# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Canonical PMC DataFrame transformations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from utils.logger import console_debug

if TYPE_CHECKING:
    import pandas as pd

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


def is_long_counter_frame(frame: pd.DataFrame) -> bool:
    """Return True when a frame stores one counter value per row."""
    counter_columns = {ROCPD_COUNTER_NAME_COLUMN, ROCPD_COUNTER_VALUE_COLUMN}
    return counter_columns.issubset(frame.columns)


def to_canonical_pmc_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the canonical wide PMC frame used by analysis."""
    if frame.empty:
        return frame

    if is_long_counter_frame(frame):
        return pivot_counter_rows(frame)

    return frame


def process_rocpd_csv(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize rocpd counter rows to the canonical PMC frame."""
    return to_canonical_pmc_frame(frame)


def pivot_counter_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-form counter rows into one row per dispatch."""
    if frame.empty:
        return frame

    import pandas as pd

    rows: list[dict] = []
    for _, group_frame in frame.groupby(ROCPD_GROUP_COLUMNS):
        row = _build_rocpd_metadata_row(group_frame)
        row.update(
            dict(
                zip(
                    group_frame[ROCPD_COUNTER_NAME_COLUMN],
                    group_frame[ROCPD_COUNTER_VALUE_COLUMN],
                )
            )
        )
        rows.append(row)

    processed_frame = pd.DataFrame(rows)
    processed_frame["GPU_ID"] = (
        processed_frame["GPU_ID"].rank(method="dense").astype(int) - 1
    )
    processed_frame["Dispatch_ID"] = range(len(processed_frame))
    return processed_frame


def prepare_pmc_frame(
    frame: pd.DataFrame,
    *,
    kernel_verbose: int,
    verbose: int,
    node_name: Optional[str] = None,
) -> pd.DataFrame:
    """Apply common post-load formatting to a PMC DataFrame."""
    if frame.empty:
        return frame

    if kernel_verbose >= 0:
        from utils.kernel_name_shortener import kernel_name_shortener

        kernel_name_shortener(frame, kernel_verbose)

    if node_name is not None:
        frame.insert(0, "Node", node_name)

    if verbose >= 2:
        console_debug(f"pmc_raw_data final_single_df {frame.info}")
    return frame


def _build_rocpd_metadata_row(group_frame: pd.DataFrame) -> dict:
    return {
        column: group_frame[column].iloc[0]
        for column in ROCPD_METADATA_COLUMNS
        if column in group_frame.columns
    }
