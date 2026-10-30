# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""PC-sampling section of the analyze report, built from the analysis database.

The rows come from the PC-sampling summary view, the same rows that
pc_sampling_summary.csv holds. The display layout is the table 21.1 layout that
the parsed tool records produce, so both sources show the same columns.
"""

import json
from typing import Collection, Dict, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from pc_sampling.pc_sampling_analysis import SOURCE_LINE_MISSING
from utils.analysis_orm import PREFIX
from utils.parser import format_pc_sampling_display_frame

# Stated wherever the PC-sampling section appears (FR-25).
PC_SAMPLING_FILTER_NOTE = (
    "PC samples are narrowed by -k only; --gpu-id and --kernel-iteration-range "
    "do not apply to them yet."
)

PC_SAMPLING_SUMMARY_VIEW = f"{PREFIX}pc_sampling_summary_view"

# The view's column names are renamed to the names the display layout reads.
SUMMARY_COLUMN_RENAMES = {
    "source": "source_line",
    "kernel_name": "Kernel_Name",
}

# Ordering by pid, code object and offset matches the order the parsed records
# produce, so the count sort breaks ties the same way.
SUMMARY_ROWS_STATEMENT = text(
    'SELECT pid, code_object_id, kernel_name, "offset", instruction, source, '
    "count, issue_count, stall_count, wave_occupancy_percent, "
    "active_thread_percent, stall_reason "
    f"FROM {PC_SAMPLING_SUMMARY_VIEW} "
    "WHERE workload_id = :workload_id "
    'ORDER BY pid, code_object_id, kernel_name, "offset"'
)


def load_pc_sampling_summary_rows(session: Session, workload_id: int) -> pd.DataFrame:
    """Return one workload's rows of the PC-sampling summary view, normalized."""
    rows = pd.read_sql_query(
        SUMMARY_ROWS_STATEMENT,
        session.connection(),
        params={"workload_id": workload_id},
    )
    return normalize_summary_rows(rows)


def normalize_summary_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Return summary-view rows under the names the display layout reads.

    A sample without a source line holds "N/A", and the stall reasons are decoded
    to {reason: count} dicts.
    """
    normalized = rows.rename(columns=SUMMARY_COLUMN_RENAMES)
    normalized["source_line"] = normalized["source_line"].fillna(SOURCE_LINE_MISSING)
    normalized["stall_reason"] = normalized["stall_reason"].apply(_decode_stall_reason)
    return normalized


def build_pc_sampling_section(
    summary_rows: pd.DataFrame,
    *,
    gpu_arch: Optional[str],
    sorting_type: str,
    num_rows: Optional[int] = None,
    kernel_names: Optional[Collection[str]] = None,
) -> pd.DataFrame:
    """Return the table 21.1 rows: the top num_rows of every selected kernel.

    kernel_names holds Kernel_Name strings, not -k indices; the caller maps each
    -k index to its name through the kernel top table. None keeps every kernel.
    num_rows of 0 or None keeps all rows of a kernel. Each kernel is sorted by
    sorting_type, offset or count, and the kernels follow in name order.
    """
    # The method is detected over every sample before the -k filter, as the
    # parsed path detects it over every record.
    method = _sampling_method(summary_rows)
    rows = summary_rows
    if kernel_names is not None:
        rows = rows[rows["Kernel_Name"].isin(list(kernel_names))]
    if rows.empty:
        return pd.DataFrame()

    kernel_sections = [
        format_pc_sampling_display_frame(
            kernel_rows.copy(),
            method,
            sorting_type,
            num_rows=num_rows,
            gpu_arch=gpu_arch,
        )
        for _, kernel_rows in rows.groupby("Kernel_Name", sort=True)
    ]
    return pd.concat(kernel_sections, ignore_index=True)


def _sampling_method(rows: pd.DataFrame) -> str:
    """Return the sampling method that produced the rows.

    Only stochastic samples record whether the wave issued, so only their issue
    counts are known.
    """
    return "stochastic" if rows["issue_count"].notna().any() else "host_trap"


def _decode_stall_reason(stall_reason: Optional[str]) -> Optional[Dict[str, int]]:
    """Decode the JSON object the view builds from one sample's stall reasons."""
    if stall_reason is None:
        return None
    return json.loads(stall_reason)
