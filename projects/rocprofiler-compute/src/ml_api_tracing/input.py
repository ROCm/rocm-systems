# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""CSV input boundary for ML API tracing analysis."""

from pathlib import Path
from typing import List, Tuple

import pandas as pd

from ml_api_tracing.analysis_models import MlApiTracePair
from ml_api_tracing.errors import MlApiTraceInputError
from utils import csv_compression

_REQUIRED_MARKER_COLUMNS = (
    "Function",
    "Thread_Id",
    "Correlation_ID",
    "GUID",
    "Start_Timestamp",
    "End_Timestamp",
)
_REQUIRED_COUNTER_COLUMNS = (
    "Correlation_ID",
    "GUID",
    "Dispatch_ID",
    "Kernel_Name",
    "Grid_Size",
    "Workgroup_Size",
    "LDS_Per_Workgroup",
    "Start_Timestamp",
    "End_Timestamp",
)


def load_ml_api_trace_pairs(workload_dir: Path) -> List[MlApiTracePair]:
    """Load all discovered marker files before returning their pass records."""
    return [
        MlApiTracePair(
            marker_df=_load_marker_trace_dataframe(marker_path),
            counter_path=counter_path,
        )
        for marker_path, counter_path in _find_ml_api_trace_csv_pairs(workload_dir)
    ]


def load_counter_trace_dataframe(counter_path: Path) -> pd.DataFrame:
    """Read the columns needed to match one pass of counter dispatches."""
    return _normalize_counter_trace_dataframe(
        pd.read_csv(counter_path), str(counter_path)
    )


def _find_ml_api_trace_csv_pairs(workload_dir: Path) -> List[Tuple[Path, Path]]:
    """Return sibling gzip marker/counter paths in lexical marker order."""
    marker_glob = f"**/ml_api_trace*_marker_api_trace.csv{csv_compression.GZIP_SUFFIX}"
    pairs: List[Tuple[Path, Path]] = []
    for marker_path in sorted(workload_dir.glob(marker_glob)):
        marker_csv_name = marker_path.name
        if marker_csv_name.endswith(csv_compression.GZIP_SUFFIX):
            marker_csv_name = marker_csv_name[: -len(csv_compression.GZIP_SUFFIX)]
        counter_csv_name = marker_csv_name.replace(
            "_marker_api_trace.csv", "_counter_collection.csv"
        )
        counter_path = csv_compression.compressed_name(
            marker_path.parent / counter_csv_name
        )
        if counter_path.is_file():
            pairs.append((marker_path, counter_path))
    return pairs


def _load_marker_trace_dataframe(marker_path: Path) -> pd.DataFrame:
    """Read the columns needed to process one pass of marker intervals."""
    return _normalize_marker_trace_dataframe(pd.read_csv(marker_path), str(marker_path))


def _normalize_marker_trace_dataframe(
    marker_df: pd.DataFrame, source_name: str
) -> pd.DataFrame:
    marker_df = _rename_column_alias(marker_df, "Correlation_ID", "Correlation_Id")
    return _required_trace_columns(
        marker_df, _REQUIRED_MARKER_COLUMNS, f"Marker CSV {source_name}"
    ).copy()


def _normalize_counter_trace_dataframe(
    counter_df: pd.DataFrame, source_name: str
) -> pd.DataFrame:
    counter_df = _rename_column_alias(counter_df, "Correlation_ID", "Correlation_Id")
    counter_df = _rename_column_alias(counter_df, "Dispatch_ID", "Dispatch_Id")
    return _required_trace_columns(
        counter_df, _REQUIRED_COUNTER_COLUMNS, f"Counter CSV {source_name}"
    )


def _required_trace_columns(
    frame: pd.DataFrame, required_columns: Tuple[str, ...], source_label: str
) -> pd.DataFrame:
    missing_columns = [
        column for column in required_columns if column not in frame.columns
    ]
    if missing_columns:
        raise MlApiTraceInputError(
            f"{source_label} is missing required columns {missing_columns}"
        )
    kept_columns = list(required_columns)
    null_columns = [column for column in kept_columns if frame[column].isna().any()]
    if null_columns:
        raise MlApiTraceInputError(f"{source_label} has null values in {null_columns}")
    return frame[kept_columns]


def _rename_column_alias(
    frame: pd.DataFrame, canonical_name: str, alias_name: str
) -> pd.DataFrame:
    """Rename alias_name to canonical_name, or drop the alias when both exist."""
    if alias_name in frame.columns and canonical_name not in frame.columns:
        return frame.rename(columns={alias_name: canonical_name})
    if alias_name in frame.columns:
        return frame.drop(columns=[alias_name])
    return frame
