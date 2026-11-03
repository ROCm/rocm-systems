# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""In-memory pass matching for ML API tracing analysis."""

from typing import Any, Dict, FrozenSet, List, Optional

import pandas as pd

from ml_api_tracing.errors import (
    MlApiTraceError,
    MlApiTraceInputError,
    PassMarkerMismatchError,
    handle_fatal_ml_api_trace_error,
)
from ml_api_tracing.marker_format import (
    parse_marker_function,
    stacked_marker_function_cell,
)

_MARKER_DISPATCH_JOIN_KEYS = ("Correlation_ID", "GUID")
_PASS_DROP_COLUMNS = (
    "Correlation_ID",
    "GUID",
    "stitch_key",
    "function_ordinal",
    "_pass_id",
    "_marker_order",
)
_COLLAPSED_MARKER_COLUMNS = (
    "Function",
    "Thread_Id",
    "Start_Timestamp",
    "End_Timestamp",
    "Kernel_Names",
    "Kernel_Start_Timestamps",
    "Kernel_End_Timestamps",
    "Kernel_Pmc_Group_Keys",
)


def join_marker_counter_pass(
    marker_df: pd.DataFrame, counter_df: pd.DataFrame
) -> pd.DataFrame:
    """Full-outer-join unique counter dispatches with one pass of markers.

    Inputs are the canonical projections returned by the input loaders.
    """
    dispatch_df = (
        counter_df
        .groupby(["Dispatch_ID", "GUID"], sort=False, dropna=False)
        .first()
        .reset_index()
        .rename(
            columns={
                "Start_Timestamp": "Kernel_Start_Timestamp",
                "End_Timestamp": "Kernel_End_Timestamp",
            }
        )
    )
    marker_ordered = marker_df.copy()
    marker_ordered["_marker_order"] = range(len(marker_ordered))
    return pd.merge(
        dispatch_df,
        marker_ordered,
        on=list(_MARKER_DISPATCH_JOIN_KEYS),
        how="outer",
    )


def unmatched_kernel_rows(joined_df: pd.DataFrame) -> pd.DataFrame:
    """Copy dispatches unmatched on Correlation_ID and GUID."""
    return joined_df[
        joined_df["Kernel_Name"].notna() & joined_df["Function"].isna()
    ].copy()


def group_marker_pass(joined_df: pd.DataFrame) -> pd.DataFrame:
    """Group the raw join's marker rows in stable occurrence order."""
    marker_rows = joined_df[joined_df["Function"].notna()].copy()
    marker_rows["Thread_Id"] = marker_rows["Thread_Id"].astype(int)
    return _add_stitch_key_and_ordinal(_group_kernels_onto_markers(marker_rows))


def collapse_matching_markers_across_passes(
    pass_frames: List[pd.DataFrame],
    errors: Optional[List[MlApiTraceError]] = None,
) -> pd.DataFrame:
    """Match grouped pass frames using the first discovered pass row.

    A supplied error list collects every mismatch. With no list, the first
    mismatch raises before further pass checks.
    """
    if not pass_frames:
        return pd.DataFrame(columns=list(_COLLAPSED_MARKER_COLUMNS))
    labeled_frames = []
    for pass_id, frame in enumerate(pass_frames):
        labeled = frame.copy()
        labeled["_pass_id"] = pass_id
        labeled_frames.append(labeled)
    concatenated = pd.concat(labeled_frames, ignore_index=True)
    if concatenated.empty or len(pass_frames) == 1:
        collapsed = concatenated
    else:
        collapsed = _collapse_multiple_marker_passes(pass_frames, concatenated, errors)
    drop_columns = [
        column for column in _PASS_DROP_COLUMNS if column in collapsed.columns
    ]
    if drop_columns:
        collapsed = collapsed.drop(columns=drop_columns)
    return collapsed


def apply_parsed_function_columns(trace_df: pd.DataFrame) -> pd.DataFrame:
    """Return trace data with parsed columns, retaining Function and Thread_Id.

    Empty traces are returned unchanged.
    """
    if trace_df.empty:
        return trace_df
    for function_value in trace_df["Function"]:
        stacked_cell = stacked_marker_function_cell(function_value)
        if stacked_cell is not None:
            raise MlApiTraceInputError(
                f"Stacked marker wire is not supported: {stacked_cell}"
            )
    parsed_rows = trace_df["Function"].map(parse_marker_function)
    parsed_df = pd.DataFrame(list(parsed_rows), index=trace_df.index)
    return pd.concat([trace_df, parsed_df], axis=1)


def _group_kernels_onto_markers(joined_df: pd.DataFrame) -> pd.DataFrame:
    """Collect kernel dispatches for each marker identity in one pass."""
    if joined_df.empty:
        return joined_df
    ordered = joined_df
    if "_marker_order" in joined_df.columns:
        ordered = joined_df.sort_values("_marker_order", kind="mergesort")
    group_keys = ["Function", "Thread_Id", "Start_Timestamp", "End_Timestamp"]
    records: List[Dict[str, Any]] = []
    for _, group in ordered.groupby(group_keys, sort=False, dropna=False):
        first = group.iloc[0]
        kernel_names: List[str] = []
        kernel_starts: List[object] = []
        kernel_ends: List[object] = []
        pmc_group_keys: List[tuple] = []
        for row in group.itertuples(index=False):
            kernel_name = row.Kernel_Name
            if pd.isna(kernel_name):
                continue
            kernel_names.append(str(kernel_name))
            kernel_starts.append(row.Kernel_Start_Timestamp)
            kernel_ends.append(row.Kernel_End_Timestamp)
            pmc_group_keys.append((
                row.Dispatch_ID,
                str(kernel_name),
                row.Grid_Size,
                row.Workgroup_Size,
                row.LDS_Per_Workgroup,
            ))
        record: Dict[str, Any] = {
            "Function": first["Function"],
            "Thread_Id": first["Thread_Id"],
            "Start_Timestamp": first["Start_Timestamp"],
            "End_Timestamp": first["End_Timestamp"],
            "Correlation_ID": first["Correlation_ID"],
            "GUID": first["GUID"],
            "Kernel_Names": kernel_names,
            "Kernel_Start_Timestamps": kernel_starts,
            "Kernel_End_Timestamps": kernel_ends,
            "Kernel_Pmc_Group_Keys": pmc_group_keys,
        }
        if "_marker_order" in group.columns:
            record["_marker_order"] = first["_marker_order"]
        records.append(record)
    grouped = pd.DataFrame.from_records(records)
    if "_marker_order" in grouped.columns:
        grouped = grouped.sort_values("_marker_order", kind="mergesort")
    return grouped


def _add_stitch_key_and_ordinal(pass_frame: pd.DataFrame) -> pd.DataFrame:
    """Number occurrences of each raw Function key with only ltid removed."""
    if pass_frame.empty:
        ordered = pass_frame.copy()
        ordered["stitch_key"] = pd.Series(dtype=str)
        ordered["function_ordinal"] = pd.Series(dtype=int)
        return ordered
    ordered = pass_frame
    if "_marker_order" in pass_frame.columns:
        ordered = pass_frame.sort_values("_marker_order", kind="mergesort")
    ordered = ordered.copy()
    ordered["stitch_key"] = ordered["Function"].map(_marker_stitch_key)
    ordered["function_ordinal"] = ordered.groupby("stitch_key", sort=False).cumcount()
    return ordered


def _marker_stitch_key(function_value: object) -> str:
    """Retain every raw Function token except ltid= launcher attribution."""
    if function_value is None or (
        isinstance(function_value, float) and pd.isna(function_value)
    ):
        return ""
    return "|".join(
        token
        for token in str(function_value).split("|")
        if not token.startswith("ltid=")
    )


def _collapse_multiple_marker_passes(
    pass_frames: List[pd.DataFrame],
    concatenated: pd.DataFrame,
    errors: Optional[List[MlApiTraceError]],
) -> pd.DataFrame:
    pass_counts = [len(frame) for frame in pass_frames]
    if len(set(pass_counts)) != 1:
        handle_fatal_ml_api_trace_error(
            PassMarkerMismatchError(
                stitch_key="",
                function_ordinal=None,
                disagreeing_values=f"per-pass marker counts {pass_counts}",
            ),
            errors,
        )
    expected_passes = set(range(len(pass_frames)))
    records: List[Dict[str, Any]] = []
    grouped = concatenated.groupby(
        ["stitch_key", "function_ordinal"], sort=False, dropna=False
    )
    for (stitch_key, ordinal), group in grouped:
        present_passes = set(group["_pass_id"].tolist())
        missing_passes = sorted(expected_passes - present_passes)
        if missing_passes:
            handle_fatal_ml_api_trace_error(
                PassMarkerMismatchError(
                    stitch_key=str(stitch_key),
                    function_ordinal=int(ordinal),
                    disagreeing_values=f"missing passes {missing_passes}",
                ),
                errors,
            )
        present_ids = sorted(present_passes)
        if len(present_ids) > 1:
            _record_kernel_name_mismatch_across_passes(
                group, stitch_key, ordinal, present_ids, errors
            )
        pass0 = group[group["_pass_id"] == 0]
        first = pass0.iloc[0] if not pass0.empty else group.iloc[0]
        records.append({column: first[column] for column in _COLLAPSED_MARKER_COLUMNS})
    return (
        pd.DataFrame.from_records(records)
        if records
        else pd.DataFrame(columns=list(_COLLAPSED_MARKER_COLUMNS))
    )


def _record_kernel_name_mismatch_across_passes(
    group: pd.DataFrame,
    stitch_key: object,
    ordinal: object,
    present_ids: List[object],
    errors: Optional[List[MlApiTraceError]],
) -> None:
    kernel_name_sets = [
        _kernel_names_as_set(
            group[group["_pass_id"] == pass_id].iloc[0]["Kernel_Names"]
        )
        for pass_id in present_ids
    ]
    if len(set(kernel_name_sets)) != 1:
        handle_fatal_ml_api_trace_error(
            PassMarkerMismatchError(
                stitch_key=str(stitch_key),
                function_ordinal=int(ordinal),
                disagreeing_values=f"Kernel_Names {list(kernel_name_sets)}",
            ),
            errors,
        )


def _kernel_names_as_set(names: object) -> FrozenSet[str]:
    if names is None or (isinstance(names, float) and pd.isna(names)):
        return frozenset()
    return frozenset(str(name) for name in names)
