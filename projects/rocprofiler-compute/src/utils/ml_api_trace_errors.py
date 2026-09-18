# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from typing import Optional

import pandas as pd


class MlApiTraceError(Exception):
    """Base class for ML API trace analyze failures."""


class UnaccountedKernelError(MlApiTraceError):
    """A GPU dispatch whose Correlation_ID is not in that pass's marker CSV."""

    def __init__(self, unmatched_rows: pd.DataFrame) -> None:
        self.unmatched_rows = unmatched_rows
        super().__init__(_format_unmatched_kernel_message(unmatched_rows))


class PassMarkerMismatchError(MlApiTraceError):
    """Operator calls or kernel-name sets disagree across profiling passes."""

    def __init__(
        self,
        stitch_key: str,
        function_ordinal: Optional[int],
        disagreeing_values: str,
    ) -> None:
        self.stitch_key = stitch_key
        self.function_ordinal = function_ordinal
        self.disagreeing_values = disagreeing_values
        ordinal_text = (
            f" function_ordinal={function_ordinal}"
            if function_ordinal is not None
            else ""
        )
        super().__init__(
            "Pass marker mismatch"
            f" stitch_key={stitch_key}{ordinal_text}: {disagreeing_values}"
        )


class OverlappingMarkerRangeError(MlApiTraceError):
    """Two markers on the same Thread_Id overlap and neither contains the other."""

    def __init__(
        self,
        thread_id: str,
        first_name: str,
        first_start: float,
        first_end: float,
        second_name: str,
        second_start: float,
        second_end: float,
    ) -> None:
        self.thread_id = thread_id
        self.first_name = first_name
        self.first_start = first_start
        self.first_end = first_end
        self.second_name = second_name
        self.second_start = second_start
        self.second_end = second_end
        super().__init__(
            "Overlapping marker ranges on "
            f"Thread_Id={thread_id}: "
            f"{first_name} [{first_start}, {first_end}] and "
            f"{second_name} [{second_start}, {second_end}]"
        )


class MissingSourceLocationError(MlApiTraceError):
    """A torch or triton marker has no file/line and no ancestor with one."""

    def __init__(
        self,
        operator_name: str,
        thread_id: str,
        start_timestamp: float,
    ) -> None:
        self.operator_name = operator_name
        self.thread_id = thread_id
        self.start_timestamp = start_timestamp
        super().__init__(
            "Missing source location for "
            f"Operator_Name={operator_name} Thread_Id={thread_id} "
            f"Start_Timestamp={start_timestamp}"
        )


def _format_unmatched_kernel_message(unmatched_rows: pd.DataFrame) -> str:
    columns = ["Kernel_Name", "Correlation_ID"]
    if "GUID" in unmatched_rows.columns:
        columns.append("GUID")
    present_columns = [column for column in columns if column in unmatched_rows.columns]
    lines = ["GPU kernel dispatches with no matching ROCTX marker:"]
    for row in unmatched_rows[present_columns].itertuples(index=False):
        parts = [f"{column}={value}" for column, value in zip(present_columns, row)]
        lines.append("  " + ", ".join(parts))
    return "\n".join(lines)
