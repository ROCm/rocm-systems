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


class ForwardThreadNotFoundError(MlApiTraceError):
    """No OS thread has T_Tid equal to the unlocated tree's usable F_Tid."""

    def __init__(
        self,
        operator_name: str,
        thread_id: str,
        start_timestamp: float,
        f_tid: str,
    ) -> None:
        self.operator_name = operator_name
        self.thread_id = thread_id
        self.start_timestamp = start_timestamp
        self.f_tid = f_tid
        super().__init__(
            "Forward thread not found for "
            f"Operator_Name={operator_name} Thread_Id={thread_id} "
            f"Start_Timestamp={start_timestamp} F_Tid={f_tid}"
        )


class UncorrelatedForwardIntervalError(MlApiTraceError):
    """The forward thread exists but no marker interval contains this tree."""

    def __init__(
        self,
        operator_name: str,
        thread_id: str,
        start_timestamp: float,
        end_timestamp: float,
        f_tid: str,
        forward_thread_id: str,
    ) -> None:
        self.operator_name = operator_name
        self.thread_id = thread_id
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.f_tid = f_tid
        self.forward_thread_id = forward_thread_id
        super().__init__(
            "Uncorrelated forward interval for "
            f"Operator_Name={operator_name} Thread_Id={thread_id} "
            f"Start_Timestamp={start_timestamp} End_Timestamp={end_timestamp} "
            f"F_Tid={f_tid} forward Thread_Id={forward_thread_id}"
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
