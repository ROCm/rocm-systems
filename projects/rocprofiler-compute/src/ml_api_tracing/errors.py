# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ML API trace errors and their reporting policies.

Input errors always raise. Integrity errors raise unless explicitly collected.
Coverage issues are collected for the Workload adapter to report as warnings.
"""

from typing import List, Optional

import pandas as pd

_UNMATCHED_KERNEL_NAME_PREVIEW = 10


def handle_fatal_ml_api_trace_error(
    error: "MlApiTraceError",
    errors: Optional[List["MlApiTraceError"]] = None,
) -> None:
    """Append an integrity error to an explicit collector, or raise immediately.

    Input errors bypass this policy; coverage issues use separate collectors.
    """
    if not isinstance(error, FATAL_ML_API_TRACE_ERRORS):
        raise TypeError("Expected a fatal ML API trace error")
    if errors is not None:
        errors.append(error)
        return
    raise error


def _format_unmatched_kernel_message(unmatched_rows: pd.DataFrame) -> str:
    count = len(unmatched_rows)
    names: List[str] = []
    if "Kernel_Name" in unmatched_rows.columns:
        names = list(
            dict.fromkeys(str(value) for value in unmatched_rows["Kernel_Name"])
        )
    preview = ", ".join(names[:_UNMATCHED_KERNEL_NAME_PREVIEW])
    if len(names) > _UNMATCHED_KERNEL_NAME_PREVIEW:
        return (
            f"{count} unmatched dispatches; first "
            f"{_UNMATCHED_KERNEL_NAME_PREVIEW} Kernel_Names: {preview}"
        )
    if preview:
        return f"{count} unmatched dispatches; Kernel_Names: {preview}"
    return f"{count} unmatched dispatches"


class MlApiTraceError(Exception):
    """Base class for ML API trace analyze failures."""


class MlApiTraceInputError(MlApiTraceError):
    """Invalid trace columns, null values, or unsupported marker wire."""


class UnaccountedKernelError(MlApiTraceError):
    """Dispatch unmatched on Correlation_ID and GUID."""

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


class LauncherThreadNotFoundError(MlApiTraceError):
    """No call-tree thread matches the unlocated tree's launcher_thread_id."""

    def __init__(
        self,
        operator_name: str,
        thread_id: str,
        start_timestamp: float,
        launcher_thread_id: str,
    ) -> None:
        self.operator_name = operator_name
        self.thread_id = thread_id
        self.start_timestamp = start_timestamp
        self.launcher_thread_id = launcher_thread_id
        super().__init__(
            "Launcher thread not found for "
            f"Operator_Name={operator_name} Thread_Id={thread_id} "
            f"Start_Timestamp={start_timestamp} "
            f"launcher_thread_id={launcher_thread_id}"
        )


class UncorrelatedLauncherIntervalError(MlApiTraceError):
    """The launcher thread exists but no marker interval contains this tree."""

    def __init__(
        self,
        operator_name: str,
        thread_id: str,
        start_timestamp: float,
        end_timestamp: float,
        launcher_thread_id: str,
    ) -> None:
        self.operator_name = operator_name
        self.thread_id = thread_id
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.launcher_thread_id = launcher_thread_id
        super().__init__(
            "Uncorrelated launcher interval for "
            f"Operator_Name={operator_name} Thread_Id={thread_id} "
            f"Start_Timestamp={start_timestamp} "
            f"End_Timestamp={end_timestamp} "
            f"launcher_thread_id={launcher_thread_id}"
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
    """Unlocated root in KNOWN_ML_API_BACKENDS with no launcher thread."""

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


class KernelSequenceLengthMismatchError(MlApiTraceError):
    """Kernel name and timestamp lists on one marker row have unequal lengths."""

    def __init__(
        self,
        operator_name: str,
        name_count: int,
        start_count: int,
        end_count: int,
    ) -> None:
        self.operator_name = operator_name
        self.name_count = name_count
        self.start_count = start_count
        self.end_count = end_count
        super().__init__(
            "Kernel name and timestamp lists have unequal lengths for "
            f"Operator_Name={operator_name} names={name_count} "
            f"starts={start_count} ends={end_count}"
        )


class MarkerNotNestedError(MlApiTraceError):
    """A consolidated marker row is missing from the nested call forest."""

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
            "Marker was not nested for "
            f"Operator_Name={operator_name} Thread_Id={thread_id} "
            f"Start_Timestamp={start_timestamp}"
        )


# The tuple uses class objects, so it must follow their definitions.
FATAL_ML_API_TRACE_ERRORS = (
    KernelSequenceLengthMismatchError,
    MarkerNotNestedError,
    OverlappingMarkerRangeError,
    PassMarkerMismatchError,
)
