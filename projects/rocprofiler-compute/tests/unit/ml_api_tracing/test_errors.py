# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for ML API trace analysis error contracts."""

import pandas as pd
import pytest

from ml_api_tracing.errors import (
    FATAL_ML_API_TRACE_ERRORS,
    KernelSequenceLengthMismatchError,
    LauncherThreadNotFoundError,
    MarkerNotNestedError,
    MissingSourceLocationError,
    MlApiTraceError,
    OverlappingMarkerRangeError,
    PassMarkerMismatchError,
    UnaccountedKernelError,
    UncorrelatedLauncherIntervalError,
)


def test_unaccounted_kernel_error_keeps_rows_and_bounded_name_order():
    unmatched_rows = pd.DataFrame({
        "Kernel_Name": [
            "z",
            "a",
            "z",
            "k8",
            "k7",
            "k6",
            "k5",
            "k4",
            "k3",
            "k2",
            "k1",
            "k0",
        ],
    })
    error = UnaccountedKernelError(unmatched_rows)
    expected_message = (
        "12 unmatched dispatches; first 10 Kernel_Names: "
        "z, a, k8, k7, k6, k5, k4, k3, k2, k1"
    )
    assert error.unmatched_rows is unmatched_rows
    assert str(error) == expected_message
    unmatched_rows.loc[0, "Kernel_Name"] = "changed"
    assert error.unmatched_rows.loc[0, "Kernel_Name"] == "changed"
    assert str(error) == expected_message


@pytest.mark.parametrize(
    ("columns", "expected_message"),
    [
        pytest.param({}, "0 unmatched dispatches", id="empty"),
        pytest.param(
            {"Dispatch_ID": [5, 9]},
            "2 unmatched dispatches",
            id="no-kernel-names",
        ),
        pytest.param(
            {"Kernel_Name": ["", ""]},
            "2 unmatched dispatches",
            id="empty-kernel-names",
        ),
        pytest.param(
            {"Kernel_Name": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]},
            "10 unmatched dispatches; Kernel_Names: a, b, c, d, e, f, g, h, i, j",
            id="exact-preview-limit",
        ),
    ],
)
def test_unaccounted_kernel_error_formats_complete_name_preview(
    columns, expected_message
):
    error = UnaccountedKernelError(pd.DataFrame(columns))
    assert str(error) == expected_message
    assert isinstance(error, MlApiTraceError)
    assert not isinstance(error, FATAL_ML_API_TRACE_ERRORS)


@pytest.mark.parametrize(
    ("error_type", "attributes", "expected_message", "fatal"),
    [
        pytest.param(
            PassMarkerMismatchError,
            {
                "stitch_key": "trace",
                "function_ordinal": None,
                "disagreeing_values": "torch != triton",
            },
            "Pass marker mismatch stitch_key=trace: torch != triton",
            True,
            id="pass-marker-without-ordinal",
        ),
        pytest.param(
            PassMarkerMismatchError,
            {
                "stitch_key": "trace",
                "function_ordinal": 0,
                "disagreeing_values": "torch != triton",
            },
            "Pass marker mismatch stitch_key=trace function_ordinal=0: torch != triton",
            True,
            id="pass-marker-zero-ordinal",
        ),
        pytest.param(
            LauncherThreadNotFoundError,
            {
                "operator_name": "aten::mm",
                "thread_id": "7",
                "start_timestamp": 1.5,
                "launcher_thread_id": "3",
            },
            "Launcher thread not found for Operator_Name=aten::mm Thread_Id=7 "
            "Start_Timestamp=1.5 launcher_thread_id=3",
            False,
            id="launcher-thread-not-found",
        ),
        pytest.param(
            UncorrelatedLauncherIntervalError,
            {
                "operator_name": "aten::mm",
                "thread_id": "7",
                "start_timestamp": 1.5,
                "end_timestamp": 2.5,
                "launcher_thread_id": "3",
            },
            "Uncorrelated launcher interval for Operator_Name=aten::mm "
            "Thread_Id=7 Start_Timestamp=1.5 End_Timestamp=2.5 "
            "launcher_thread_id=3",
            False,
            id="uncorrelated-launcher-interval",
        ),
        pytest.param(
            OverlappingMarkerRangeError,
            {
                "thread_id": "7",
                "first_name": "first",
                "first_start": 1.5,
                "first_end": 3.5,
                "second_name": "second",
                "second_start": 2.5,
                "second_end": 4.5,
            },
            "Overlapping marker ranges on Thread_Id=7: "
            "first [1.5, 3.5] and second [2.5, 4.5]",
            True,
            id="overlapping-marker-ranges",
        ),
        pytest.param(
            MissingSourceLocationError,
            {
                "operator_name": "aten::mm",
                "thread_id": "7",
                "start_timestamp": 1.5,
            },
            "Missing source location for Operator_Name=aten::mm Thread_Id=7 "
            "Start_Timestamp=1.5",
            False,
            id="missing-source-location",
        ),
        pytest.param(
            KernelSequenceLengthMismatchError,
            {
                "operator_name": "aten::mm",
                "name_count": 2,
                "start_count": 3,
                "end_count": 4,
            },
            "Kernel name and timestamp lists have unequal lengths for "
            "Operator_Name=aten::mm names=2 starts=3 ends=4",
            True,
            id="kernel-sequence-length-mismatch",
        ),
        pytest.param(
            MarkerNotNestedError,
            {
                "operator_name": "aten::mm",
                "thread_id": "7",
                "start_timestamp": 1.5,
            },
            "Marker was not nested for Operator_Name=aten::mm Thread_Id=7 "
            "Start_Timestamp=1.5",
            True,
            id="marker-not-nested",
        ),
    ],
)
def test_error_constructors_preserve_attributes_messages_and_classification(
    error_type, attributes, expected_message, fatal
):
    error = error_type(**attributes)
    assert vars(error) == attributes
    assert str(error) == expected_message
    assert error.args == (expected_message,)
    assert isinstance(error, MlApiTraceError)
    assert isinstance(error, FATAL_ML_API_TRACE_ERRORS) is fatal


def test_fatal_error_classes_keep_analysis_policy_order():
    assert FATAL_ML_API_TRACE_ERRORS == (
        KernelSequenceLengthMismatchError,
        MarkerNotNestedError,
        OverlappingMarkerRangeError,
        PassMarkerMismatchError,
    )
