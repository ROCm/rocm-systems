# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for matching ML API trace passes."""

import pandas as pd
import pytest

from ml_api_tracing.analyze.errors import PassMarkerMismatchError
from ml_api_tracing.analyze.pass_matching import (
    collapse_matching_markers_across_passes,
    group_marker_pass,
    join_marker_counter_pass,
)

COUNTER_COLUMNS = [
    "Correlation_ID",
    "GUID",
    "Dispatch_ID",
    "Kernel_Name",
    "Grid_Size",
    "Workgroup_Size",
    "LDS_Per_Workgroup",
    "Start_Timestamp",
    "End_Timestamp",
]


def counter_frame(rows):
    return pd.DataFrame(rows, columns=COUNTER_COLUMNS)


def grouped_pass(markers, counters):
    return group_marker_pass(
        join_marker_counter_pass(pd.DataFrame(markers), counter_frame(counters))
    )


def test_group_preserves_marker_order_kernel_order_and_pmc_keys(
    linear_marker_row, counter_row
):
    markers = pd.DataFrame([
        linear_marker_row(9, 100, 200, thread_id=42),
        linear_marker_row(1, 0, 90, thread_id=42),
    ])
    counters = counter_frame([
        counter_row(1, "earlier_marker_kernel", 10, 20, dispatch_id=4),
        counter_row(9, "kernel_b", 140, 150, dispatch_id=8),
        counter_row(9, "kernel_a", 110, 130, dispatch_id=2),
    ])
    joined = join_marker_counter_pass(markers, counters)

    grouped = group_marker_pass(joined)

    assert grouped["Start_Timestamp"].tolist() == [100, 0]
    assert grouped["function_ordinal"].tolist() == [0, 1]
    assert grouped["Kernel_Names"].tolist() == [
        ["kernel_b", "kernel_a"],
        ["earlier_marker_kernel"],
    ]
    assert grouped["Kernel_Pmc_Group_Keys"].iloc[0] == [
        (8, "kernel_b", 256, 64, 0),
        (2, "kernel_a", 256, 64, 0),
    ]
    assert pd.api.types.is_integer_dtype(grouped["Thread_Id"].dtype)


def test_marker_stitch_key_keeps_seqnr_tid_ftid_omits_ltid(linear_marker_row):
    functions = [
        "aten::addmm:n/a|seqNr=1|tid=1|ftid=0|ltid=7|scope=FUNCTION|args=()|torch",
        "aten::addmm:n/a|seqNr=1|tid=1|ftid=0|ltid=9|scope=FUNCTION|args=()|torch",
    ]
    grouped = grouped_pass(
        [
            linear_marker_row(index, index * 100, index * 100 + 90, function=function)
            for index, function in enumerate(functions)
        ],
        [],
    )

    expected = "aten::addmm:n/a|seqNr=1|tid=1|ftid=0|scope=FUNCTION|args=()|torch"
    assert grouped["stitch_key"].tolist() == [expected, expected]
    assert grouped["function_ordinal"].tolist() == [0, 1]


@pytest.mark.parametrize(
    ("first_wire", "second_wire"),
    [
        ("op%2Fpart:n/a|args=(a%7Cb)|torch", "op/part:n/a|args=(a%7Cb)|torch"),
        ("op:n/a|args=(a%7Cb)|torch", "op:n/a|args=(a|b)|torch"),
        ("op:n/a|seqNr=1|torch", "op:n/a|seqNr=2|torch"),
        ("op:n/a|tid=1|torch", "op:n/a|tid=2|torch"),
        ("op:n/a|ftid=1|torch", "op:n/a|ftid=2|torch"),
        ("op:n/a|scope=FUNCTION|torch", "op:n/a|scope=BACKWARD_FUNCTION|torch"),
    ],
)
def test_stitch_matching_keeps_raw_wire_identity(
    linear_marker_row, first_wire, second_wire
):
    passes = [
        grouped_pass([linear_marker_row(1, 0, 100, function=function)], [])
        for function in (first_wire, second_wire)
    ]

    with pytest.raises(PassMarkerMismatchError, match="missing passes"):
        collapse_matching_markers_across_passes(passes)


def test_collapse_matches_kernel_name_sets_without_reordering_first_pass(
    linear_marker_row, counter_row
):
    passes = [
        grouped_pass(
            [linear_marker_row(1, 0, 100)],
            [
                counter_row(1, name, index * 10, index * 10 + 5, dispatch_id=index)
                for index, name in enumerate(names)
            ],
        )
        for names in (["kernel_b", "kernel_a", "kernel_b"], ["kernel_a", "kernel_b"])
    ]

    collapsed = collapse_matching_markers_across_passes(passes)

    assert collapsed["Kernel_Names"].tolist() == [["kernel_b", "kernel_a", "kernel_b"]]
