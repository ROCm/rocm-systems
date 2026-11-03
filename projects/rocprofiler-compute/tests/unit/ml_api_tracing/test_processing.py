# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for ml_api_tracing/processing.py."""

import pandas as pd
import pytest

import ml_api_tracing.processing as processing
from ml_api_tracing.errors import MlApiTraceInputError, PassMarkerMismatchError
from ml_api_tracing.processing import (
    apply_parsed_function_columns,
    collapse_matching_markers_across_passes,
    group_marker_pass,
    join_marker_counter_pass,
    unmatched_kernel_rows,
)

COLLAPSED_COLUMNS = [
    "Function",
    "Thread_Id",
    "Start_Timestamp",
    "End_Timestamp",
    "Kernel_Names",
    "Kernel_Start_Timestamps",
    "Kernel_End_Timestamps",
    "Kernel_Pmc_Group_Keys",
]
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


def forbidden_later_kernel_check(*args):
    pytest.fail("Kernel-name comparison ran after the first fatal mismatch")


def forbidden_marker_parser(function_value):
    pytest.fail("Parsing ran before the complete legacy-wire check")


def test_join_collapses_long_counter_rows_with_first_dispatch_values(
    linear_marker_row, counter_row
):
    markers = pd.DataFrame([linear_marker_row(1, 0, 200)])
    counters = counter_frame([
        counter_row(1, "first_kernel", 100, 150, dispatch_id=7),
        counter_row(1, "ignored_duplicate", 120, 190, dispatch_id=7),
        counter_row(1, "second_kernel", 160, 180, dispatch_id=2),
    ])
    original_markers = markers.copy(deep=True)
    original_counters = counters.copy(deep=True)

    joined = join_marker_counter_pass(markers, counters)

    assert joined["Dispatch_ID"].tolist() == [7, 2]
    assert joined["Kernel_Name"].tolist() == ["first_kernel", "second_kernel"]
    assert joined["Kernel_Start_Timestamp"].tolist() == [100, 160]
    assert joined["Kernel_End_Timestamp"].tolist() == [150, 180]
    assert list(joined.columns) == [
        "Dispatch_ID",
        "GUID",
        "Correlation_ID",
        "Kernel_Name",
        "Grid_Size",
        "Workgroup_Size",
        "LDS_Per_Workgroup",
        "Kernel_Start_Timestamp",
        "Kernel_End_Timestamp",
        "Function",
        "Thread_Id",
        "Start_Timestamp",
        "End_Timestamp",
        "_marker_order",
    ]
    pd.testing.assert_frame_equal(markers, original_markers)
    pd.testing.assert_frame_equal(counters, original_counters)


def test_join_uses_guid_for_dispatch_collapse_and_marker_matching(
    linear_marker_row, counter_row, test_guid, test_guid_other
):
    markers = pd.DataFrame([linear_marker_row(1, 0, 100, guid=test_guid)])
    counters = counter_frame([
        counter_row(1, "matched", 10, 20, dispatch_id=3, guid=test_guid),
        counter_row(1, "orphan", 30, 40, dispatch_id=3, guid=test_guid_other),
    ])

    joined = join_marker_counter_pass(markers, counters)
    grouped = group_marker_pass(joined)
    unmatched = unmatched_kernel_rows(joined)

    assert len(joined) == 2
    assert grouped["Kernel_Names"].tolist() == [["matched"]]
    assert unmatched["Kernel_Name"].tolist() == ["orphan"]
    assert unmatched["GUID"].tolist() == [test_guid_other]


def test_group_keeps_marker_without_kernels(linear_marker_row):
    grouped = grouped_pass([linear_marker_row(1, 0, 100)], [])

    assert grouped["Kernel_Names"].tolist() == [[]]
    assert grouped["Kernel_Start_Timestamps"].tolist() == [[]]
    assert grouped["Kernel_End_Timestamps"].tolist() == [[]]
    assert grouped["Kernel_Pmc_Group_Keys"].tolist() == [[]]
    assert grouped["Thread_Id"].tolist() == [1]


def test_unmatched_rows_are_collapsed_dispatch_copies(linear_marker_row, counter_row):
    joined = join_marker_counter_pass(
        pd.DataFrame([linear_marker_row(1, 0, 100)]),
        counter_frame([
            counter_row(99, "orphan", 10, 20, dispatch_id=3),
            counter_row(99, "orphan", 10, 20, dispatch_id=3),
        ]),
    )
    original = joined.copy(deep=True)

    unmatched = unmatched_kernel_rows(joined)
    unmatched.loc[:, "Kernel_Name"] = "changed"
    grouped = group_marker_pass(joined)

    assert len(unmatched) == 1
    assert grouped["Kernel_Names"].tolist() == [[]]
    pd.testing.assert_frame_equal(joined, original)


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
    original = joined.copy(deep=True)

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
    pd.testing.assert_frame_equal(joined, original)


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


def test_collapse_uses_first_pass_times_thread_and_dispatch_metadata(
    linear_marker_row, counter_row
):
    first_pass = grouped_pass(
        [linear_marker_row(11, 0, 100, thread_id=42)],
        [counter_row(11, "kernel", 10, 40, dispatch_id=7)],
    )
    second_pass = grouped_pass(
        [linear_marker_row(22, 1000, 1100, thread_id=99)],
        [counter_row(22, "kernel", 1010, 1100, dispatch_id=9)],
    )
    originals = [frame.copy(deep=True) for frame in (first_pass, second_pass)]

    collapsed = collapse_matching_markers_across_passes([first_pass, second_pass])

    pd.testing.assert_frame_equal(collapsed, first_pass[COLLAPSED_COLUMNS])
    pd.testing.assert_frame_equal(first_pass, originals[0])
    pd.testing.assert_frame_equal(second_pass, originals[1])


def test_collapse_matches_occurrence_ordinals(linear_marker_row, counter_row):
    passes = [
        grouped_pass(
            [
                linear_marker_row(1, offset, offset + 50),
                linear_marker_row(2, offset + 60, offset + 110),
            ],
            [
                counter_row(1, "kernel_a", offset + 10, offset + 20, dispatch_id=1),
                counter_row(2, "kernel_b", offset + 70, offset + 80, dispatch_id=2),
            ],
        )
        for offset in (0, 200)
    ]

    collapsed = collapse_matching_markers_across_passes(passes)

    assert collapsed["Start_Timestamp"].tolist() == [0, 60]
    assert collapsed["Kernel_Names"].tolist() == [["kernel_a"], ["kernel_b"]]


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


def test_collapse_reports_first_fatal_before_later_kernel_checks(
    linear_marker_row, counter_row, monkeypatch
):
    passes = [
        grouped_pass(
            [linear_marker_row(1, 0, 50), linear_marker_row(2, 60, 110)],
            [counter_row(1, "kernel_a", 10, 20)],
        ),
        grouped_pass(
            [linear_marker_row(1, 0, 50)], [counter_row(1, "wrong_kernel", 10, 20)]
        ),
    ]

    monkeypatch.setattr(
        processing,
        "_record_kernel_name_mismatch_across_passes",
        forbidden_later_kernel_check,
    )

    with pytest.raises(PassMarkerMismatchError) as excinfo:
        collapse_matching_markers_across_passes(passes)

    assert excinfo.value.function_ordinal is None
    assert excinfo.value.disagreeing_values == "per-pass marker counts [2, 1]"


def test_collapse_explicit_error_list_collects_mismatches_in_order(
    linear_marker_row, counter_row
):
    passes = [
        grouped_pass(
            [linear_marker_row(1, 0, 50), linear_marker_row(2, 60, 110)],
            [counter_row(1, "kernel_a", 10, 20)],
        ),
        grouped_pass(
            [linear_marker_row(1, 0, 50)], [counter_row(1, "wrong_kernel", 10, 20)]
        ),
    ]
    errors = []

    collapsed = collapse_matching_markers_across_passes(passes, errors=errors)

    assert len(collapsed) == 2
    assert len(errors) == 3
    assert errors[0].disagreeing_values == "per-pass marker counts [2, 1]"
    assert errors[1].disagreeing_values.startswith("Kernel_Names ")
    assert errors[1].function_ordinal == 0
    assert errors[2].disagreeing_values == "missing passes [1]"
    assert errors[2].function_ordinal == 1


def test_collapse_rejects_empty_kernel_set_mismatch(linear_marker_row, counter_row):
    passes = [
        grouped_pass([linear_marker_row(1, 0, 100)], counters)
        for counters in ([counter_row(1, "kernel", 10, 20)], [])
    ]

    with pytest.raises(PassMarkerMismatchError, match="Kernel_Names"):
        collapse_matching_markers_across_passes(passes)


def test_empty_pass_retains_existing_columns_and_stitch_dtypes(linear_marker_row):
    marker_columns = list(linear_marker_row(1, 0, 100))
    joined = join_marker_counter_pass(
        pd.DataFrame(columns=marker_columns), counter_frame([])
    )
    grouped = group_marker_pass(joined)
    collapsed = collapse_matching_markers_across_passes([grouped, grouped])

    assert grouped.empty
    assert pd.api.types.is_integer_dtype(grouped["Thread_Id"].dtype)
    assert pd.api.types.is_integer_dtype(grouped["function_ordinal"].dtype)
    assert grouped["stitch_key"].dtype == object
    assert list(collapsed.columns) == [
        column
        for column in grouped.columns
        if column
        not in (
            "Correlation_ID",
            "GUID",
            "stitch_key",
            "function_ordinal",
            "_marker_order",
        )
    ]
    assert apply_parsed_function_columns(collapsed) is collapsed
    pd.testing.assert_frame_equal(
        collapse_matching_markers_across_passes([]),
        pd.DataFrame(columns=COLLAPSED_COLUMNS),
    )


def test_apply_parsed_columns_preserves_raw_wire_thread_and_index():
    function = (
        "aten::addmm:n/a|seqNr=1|tid=1|ftid=1|ltid=7|scope=FUNCTION|args=(a%7Cb)|torch"
    )
    trace_df = pd.DataFrame({"Function": [function], "Thread_Id": [42]}, index=[5])
    original = trace_df.copy(deep=True)

    parsed = apply_parsed_function_columns(trace_df)

    assert parsed.index.tolist() == [5]
    assert parsed["Function"].tolist() == [function]
    assert parsed["Thread_Id"].tolist() == [42]
    assert parsed["launcher_thread_id"].tolist() == ["7"]
    assert parsed["args"].tolist() == ["(a|b)"]
    assert parsed["Operator_Name"].tolist() == ["aten::addmm"]
    pd.testing.assert_frame_equal(trace_df, original)


def test_apply_parsed_columns_scans_all_legacy_wire_before_parsing(monkeypatch):
    trace_df = pd.DataFrame({
        "Function": ["valid:n/a|torch", "triton.JITFunction.foo:#1@file.py:1"],
    })

    monkeypatch.setattr(processing, "parse_marker_function", forbidden_marker_parser)

    with pytest.raises(MlApiTraceInputError) as excinfo:
        apply_parsed_function_columns(trace_df)

    assert str(excinfo.value) == (
        "Stacked marker wire is not supported: triton.JITFunction.foo:#1@file.py:1"
    )


@pytest.mark.parametrize(
    "function", ["triton.JITFunction.foo@file.py", "triton.JITFunction.foo:#file.py"]
)
def test_apply_parsed_columns_accepts_nonstacked_similar_names(function):
    parsed = apply_parsed_function_columns(pd.DataFrame({"Function": [function]}))

    assert parsed["Operator_Name"].tolist() == [function]
