# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Workload state and diagnostics for ML API trace analysis."""

import pandas as pd
import pytest

from ml_api_tracing import analysis, processing
from ml_api_tracing.analysis import process_ml_api_trace_output
from ml_api_tracing.analysis_models import MlApiTracePair
from ml_api_tracing.errors import LauncherThreadNotFoundError, MarkerNotNestedError
from ml_api_tracing.marker_format import parse_marker_function
from ml_api_tracing.models import CallTreeNode
from utils import csv_compression, schema


def build_record_console_error_and_exit(monkeypatch):
    """Replace console_error so tests see the message and SystemExit."""
    messages = []

    def console_error(*argv, exit=True, exit_code=1):
        if len(argv) > 1:
            messages.append(str(argv[1]))
        elif argv:
            messages.append(str(argv[0]))
        else:
            messages.append("")
        if exit:
            raise SystemExit(exit_code)

    monkeypatch.setattr(analysis, "console_error", console_error)
    return messages


@pytest.fixture
def record_console_error_and_exit():
    return build_record_console_error_and_exit


def fail_unmatched(_joined):
    raise ValueError("unmatched extraction failed")


def fail_parse(_function):
    raise ValueError("marker parsing failed")


def fail_validation(_trace, _forest, skipped_keys):
    raise MarkerNotNestedError("missing", "42", 20.0)


@pytest.fixture
def populated_workload(tmp_path):
    return schema.Workload(
        ml_api_trace_pairs=[MlApiTracePair(pd.DataFrame(), tmp_path / "old.csv")],
        unmatched_kernel_frames=[pd.DataFrame({"Kernel_Name": ["old"]})],
        ml_api_trace_df=pd.DataFrame({"Function": ["old"]}),
        ml_api_call_trees={"old": [CallTreeNode(name="old")]},
    )


def test_later_marker_failure_preserves_all_previous_state(
    tmp_path,
    monkeypatch,
    populated_workload,
    record_console_error_and_exit,
    write_ml_api_pass,
    linear_marker_row,
    counter_row,
):
    workload = populated_workload
    previous = vars(workload).copy()
    messages = record_console_error_and_exit(monkeypatch)
    bad_counter = counter_row(1, "k", 10, 20)
    del bad_counter["GUID"]
    bad_marker = linear_marker_row(2, 30, 60)
    del bad_marker["GUID"]
    write_ml_api_pass(tmp_path, 0, [linear_marker_row(1, 0, 25)], [bad_counter])
    write_ml_api_pass(tmp_path, 1, [bad_marker], [counter_row(2, "k", 40, 50)])

    with pytest.raises(SystemExit) as failure:
        process_ml_api_trace_output(workload, str(tmp_path))

    assert failure.value.code == 1
    assert len(messages) == 1
    assert "Marker CSV" in messages[0]
    assert "pmc_perf_1_marker" in messages[0]
    for field in (
        "ml_api_trace_pairs",
        "unmatched_kernel_frames",
        "ml_api_trace_df",
        "ml_api_call_trees",
    ):
        assert getattr(workload, field) is previous[field]


def test_later_counter_failure_keeps_earlier_raw_join(
    tmp_path,
    monkeypatch,
    populated_workload,
    record_console_error_and_exit,
    write_ml_api_pass,
    linear_marker_row,
    counter_row,
):
    workload = populated_workload
    previous = vars(workload).copy()
    messages = record_console_error_and_exit(monkeypatch)
    warnings = []
    monkeypatch.setattr(
        analysis, "console_warning", lambda *args: warnings.append(args)
    )
    bad_counter = counter_row(2, "bad", 40, 50)
    del bad_counter["GUID"]
    write_ml_api_pass(
        tmp_path, 0, [linear_marker_row(1, 0, 25)], [counter_row(99, "orphan", 10, 20)]
    )
    write_ml_api_pass(tmp_path, 1, [linear_marker_row(2, 30, 60)], [bad_counter])

    with pytest.raises(SystemExit):
        process_ml_api_trace_output(workload, str(tmp_path))

    assert "Counter CSV" in messages[0]
    assert len(messages) == 1
    assert warnings == []
    assert workload.ml_api_trace_pairs is not previous["ml_api_trace_pairs"]
    first, second = workload.ml_api_trace_pairs
    assert "Kernel_Name" in first.joined_df.columns
    assert "Kernel_Names" not in first.joined_df.columns
    assert second.joined_df is None
    for field in ("unmatched_kernel_frames", "ml_api_trace_df", "ml_api_call_trees"):
        assert getattr(workload, field) is previous[field]


def test_unmatched_extraction_failure_keeps_raw_join(
    tmp_path,
    monkeypatch,
    populated_workload,
    write_ml_api_pass,
    linear_marker_row,
    counter_row,
):
    previous_unmatched = populated_workload.unmatched_kernel_frames
    previous_trace = populated_workload.ml_api_trace_df
    previous_forest = populated_workload.ml_api_call_trees
    write_ml_api_pass(
        tmp_path, 0, [linear_marker_row(1, 0, 100)], [counter_row(1, "k", 10, 20)]
    )

    monkeypatch.setattr(analysis, "unmatched_kernel_rows", fail_unmatched)
    with pytest.raises(ValueError, match="unmatched extraction failed"):
        process_ml_api_trace_output(populated_workload, str(tmp_path))

    assert populated_workload.ml_api_trace_pairs[0].joined_df is not None
    assert populated_workload.unmatched_kernel_frames is previous_unmatched
    assert populated_workload.ml_api_trace_df is previous_trace
    assert populated_workload.ml_api_call_trees is previous_forest


def test_later_grouping_failure_keeps_earlier_grouped_frame(
    tmp_path,
    populated_workload,
    write_ml_api_pass,
    linear_marker_row,
    counter_row,
):
    previous_unmatched = populated_workload.unmatched_kernel_frames
    previous_trace = populated_workload.ml_api_trace_df
    previous_forest = populated_workload.ml_api_call_trees
    write_ml_api_pass(
        tmp_path, 0, [linear_marker_row(1, 0, 100)], [counter_row(1, "k", 10, 20)]
    )
    write_ml_api_pass(
        tmp_path,
        1,
        [linear_marker_row(2, 200, 300, thread_id="invalid")],
        [counter_row(2, "k", 210, 220)],
    )

    with pytest.raises(ValueError, match="invalid"):
        process_ml_api_trace_output(populated_workload, str(tmp_path))

    first, second = populated_workload.ml_api_trace_pairs
    assert "Kernel_Names" in first.joined_df.columns
    assert "Kernel_Name" in second.joined_df.columns
    assert "Kernel_Names" not in second.joined_df.columns
    assert populated_workload.unmatched_kernel_frames is not previous_unmatched
    assert populated_workload.ml_api_trace_df is previous_trace
    assert populated_workload.ml_api_call_trees is previous_forest


@pytest.mark.parametrize("failure_stage", ["collapse", "legacy", "parse"])
def test_failure_before_trace_assignment_preserves_previous_trace_and_forest(
    tmp_path,
    monkeypatch,
    populated_workload,
    record_console_error_and_exit,
    write_ml_api_pass,
    linear_marker_row,
    counter_row,
    failure_stage,
):
    previous_trace = populated_workload.ml_api_trace_df
    previous_forest = populated_workload.ml_api_call_trees
    messages = record_console_error_and_exit(monkeypatch)
    marker = linear_marker_row(1, 0, 100)
    if failure_stage == "legacy":
        marker["Function"] = "triton.JITFunction.foo:#1@file.py:1"
    write_ml_api_pass(tmp_path, 0, [marker], [counter_row(1, "k", 10, 20)])
    if failure_stage == "collapse":
        write_ml_api_pass(
            tmp_path,
            1,
            [linear_marker_row(2, 200, 300), linear_marker_row(3, 400, 500)],
            [
                counter_row(2, "k", 210, 220),
                counter_row(3, "k", 410, 420, dispatch_id=2),
            ],
        )
    if failure_stage == "parse":
        monkeypatch.setattr(processing, "parse_marker_function", fail_parse)
        expected_error = ValueError
    else:
        expected_error = SystemExit

    with pytest.raises(expected_error):
        process_ml_api_trace_output(populated_workload, str(tmp_path))

    assert populated_workload.ml_api_trace_df is previous_trace
    assert populated_workload.ml_api_call_trees is previous_forest
    assert all(
        "Kernel_Names" in p.joined_df for p in populated_workload.ml_api_trace_pairs
    )
    if failure_stage == "legacy":
        assert messages == [
            "Stacked marker wire is not supported: triton.JITFunction.foo:#1@file.py:1"
        ]
    elif failure_stage == "collapse":
        assert len(messages) == 1
        assert "per-pass marker counts" in messages[0]
    else:
        assert messages == []


def test_nesting_failure_assigns_trace_and_keeps_previous_forest(
    tmp_path,
    monkeypatch,
    populated_workload,
    record_console_error_and_exit,
    write_ml_api_pass,
    linear_marker_row,
    counter_row,
):
    previous_trace = populated_workload.ml_api_trace_df
    previous_forest = populated_workload.ml_api_call_trees
    messages = record_console_error_and_exit(monkeypatch)
    write_ml_api_pass(
        tmp_path,
        0,
        [linear_marker_row(1, 0, 100), linear_marker_row(2, 50, 150)],
        [counter_row(1, "k", 10, 20), counter_row(2, "k", 60, 70, dispatch_id=2)],
    )

    with pytest.raises(SystemExit):
        process_ml_api_trace_output(populated_workload, str(tmp_path))

    assert populated_workload.ml_api_trace_df is not previous_trace
    assert populated_workload.ml_api_call_trees is previous_forest
    assert len(messages) == 1
    assert "Overlapping marker ranges" in messages[0]


def test_no_pairs_resets_only_the_pair_list(tmp_path, populated_workload):
    previous = vars(populated_workload).copy()

    process_ml_api_trace_output(populated_workload, str(tmp_path))

    assert populated_workload.ml_api_trace_pairs == []
    assert populated_workload.ml_api_trace_pairs is not previous["ml_api_trace_pairs"]
    for field in ("unmatched_kernel_frames", "ml_api_trace_df", "ml_api_call_trees"):
        assert getattr(populated_workload, field) is previous[field]


def test_header_only_pair_replaces_completed_stages(
    tmp_path,
    populated_workload,
    write_ml_api_pass,
    linear_marker_row,
    counter_row,
):
    write_ml_api_pass(
        tmp_path,
        0,
        pd.DataFrame(columns=list(linear_marker_row(1, 0, 100))),
        pd.DataFrame(columns=list(counter_row(1, "k", 10, 20))),
    )
    previous = vars(populated_workload).copy()

    process_ml_api_trace_output(populated_workload, str(tmp_path))

    assert len(populated_workload.ml_api_trace_pairs) == 1
    assert populated_workload.ml_api_trace_df.empty
    assert populated_workload.ml_api_call_trees == {}
    assert len(populated_workload.unmatched_kernel_frames) == 1
    for field in ("unmatched_kernel_frames", "ml_api_trace_df", "ml_api_call_trees"):
        assert getattr(populated_workload, field) is not previous[field]


def test_unmatched_dispatch_preserves_integer_thread_ids_and_launcher_attachment(
    tmp_path,
    monkeypatch,
    write_ml_api_pass,
    linear_marker_row,
    counter_row,
):
    warnings = []
    monkeypatch.setattr(
        analysis, "console_warning", lambda *args: warnings.append(args)
    )
    worker = "aten::mm:n/a|ltid=7|args=()|torch"
    write_ml_api_pass(
        tmp_path,
        0,
        [
            linear_marker_row(1, 0, 100, thread_id=7),
            linear_marker_row(2, 20, 60, thread_id=42, function=worker),
        ],
        [
            counter_row(2, "worker_kernel", 30, 40),
            counter_row(99, "orphan", 70, 80, dispatch_id=2),
        ],
    )
    workload = schema.Workload()

    process_ml_api_trace_output(workload, str(tmp_path))

    assert str(workload.ml_api_trace_df["Thread_Id"].dtype) == "int64"
    assert list(workload.ml_api_call_trees) == ["7"]
    assert workload.ml_api_call_trees["7"][0].children[0].thread_id == "42"
    assert (
        workload.ml_api_call_trees["7"][0].children[0].kernels["worker_kernel"].launches
        == 1
    )
    assert len(warnings) == 1
    assert "unmatched dispatches" in warnings[0][1]


def test_unmatched_warning_precedes_pass_matching_fatal(
    tmp_path, caplog, write_ml_api_pass, linear_marker_row, counter_row
):
    write_ml_api_pass(
        tmp_path,
        0,
        [linear_marker_row(1, 0, 100)],
        [
            counter_row(1, "k", 10, 20),
            counter_row(99, "orphan", 30, 40, dispatch_id=2),
        ],
    )
    write_ml_api_pass(
        tmp_path,
        1,
        [linear_marker_row(2, 200, 300), linear_marker_row(3, 400, 500)],
        [
            counter_row(2, "k", 210, 220),
            counter_row(3, "k", 410, 420, dispatch_id=2),
        ],
    )

    with caplog.at_level("WARNING"), pytest.raises(SystemExit) as failure:
        process_ml_api_trace_output(schema.Workload(), str(tmp_path))

    assert failure.value.code == 1
    diagnostics = [(record.levelname, record.getMessage()) for record in caplog.records]
    assert len(diagnostics) == 2
    assert diagnostics[0] == (
        "WARNING",
        "[analysis] 1 unmatched dispatches; Kernel_Names: orphan",
    )
    assert diagnostics[1][0] == "ERROR"
    assert "per-pass marker counts [1, 2]" in diagnostics[1][1]


def test_attachment_warning_precedes_validation_fatal_and_stops_later_stages(
    tmp_path,
    monkeypatch,
    populated_workload,
    write_ml_api_pass,
    linear_marker_row,
    counter_row,
):
    events = []
    monkeypatch.setattr(analysis, "console_warning", lambda *args: events.append(args))

    def fatal_exit(*args):
        events.append(args)
        raise SystemExit(1)

    monkeypatch.setattr(analysis, "console_error", fatal_exit)
    monkeypatch.setattr(analysis, "validate_all_markers_nested", fail_validation)
    write_ml_api_pass(
        tmp_path,
        0,
        [
            linear_marker_row(
                1, 20, 60, thread_id=42, function="aten::mm:n/a|ltid=7|torch"
            )
        ],
        [counter_row(1, "k", 30, 40)],
    )
    previous_forest = populated_workload.ml_api_call_trees

    with pytest.raises(SystemExit):
        process_ml_api_trace_output(populated_workload, str(tmp_path))

    assert populated_workload.ml_api_call_trees is not previous_forest
    root = populated_workload.ml_api_call_trees["42"][0]
    assert root.kernel_launches == 0
    assert root.kernels["k"].launches == 1
    assert len(events) == 2
    assert "Launcher thread not found" in events[0][1]
    assert "Marker was not nested" in events[1][1]


def test_attachment_flushes_accumulated_warnings_on_unexpected_failure(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        analysis, "console_warning", lambda *args: warnings.append(args)
    )
    issue = LauncherThreadNotFoundError("partial", "42", 20.0, "7")

    def partial_attachment(_forest, issues):
        issues.append(issue)
        raise ValueError("attachment failed")

    monkeypatch.setattr(
        analysis.call_tree,
        "attach_unlocated_trees_by_launcher_thread",
        partial_attachment,
    )
    with pytest.raises(ValueError, match="attachment failed"):
        analysis.attach_unlocated_trees_by_launcher_thread({})
    assert warnings == [("analysis", str(issue))]

    issues = []
    warnings.clear()
    with pytest.raises(ValueError, match="attachment failed"):
        analysis.attach_unlocated_trees_by_launcher_thread({}, issues)
    assert issues == [issue]
    assert warnings == []


def test_missing_source_warning_follows_pruning_and_rollup(
    tmp_path,
    monkeypatch,
    write_ml_api_pass,
    linear_marker_row,
    counter_row,
):
    workload = schema.Workload()
    warning_states = []

    def record_warning(*args):
        roots = workload.ml_api_call_trees["1"]
        warning_states.append((
            args,
            [
                (root.name, root.kernel_launches, root.total_duration_ms)
                for root in roots
            ],
        ))

    monkeypatch.setattr(analysis, "console_warning", record_warning)
    write_ml_api_pass(
        tmp_path,
        0,
        [
            linear_marker_row(1, 0, 10, function="aten::cpu:n/a|torch"),
            linear_marker_row(2, 20, 60, function="aten::mm:n/a|torch"),
        ],
        [counter_row(2, "k", 30, 40)],
    )

    process_ml_api_trace_output(workload, str(tmp_path))

    assert len(warning_states) == 1
    warning, roots_at_warning = warning_states[0]
    assert warning == (
        "analysis",
        "Missing source location for Operator_Name=aten::mm Thread_Id=1 "
        "Start_Timestamp=20.0",
    )
    assert roots_at_warning == [("aten::mm", 1, pytest.approx(0.00001))]
    assert workload.ml_api_call_trees["1"][0].kernels["k"].launches == 1


def test_parse_marker_function_ltid_and_thread_id_unchanged(
    tmp_path, write_ml_api_pass, counter_row, test_guid
):
    parsed = parse_marker_function(
        "aten::addmm:n/a|seqNr=1|tid=1|ftid=1|ltid=7|scope=FUNCTION|args=()|torch"
    )
    assert parsed["launcher_thread_id"] == "7"
    workload_dir = tmp_path / "ltid_thread"
    function = (
        "nn.Module.Linear.forward:simple_torch_code.py:19"
        "|seqNr=n/a|tid=n/a|ftid=1|ltid=7|scope=n/a|args=()|torch"
    )
    write_ml_api_pass(
        workload_dir,
        0,
        [
            {
                "Function": function,
                "Thread_Id": 42,
                "Correlation_ID": 8,
                "GUID": test_guid,
                "Start_Timestamp": 20,
                "End_Timestamp": 70,
            }
        ],
        [counter_row(8, "addmm_kernel", 30, 60, dispatch_id=1)],
    )
    workload = schema.Workload()
    process_ml_api_trace_output(workload, str(workload_dir))
    assert workload.ml_api_trace_df["Thread_Id"].tolist() == [42]
    assert workload.ml_api_trace_df["launcher_thread_id"].tolist() == ["7"]


def test_process_outer_join_keeps_wrap_with_empty_kernel_names(
    tmp_path, linear_marker_row
):
    workload_dir = tmp_path / "empty_kernels"
    workload_dir.mkdir()
    pd.DataFrame([linear_marker_row(1, 10, 80)]).to_csv(
        csv_compression.compressed_name(
            workload_dir / "ml_api_trace_pmc_perf_0_marker_api_trace.csv"
        ),
        index=False,
        compression="gzip",
    )
    pd.DataFrame(
        columns=[
            "Correlation_ID",
            "GUID",
            "Kernel_Name",
            "Dispatch_ID",
            "Grid_Size",
            "Workgroup_Size",
            "LDS_Per_Workgroup",
            "Start_Timestamp",
            "End_Timestamp",
            "Counter_Name",
        ]
    ).to_csv(
        csv_compression.compressed_name(
            workload_dir / "ml_api_trace_pmc_perf_0_counter_collection.csv"
        ),
        index=False,
        compression="gzip",
    )
    workload = schema.Workload()
    process_ml_api_trace_output(workload, str(workload_dir))
    assert list(workload.ml_api_trace_df["Kernel_Names"].iloc[0]) == []
    assert workload.ml_api_call_trees == {}


def test_process_collapses_two_counter_rows_to_one_dispatch_duration(
    tmp_path, write_ml_api_pass, linear_marker_row, counter_row
):
    workload_dir = tmp_path / "two_counters"
    write_ml_api_pass(
        workload_dir,
        0,
        [linear_marker_row(1, 0, 200)],
        [
            counter_row(
                1,
                "addmm_kernel",
                100,
                150,
                dispatch_id=7,
                counter_name="SQ_WAVES",
            ),
            counter_row(
                1,
                "addmm_kernel",
                100,
                150,
                dispatch_id=7,
                counter_name="GRBM_GUI_ACTIVE",
            ),
        ],
    )
    workload = schema.Workload()
    process_ml_api_trace_output(workload, str(workload_dir))
    stats = workload.ml_api_call_trees["1"][0].kernels["addmm_kernel"]
    assert stats.launches == 1
    assert stats.total_duration_ns == 50


def test_process_two_dispatches_same_marker_keep_both_kernel_names(
    tmp_path, write_ml_api_pass, linear_marker_row, counter_row
):
    workload_dir = tmp_path / "two_kernels"
    write_ml_api_pass(
        workload_dir,
        0,
        [linear_marker_row(5, 0, 200)],
        [
            counter_row(5, "kernel_a", 10, 20, dispatch_id=1),
            counter_row(5, "kernel_b", 30, 40, dispatch_id=2),
        ],
    )
    workload = schema.Workload()
    process_ml_api_trace_output(workload, str(workload_dir))
    names = list(workload.ml_api_trace_df["Kernel_Names"].iloc[0])
    assert names == ["kernel_a", "kernel_b"]
    assert set(workload.ml_api_call_trees["1"][0].kernels) == {"kernel_a", "kernel_b"}


def test_process_unmatched_kernel_warns_and_continues(
    tmp_path, monkeypatch, write_ml_api_pass, linear_marker_row, counter_row
):
    warnings = []
    monkeypatch.setattr(
        analysis,
        "console_warning",
        lambda *argv: warnings.append(argv),
    )
    workload_dir = tmp_path / "unmatched"
    write_ml_api_pass(
        workload_dir,
        0,
        [linear_marker_row(1, 0, 100)],
        [counter_row(99, "orphan_kernel", 10, 20)],
    )
    workload = schema.Workload()
    process_ml_api_trace_output(workload, str(workload_dir))
    assert len(workload.unmatched_kernel_frames) == 1
    assert not workload.unmatched_kernel_frames[0].empty
    warning_text = str(warnings)
    assert "orphan_kernel" in warning_text
    assert "unmatched dispatches" in warning_text
    assert not workload.ml_api_trace_df.empty


def test_process_unmatched_dispatch_two_counter_rows_is_one_row(
    tmp_path, write_ml_api_pass, linear_marker_row, counter_row
):
    workload_dir = tmp_path / "unmatched_two_counters"
    write_ml_api_pass(
        workload_dir,
        0,
        [linear_marker_row(1, 0, 100)],
        [
            counter_row(
                99,
                "orphan_kernel",
                10,
                20,
                dispatch_id=3,
                counter_name="SQ_WAVES",
            ),
            counter_row(
                99,
                "orphan_kernel",
                10,
                20,
                dispatch_id=3,
                counter_name="GRBM_GUI_ACTIVE",
            ),
        ],
    )
    workload = schema.Workload()
    process_ml_api_trace_output(workload, str(workload_dir))
    unmatched = workload.unmatched_kernel_frames[0]
    assert len(unmatched) == 1
    assert unmatched.iloc[0]["Kernel_Name"] == "orphan_kernel"


def test_process_joins_on_guid_and_correlation_id(
    tmp_path, write_ml_api_pass, linear_marker_row, counter_row, test_guid
):
    workload_dir = tmp_path / "guid_join"
    write_ml_api_pass(
        workload_dir,
        0,
        [linear_marker_row(1, 0, 100, guid=test_guid)],
        [counter_row(1, "addmm_kernel", 10, 40, guid=test_guid)],
    )
    workload = schema.Workload()
    process_ml_api_trace_output(workload, str(workload_dir))
    assert list(workload.ml_api_trace_df["Kernel_Names"].iloc[0]) == ["addmm_kernel"]


def test_process_same_correlation_id_different_guid_does_not_join(
    tmp_path,
    write_ml_api_pass,
    linear_marker_row,
    counter_row,
    test_guid,
    test_guid_other,
):
    workload_dir = tmp_path / "guid_mismatch"
    write_ml_api_pass(
        workload_dir,
        0,
        [linear_marker_row(1, 0, 100, guid=test_guid)],
        [counter_row(1, "addmm_kernel", 10, 40, guid=test_guid_other)],
    )
    workload = schema.Workload()
    process_ml_api_trace_output(workload, str(workload_dir))
    assert list(workload.ml_api_trace_df["Kernel_Names"].iloc[0]) == []
    unmatched = workload.unmatched_kernel_frames[0]
    assert unmatched.iloc[0]["Kernel_Name"] == "addmm_kernel"


def test_process_one_pass_builds_ml_api_trace_df(
    tmp_path, write_ml_api_pass, linear_marker_row, counter_row
):
    workload_dir = tmp_path / "one_pass"
    write_ml_api_pass(
        workload_dir,
        0,
        [linear_marker_row(1, 0, 100)],
        [counter_row(1, "addmm_kernel", 10, 40)],
    )
    workload = schema.Workload()
    process_ml_api_trace_output(workload, str(workload_dir))
    assert len(workload.ml_api_trace_df) == 1
    assert workload.ml_api_call_trees["1"][0].kernel_launches == 1


@pytest.mark.parametrize("csv_kind", ["marker", "counter"])
def test_process_exits_when_guid_column_is_missing(
    tmp_path,
    monkeypatch,
    csv_kind,
    record_console_error_and_exit,
    write_ml_api_pass,
    linear_marker_row,
    counter_row,
):
    messages = record_console_error_and_exit(monkeypatch)
    marker = linear_marker_row(1, 0, 100)
    counter = counter_row(1, "addmm_kernel", 10, 40)
    row_without_guid = marker if csv_kind == "marker" else counter
    del row_without_guid["GUID"]
    write_ml_api_pass(tmp_path, 0, [marker], [counter])
    with pytest.raises(SystemExit):
        process_ml_api_trace_output(schema.Workload(), str(tmp_path))
    assert "missing required columns ['GUID']" in messages[-1]


def test_process_two_passes_collapse_to_one_launch(
    tmp_path, write_ml_api_pass, linear_marker_row, counter_row
):
    workload_dir = tmp_path / "two_pass_collapse"
    write_ml_api_pass(
        workload_dir,
        0,
        [linear_marker_row(11, 0, 100)],
        [counter_row(11, "addmm_kernel", 10, 40, dispatch_id=1)],
    )
    write_ml_api_pass(
        workload_dir,
        1,
        [linear_marker_row(22, 1000, 1100)],
        [counter_row(22, "addmm_kernel", 1010, 1040, dispatch_id=9)],
    )
    workload = schema.Workload()
    process_ml_api_trace_output(workload, str(workload_dir))
    assert len(workload.ml_api_trace_df) == 1
    assert workload.ml_api_call_trees["1"][0].kernel_launches == 1


def test_process_collapse_keeps_pass0_gpu_time(
    tmp_path, write_ml_api_pass, linear_marker_row, counter_row
):
    workload_dir = tmp_path / "pass0_gpu_time"
    write_ml_api_pass(
        workload_dir,
        0,
        [linear_marker_row(11, 0, 100)],
        [counter_row(11, "addmm_kernel", 10, 40, dispatch_id=1)],
    )
    write_ml_api_pass(
        workload_dir,
        1,
        [linear_marker_row(22, 1000, 1100)],
        [counter_row(22, "addmm_kernel", 1010, 1100, dispatch_id=9)],
    )
    workload = schema.Workload()
    process_ml_api_trace_output(workload, str(workload_dir))
    stats = workload.ml_api_call_trees["1"][0].kernels["addmm_kernel"]
    assert stats.total_duration_ns == 30


def test_process_two_wraps_same_stitch_key_keep_ordinals(
    tmp_path, write_ml_api_pass, linear_marker_row, counter_row
):
    workload_dir = tmp_path / "two_ordinals"
    first = linear_marker_row(1, 0, 50)
    second = linear_marker_row(2, 60, 110)
    write_ml_api_pass(
        workload_dir,
        0,
        [first, second],
        [
            counter_row(1, "kernel_a", 10, 20, dispatch_id=1),
            counter_row(2, "kernel_b", 70, 80, dispatch_id=2),
        ],
    )
    write_ml_api_pass(
        workload_dir,
        1,
        [linear_marker_row(8, 200, 250), linear_marker_row(9, 260, 310)],
        [
            counter_row(8, "kernel_a", 210, 220, dispatch_id=1),
            counter_row(9, "kernel_b", 270, 280, dispatch_id=2),
        ],
    )
    workload = schema.Workload()
    process_ml_api_trace_output(workload, str(workload_dir))
    assert len(workload.ml_api_trace_df) == 2
    roots = workload.ml_api_call_trees["1"]
    assert len(roots) == 2
    assert [root.call_count for root in roots] == [1, 1]


def test_process_pass_null_kernel_name_mismatch_exits(
    tmp_path,
    monkeypatch,
    record_console_error_and_exit,
    write_ml_api_pass,
    linear_marker_row,
    counter_row,
):
    messages = record_console_error_and_exit(monkeypatch)
    workload_dir = tmp_path / "kernel_mismatch"
    write_ml_api_pass(
        workload_dir,
        0,
        [linear_marker_row(1, 0, 100)],
        [counter_row(1, "kernel_a", 10, 40)],
    )
    workload_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([linear_marker_row(1, 0, 100)]).to_csv(
        csv_compression.compressed_name(
            workload_dir / "ml_api_trace_pmc_perf_1_marker_api_trace.csv"
        ),
        index=False,
        compression="gzip",
    )
    pd.DataFrame(
        columns=[
            "Correlation_ID",
            "GUID",
            "Kernel_Name",
            "Dispatch_ID",
            "Grid_Size",
            "Workgroup_Size",
            "LDS_Per_Workgroup",
            "Start_Timestamp",
            "End_Timestamp",
            "Counter_Name",
        ]
    ).to_csv(
        csv_compression.compressed_name(
            workload_dir / "ml_api_trace_pmc_perf_1_counter_collection.csv"
        ),
        index=False,
        compression="gzip",
    )
    workload = schema.Workload()
    with pytest.raises(SystemExit) as excinfo:
        process_ml_api_trace_output(workload, str(workload_dir))
    assert excinfo.value.code == 1
    assert "Pass marker mismatch" in messages[0]
    assert "Kernel_Names" in messages[0]


def test_process_pass_marker_count_mismatch_exits(
    tmp_path,
    monkeypatch,
    record_console_error_and_exit,
    write_ml_api_pass,
    linear_marker_row,
    counter_row,
):
    messages = record_console_error_and_exit(monkeypatch)
    workload_dir = tmp_path / "count_mismatch"
    write_ml_api_pass(
        workload_dir,
        0,
        [linear_marker_row(1, 0, 50), linear_marker_row(2, 60, 110)],
        [
            counter_row(1, "kernel_a", 10, 20, dispatch_id=1),
            counter_row(2, "kernel_b", 70, 80, dispatch_id=2),
        ],
    )
    write_ml_api_pass(
        workload_dir,
        1,
        [linear_marker_row(1, 0, 50)],
        [counter_row(1, "kernel_a", 10, 20)],
    )
    workload = schema.Workload()
    with pytest.raises(SystemExit) as excinfo:
        process_ml_api_trace_output(workload, str(workload_dir))
    assert excinfo.value.code == 1
    assert "per-pass marker counts" in messages[0]
