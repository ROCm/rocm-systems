# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Workload state and diagnostics for ML API trace analysis."""

import pandas as pd
import pytest

from ml_api_tracing.analyze import orchestrator
from ml_api_tracing.analyze.errors import MarkerNotNestedError
from ml_api_tracing.analyze.orchestrator import process_ml_api_trace_output
from ml_api_tracing.marker_format import parse_marker_function
from utils import csv_compression, schema


@pytest.fixture
def record_console_error_and_exit():
    return build_record_console_error_and_exit


class ConsoleErrorRecorder:
    """Stand in for console_error: keep the message and exit like the real one."""

    def __init__(self):
        self.messages = []

    def __call__(self, *argv, exit=True, exit_code=1):
        self.messages.append(str(argv[-1]) if argv else "")
        if exit:
            raise SystemExit(exit_code)


class EventRecorder:
    """Record every console call in order, exiting when asked to."""

    def __init__(self, events, exit_code=None):
        self.events = events
        self.exit_code = exit_code

    def __call__(self, *args):
        self.events.append(args)
        if self.exit_code is not None:
            raise SystemExit(self.exit_code)


class WorkloadStateRecorder:
    """Record each warning with the workload's root nodes at that moment."""

    def __init__(self, workload):
        self.workload = workload
        self.states = []

    def __call__(self, *args):
        roots = self.workload.ml_api_call_trees["1"]
        self.states.append((
            args,
            [(r.name, r.kernel_launches, r.total_duration_ms) for r in roots],
        ))


def build_record_console_error_and_exit(monkeypatch):
    """Replace console_error so tests see the message and SystemExit."""
    recorder = ConsoleErrorRecorder()
    monkeypatch.setattr(orchestrator, "console_error", recorder)
    return recorder.messages


def fail_validation(_trace, _forest):
    raise MarkerNotNestedError("missing", "42", 20.0)


def test_unmatched_dispatch_preserves_integer_thread_ids_and_launcher_attachment(
    tmp_path,
    monkeypatch,
    write_ml_api_pass,
    linear_marker_row,
    counter_row,
):
    warnings = []
    monkeypatch.setattr(orchestrator, "console_warning", EventRecorder(warnings))
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
    tmp_path, monkeypatch, write_ml_api_pass, linear_marker_row, counter_row
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

    events = []
    monkeypatch.setattr(orchestrator, "console_warning", EventRecorder(events))
    monkeypatch.setattr(orchestrator, "console_error", EventRecorder(events, 1))

    with pytest.raises(SystemExit) as failure:
        process_ml_api_trace_output(schema.Workload(), str(tmp_path))

    assert failure.value.code == 1
    assert len(events) == 2
    assert events[0] == (
        "analysis",
        "1 unmatched dispatches; Kernel_Names: orphan",
    )
    assert "per-pass marker counts [1, 2]" in events[1][1]


def test_attachment_warning_precedes_validation_fatal(
    tmp_path,
    monkeypatch,
    write_ml_api_pass,
    linear_marker_row,
    counter_row,
):
    events = []
    monkeypatch.setattr(orchestrator, "console_warning", EventRecorder(events))

    monkeypatch.setattr(orchestrator, "console_error", EventRecorder(events, 1))
    monkeypatch.setattr(orchestrator, "validate_all_markers_nested", fail_validation)
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
    with pytest.raises(SystemExit):
        process_ml_api_trace_output(schema.Workload(), str(tmp_path))

    assert len(events) == 2
    assert "Launcher thread not found" in events[0][1]
    assert "Marker was not nested" in events[1][1]


def test_missing_source_warning_follows_pruning_and_rollup(
    tmp_path,
    monkeypatch,
    write_ml_api_pass,
    linear_marker_row,
    counter_row,
):
    workload = schema.Workload()
    recorder = WorkloadStateRecorder(workload)
    monkeypatch.setattr(orchestrator, "console_warning", recorder)
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

    assert len(recorder.states) == 1
    warning, roots_at_warning = recorder.states[0]
    assert warning == (
        "analysis",
        "Missing source location for Operator_Name=aten::mm Thread_Id=1 "
        "Start_Timestamp=20.0",
    )
    assert roots_at_warning == [("aten::mm", 1, pytest.approx(0.00001))]
    assert workload.ml_api_call_trees["1"][0].kernels["k"].launches == 1


def test_parse_marker_function_ltid_and_thread_id_unchanged(
    tmp_path, write_ml_api_pass, counter_row, guid
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
                "GUID": guid,
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
    monkeypatch.setattr(orchestrator, "console_warning", EventRecorder(warnings))
    workload_dir = tmp_path / "unmatched"
    write_ml_api_pass(
        workload_dir,
        0,
        [linear_marker_row(1, 0, 100)],
        [counter_row(99, "orphan_kernel", 10, 20)],
    )
    workload = schema.Workload()
    process_ml_api_trace_output(workload, str(workload_dir))
    warning_text = str(warnings)
    assert "orphan_kernel" in warning_text
    assert "unmatched dispatches" in warning_text
    assert not workload.ml_api_trace_df.empty


def test_process_unmatched_dispatch_two_counter_rows_is_one_row(
    tmp_path, monkeypatch, write_ml_api_pass, linear_marker_row, counter_row
):
    warnings = []
    monkeypatch.setattr(orchestrator, "console_warning", EventRecorder(warnings))
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
    assert warnings == [
        ("analysis", "1 unmatched dispatches; Kernel_Names: orphan_kernel")
    ]


def test_process_joins_on_guid_and_correlation_id(
    tmp_path, write_ml_api_pass, linear_marker_row, counter_row, guid
):
    workload_dir = tmp_path / "guid_join"
    write_ml_api_pass(
        workload_dir,
        0,
        [linear_marker_row(1, 0, 100, guid=guid)],
        [counter_row(1, "addmm_kernel", 10, 40, guid=guid)],
    )
    workload = schema.Workload()
    process_ml_api_trace_output(workload, str(workload_dir))
    assert list(workload.ml_api_trace_df["Kernel_Names"].iloc[0]) == ["addmm_kernel"]


def test_process_same_correlation_id_different_guid_does_not_join(
    tmp_path,
    monkeypatch,
    write_ml_api_pass,
    linear_marker_row,
    counter_row,
    guid,
    other_guid,
):
    warnings = []
    monkeypatch.setattr(orchestrator, "console_warning", EventRecorder(warnings))
    workload_dir = tmp_path / "guid_mismatch"
    write_ml_api_pass(
        workload_dir,
        0,
        [linear_marker_row(1, 0, 100, guid=guid)],
        [counter_row(1, "addmm_kernel", 10, 40, guid=other_guid)],
    )
    workload = schema.Workload()
    process_ml_api_trace_output(workload, str(workload_dir))
    assert list(workload.ml_api_trace_df["Kernel_Names"].iloc[0]) == []
    assert "addmm_kernel" in warnings[0][1]


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
