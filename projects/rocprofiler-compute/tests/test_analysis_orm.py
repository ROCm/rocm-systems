# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for analysis_orm.py static methods."""

import json

import numpy as np
import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from utils.analysis_orm import (
    CodeObjectStore,
    Database,
    Dispatch,
    InstructionLine,
    Kernel,
    PCSampleStallReason,
    PCSampleStallReasonLookup,
    PCSampleState,
    Workload,
)

PC_SAMPLING_VIEW_COLUMNS = [
    "workload_id",
    "kernel_uuid",
    "kernel_name",
    "offset",
    "instruction",
    "source",
    "count",
    "count_issue",
    "count_stall",
    "stall_reason",
]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"k": float("nan")}, {"k": None}),
        ({"k": float("inf")}, {"k": None}),
        ({"k": float("-inf")}, {"k": None}),
        ({"k": np.float64("nan")}, {"k": None}),
        (
            {"i": 1, "f": 2.5, "s": "text", "n": None},
            {"i": 1, "f": 2.5, "s": "text", "n": None},
        ),
        ({"a": [{"b": float("nan")}]}, {"a": [{"b": None}]}),
        ({"a": ({"b": float("inf")},)}, {"a": [{"b": None}]}),
    ],
    ids=[
        "nan",
        "inf",
        "neg_inf",
        "numpy_nan",
        "valid_passthrough",
        "nested_list",
        "nested_tuple",
    ],
)
def test_json_sanitize(value, expected):
    assert Database._json_sanitize(value) == expected


def add_kernel_with_durations(
    session, workload: Workload, name: str, durations: list[int]
) -> Kernel:
    """Add a kernel to *workload* with one dispatch per entry in *durations*."""
    kernel = Kernel(kernel_name=name, workload=workload)
    session.add(kernel)
    for dispatch_id, duration in enumerate(durations):
        session.add(
            Dispatch(
                dispatch_id=dispatch_id,
                pid=1,
                gpu_id=0,
                start_timestamp=0,
                end_timestamp=duration,
                kernel=kernel,
            )
        )
    return kernel


def add_pc_sampling_state(
    session: Session,
    *,
    workload: Workload,
    kernel: Kernel,
    pid: int,
    offset: int | None = 0x10,
    instruction: str | None = "v_mov",
    source: str | None = "/s/a.cpp:1",
    total_count: int = 3,
    issue_count: int | None = 1,
    stall_count: int | None = 2,
    stall_reasons: dict[str, int] | None = None,
    code_object_id: int = 5,
) -> PCSampleState:
    code_object = CodeObjectStore(
        pid=pid,
        code_object_id=code_object_id,
        load_base=0x1000,
        workload=workload,
    )
    instruction_line = InstructionLine(
        code_object_offset=offset,
        comment=source,
        instruction=instruction,
        code_object_store=code_object,
        kernel=kernel,
    )
    sample_state = PCSampleState(
        total_count=total_count,
        issue_count=issue_count,
        stall_count=stall_count,
        instruction_line=instruction_line,
    )
    session.add(sample_state)
    for reason_text, reason_count in (stall_reasons or {}).items():
        reason_lookup = Database.get_or_create_type(
            PCSampleStallReasonLookup, reason_text
        )
        session.add(
            PCSampleStallReason(
                pc_sample_state=sample_state,
                stall_reason_lookup=reason_lookup,
                count=reason_count,
            )
        )
    return sample_state


def fetch_pc_sampling_rows(session: Session) -> list[dict[str, object]]:
    selected_columns = ", ".join(PC_SAMPLING_VIEW_COLUMNS)
    rows = session.execute(
        text(
            f"SELECT {selected_columns} FROM compute_pc_sampling_view "
            "ORDER BY workload_id, kernel_uuid, offset, instruction, source"
        )
    ).mappings()
    decoded_rows = []
    for row in rows:
        decoded_row = dict(row)
        stall_reason = decoded_row["stall_reason"]
        if isinstance(stall_reason, str):
            decoded_row["stall_reason"] = json.loads(stall_reason)
        decoded_rows.append(decoded_row)
    return decoded_rows


# =============================================================================
# kernel view: median duration algorithm
# =============================================================================


@pytest.mark.parametrize(
    ("durations", "expected_median"),
    [
        ([30, 10, 20], 20.0),  # odd: middle value
        ([40, 10, 30, 20], 25.0),  # even: mean of two middle values
        ([42], 42.0),  # single dispatch
    ],
    ids=["odd", "even", "single"],
)
def test_kernel_view_median(db_session, durations, expected_median):
    """The kernel view computes median duration for odd/even/single counts."""
    workload = Workload(name="w", sub_name="s")
    db_session.add(workload)
    add_kernel_with_durations(db_session, workload, "k", durations)
    Database.create_views()
    db_session.commit()

    row = db_session.execute(
        text("SELECT duration_ns_median FROM compute_kernel_view")
    ).fetchone()
    assert row[0] == expected_median


def test_kernel_view_aggregates(db_session):
    """The kernel view reports count/sum/min/max/mean over dispatch durations."""
    workload = Workload(name="w", sub_name="s")
    db_session.add(workload)
    add_kernel_with_durations(db_session, workload, "k", [10, 20, 30])
    Database.create_views()
    db_session.commit()

    row = db_session.execute(
        text(
            "SELECT dispatch_count, duration_ns_sum, duration_ns_min, "
            "duration_ns_max, duration_ns_mean FROM compute_kernel_view"
        )
    ).fetchone()
    assert row == (3, 60, 10, 30, 20.0)


# =============================================================================
# unique constraints
# =============================================================================


def test_duplicate_kernel_rejected(db_session):
    """A second kernel with the same (workload_id, kernel_name) is rejected."""
    workload = Workload(name="w", sub_name="s")
    db_session.add(workload)
    db_session.add(Kernel(kernel_name="k", workload=workload))
    db_session.add(Kernel(kernel_name="k", workload=workload))
    with pytest.raises(IntegrityError):
        db_session.commit()


def test_duplicate_dispatch_rejected(db_session):
    """A duplicate (kernel_uuid, pid, dispatch_id) is rejected."""
    workload = Workload(name="w", sub_name="s")
    db_session.add(workload)
    kernel = Kernel(kernel_name="k", workload=workload)
    db_session.add(kernel)
    db_session.add(Dispatch(dispatch_id=0, pid=101, kernel=kernel))
    db_session.add(Dispatch(dispatch_id=0, pid=101, kernel=kernel))
    with pytest.raises(IntegrityError):
        db_session.commit()


def test_dispatch_id_can_repeat_across_processes(db_session):
    """Process-local dispatch IDs remain distinct and aggregate together."""
    workload = Workload(name="w", sub_name="s")
    kernel = Kernel(kernel_name="k", workload=workload)
    db_session.add_all([
        Dispatch(
            dispatch_id=0,
            pid=101,
            start_timestamp=10,
            end_timestamp=20,
            kernel=kernel,
        ),
        Dispatch(
            dispatch_id=0,
            pid=202,
            start_timestamp=30,
            end_timestamp=50,
            kernel=kernel,
        ),
    ])
    Database.create_views()
    db_session.commit()

    row = db_session.execute(
        text("SELECT dispatch_count, duration_ns_sum FROM compute_kernel_view")
    ).one()
    assert row == (2, 30)


def test_duplicate_stall_reason_lookup_rejected(db_session):
    """A second stall-reason lookup with the same text is rejected."""
    db_session.add(PCSampleStallReasonLookup(text="WAITCNT"))
    db_session.add(PCSampleStallReasonLookup(text="WAITCNT"))
    with pytest.raises(IntegrityError):
        db_session.commit()


# =============================================================================
# get_view_sql
# =============================================================================


def test_get_view_sql_returns_copy(db_session):
    """Mutating the returned dict does not poison the cached view SQL."""
    view_sql = Database.get_view_sql()
    view_sql.clear()
    assert Database.get_view_sql()  # cache still populated


# =============================================================================
# get_or_create_type
# =============================================================================


def test_get_or_create_type_dedups(db_session):
    """The same text returns one cached row; a new text creates another."""
    first = Database.get_or_create_type(PCSampleStallReasonLookup, "WAITCNT")
    again = Database.get_or_create_type(PCSampleStallReasonLookup, "WAITCNT")
    other = Database.get_or_create_type(PCSampleStallReasonLookup, "BARRIER_WAIT")
    db_session.commit()

    assert first is again
    assert other is not first
    assert db_session.query(PCSampleStallReasonLookup).count() == 2


# =============================================================================
# pc_sampling view
# =============================================================================


def test_pc_sampling_view_flattens_normalized_tables(db_session):
    """The pc_sampling view flattens the normalized tables and rebuilds
    stall_reason as a JSON dict."""
    workload = Workload(name="w", sub_name="s")
    kernel = Kernel(kernel_name="vecCopy", workload=workload)
    add_pc_sampling_state(
        db_session,
        workload=workload,
        kernel=kernel,
        pid=1,
        stall_reasons={"WAITCNT": 2},
    )
    Database.create_views()
    db_session.commit()

    assert fetch_pc_sampling_rows(db_session) == [
        {
            "workload_id": workload.workload_id,
            "kernel_uuid": kernel.kernel_uuid,
            "kernel_name": "vecCopy",
            "offset": 0x10,
            "instruction": "v_mov",
            "source": "/s/a.cpp:1",
            "count": 3,
            "count_issue": 1,
            "count_stall": 2,
            "stall_reason": {"WAITCNT": 2},
        }
    ]


def test_pc_sampling_view_aggregates_matching_states_across_processes(db_session):
    workload = Workload(name="w", sub_name="s")
    kernel = Kernel(kernel_name="vecCopy", workload=workload)
    add_pc_sampling_state(
        db_session,
        workload=workload,
        kernel=kernel,
        pid=101,
        total_count=8,
        issue_count=2,
        stall_count=6,
        stall_reasons={"WAITCNT": 2, "MEMORY": 4},
    )
    add_pc_sampling_state(
        db_session,
        workload=workload,
        kernel=kernel,
        pid=202,
        total_count=9,
        issue_count=3,
        stall_count=6,
        stall_reasons={"WAITCNT": 5, "BARRIER": 1},
    )
    Database.create_views()
    db_session.commit()

    rows = fetch_pc_sampling_rows(db_session)

    assert len(rows) == 1
    assert rows[0]["count"] == 17
    assert rows[0]["count_issue"] == 5
    assert rows[0]["count_stall"] == 12
    assert rows[0]["stall_reason"] == {
        "BARRIER": 1,
        "MEMORY": 4,
        "WAITCNT": 7,
    }


@pytest.mark.parametrize(
    ("identity_field", "first_value", "second_value"),
    [
        pytest.param("instruction", "v_mov", "v_add", id="instruction"),
        pytest.param(
            "source",
            "/s/a.cpp:1",
            "/s/b.cpp:1",
            id="source",
        ),
    ],
)
def test_pc_sampling_view_keeps_display_identity_fields_separate(
    db_session,
    identity_field: str,
    first_value: str,
    second_value: str,
):
    workload = Workload(name="w", sub_name="s")
    kernel = Kernel(kernel_name="vecCopy", workload=workload)
    add_pc_sampling_state(
        db_session,
        workload=workload,
        kernel=kernel,
        pid=101,
        total_count=3,
        issue_count=3,
        stall_count=0,
        **{identity_field: first_value},
    )
    add_pc_sampling_state(
        db_session,
        workload=workload,
        kernel=kernel,
        pid=202,
        total_count=5,
        issue_count=5,
        stall_count=0,
        **{identity_field: second_value},
    )
    Database.create_views()
    db_session.commit()

    rows = fetch_pc_sampling_rows(db_session)

    assert {(row[identity_field], row["count"]) for row in rows} == {
        (first_value, 3),
        (second_value, 5),
    }


def test_pc_sampling_view_keeps_different_kernels_separate(db_session):
    workload = Workload(name="w", sub_name="s")
    first_kernel = Kernel(kernel_name="first", workload=workload)
    second_kernel = Kernel(kernel_name="second", workload=workload)
    add_pc_sampling_state(
        db_session,
        workload=workload,
        kernel=first_kernel,
        pid=101,
        total_count=3,
        issue_count=3,
        stall_count=0,
    )
    add_pc_sampling_state(
        db_session,
        workload=workload,
        kernel=second_kernel,
        pid=202,
        total_count=5,
        issue_count=5,
        stall_count=0,
    )
    Database.create_views()
    db_session.commit()

    rows = fetch_pc_sampling_rows(db_session)

    assert {(row["kernel_name"], row["count"]) for row in rows} == {
        ("first", 3),
        ("second", 5),
    }


def test_pc_sampling_view_keeps_different_workloads_separate(db_session):
    first_workload = Workload(name="first", sub_name="s")
    second_workload = Workload(name="second", sub_name="s")
    first_kernel = Kernel(kernel_name="vecCopy", workload=first_workload)
    second_kernel = Kernel(kernel_name="vecCopy", workload=second_workload)
    add_pc_sampling_state(
        db_session,
        workload=first_workload,
        kernel=first_kernel,
        pid=101,
        total_count=3,
        issue_count=3,
        stall_count=0,
    )
    add_pc_sampling_state(
        db_session,
        workload=second_workload,
        kernel=second_kernel,
        pid=202,
        total_count=5,
        issue_count=5,
        stall_count=0,
    )
    Database.create_views()
    db_session.commit()

    rows = fetch_pc_sampling_rows(db_session)

    assert {(row["workload_id"], row["count"]) for row in rows} == {
        (first_workload.workload_id, 3),
        (second_workload.workload_id, 5),
    }


def test_pc_sampling_view_aggregates_host_trap_states_with_null_counts(db_session):
    workload = Workload(name="w", sub_name="s")
    kernel = Kernel(kernel_name="vecCopy", workload=workload)
    add_pc_sampling_state(
        db_session,
        workload=workload,
        kernel=kernel,
        pid=101,
        total_count=3,
        issue_count=None,
        stall_count=None,
    )
    add_pc_sampling_state(
        db_session,
        workload=workload,
        kernel=kernel,
        pid=202,
        total_count=5,
        issue_count=None,
        stall_count=None,
    )
    Database.create_views()
    db_session.commit()

    rows = fetch_pc_sampling_rows(db_session)

    assert len(rows) == 1
    assert rows[0]["count"] == 8
    assert rows[0]["count_issue"] is None
    assert rows[0]["count_stall"] is None
    assert rows[0]["stall_reason"] is None


@pytest.mark.parametrize("nullable_field", ["offset", "instruction", "source"])
def test_pc_sampling_view_attaches_reasons_with_nullable_identity(
    db_session, nullable_field
):
    workload = Workload(name="w", sub_name="s")
    kernel = Kernel(kernel_name="vecCopy", workload=workload)
    identity = {
        "offset": 0x10,
        "instruction": "v_mov",
        "source": "/s/a.cpp:1",
    }
    identity[nullable_field] = None
    add_pc_sampling_state(
        db_session,
        workload=workload,
        kernel=kernel,
        pid=101,
        total_count=3,
        issue_count=1,
        stall_count=2,
        stall_reasons={"WAITCNT": 2},
        **identity,
    )
    add_pc_sampling_state(
        db_session,
        workload=workload,
        kernel=kernel,
        pid=202,
        total_count=5,
        issue_count=1,
        stall_count=4,
        stall_reasons={"WAITCNT": 4},
        **identity,
    )
    Database.create_views()
    db_session.commit()

    rows = fetch_pc_sampling_rows(db_session)

    assert len(rows) == 1
    assert rows[0][nullable_field] is None
    assert rows[0]["count"] == 8
    assert rows[0]["stall_reason"] == {"WAITCNT": 6}
