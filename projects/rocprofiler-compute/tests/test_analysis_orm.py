# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for analysis_orm.py static methods."""

import numpy as np
import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

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
                gpu_id=0,
                start_timestamp=0,
                end_timestamp=duration,
                kernel=kernel,
            )
        )
    return kernel


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
    """A second dispatch with the same (kernel_uuid, dispatch_id) is rejected."""
    workload = Workload(name="w", sub_name="s")
    db_session.add(workload)
    kernel = Kernel(kernel_name="k", workload=workload)
    db_session.add(kernel)
    db_session.add(Dispatch(dispatch_id=0, kernel=kernel))
    db_session.add(Dispatch(dispatch_id=0, kernel=kernel))
    with pytest.raises(IntegrityError):
        db_session.commit()


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


def test_get_or_create_type_dedups_across_calls(db_session):
    """Reusing a text after a commit (e.g. a second workload) creates no
    duplicate row, respecting the unique constraint."""
    Database.get_or_create_type(PCSampleStallReasonLookup, "WAITCNT")
    db_session.commit()
    Database.get_or_create_type(PCSampleStallReasonLookup, "WAITCNT")
    db_session.commit()

    assert db_session.query(PCSampleStallReasonLookup).count() == 1


# =============================================================================
# pc_sampling view
# =============================================================================


def test_pc_sampling_view_flattens_normalized_tables(db_session):
    """The pc_sampling view flattens the normalized tables and rebuilds
    stall_reason as a JSON dict."""
    workload = Workload(name="w", sub_name="s")
    db_session.add(workload)
    kernel = Kernel(kernel_name="vecCopy", workload=workload)
    db_session.add(kernel)
    code_object = CodeObjectStore(
        pid=1, code_object_id=5, load_base=0x1000, workload=workload
    )
    db_session.add(code_object)
    line = InstructionLine(
        code_object_offset=0x10,
        comment="/s/a.cpp:1",
        instruction="v_mov",
        code_object_store=code_object,
        kernel=kernel,
    )
    db_session.add(line)
    state = PCSampleState(
        total_count=3, issue_count=1, stall_count=2, instruction_line=line
    )
    db_session.add(state)
    reason_lookup = PCSampleStallReasonLookup(text="WAITCNT")
    db_session.add(reason_lookup)
    db_session.add(
        PCSampleStallReason(
            pc_sample_state=state, stall_reason_lookup=reason_lookup, count=2
        )
    )
    Database.create_views()
    db_session.commit()

    row = db_session.execute(
        text(
            "SELECT kernel_name, offset, instruction, source, count, "
            "count_issue, count_stall, stall_reason FROM compute_pc_sampling_view"
        )
    ).fetchone()
    assert row == (
        "vecCopy",
        0x10,
        "v_mov",
        "/s/a.cpp:1",
        3,
        1,
        2,
        '{"WAITCNT":2}',
    )
