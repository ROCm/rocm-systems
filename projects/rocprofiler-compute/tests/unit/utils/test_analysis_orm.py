# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for analysis_orm.py static methods."""

import json

import numpy as np
import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from pc_sampling.source_snapshot_analysis import parse_source_frames
from utils.analysis_orm import (
    CodeObjectStore,
    Database,
    Dispatch,
    InstructionLine,
    InstructionSourceLine,
    Kernel,
    KernelSymbol,
    PCSampleStallReason,
    PCSampleStallReasonLookup,
    PCSampleState,
    SourceFile,
    SourceLine,
    Workload,
)

PC_SAMPLING_SUMMARY_VIEW_COLUMNS = [
    "workload_id",
    "pid",
    "code_object_id",
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
                gpu_id=0,
                start_timestamp=0,
                end_timestamp=duration,
                kernel=kernel,
            )
        )
    return kernel


def add_source_frames(
    session: Session,
    workload: Workload,
    instruction_line: InstructionLine,
    source: str | None,
) -> None:
    """Attach an instruction's inline stack, reusing rows already created."""
    for frame_index, (file_path, line_number) in enumerate(parse_source_frames(source)):
        source_file = next(
            (
                candidate
                for candidate in workload.source_files
                if candidate.file_path == file_path
            ),
            None,
        )
        if source_file is None:
            source_file = SourceFile(workload=workload, file_path=file_path)
            session.add(source_file)

        source_line = next(
            (
                candidate
                for candidate in source_file.source_lines
                if candidate.line_number == line_number
            ),
            None,
        )
        if source_line is None:
            source_line = SourceLine(source_file=source_file, line_number=line_number)
            session.add(source_line)

        session.add(
            InstructionSourceLine(
                instruction_line=instruction_line,
                source_line=source_line,
                frame_index=frame_index,
            )
        )


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
    """Insert one sampled instruction state with optional stall-reason children."""
    code_object = CodeObjectStore(
        code_object_id=code_object_id,
        pid=pid,
        load_base=0x1000,
        workload=workload,
    )
    instruction_line = InstructionLine(
        code_object_offset=offset,
        instruction=instruction,
        kernel_symbol=KernelSymbol(code_object_store=code_object, kernel=kernel),
    )
    add_source_frames(session, workload, instruction_line, source)
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


def fetch_pc_sampling_summary_rows(session: Session) -> list[dict[str, object]]:
    """Fetch sampling summary rows with decoded stall-reason JSON."""
    selected_columns = ", ".join(PC_SAMPLING_SUMMARY_VIEW_COLUMNS)
    rows = session.execute(
        text(
            f"SELECT {selected_columns} FROM compute_pc_sampling_summary_view "
            "ORDER BY workload_id, pid, code_object_id, kernel_uuid, offset, "
            "instruction, source"
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


def test_duplicate_dispatch_id_under_same_kernel_rejected(db_session):
    """Reject duplicate dispatch identities within one kernel."""
    workload = Workload(name="w", sub_name="s")
    db_session.add(workload)
    kernel = Kernel(kernel_name="k", workload=workload)
    db_session.add(kernel)
    db_session.add(Dispatch(dispatch_id=0, kernel=kernel))
    db_session.add(Dispatch(dispatch_id=0, kernel=kernel))
    with pytest.raises(IntegrityError):
        db_session.commit()


def test_duplicate_instruction_identity_under_same_symbol_rejected(db_session):
    """Reject two instruction lines at one offset within a single symbol."""
    workload = Workload(name="w", sub_name="s")
    kernel = Kernel(kernel_name="k", workload=workload)
    code_object = CodeObjectStore(
        code_object_id=5,
        pid=42,
        load_base=0x1000,
        workload=workload,
    )
    kernel_symbol = KernelSymbol(code_object_store=code_object, kernel=kernel)
    db_session.add(Dispatch(dispatch_id=0, kernel=kernel))
    db_session.add_all([
        InstructionLine(
            code_object_offset=0x10,
            instruction="v_mov",
            kernel_symbol=kernel_symbol,
        ),
        InstructionLine(
            code_object_offset=0x10,
            instruction="v_mov",
            kernel_symbol=kernel_symbol,
        ),
    ])

    with pytest.raises(IntegrityError):
        db_session.commit()


def test_duplicate_kernel_symbol_under_same_parents_rejected(db_session):
    """Reject two symbols for one kernel within a single code object."""
    workload = Workload(name="w", sub_name="s")
    kernel = Kernel(kernel_name="k", workload=workload)
    code_object = CodeObjectStore(
        code_object_id=5,
        pid=42,
        load_base=0x1000,
        workload=workload,
    )
    db_session.add_all([
        KernelSymbol(code_object_store=code_object, kernel=kernel),
        KernelSymbol(code_object_store=code_object, kernel=kernel),
    ])

    with pytest.raises(IntegrityError):
        db_session.commit()


def test_duplicate_symbol_offset_in_one_code_object_rejected(db_session):
    """Reject two kernels claiming one offset within a single code object."""
    workload = Workload(name="w", sub_name="s")
    code_object = CodeObjectStore(
        code_object_id=5,
        pid=42,
        load_base=0x1000,
        workload=workload,
    )
    for kernel_name in ("k1", "k2"):
        db_session.add(
            KernelSymbol(
                code_object_store=code_object,
                kernel=Kernel(kernel_name=kernel_name, workload=workload),
                code_object_offset=0x100,
            )
        )

    with pytest.raises(IntegrityError):
        db_session.commit()


def test_offsetless_symbols_in_one_code_object_allowed(db_session):
    """Symbols awaiting an ISA backfill do not collide on their null offset."""
    workload = Workload(name="w", sub_name="s")
    code_object = CodeObjectStore(
        code_object_id=5,
        pid=42,
        load_base=0x1000,
        workload=workload,
    )
    for kernel_name in ("k1", "k2"):
        db_session.add(
            KernelSymbol(
                code_object_store=code_object,
                kernel=Kernel(kernel_name=kernel_name, workload=workload),
            )
        )
    db_session.commit()

    symbols = db_session.query(KernelSymbol).all()
    assert len(symbols) == 2
    assert {symbol.code_object_offset for symbol in symbols} == {None}


def test_two_kernels_in_one_code_object_get_separate_symbols(db_session):
    """One code object holding two kernels yields one symbol per kernel."""
    workload = Workload(name="w", sub_name="s")
    code_object = CodeObjectStore(
        code_object_id=5,
        pid=42,
        load_base=0x1000,
        workload=workload,
    )
    for kernel_name, symbol_offset in (("k1", 0x100), ("k2", 0x200)):
        db_session.add(
            KernelSymbol(
                code_object_store=code_object,
                kernel=Kernel(kernel_name=kernel_name, workload=workload),
                code_object_offset=symbol_offset,
            )
        )
    db_session.commit()

    symbols = db_session.query(KernelSymbol).all()
    assert len(symbols) == 2
    assert {symbol.kernel.kernel_name for symbol in symbols} == {"k1", "k2"}
    assert {symbol.code_object_uuid for symbol in symbols} == {
        code_object.code_object_uuid
    }


def test_duplicate_code_object_identity_within_process_rejected(db_session):
    """Reject duplicate code-object IDs within one process and workload."""
    workload = Workload(name="w", sub_name="s")
    db_session.add_all([
        CodeObjectStore(code_object_id=5, pid=42, workload=workload),
        CodeObjectStore(code_object_id=5, pid=42, workload=workload),
    ])

    with pytest.raises(IntegrityError):
        db_session.commit()


def test_equal_identities_under_distinct_parent_chains_get_distinct_uuids(
    db_session,
):
    """Allow equal child identities under distinct ownership chains."""
    workload = Workload(name="w", sub_name="s")
    kernel = Kernel(kernel_name="k", workload=workload)
    first_dispatch = Dispatch(dispatch_id=0, kernel=kernel)
    second_dispatch = Dispatch(dispatch_id=1, kernel=kernel)
    first_code_object = CodeObjectStore(
        code_object_id=5,
        pid=42,
        load_base=0x1000,
        workload=workload,
    )
    second_code_object = CodeObjectStore(
        code_object_id=5,
        pid=99,
        load_base=0x1000,
        workload=workload,
    )
    first_symbol = KernelSymbol(code_object_store=first_code_object, kernel=kernel)
    second_symbol = KernelSymbol(code_object_store=second_code_object, kernel=kernel)
    first_instruction = InstructionLine(
        code_object_offset=0x10,
        instruction="v_mov",
        kernel_symbol=first_symbol,
    )
    second_instruction = InstructionLine(
        code_object_offset=0x10,
        instruction="v_mov",
        kernel_symbol=second_symbol,
    )
    db_session.add_all([
        first_dispatch,
        second_dispatch,
        first_instruction,
        second_instruction,
    ])
    db_session.commit()

    assert first_dispatch.dispatch_uuid != second_dispatch.dispatch_uuid
    assert first_code_object.code_object_uuid != second_code_object.code_object_uuid
    assert (first_code_object.pid, second_code_object.pid) == (42, 99)
    assert first_instruction.instruction_uuid != second_instruction.instruction_uuid
    # Same kernel and offset; the process-scoped code object is what separates them.
    assert first_symbol.kernel is second_symbol.kernel
    assert first_symbol.code_object_store is first_code_object
    assert second_symbol.code_object_store is second_code_object


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
# pc_sampling_summary view
# =============================================================================


def test_pc_sampling_summary_view_flattens_normalized_tables(db_session):
    """The summary view flattens the normalized tables and rebuilds
    stall_reason as a JSON dict."""
    workload = Workload(name="w", sub_name="s")
    kernel = Kernel(kernel_name="vecCopy", workload=workload)
    add_pc_sampling_state(
        db_session,
        workload=workload,
        kernel=kernel,
        pid=42,
        stall_reasons={"WAITCNT": 2},
    )
    Database.create_views()
    db_session.commit()

    assert fetch_pc_sampling_summary_rows(db_session) == [
        {
            "workload_id": workload.workload_id,
            "pid": 42,
            "code_object_id": 5,
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


def test_pc_sample_state_placeholder_columns_default_to_none(db_session):
    """The unpopulated PCSampleState columns exist and stay null."""
    workload = Workload(name="w", sub_name="s")
    kernel = Kernel(kernel_name="vecCopy", workload=workload)
    sample_state = add_pc_sampling_state(
        db_session, workload=workload, kernel=kernel, pid=42
    )
    db_session.commit()

    assert sample_state.active_thread_percent is None
    assert sample_state.wave_occupancy_percent is None
    assert sample_state.dispatch_uuid is None


def test_instruction_line_static_type_defaults_to_none(db_session):
    """The unpopulated static instruction type exists and stays null."""
    workload = Workload(name="w", sub_name="s")
    kernel = Kernel(kernel_name="vecCopy", workload=workload)
    sample_state = add_pc_sampling_state(
        db_session, workload=workload, kernel=kernel, pid=42
    )
    db_session.commit()

    assert sample_state.instruction_line.instruction_type_uuid is None
    assert sample_state.instruction_line.instruction_type_lookup is None


def test_pc_sampling_summary_view_separates_states_by_code_object(db_session):
    """States under one kernel stay separate per code object."""
    workload = Workload(name="w", sub_name="s")
    kernel = Kernel(kernel_name="vecCopy", workload=workload)
    add_pc_sampling_state(
        db_session,
        workload=workload,
        kernel=kernel,
        pid=42,
        total_count=8,
        issue_count=2,
        stall_count=6,
        stall_reasons={"WAITCNT": 2, "MEMORY": 4},
    )
    add_pc_sampling_state(
        db_session,
        workload=workload,
        kernel=kernel,
        pid=42,
        code_object_id=6,
        total_count=9,
        issue_count=3,
        stall_count=6,
        stall_reasons={"WAITCNT": 5, "BARRIER": 1},
    )
    Database.create_views()
    db_session.commit()

    rows = fetch_pc_sampling_summary_rows(db_session)

    # The same offset in two code objects can be different code, so the rows are
    # never summed together.
    assert [row["code_object_id"] for row in rows] == [5, 6]
    assert [row["count"] for row in rows] == [8, 9]
    assert [row["count_issue"] for row in rows] == [2, 3]
    assert [row["count_stall"] for row in rows] == [6, 6]
    assert [row["stall_reason"] for row in rows] == [
        {"MEMORY": 4, "WAITCNT": 2},
        {"BARRIER": 1, "WAITCNT": 5},
    ]


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
def test_pc_sampling_summary_view_keeps_display_identity_fields_separate(
    db_session,
    identity_field: str,
    first_value: str,
    second_value: str,
):
    """Different instruction or source values remain separate view rows."""
    workload = Workload(name="w", sub_name="s")
    kernel = Kernel(kernel_name="vecCopy", workload=workload)
    add_pc_sampling_state(
        db_session,
        workload=workload,
        kernel=kernel,
        pid=42,
        total_count=3,
        issue_count=3,
        stall_count=0,
        **{identity_field: first_value},
    )
    add_pc_sampling_state(
        db_session,
        workload=workload,
        kernel=kernel,
        pid=42,
        code_object_id=6,
        total_count=5,
        issue_count=5,
        stall_count=0,
        **{identity_field: second_value},
    )
    Database.create_views()
    db_session.commit()

    rows = fetch_pc_sampling_summary_rows(db_session)

    assert {(row[identity_field], row["count"]) for row in rows} == {
        (first_value, 3),
        (second_value, 5),
    }


def test_pc_sampling_summary_view_keeps_processes_separate_under_one_kernel(db_session):
    """Identical states in two processes remain separate view rows."""
    workload = Workload(name="w", sub_name="s")
    kernel = Kernel(kernel_name="vecCopy", workload=workload)
    add_pc_sampling_state(
        db_session,
        workload=workload,
        kernel=kernel,
        pid=42,
        total_count=3,
        issue_count=3,
        stall_count=0,
    )
    add_pc_sampling_state(
        db_session,
        workload=workload,
        kernel=kernel,
        pid=99,
        total_count=5,
        issue_count=5,
        stall_count=0,
    )
    Database.create_views()
    db_session.commit()

    rows = fetch_pc_sampling_summary_rows(db_session)

    # One kernel row, one view row per process.
    assert {row["kernel_uuid"] for row in rows} == {kernel.kernel_uuid}
    assert {(row["pid"], row["count"]) for row in rows} == {(42, 3), (99, 5)}


def test_pc_sampling_summary_view_keeps_different_workloads_separate(db_session):
    """Matching states for different workloads remain separate view rows."""
    first_workload = Workload(name="first", sub_name="s")
    second_workload = Workload(name="second", sub_name="s")
    first_kernel = Kernel(kernel_name="vecCopy", workload=first_workload)
    second_kernel = Kernel(kernel_name="vecCopy", workload=second_workload)
    add_pc_sampling_state(
        db_session,
        workload=first_workload,
        kernel=first_kernel,
        pid=42,
        total_count=3,
        issue_count=3,
        stall_count=0,
    )
    add_pc_sampling_state(
        db_session,
        workload=second_workload,
        kernel=second_kernel,
        pid=42,
        total_count=5,
        issue_count=5,
        stall_count=0,
    )
    Database.create_views()
    db_session.commit()

    rows = fetch_pc_sampling_summary_rows(db_session)

    assert {(row["workload_id"], row["count"]) for row in rows} == {
        (first_workload.workload_id, 3),
        (second_workload.workload_id, 5),
    }


def test_pc_sampling_summary_view_keeps_host_trap_states_with_null_counts(db_session):
    """Host-trap rows stay per code object with null issue, stall, and reason."""
    workload = Workload(name="w", sub_name="s")
    kernel = Kernel(kernel_name="vecCopy", workload=workload)
    add_pc_sampling_state(
        db_session,
        workload=workload,
        kernel=kernel,
        pid=42,
        total_count=3,
        issue_count=None,
        stall_count=None,
    )
    add_pc_sampling_state(
        db_session,
        workload=workload,
        kernel=kernel,
        pid=42,
        code_object_id=6,
        total_count=5,
        issue_count=None,
        stall_count=None,
    )
    Database.create_views()
    db_session.commit()

    rows = fetch_pc_sampling_summary_rows(db_session)

    assert [row["code_object_id"] for row in rows] == [5, 6]
    assert [row["count"] for row in rows] == [3, 5]
    assert all(row["count_issue"] is None for row in rows)
    assert all(row["count_stall"] is None for row in rows)
    assert all(row["stall_reason"] is None for row in rows)


@pytest.mark.parametrize("nullable_field", ["offset", "instruction", "source"])
def test_pc_sampling_summary_view_attaches_reasons_with_nullable_identity(
    db_session, nullable_field
):
    """Null identity fields keep their rows and their own stall reasons."""
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
        pid=42,
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
        pid=42,
        code_object_id=6,
        total_count=5,
        issue_count=1,
        stall_count=4,
        stall_reasons={"WAITCNT": 4},
        **identity,
    )
    Database.create_views()
    db_session.commit()

    rows = fetch_pc_sampling_summary_rows(db_session)

    assert [row["code_object_id"] for row in rows] == [5, 6]
    assert all(row[nullable_field] is None for row in rows)
    assert [row["count"] for row in rows] == [3, 5]
    assert [row["stall_reason"] for row in rows] == [{"WAITCNT": 2}, {"WAITCNT": 4}]


# =============================================================================
# source tables and source_lines view
# =============================================================================


def test_duplicate_source_file_under_same_workload_rejected(db_session):
    """One workload cannot hold the same file path twice."""
    workload = Workload(name="w", sub_name="s")
    db_session.add_all([
        SourceFile(workload=workload, file_path="a.cpp", md5_checksum="aa"),
        SourceFile(workload=workload, file_path="a.cpp", md5_checksum="bb"),
    ])

    with pytest.raises(IntegrityError):
        db_session.commit()


def test_same_source_file_under_distinct_workloads_allowed(db_session):
    """Two workloads can each hold their own a.cpp with different content."""
    first_workload = Workload(name="first", sub_name="s")
    second_workload = Workload(name="second", sub_name="s")
    db_session.add_all([
        SourceFile(workload=first_workload, file_path="a.cpp", md5_checksum="aa"),
        SourceFile(workload=second_workload, file_path="a.cpp", md5_checksum="bb"),
    ])
    db_session.commit()

    assert db_session.query(SourceFile).count() == 2


def test_duplicate_source_line_under_same_file_rejected(db_session):
    """One file cannot hold the same line number twice."""
    workload = Workload(name="w", sub_name="s")
    source_file = SourceFile(workload=workload, file_path="a.cpp")
    db_session.add_all([
        SourceLine(source_file=source_file, line_number=1, content="int a;"),
        SourceLine(source_file=source_file, line_number=1, content="int b;"),
    ])

    with pytest.raises(IntegrityError):
        db_session.commit()


def test_duplicate_frame_index_under_same_instruction_rejected(db_session):
    """One instruction cannot hold two frames at the same depth."""
    workload = Workload(name="w", sub_name="s")
    kernel = Kernel(kernel_name="k", workload=workload)
    code_object = CodeObjectStore(
        code_object_id=5, pid=42, load_base=0x1000, workload=workload
    )
    instruction_line = InstructionLine(
        code_object_offset=0x10,
        instruction="v_mov",
        kernel_symbol=KernelSymbol(code_object_store=code_object, kernel=kernel),
    )
    source_file = SourceFile(workload=workload, file_path="a.cpp")
    db_session.add_all([
        InstructionSourceLine(
            instruction_line=instruction_line,
            source_line=SourceLine(source_file=source_file, line_number=1),
            frame_index=0,
        ),
        InstructionSourceLine(
            instruction_line=instruction_line,
            source_line=SourceLine(source_file=source_file, line_number=2),
            frame_index=0,
        ),
    ])

    with pytest.raises(IntegrityError):
        db_session.commit()


def test_source_lines_view_joins_lines_to_their_file(db_session):
    """The view exposes each stored line beside its file path and digest."""
    workload = Workload(name="w", sub_name="s")
    source_file = SourceFile(workload=workload, file_path="a.cpp", md5_checksum="aa")
    db_session.add_all([
        SourceLine(source_file=source_file, line_number=1, content="int a;"),
        # A dropped line number keeps the file identity with no content.
        SourceLine(source_file=source_file, line_number=None, content=None),
    ])
    Database.create_views()
    db_session.commit()

    rows = [
        tuple(row)
        for row in db_session.execute(
            text(
                "SELECT workload_id, file_path, md5_checksum, line_number, content "
                "FROM compute_source_lines_view ORDER BY line_number"
            )
        )
    ]

    assert rows == [
        (workload.workload_id, "a.cpp", "aa", None, None),
        (workload.workload_id, "a.cpp", "aa", 1, "int a;"),
    ]


@pytest.mark.parametrize(
    ("source", "expected_source"),
    [
        pytest.param("/s/a.cpp:1", "/s/a.cpp:1", id="single_frame"),
        pytest.param(
            "/s/hip.h:258 -> /s/hip.h:317 -> /s/a.cpp:36",
            "/s/hip.h:258 -> /s/hip.h:317 -> /s/a.cpp:36",
            id="chain_rebuilt_innermost_first",
        ),
        pytest.param(
            "/s/hip.h:? -> /s/a.cpp:36",
            "/s/hip.h:? -> /s/a.cpp:36",
            id="dropped_line_number_renders_question_mark",
        ),
        pytest.param(None, None, id="no_frames_yields_null"),
    ],
)
def test_pc_sampling_summary_view_rebuilds_source_from_frames(
    db_session, source, expected_source
):
    """The summary view reconstructs the comment text from ordered frames."""
    workload = Workload(name="w", sub_name="s")
    kernel = Kernel(kernel_name="vecCopy", workload=workload)
    db_session.add(Dispatch(dispatch_id=0, kernel=kernel))
    add_pc_sampling_state(
        db_session, workload=workload, kernel=kernel, pid=42, source=source
    )
    Database.create_views()
    db_session.commit()

    rows = fetch_pc_sampling_summary_rows(db_session)

    assert [row["source"] for row in rows] == [expected_source]


def test_pc_sampling_summary_view_keeps_frames_of_one_instruction_together(db_session):
    """Two instructions sharing a source line keep their own chains."""
    workload = Workload(name="w", sub_name="s")
    kernel = Kernel(kernel_name="vecCopy", workload=workload)
    db_session.add(Dispatch(dispatch_id=0, kernel=kernel))
    add_pc_sampling_state(
        db_session,
        workload=workload,
        kernel=kernel,
        pid=42,
        offset=0x10,
        source="/s/hip.h:317 -> /s/a.cpp:36",
    )
    add_pc_sampling_state(
        db_session,
        workload=workload,
        kernel=kernel,
        pid=42,
        code_object_id=6,
        offset=0x20,
        source="/s/a.cpp:36",
    )
    Database.create_views()
    db_session.commit()

    rows = fetch_pc_sampling_summary_rows(db_session)

    assert [row["source"] for row in rows] == [
        "/s/hip.h:317 -> /s/a.cpp:36",
        "/s/a.cpp:36",
    ]
