#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# OMPT is a rocpd-only trace: rocprofv3 writes OMPT records to the rocpd SQLite
# database (and they are exported to csv/perfetto/otf2 via `rocpd convert`). The
# direct json/csv/perfetto/otf2 generators do not contain OMPT records, so these
# validators read OMPT from the rocpd database.
#
# In rocpd, ranged OMPT operations are stored as named rows in the `regions` view
# and instantaneous operations (start == end, e.g. omp_parallel_begin) as named
# rows in the `samples` view. The `regions_and_samples` view unions both, so the
# OMPT operation set is queried from there. The "OMPT" category string is set in
# source/lib/output/domain_type.cpp.

import sys
from collections import Counter

import pytest

OMPT_CATEGORY = "OMPT"

# The "parallel" --ompt-trace category resolves (in tool.cpp::resolve_ompt_ops) to
# exactly these host-side operations. A --ompt-trace parallel run must not record
# any operation outside this set.
PARALLEL_CATEGORY_OPS = {
    "omp_parallel_begin",
    "omp_parallel_end",
    "omp_implicit_task",
    "omp_work",
    "omp_dispatch",
    "omp_reduction",
    "omp_masked",
}

# Representative host-side (CPU) OpenMP operations
HOST_PARALLEL_OPS = ("omp_parallel_begin", "omp_parallel_end")
HOST_WORK_OPS = ("omp_work", "omp_sync_region", "omp_dispatch")


def _ompt_op_counts(conn):
    """Return Counter[operation_name] -> row count for OMPT records in rocpd,
    spanning both ranged regions and instant samples."""
    rows = conn.execute(
        "SELECT name, COUNT(*) FROM regions_and_samples WHERE category = ? GROUP BY name",
        (OMPT_CATEGORY,),
    ).fetchall()
    return Counter(dict(rows))


def _is_target_submit_operation(op_name):
    """The SDK enum spells the operation 'target_submit_emi'; accept anything that
    contains 'target_submit'."""
    return "target_submit" in op_name.lower()


def test_agent_info(rocpd_conn):
    """Every agent recorded in the rocpd database must be a well-formed CPU or GPU.
    Catches regressions where rocprofv3 fails to enumerate agents alongside OMPT
    tracing."""
    rows = rocpd_conn.execute("SELECT type FROM rocpd_info_agent").fetchall()
    assert len(rows) > 0, "rocpd database contains no agents"
    for (agent_type,) in rows:
        assert agent_type in ("CPU", "GPU"), f"unexpected agent type {agent_type!r}"


def test_rocpd_contains_ompt_records(rocpd_conn):
    """The rocpd database must contain well-formed OMPT records (regions + samples)."""
    counts = _ompt_op_counts(rocpd_conn)
    assert sum(counts.values()) > 0, "rocpd database has no OMPT records"

    rows = rocpd_conn.execute(
        "SELECT tid, start, end FROM regions_and_samples WHERE category = ?",
        (OMPT_CATEGORY,),
    ).fetchall()
    for tid, start, end in rows:
        assert tid > 0, "OMPT record has non-positive thread id"
        assert start > 0, "OMPT record has non-positive start timestamp"
        # instant records have start == end; ranged records have start < end
        assert end >= start, "OMPT record has end before start"


def test_ompt_host_side_events_present(rocpd_conn):
    """A default (unfiltered) --ompt-trace should capture host-side OpenMP activity,
    not just target-offload events: the parallel-region lifecycle (parallel begin/end
    + implicit_task) and at least one work-related callback."""
    ops = set(_ompt_op_counts(rocpd_conn))
    if not ops:
        pytest.skip("no OMPT records present; OMPT tracing was likely not run")

    assert any(op in ops for op in HOST_PARALLEL_OPS), (
        f"expected a host-side parallel-region event ({HOST_PARALLEL_OPS}); "
        f"saw: {sorted(ops)}"
    )
    assert (
        "omp_implicit_task" in ops
    ), f"expected host-side 'omp_implicit_task' events; saw: {sorted(ops)}"
    assert any(op in ops for op in HOST_WORK_OPS), (
        f"expected at least one work-related host event ({HOST_WORK_OPS}); "
        f"saw: {sorted(ops)}"
    )


def test_ompt_target_correlates_with_kernel_dispatch(rocpd_conn):
    """AC5 -- every OMPT target_submit record must share its internal correlation id
    (stored as the rocpd event/region stack_id) with a kernel-dispatch record. This
    validates that the rocprofv3 pipeline preserves the OMPT <-> kernel-dispatch
    correlation the SDK promises. Skipped when kernel_dispatch records are absent (the
    granular-target variant runs without --kernel-trace)."""
    kernel_ids = {r[0] for r in rocpd_conn.execute("SELECT stack_id FROM kernels")}
    if not kernel_ids:
        pytest.skip("no kernel dispatch records; this run was without --kernel-trace")

    submit_ids = {
        r[0]
        for r in rocpd_conn.execute(
            "SELECT stack_id FROM regions WHERE category = ? AND name LIKE '%target_submit%'",
            (OMPT_CATEGORY,),
        )
    }
    assert submit_ids, (
        "Expected at least one OMPT target_submit_emi record on a target-offload "
        "program"
    )

    unmatched = submit_ids - kernel_ids
    assert not unmatched, (
        f"{len(unmatched)} OMPT target_submit correlation id(s) did not match any "
        f"kernel_dispatch; first unmatched: {sorted(unmatched)[:5]}"
    )


def test_granular_target_filter_only_target_ops(rocpd_conn):
    """AC2 -- when --ompt-trace target is supplied, the captured records must only be
    target-category operations (target_emi, target_data_op_emi, target_submit_emi)."""
    ops = set(_ompt_op_counts(rocpd_conn))
    if not ops:
        pytest.skip("no OMPT records present; the granular-filter run was likely not run")

    leaks = {op for op in ops if "target" not in op.lower()}
    assert not leaks, (
        f"--ompt-trace target leaked non-target operations into the trace: "
        f"{sorted(leaks)}"
    )
    assert any(
        _is_target_submit_operation(op) for op in ops
    ), f"--ompt-trace target produced no target_submit records; saw: {sorted(ops)}"


def test_granular_host_filter_only_parallel_ops(rocpd_conn):
    """Counterpart to the target filter for a host-side filter: --ompt-trace parallel
    must record only the parallel-category operations and must not leak
    task/sync/mutex/target ops."""
    ops = set(_ompt_op_counts(rocpd_conn))
    if not ops:
        pytest.skip("no OMPT records present; the host-filter run was likely not run")

    leaks = {op for op in ops if op not in PARALLEL_CATEGORY_OPS}
    assert not leaks, (
        f"--ompt-trace parallel leaked non-parallel-category operations into the "
        f"trace: {sorted(leaks)}"
    )
    assert any(
        op in ops
        for op in ("omp_parallel_begin", "omp_parallel_end", "omp_implicit_task")
    ), f"--ompt-trace parallel produced no parallel-region records; saw: {sorted(ops)}"


def test_ompt_all_form_is_complete(rocpd_conn):
    """--ompt-trace all must behave like bare --ompt-trace and produce the complete
    OpenMP trace: both host-side (parallel/implicit-task) and, on a target-offload
    workload, target_submit events."""
    ops = set(_ompt_op_counts(rocpd_conn))
    if not ops:
        pytest.skip("no OMPT records present; OMPT tracing was likely not run")

    assert any(op in ops for op in HOST_PARALLEL_OPS), (
        f"--ompt-trace all is missing host-side parallel events ({HOST_PARALLEL_OPS}); "
        f"saw: {sorted(ops)}"
    )
    assert (
        "omp_implicit_task" in ops
    ), f"--ompt-trace all is missing host-side 'omp_implicit_task'; saw: {sorted(ops)}"
    assert any(_is_target_submit_operation(op) for op in ops), (
        f"--ompt-trace all is missing target_submit events on a target-offload "
        f"workload; saw: {sorted(ops)}"
    )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
