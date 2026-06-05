# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Post-load sanity checks for Perfetto traces.

Provides:
- run_sanity_checks(tp): run both sanity queries on a loaded TraceProcessor;
  raise RuntimeError with descriptive message on any malformed trace condition
- _query_to_dataframe(tp, sql): production workaround for the
  QueryResultIterator.as_pandas_dataframe() pandas bug (RepeatedScalarContainer)

Design decisions:
- D-15: Sanity checks run in fixture setup (before returning to test), not lazily
- Failures raise RuntimeError (infrastructure failure), not AssertionError
- No runtime import of perfetto at module level (avoids ImportError if not installed)
- Both SQL strings are hardcoded constants (T-03-04: no injection vector)
- Row access uses attribute style (row.count, row.value) per production pattern;
  if attribute access fails with a real trace, use row[0] as fallback
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from perfetto.trace_processor import TraceProcessor

# ---------------------------------------------------------------------------
# SQL constants — hardcoded; no user input (T-03-04)
# ---------------------------------------------------------------------------

# Columns are explicitly aliased (AS cnt / AS val) so attribute access on the
# returned rows is deterministic. A bare "SELECT COUNT(*)" yields a column named
# "COUNT(*)", which is not a valid attribute name — row.count then raises
# AttributeError on a real TraceProcessor.
_SQL_NEGATIVE_TIMESTAMPS = "SELECT COUNT(*) AS cnt FROM slice WHERE ts < 0"

_SQL_SEQUENCE_FAILURES = (
    "SELECT value AS val FROM stats WHERE name = 'traced_buf_sequence_failures'"
)

# ---------------------------------------------------------------------------
# QueryResultIterator pandas bug workaround
# ---------------------------------------------------------------------------

def _query_to_dataframe(tp: Any, sql: str):
    """Execute SQL and return a pandas DataFrame, applying the RepeatedScalarContainer fix.

    This is the exact production workaround from:
    rocm-systems/projects/rocprofiler-sdk/tests/pytest-packages/pytest_utils/perfetto_reader.py

    The bug: QueryResultIterator.__column_names may be a protobuf RepeatedScalarContainer
    instead of a list, causing as_pandas_dataframe() to raise TypeError. Patching it to
    a plain list before the call resolves the issue.

    Args:
        tp: TraceProcessor instance (or compatible mock).
        sql: PerfettoSQL query string.

    Returns:
        pandas DataFrame with query results.
    """
    query_itr = tp.query(sql)
    _buggy_key = "_QueryResultIterator__column_names"
    if (
        _buggy_key in query_itr.__dict__
        and type(query_itr.__dict__[_buggy_key]).__name__ == "RepeatedScalarContainer"
    ):
        query_itr.__dict__[_buggy_key] = list(query_itr.__dict__[_buggy_key])
    return query_itr.as_pandas_dataframe()

# ---------------------------------------------------------------------------
# Post-load sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks(tp: Any) -> None:
    """Run post-load sanity queries on a loaded Perfetto trace.

    Both queries run after every TraceProcessor load (D-15). Raises RuntimeError
    on the first failing check — test fixture must not be returned to test body
    when the trace is malformed.

    Checks performed:
    1. Negative timestamps: indicates clock domain conversion failure
    2. Sequence failures: indicates ring-buffer overflow / silent event dropping

    Args:
        tp: TraceProcessor instance (or compatible mock with .query() method).

    Raises:
        RuntimeError: If any sanity check finds a malformed trace condition.
            Message includes count and actionable remediation steps.
    """
    # Check 1: Negative timestamps indicate clock domain conversion failure
    result = tp.query(_SQL_NEGATIVE_TIMESTAMPS)
    neg_count = next(iter(result)).cnt
    if neg_count > 0:
        raise RuntimeError(
            f"Perfetto trace has {neg_count} slice(s) with negative timestamps. "
            "This indicates a clock domain conversion failure. "
            "Check ROCPROFSYS_PERFETTO_BUFFER_SIZE_KB and clock source configuration."
        )

    # Check 2: Sequence failures indicate ring-buffer interning loss (silent data loss)
    rows = list(tp.query(_SQL_SEQUENCE_FAILURES))
    if rows and rows[0].val > 0:
        raise RuntimeError(
            f"Perfetto trace has {rows[0].value} sequence failure(s). "
            "Events were silently lost due to ring-buffer overflow. "
            "Increase ROCPROFSYS_PERFETTO_BUFFER_SIZE_KB before re-collecting."
        )
