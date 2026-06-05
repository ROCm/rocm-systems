# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cross-format correlation validators for rocprof-sys profiling outputs.

Implements four standalone CROSS-* validators that assert the same GPU operations
appear consistently across Perfetto and RocPD output formats:

  CROSS-01 (assert_kernel_correlation):
      Assert that kernel operations present in RocPD are also present in Perfetto.
      Uses name-first alignment: exact match, then normalized fallback (strip <>,
      lowercase, trim).

  CROSS-02 (assert_temporal_ordering):
      Assert temporal ordering consistency between Perfetto and RocPD events.
      Checks both absolute timestamp delta <= tolerance_ns AND relative ordering
      consistency for matched kernel pairs.

  CROSS-03 (assert_hip_correlation):
      Assert that HIP API calls present in RocPD (category='rocm_hip_api') are
      also present in Perfetto (category='hip_api'). Same alignment logic as CROSS-01.

  CROSS-04 (assert_timemory_perfetto_correlation):
      Assert that timemory wall-clock totals for a label align with the sum of
      Perfetto slice durations for matching slices (within tolerance_pct %).

Clock source note [ASSUMED]: Both Perfetto and RocPD are assumed to use
CLOCK_BOOTTIME nanosecond timestamps. This cannot be verified with CPU-only
cross-format artifacts. The tolerance_ns parameter is the escape valve for
clock drift between formats.

All functions return a CheckResult — callers can check .passed and use
.assert_or_raise() to convert a failure to AssertionError in pytest.
"""
from __future__ import annotations

import re
import sqlite3
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from rocprofsys_validator.core import CheckResult

if TYPE_CHECKING:
    from rocprofsys_validator.readers.perfetto import PerfettoReader
    from rocprofsys_validator.readers.rocpd import RocpdReader
    from rocprofsys_validator.readers.timemory import TimemoryReader

# ---------------------------------------------------------------------------
# Module-level SQL constants
# ---------------------------------------------------------------------------

# Default category vocabularies. These vary by tool/version, so every public
# correlation function exposes them as parameters rather than hardcoding the
# literal (CLAUDE.md adaptability constraint). The constants below are only the
# defaults used when the caller does not override them.
_DEFAULT_PERFETTO_KERNEL_CATEGORY = "kernel_dispatch"
_DEFAULT_PERFETTO_HIP_CATEGORY = "hip_api"
_DEFAULT_ROCPD_HIP_CATEGORY = "rocm_hip_api"

_ROCPD_KERNEL_SQL = "SELECT name, start, end FROM kernels"

def _perfetto_slice_sql(category: str) -> str:
    """Build a Perfetto slice query filtered to a single category (quote-escaped)."""
    safe = category.replace("'", "''")
    return f"SELECT name, ts, dur FROM slice WHERE category = '{safe}'"

def _rocpd_region_sql(category: str) -> str:
    """Build a RocPD regions query filtered to a single category (quote-escaped)."""
    safe = category.replace("'", "''")
    return f"SELECT name, start, end FROM regions WHERE category = '{safe}'"

def _empty_rocpd_result(
    validator_name: str, kind: str, require_records: bool
) -> CheckResult:
    """Result for the 'no RocPD records to correlate' case.

    A validation framework must not silently pass when it matched nothing. When
    require_records is True (the default) this is a hard failure; callers who
    genuinely expect zero records (e.g. a CPU-only run) opt out with
    require_records=False to get an explicit vacuous pass.
    """
    if require_records:
        return CheckResult(
            passed=False,
            validator_name=validator_name,
            message=(
                f"no RocPD {kind} found to correlate "
                f"(require_records=True). Pass require_records=False if zero {kind} "
                f"is expected for this run."
            ),
            expected=f">= 1 RocPD {kind}",
            actual="0",
            details={"require_records": require_records},
        )
    return CheckResult(
        passed=True,
        validator_name=validator_name,
        message=f"PASSED (vacuous): no RocPD {kind} to correlate",
        details={"require_records": require_records},
    )

# ---------------------------------------------------------------------------
# NormalizedEvent dataclass
# ---------------------------------------------------------------------------

@dataclass
class NormalizedEvent:
    """Common representation of a timed event from any profiling format.

    Used internally by correlation functions to align events across formats.
    Both Perfetto and RocPD record timestamps in nanoseconds; NormalizedEvent
    stores them in a uniform ns representation.

    Attributes:
        name: Event or kernel name (as recorded in the source format).
        source: Origin format; one of "perfetto" or "rocpd".
        start_ns: Start timestamp in nanoseconds.
        end_ns: End timestamp in nanoseconds (= start_ns + duration for Perfetto).
        metadata: Additional format-specific fields (e.g., {"dur": int} for Perfetto).
    """

    name: str
    source: str
    start_ns: int
    end_ns: int
    metadata: dict[str, Any] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    """Strip C++ template parameters, lowercase, and trim surrounding whitespace.

    Used as a fallback normalization step when exact name matching fails.
    Examples::

        _normalize_name('vectorAdd<float>')         -> 'vectoradd'
        _normalize_name('kernel<std::vector<float>>') -> 'kernel'
        _normalize_name('hipLaunchKernel')           -> 'hiplaunchkernel'
        _normalize_name('  MyKernel  ')              -> 'mykernel'

    Args:
        name: Raw event or kernel name as recorded in the profiling format.

    Returns:
        Normalized form: template parameters removed, lowercased, whitespace stripped.
    """
    # Iterative substitution: remove innermost <...> first, then outer levels.
    # A single pass of <[^>]*> leaves trailing '>' for nested templates like
    # kernel<std::vector<float>>. Looping until stable handles all nesting depths.
    prev = None
    result = name
    while prev != result:
        prev = result
        result = re.sub(r'<[^<>]*>', '', prev)
    return result.lower().strip()

def _from_perfetto_slice(row: Any) -> NormalizedEvent:
    """Convert a pd.Series row (from df.iterrows()) to a NormalizedEvent.

    Args:
        row: A pd.Series from iterating over execute_sql() DataFrame output.
             Use bracket notation (row["name"]) — row.name returns the index.

    Returns:
        NormalizedEvent with source="perfetto", timestamps derived from ts and dur.
    """
    return NormalizedEvent(
        name=row["name"],
        source="perfetto",
        start_ns=int(row["ts"]),
        end_ns=int(row["ts"]) + int(row["dur"]),
        metadata={"dur": int(row["dur"])},
    )

def _from_rocpd_region(row: Any) -> NormalizedEvent:
    """Convert a sqlite3.Row (from execute_sql() list) to a NormalizedEvent.

    Args:
        row: A sqlite3.Row from RocpdReader.execute_sql() return value.
             row["end"] works correctly despite "end" being a SQL keyword.

    Returns:
        NormalizedEvent with source="rocpd", timestamps from start/end columns.
    """
    return NormalizedEvent(
        name=row["name"],
        source="rocpd",
        start_ns=int(row["start"]),
        end_ns=int(row["end"]),
        metadata={},
    )

def _align_events(
    perfetto_events: list[NormalizedEvent],
    rocpd_events: list[NormalizedEvent],
) -> tuple[list[tuple[NormalizedEvent, NormalizedEvent]], list[str]]:
    """Align RocPD events against Perfetto events by name, with normalized fallback.

    Alignment strategy:
      Step 1 (exact): For each RocPD event, check if its name is in the set of
                      Perfetto event names.
      Step 2 (normalized fallback): For remaining unmatched RocPD events, check
                      if _normalize_name(rocpd.name) matches any
                      _normalize_name(perfetto.name).
      Unmatched: Events that fail both steps are added to unmatched list.

    Args:
        perfetto_events: List of NormalizedEvent from Perfetto.
        rocpd_events: List of NormalizedEvent from RocPD.

    Returns:
        Tuple of (matched_pairs, unmatched_rocpd_names) where matched_pairs is
        a list of (rocpd_event, perfetto_event) tuples and unmatched_rocpd_names
        is a list of RocPD event names with no Perfetto counterpart.
    """
    # Build lookup maps for Perfetto events
    perfetto_by_name: dict[str, NormalizedEvent] = {}
    for evt in perfetto_events:
        if evt.name not in perfetto_by_name:
            perfetto_by_name[evt.name] = evt

    perfetto_by_normalized: dict[str, NormalizedEvent] = {}
    for evt in perfetto_events:
        normalized = _normalize_name(evt.name)
        if normalized not in perfetto_by_normalized:
            perfetto_by_normalized[normalized] = evt

    matched_pairs: list[tuple[NormalizedEvent, NormalizedEvent]] = []
    unmatched_rocpd_names: list[str] = []

    for rocpd_evt in rocpd_events:
        # Step 1: exact match
        if rocpd_evt.name in perfetto_by_name:
            matched_pairs.append((rocpd_evt, perfetto_by_name[rocpd_evt.name]))
            continue

        # Step 2: normalized fallback
        normalized_rocpd = _normalize_name(rocpd_evt.name)
        if normalized_rocpd in perfetto_by_normalized:
            matched_pairs.append((rocpd_evt, perfetto_by_normalized[normalized_rocpd]))
            continue

        # No match found
        unmatched_rocpd_names.append(rocpd_evt.name)

    return matched_pairs, unmatched_rocpd_names

# ---------------------------------------------------------------------------
# Public correlation functions
# ---------------------------------------------------------------------------

def assert_kernel_correlation(
    perfetto_reader: "PerfettoReader",
    rocpd_reader: "RocpdReader",
    tolerance_ns: int = 1_000_000,
    *,
    perfetto_category: str = _DEFAULT_PERFETTO_KERNEL_CATEGORY,
    require_records: bool = True,
) -> CheckResult:
    """Assert that all kernel operations in RocPD also appear in Perfetto (CROSS-01).

    Queries Perfetto for slices with the given perfetto_category and RocPD for
    all rows in the kernels view, then aligns by name (exact first, normalized
    fallback). Reports unmatched kernel names in details['unmatched'].

    Args:
        perfetto_reader: PerfettoReader with an active trace connection.
        rocpd_reader: RocpdReader with an active SQLite connection.
        tolerance_ns: Timestamp tolerance in nanoseconds (unused here; reserved
                      for future multi-step alignment). Default: 1ms.
        perfetto_category: Perfetto slice category for kernels (default
                      'kernel_dispatch'). Configurable per tool/version.
        require_records: When True (default), zero RocPD kernels is a hard
                      failure (the framework refuses to pass on nothing). Set
                      False to allow an explicit vacuous pass for runs that
                      legitimately have no kernels.

    Returns:
        CheckResult with passed=True if all RocPD kernels have a Perfetto
        counterpart, or passed=False with details['unmatched'] listing missing names.
        When RocPD has no kernels, the outcome depends on require_records.
    """
    perf_df: pd.DataFrame = perfetto_reader.execute_sql(
        _perfetto_slice_sql(perfetto_category)
    )
    rocpd_rows: list[sqlite3.Row] = rocpd_reader.execute_sql(_ROCPD_KERNEL_SQL)

    if not rocpd_rows:
        return _empty_rocpd_result(
            "assert_kernel_correlation", "kernels", require_records
        )

    perfetto_events = [
        _from_perfetto_slice(row) for _, row in perf_df.iterrows()
    ]
    rocpd_events = [_from_rocpd_region(row) for row in rocpd_rows]

    matched_pairs, unmatched = _align_events(perfetto_events, rocpd_events)

    if not unmatched:
        return CheckResult(
            passed=True,
            validator_name="assert_kernel_correlation",
            message=f"all {len(rocpd_rows)} RocPD kernels matched in Perfetto",
            details={
                "matched_count": len(matched_pairs),
                "total_rocpd": len(rocpd_rows),
                "total_perfetto": len(perf_df),
            },
        )

    return CheckResult(
        passed=False,
        validator_name="assert_kernel_correlation",
        message=f"{len(unmatched)} RocPD kernels have no Perfetto counterpart",
        expected="all kernels matched",
        actual=f"unmatched {unmatched}",
        details={
            "unmatched": unmatched,
            "total_rocpd": len(rocpd_rows),
            "total_perfetto": len(perf_df),
        },
    )

def assert_temporal_ordering(
    perfetto_reader: "PerfettoReader",
    rocpd_reader: "RocpdReader",
    tolerance_ns: int = 1_000_000,
    *,
    perfetto_category: str = _DEFAULT_PERFETTO_KERNEL_CATEGORY,
    require_records: bool = True,
) -> CheckResult:
    """Assert temporal ordering consistency between Perfetto and RocPD events (CROSS-02).

    For each matched kernel pair, checks:
      1. Absolute timestamp delta: abs(rocpd.start_ns - perfetto.start_ns) <= tolerance_ns
      2. Relative ordering consistency: if A before B in RocPD, then A before B in Perfetto

    Uses the same kernel queries as assert_kernel_correlation.

    Args:
        perfetto_reader: PerfettoReader with an active trace connection.
        rocpd_reader: RocpdReader with an active SQLite connection.
        tolerance_ns: Maximum allowed timestamp delta between matched pairs. Default: 1ms.
        perfetto_category: Perfetto slice category for kernels (default
                      'kernel_dispatch'). Configurable per tool/version.
        require_records: When True (default), zero RocPD kernels is a hard failure.

    Returns:
        CheckResult with passed=True if all matched pairs are within tolerance and
        ordering is consistent, or passed=False with details['violations'] describing each
        violation found.
    """
    perf_df: pd.DataFrame = perfetto_reader.execute_sql(
        _perfetto_slice_sql(perfetto_category)
    )
    rocpd_rows: list[sqlite3.Row] = rocpd_reader.execute_sql(_ROCPD_KERNEL_SQL)

    if not rocpd_rows:
        return _empty_rocpd_result(
            "assert_temporal_ordering", "kernels", require_records
        )

    perfetto_events = [
        _from_perfetto_slice(row) for _, row in perf_df.iterrows()
    ]
    rocpd_events = [_from_rocpd_region(row) for row in rocpd_rows]

    matched_pairs, _ = _align_events(perfetto_events, rocpd_events)

    violations: list[str] = []

    # Check 1: Absolute timestamp delta for each matched pair
    for rocpd_evt, perf_evt in matched_pairs:
        delta = abs(rocpd_evt.start_ns - perf_evt.start_ns)
        if delta > tolerance_ns:
            violations.append(
                f"timestamp delta {delta}ns exceeds tolerance {tolerance_ns}ns "
                f"for kernel '{rocpd_evt.name}' "
                f"(rocpd={rocpd_evt.start_ns}, perfetto={perf_evt.start_ns})"
            )

    # Check 2: Relative ordering consistency
    # Sort matched pairs by RocPD start time to get RocPD ordering
    matched_by_rocpd = sorted(matched_pairs, key=lambda p: p[0].start_ns)
    # Sort matched pairs by Perfetto start time to get Perfetto ordering
    matched_by_perfetto = sorted(matched_pairs, key=lambda p: p[1].start_ns)

    rocpd_order = [p[0].name for p in matched_by_rocpd]
    perfetto_order = [p[0].name for p in matched_by_perfetto]

    # LIMITATION (WR-04): When duplicate kernel names exist, this ordering check
    # compares name sequences — swapped invocations of the same kernel are not
    # detected. For example, two 'vectorAdd' invocations swapped between formats
    # produce ['vectorAdd', 'vectorAdd'] on both sides — no violation detected.
    # Future work: use per-event identity (index or timestamp) for the comparison.
    if rocpd_order != perfetto_order:
        violations.append(
            f"relative ordering inconsistency: "
            f"RocPD order {rocpd_order} != Perfetto order {perfetto_order}"
        )

    if not violations:
        return CheckResult(
            passed=True,
            validator_name="assert_temporal_ordering",
            message=(
                f"all {len(matched_pairs)} matched pairs within tolerance={tolerance_ns}ns "
                f"with consistent ordering"
            ),
            details={
                "matched_count": len(matched_pairs),
                "tolerance_ns": tolerance_ns,
            },
        )

    return CheckResult(
        passed=False,
        validator_name="assert_temporal_ordering",
        message=f"{len(violations)} temporal ordering violation(s) found",
        details={
            "violations": violations,
            "matched_count": len(matched_pairs),
            "tolerance_ns": tolerance_ns,
        },
    )

def assert_hip_correlation(
    perfetto_reader: "PerfettoReader",
    rocpd_reader: "RocpdReader",
    *,
    perfetto_category: str = _DEFAULT_PERFETTO_HIP_CATEGORY,
    rocpd_category: str = _DEFAULT_ROCPD_HIP_CATEGORY,
    require_records: bool = True,
) -> CheckResult:
    """Assert that HIP API calls in RocPD also appear in Perfetto (CROSS-03).

    Queries Perfetto for slices with perfetto_category (default 'hip_api') and
    RocPD for regions with rocpd_category (default 'rocm_hip_api'), then aligns
    by name using the same exact + normalized fallback strategy as
    assert_kernel_correlation. Both category names are configurable because they
    differ across tools and versions (CLAUDE.md adaptability constraint).

    Args:
        perfetto_reader: PerfettoReader with an active trace connection.
        rocpd_reader: RocpdReader with an active SQLite connection.
        perfetto_category: Perfetto slice category for HIP calls (default 'hip_api').
        rocpd_category: RocPD region category for HIP calls (default 'rocm_hip_api').
        require_records: When True (default), zero RocPD HIP calls is a hard failure.

    Returns:
        CheckResult with passed=True if all RocPD HIP calls have a Perfetto
        counterpart, or passed=False with details['unmatched'] listing missing names.
        When RocPD has no HIP calls, the outcome depends on require_records.
    """
    perf_df: pd.DataFrame = perfetto_reader.execute_sql(
        _perfetto_slice_sql(perfetto_category)
    )
    rocpd_rows: list[sqlite3.Row] = rocpd_reader.execute_sql(
        _rocpd_region_sql(rocpd_category)
    )

    if not rocpd_rows:
        return _empty_rocpd_result(
            "assert_hip_correlation", "HIP calls", require_records
        )

    perfetto_events = [
        _from_perfetto_slice(row) for _, row in perf_df.iterrows()
    ]
    rocpd_events = [_from_rocpd_region(row) for row in rocpd_rows]

    matched_pairs, unmatched = _align_events(perfetto_events, rocpd_events)

    if not unmatched:
        return CheckResult(
            passed=True,
            validator_name="assert_hip_correlation",
            message=f"all {len(rocpd_rows)} RocPD HIP calls matched in Perfetto",
            details={
                "matched_count": len(matched_pairs),
                "total_rocpd": len(rocpd_rows),
                "total_perfetto": len(perf_df),
            },
        )

    return CheckResult(
        passed=False,
        validator_name="assert_hip_correlation",
        message=f"{len(unmatched)} RocPD HIP calls have no Perfetto counterpart",
        expected="all HIP calls matched",
        actual=f"unmatched {unmatched}",
        details={
            "unmatched": unmatched,
            "total_rocpd": len(rocpd_rows),
            "total_perfetto": len(perf_df),
        },
    )

def assert_record_count_parity(
    counts: dict[str, int],
    tolerance: int = 0,
) -> CheckResult:
    """Assert record counts agree across formats (the primary cross-format check).

    The strongest cross-format correctness guarantee is that no records are
    dropped or duplicated: the count of a given record kind must match across
    every format that emits it (Perfetto slices, RocPD view rows, rocprofiler
    JSON records). Callers build the dict from whichever sources apply, e.g.::

        assert_record_count_parity({
            "json":     json_reader.record_count("kernel_dispatch"),
            "perfetto": perfetto_count,
            "rocpd":    rocpd_count,
        })

    Args:
        counts: Mapping of source label -> record count. At least two sources
                are required.
        tolerance: Maximum allowed spread (max - min) between counts. Default 0
                (exact equality).

    Returns:
        CheckResult with passed=True when (max - min) <= tolerance, else
        passed=False with the per-source counts and spread in details.
    """
    if len(counts) < 2:
        return CheckResult(
            passed=False,
            validator_name="assert_record_count_parity",
            message=(
                "Record-count parity needs at least two sources to compare; "
                f"got {len(counts)} ({list(counts)})"
            ),
            details={"counts": dict(counts)},
        )

    values = list(counts.values())
    spread = max(values) - min(values)
    passed = spread <= tolerance

    return CheckResult(
        passed=passed,
        validator_name="assert_record_count_parity",
        message=(
            f"Record counts agree across {len(counts)} sources "
            f"(spread={spread} <= tolerance={tolerance})"
            if passed
            else f"Record-count mismatch across sources: {dict(counts)} "
            f"(spread={spread} > tolerance={tolerance})"
        ),
        expected=f"spread <= {tolerance}",
        actual=f"spread = {spread}",
        details={"counts": dict(counts), "spread": spread, "tolerance": tolerance},
    )

def assert_timemory_perfetto_correlation(
    timemory_reader: "TimemoryReader",
    perfetto_reader: "PerfettoReader",
    label: str,
    tolerance_pct: float = 5.0,
) -> CheckResult:
    """Compare timemory wall-clock totals to Perfetto slice durations (CROSS-04).

    Finds the label in timemory's wall_clock output (substring match), then finds
    Perfetto slices whose names contain the label (LIKE '%label%'), sums Perfetto
    durations (ns -> seconds), and checks whether the two totals agree within
    tolerance_pct percent.

    Security: The label is embedded in a LIKE query via PerfettoReader.execute_sql.
    The ', %, and _ characters are escaped (single-quote doubling and LIKE metacharacter
    escaping with ESCAPE '\'). The label is treated as trusted input from validator
    code, not arbitrary user input.

    Important: This function NEVER passes silently when zero Perfetto slices
    match the label. A result with 0 matched slices always returns passed=False.

    Args:
        timemory_reader: TimemoryReader with a valid directory path.
        perfetto_reader: PerfettoReader with an active trace connection.
        label: Substring label to search for in timemory LABEL column and Perfetto names.
        tolerance_pct: Maximum allowed percentage difference between timemory SUM and
                       Perfetto duration sum. Default: 5.0%.

    Returns:
        CheckResult with:
          - passed=True if abs(pct_diff) <= tolerance_pct
          - passed=False with message if label not found in timemory
          - passed=False with message "No Perfetto slices matched label '{label}'"
            if zero Perfetto slices match (NEVER silently passes)
          - details containing: pct_diff, tolerance_pct, timemory_sum_sec,
            perfetto_dur_sec
    """
    tim_df: pd.DataFrame = timemory_reader._parse_file("wall_clock")

    # Guard: file absent or unparseable returns an empty column-less DataFrame
    if tim_df.empty or "LABEL" not in tim_df.columns:
        return CheckResult(
            passed=False,
            validator_name="assert_timemory_perfetto_correlation",
            message="wall_clock.txt could not be parsed or is absent",
            details={"label": label},
        )

    # Substring match against LABEL column
    matched_tim = tim_df[tim_df["LABEL"].str.contains(label, regex=False)]

    if matched_tim.empty:
        return CheckResult(
            passed=False,
            validator_name="assert_timemory_perfetto_correlation",
            message=f"Label '{label}' not found in timemory wall_clock output",
            details={"label": label},
        )

    # Warn when multiple rows match — only the first is used; caller should use a
    # more specific label if this is unexpected.
    if len(matched_tim) > 1:
        warnings.warn(
            f"assert_timemory_perfetto_correlation: label '{label}' matched "
            f"{len(matched_tim)} rows in timemory wall_clock output; using first match. "
            f"Matched labels: {matched_tim['LABEL'].tolist()}",
            stacklevel=2,
        )

    # SUM dtype is str — always cast to float before arithmetic (Pitfall 2)
    sum_sec = float(matched_tim["SUM"].iloc[0])

    # Build LIKE pattern with metacharacter escaping (T-04-01 threat mitigation)
    safe_label = (
        label
        .replace("'", "''")      # SQL single-quote escape
        .replace("%", "\\%")     # LIKE wildcard
        .replace("_", "\\_")     # LIKE single-char wildcard
    )
    sql = f"SELECT name, ts, dur FROM slice WHERE name LIKE '%{safe_label}%' ESCAPE '\\'"

    perf_df: pd.DataFrame = perfetto_reader.execute_sql(sql)

    # Guard: zero matches must never silently pass (per D-01 in CONTEXT.md)
    if perf_df.empty:
        return CheckResult(
            passed=False,
            validator_name="assert_timemory_perfetto_correlation",
            message=f"No Perfetto slices matched label '{label}'",
            details={"label": label, "timemory_sum_sec": sum_sec},
        )

    # Perfetto dur is in nanoseconds; divide by 1e9 for seconds conversion
    perfetto_dur_sec = float(perf_df["dur"].sum()) / 1_000_000_000

    abs_diff = abs(perfetto_dur_sec - sum_sec)
    pct_diff = (abs_diff / sum_sec) * 100 if sum_sec != 0 else float("inf")
    passed = bool(pct_diff <= tolerance_pct)

    message_verb = "PASSED" if passed else "FAILED"
    return CheckResult(
        passed=passed,
        validator_name="assert_timemory_perfetto_correlation",
        message=(
            f"[assert_timemory_perfetto_correlation] {message_verb}: "
            f"label='{label}', pct_diff={round(pct_diff, 4)}%, tolerance={tolerance_pct}%"
        ),
        expected=f"pct_diff <= {tolerance_pct}%",
        actual=f"pct_diff = {round(pct_diff, 4)}%",
        details={
            "pct_diff": round(pct_diff, 4),
            "tolerance_pct": tolerance_pct,
            "timemory_sum_sec": sum_sec,
            "perfetto_dur_sec": round(perfetto_dur_sec, 9),
        },
    )
