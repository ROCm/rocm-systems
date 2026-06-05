# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PerfettoReader — wraps TraceProcessor for single-file trace validation.

Implements all 15 PERF-* validator methods. Accepts a pre-created TraceProcessor
instance (or Mock) for unit testing without a real binary.

Design decisions:
- _query delegates to _query_to_dataframe from sanity.py (never re-implemented here)
- _find_tracks uses SQL LIKE (not GLOB) for bracket-containing track names
- close() only calls tp.close() when _owns_tp=True (ownership semantics)
- assert_flow_events_connect_tracks() skips gracefully when no flow events present

PITFALL: SQL GLOB treats [N] as character class — breaks 'GPU [0] GFX Busy (S)' queries.
CORRECT: Use LIKE '%pattern%' for SQL pre-filter; Python re.compile for regex matching.
"""
from __future__ import annotations

import math
import os
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from perfetto.trace_processor import TraceProcessor

from rocprofsys_validator.core import FormatReader, CheckResult
from rocprofsys_validator.registry import reader
from rocprofsys_validator.sanity import _query_to_dataframe, run_sanity_checks

# Wildcard sentinel for assert_slice_order: matches any run (>= 0) of slices.
# Aliased to Ellipsis so callers can write either ``...`` or ``ANYTHING``.
ANYTHING = Ellipsis

def _expand_slice_order_steps(steps: tuple) -> list[tuple[str, str | None]]:
    """Expand a slice-order step list into a flat token stream.

    Each step is one of:
    - ``"name"``            → one slice named ``name``
    - ``["name", count]``   → ``count`` consecutive slices named ``name``
    - ``...`` / ``ANYTHING``→ any run (>= 0) of slices (a wildcard gap)

    Returns tokens as ``("lit", name)`` / ``("star", None)``; consecutive
    wildcards are collapsed.
    """
    tokens: list[tuple[str, str | None]] = []
    for step in steps:
        if step is Ellipsis:
            if not (tokens and tokens[-1] == ("star", None)):
                tokens.append(("star", None))
        elif isinstance(step, str):
            tokens.append(("lit", step))
        elif isinstance(step, (list, tuple)) and len(step) == 2:
            name, count = step
            if not isinstance(name, str):
                raise ValueError(f"slice-order step name must be a string: {step!r}")
            count = int(count)
            if count < 0:
                raise ValueError(f"slice-order step count must be >= 0: {step!r}")
            tokens.extend(("lit", name) for _ in range(count))
        else:
            raise ValueError(
                f"invalid slice-order step {step!r}; expected a name, "
                "[name, count], or ... / ANYTHING"
            )
    return tokens

def _slice_name_match(actual: str | None, expected: str, match: str) -> bool:
    """Compare one slice name against an expected literal under the given match mode."""
    if actual is None:
        return False
    if match == "exact":
        return actual == expected
    if match == "substring":
        return expected in actual
    if match == "regex":
        return re.search(expected, actual) is not None
    raise ValueError(f"invalid match type: {match!r}")

def _match_slice_order(seq: list, tokens: list[tuple[str, str | None]], match: str) -> bool:
    """Full-sequence wildcard match (anchored both ends; ``star`` = any run).

    Standard linear wildcard matcher: literals must align consecutively; a
    ``star`` token absorbs any number of slices, with backtracking. The whole
    ``seq`` must be consumed, so callers add a leading/trailing ``...`` when they
    want slack at the start/end.
    """
    n, m = len(seq), len(tokens)
    i = j = 0
    star_j = -1
    star_i = 0
    while i < n:
        if j < m and tokens[j][0] == "lit" and _slice_name_match(seq[i], tokens[j][1], match):
            i += 1
            j += 1
        elif j < m and tokens[j][0] == "star":
            star_j = j
            star_i = i
            j += 1
        elif star_j != -1:
            j = star_j + 1
            star_i += 1
            i = star_i
        else:
            return False
    while j < m and tokens[j][0] == "star":
        j += 1
    return j == m

def _fmt_slice_order(steps: tuple) -> str:
    """Render a slice-order step list as a readable expectation string."""
    parts = []
    for s in steps:
        if s is Ellipsis:
            parts.append("...")
        elif isinstance(s, str):
            parts.append(f"{s}×1")
        elif isinstance(s, (list, tuple)) and len(s) == 2:
            parts.append(f"{s[0]}×{s[1]}")
        else:
            parts.append(repr(s))
    return " , ".join(parts)

# ---------------------------------------------------------------------------
# Pure interval / statistics helpers (timeline math, unit-tested directly)
# ---------------------------------------------------------------------------

def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge (start, dur) pairs into sorted, non-overlapping (start, end) ranges.

    Negative or None durations are ignored (they are caught by other checks).
    """
    iv = sorted((s, s + d) for s, d in intervals if d is not None and d >= 0)
    out: list[tuple[int, int]] = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out

def _interval_union_ns(intervals: list[tuple[int, int]]) -> int:
    """Total time covered by the union of the intervals (overlaps counted once)."""
    return sum(e - s for s, e in _merge_intervals(intervals))

def _interval_span_ns(intervals: list[tuple[int, int]]) -> int:
    """Wall-clock span from earliest start to latest end (0 if empty)."""
    merged = _merge_intervals(intervals)
    if not merged:
        return 0
    return merged[-1][1] - merged[0][0]

def _interval_intersect_ns(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    """Total time where the union of A overlaps the union of B."""
    A, B = _merge_intervals(a), _merge_intervals(b)
    i = j = 0
    total = 0
    while i < len(A) and j < len(B):
        lo = max(A[i][0], B[j][0])
        hi = min(A[i][1], B[j][1])
        if hi > lo:
            total += hi - lo
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1
    return total

def _max_gap_ns(intervals: list[tuple[int, int]]) -> int:
    """Largest idle gap between consecutive merged intervals (0 if < 2)."""
    merged = _merge_intervals(intervals)
    return max((merged[k][0] - merged[k - 1][1] for k in range(1, len(merged))), default=0)

def _count_overlapping(intervals: list[tuple[int, int]]) -> int:
    """Number of intervals that start before the running max-end of earlier ones."""
    iv = sorted((s, s + d) for s, d in intervals if d is not None and d >= 0)
    violations = 0
    prev_end: int | None = None
    for s, e in iv:
        if prev_end is not None and s < prev_end:
            violations += 1
        prev_end = e if prev_end is None else max(prev_end, e)
    return violations

def _percentile(values: list[float], pct: float) -> float | None:
    """Linear-interpolation percentile (pct in [0, 100]); None for empty input."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * (pct / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] + (xs[hi] - xs[lo]) * (k - lo))

def _mean_std(values: list[float]) -> tuple[float, float]:
    """Population mean and standard deviation (0.0, 0.0 for empty input)."""
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, math.sqrt(var)

@reader("perfetto")
class PerfettoReader(FormatReader):
    """Validates a Perfetto trace file using TraceProcessor SQL queries.

    Accepts either a file path (str/Path) or a pre-created TraceProcessor instance
    (including Mock objects for unit tests).

    Usage::

        # With a real trace file
        with PerfettoReader("/path/to/trace.proto") as r:
            r.assert_track_exists("HIP API")
            results = r.validate()

        # With a Mock for unit tests
        from unittest.mock import Mock
        reader = PerfettoReader(Mock())
    """

    def __init__(
        self,
        trace: Any,  # str | Path | TraceProcessor | Mock
        tp_bin: str | None = None,
    ) -> None:
        """Initialize PerfettoReader.

        Args:
            trace: File path (str/Path) → creates own TraceProcessor (_owns_tp=True).
                   Any other object (TraceProcessor, Mock) → used as-is (_owns_tp=False).
            tp_bin: Optional path to trace_processor_shell binary. Falls back to
                    ROCPROFSYS_TRACE_PROCESSOR_SHELL env var.
        """
        self._owns_tp = False
        if isinstance(trace, (str, Path)):
            from perfetto.trace_processor import TraceProcessor, TraceProcessorConfig
            bin_path = tp_bin or os.environ.get("ROCPROFSYS_TRACE_PROCESSOR_SHELL")
            # bin_path=None is intentional when neither tp_bin nor the env var is set.
            # The perfetto package includes a bundled trace_processor_shell binary that
            # TraceProcessor will use automatically — no explicit path required.
            cfg = TraceProcessorConfig(bin_path=bin_path, load_timeout=30)
            self._tp = TraceProcessor(trace=str(trace), config=cfg)
            self._owns_tp = True
            # Run the post-load data-integrity guard (negative timestamps,
            # ring-buffer sequence failures) so standalone PerfettoReader use
            # gets the same protection as the pytest fixture. Close the
            # subprocess if the trace is malformed to avoid leaking it.
            try:
                run_sanity_checks(self._tp)
            except BaseException:
                self._tp.close()
                raise
        else:
            self._tp = trace  # pre-created TP or Mock for unit tests
        self._results: list[CheckResult] = []
        self._path = str(trace) if isinstance(trace, (str, Path)) else "<mock>"

    def close(self) -> None:
        """Release the TraceProcessor subprocess — only when we created it."""
        if self._owns_tp and self._tp is not None:
            self._tp.close()

    def validate(self) -> list[CheckResult]:
        """Return all accumulated CheckResult objects."""
        return list(self._results)

    # ---------------------------------------------------------------------------
    # Internal query helpers
    # ---------------------------------------------------------------------------

    def _query(self, sql: str) -> pd.DataFrame:
        """Execute SQL and return a DataFrame. Delegates to _query_to_dataframe."""
        return _query_to_dataframe(self._tp, sql)

    def _escape_like(self, pattern: str) -> str:
        """Escape a pattern for safe interpolation into a single-quoted SQL LIKE clause.

        Escapes the LIKE wildcards ``%`` and ``_`` (paired with ``ESCAPE '\\'`` at
        the call site) and doubles single quotes so a pattern containing an
        apostrophe (e.g. ``O'Brien``) cannot terminate the string literal and
        cause a SQL syntax error. The three escaped character sets are disjoint,
        so escaping order does not matter.
        """
        return (
            pattern.replace("%", "\\%").replace("_", "\\_").replace("'", "''")
        )

    def _sql_in_list(self, names: list[str]) -> str:
        """Format names as a quoted, comma-separated SQL IN list (single quotes doubled).

        The names come from the trace's own ``track`` table, so exact equality
        via ``IN (...)`` is correct; quote-doubling keeps names containing an
        apostrophe from breaking the literal.
        """
        return ", ".join("'" + n.replace("'", "''") + "'" for n in names)

    def _find_tracks(self, pattern: str, match_type: str = "exact") -> list[str]:
        """Return matching track names from the trace.

        CRITICAL: Never use SQL GLOB — square brackets in GPU track names
        ('GPU [0] GFX Busy (S)') are treated as character classes by GLOB.
        Always use LIKE for SQL pre-filtering.

        Args:
            pattern: Track name pattern.
            match_type: 'exact', 'substring', or 'regex'.

        Returns:
            List of matched track name strings (None values excluded).
        """
        if match_type == "exact":
            safe = pattern.replace("'", "''")
            df = self._query(
                f"SELECT name FROM track WHERE name = '{safe}' AND name IS NOT NULL"
            )
        elif match_type == "substring":
            safe = self._escape_like(pattern)
            df = self._query(
                f"SELECT name FROM track WHERE name LIKE '%{safe}%' ESCAPE '\\' AND name IS NOT NULL"
            )
        else:  # regex
            df = self._query("SELECT DISTINCT name FROM track WHERE name IS NOT NULL")
            compiled = re.compile(pattern)
            df = df[df["name"].apply(lambda n: bool(compiled.search(str(n) if n else "")))]
        return [n for n in df["name"].tolist() if n is not None]

    def _find_counter_tracks(self, pattern: str, match_type: str = "substring") -> list[str]:
        """Return matching counter track names from the counter_track table.

        For GPU patterns like 'GPU [0] GFX Busy (S)', uses LIKE 'GPU % (S)' as
        the SQL pre-filter (GLOB would break on the square brackets).

        Args:
            pattern: Counter track name pattern.
            match_type: 'exact', 'substring', or 'regex'.

        Returns:
            List of matched counter track name strings.
        """
        if match_type == "exact":
            safe = pattern.replace("'", "''")
            df = self._query(
                f"SELECT name FROM counter_track WHERE name = '{safe}' AND name IS NOT NULL"
            )
        elif match_type == "substring":
            safe = self._escape_like(pattern)
            df = self._query(
                f"SELECT name FROM counter_track WHERE name LIKE '%{safe}%' ESCAPE '\\' AND name IS NOT NULL"
            )
        else:  # regex
            df = self._query("SELECT DISTINCT name FROM counter_track WHERE name IS NOT NULL")
            compiled = re.compile(pattern)
            df = df[df["name"].apply(lambda n: bool(compiled.search(str(n) if n else "")))]
        return [n for n in df["name"].tolist() if n is not None]

    # ---------------------------------------------------------------------------
    # PERF-01: Assert named track exists
    # ---------------------------------------------------------------------------

    def assert_track_exists(self, pattern: str, match: str = "exact") -> CheckResult:
        """Assert at least one track matching the pattern exists in the trace.

        Args:
            pattern: Track name to search for.
            match: 'exact' (default), 'substring', or 'regex'.

        Returns:
            CheckResult with passed=True if any matching track found.
        """
        found = self._find_tracks(pattern, match_type=match)
        if found:
            result = CheckResult(
                passed=True,
                validator_name="assert_track_exists",
                message=f"Track matching {pattern!r} found: {found[0]!r}",
                details={"match_type": match, "trace": self._path},
            )
        else:
            result = CheckResult(
                passed=False,
                validator_name="assert_track_exists",
                message=f"No track matching {pattern!r} found",
                expected=pattern,
                actual="(none)",
                details={"match_type": match, "trace": self._path},
            )
        self._results.append(result)
        return result

    def assert_track_absent(self, pattern: str, match: str = "exact") -> CheckResult:
        """Assert that NO track matching the pattern exists (negative test).

        The explicit complement of assert_track_exists — for asserting, e.g.,
        that a disabled-feature run contains no GPU track.

        Args:
            pattern: Track name pattern.
            match: 'exact' (default), 'substring', or 'regex'.

        Returns:
            CheckResult with passed=True when zero matching tracks are found.
        """
        found = self._find_tracks(pattern, match_type=match)
        passed = not found
        result = CheckResult(
            passed=passed,
            validator_name="assert_track_absent",
            message=(
                f"No track matching {pattern!r} present (as expected)"
                if passed
                else f"Track matching {pattern!r} unexpectedly present: {found[0]!r}"
            ),
            expected=f"no track matching {pattern!r}" if not passed else None,
            actual=found[0] if not passed else None,
            details={"match_type": match, "trace": self._path, "found": found},
        )
        self._results.append(result)
        return result

    # ---------------------------------------------------------------------------
    # PERF-02: Assert process/thread track exists
    # ---------------------------------------------------------------------------

    def assert_process_track_exists(
        self, name_pattern: str, match: str = "substring"
    ) -> CheckResult:
        """Assert at least one process track with the given process name pattern exists.

        Args:
            name_pattern: Process name to search for.
            match: 'exact', 'substring' (default), or 'regex'.

        Returns:
            CheckResult with passed=True if any matching process track found.
        """
        df = self._query(
            "SELECT pt.id, p.name FROM process_track pt "
            "JOIN process p ON pt.upid = p.upid "
            "WHERE p.name IS NOT NULL"
        )
        names = [n for n in df["name"].tolist() if n is not None] if not df.empty else []
        if match == "exact":
            matched = [n for n in names if n == name_pattern]
        elif match == "substring":
            matched = [n for n in names if name_pattern in str(n)]
        else:  # regex
            compiled = re.compile(name_pattern)
            matched = [n for n in names if compiled.search(str(n) if n else "")]

        passed = bool(matched)
        result = CheckResult(
            passed=passed,
            validator_name="assert_process_track_exists",
            message=(
                f"Process track for {name_pattern!r} found: {matched[0]!r}"
                if passed
                else f"No process track matching {name_pattern!r} found"
            ),
            expected=name_pattern if not passed else None,
            actual="(none)" if not passed else None,
            details={"match_type": match, "trace": self._path},
        )
        self._results.append(result)
        return result

    def assert_thread_track_exists(
        self, name_pattern: str, match: str = "substring"
    ) -> CheckResult:
        """Assert at least one thread track with the given thread name pattern exists.

        Args:
            name_pattern: Thread name to search for.
            match: 'exact', 'substring' (default), or 'regex'.

        Returns:
            CheckResult with passed=True if any matching thread track found.
        """
        df = self._query(
            "SELECT tt.id, t.name FROM thread_track tt "
            "JOIN thread t ON tt.utid = t.utid "
            "WHERE t.name IS NOT NULL"
        )
        names = [n for n in df["name"].tolist() if n is not None] if not df.empty else []
        if match == "exact":
            matched = [n for n in names if n == name_pattern]
        elif match == "substring":
            matched = [n for n in names if name_pattern in str(n)]
        else:  # regex
            compiled = re.compile(name_pattern)
            matched = [n for n in names if compiled.search(str(n) if n else "")]

        passed = bool(matched)
        result = CheckResult(
            passed=passed,
            validator_name="assert_thread_track_exists",
            message=(
                f"Thread track for {name_pattern!r} found: {matched[0]!r}"
                if passed
                else f"No thread track matching {name_pattern!r} found"
            ),
            expected=name_pattern if not passed else None,
            actual="(none)" if not passed else None,
            details={"match_type": match, "trace": self._path},
        )
        self._results.append(result)
        return result

    # ---------------------------------------------------------------------------
    # PERF-03: Assert slices exist
    # ---------------------------------------------------------------------------

    def _build_slice_query(
        self,
        track_pattern: str | None,
        category: str | None,
        slice_name_pattern: str | None,
    ) -> str:
        """Build a COUNT(*) query over slices filtered by any of track/category/name.

        The track join is only added when a track pattern is supplied, so a
        category-only count (the production per-kind pattern) does not require a
        track. All three filters are AND-combined.
        """
        join = ""
        conds: list[str] = []
        if track_pattern is not None:
            safe_track = self._escape_like(track_pattern)
            join = "JOIN track t ON s.track_id = t.id "
            conds.append(
                f"t.name LIKE '%{safe_track}%' ESCAPE '\\' AND t.name IS NOT NULL"
            )
        if category is not None:
            safe_cat = category.replace("'", "''")
            conds.append(f"s.category = '{safe_cat}'")
        if slice_name_pattern is not None:
            safe_slice = self._escape_like(slice_name_pattern)
            conds.append(f"s.name LIKE '%{safe_slice}%' ESCAPE '\\'")
        where = (" WHERE " + " AND ".join(conds)) if conds else ""
        return f"SELECT COUNT(*) AS cnt FROM slice s {join}{where}"

    @staticmethod
    def _slice_filter_desc(
        track_pattern: str | None, category: str | None
    ) -> str:
        """Human-readable description of the active slice filter for messages."""
        parts = []
        if track_pattern is not None:
            parts.append(f"track matching {track_pattern!r}")
        if category is not None:
            parts.append(f"category {category!r}")
        return " and ".join(parts) if parts else "any slice"

    def assert_slices_exist(
        self,
        track_pattern: str | None = None,
        slice_name_pattern: str | None = None,
        category: str | None = None,
    ) -> CheckResult:
        """Assert at least one slice exists matching track, category, and/or name.

        Args:
            track_pattern: Optional track name pattern (substring match).
            slice_name_pattern: Optional slice name pattern (substring match).
            category: Optional slice category (exact match) — lets you assert,
                      e.g., that any 'kernel_dispatch' slice exists.

        Returns:
            CheckResult with passed=True if slices found.
        """
        df = self._query(
            self._build_slice_query(track_pattern, category, slice_name_pattern)
        )
        count = int(df["cnt"].iloc[0]) if not df.empty else 0
        passed = count > 0
        desc = self._slice_filter_desc(track_pattern, category)

        result = CheckResult(
            passed=passed,
            validator_name="assert_slices_exist",
            message=(
                f"Found {count} slice(s) on {desc}"
                if passed
                else f"No slices found on {desc}"
            ),
            expected=f">= 1 slice on {desc}" if not passed else None,
            actual=count if not passed else None,
            details={
                "track_pattern": track_pattern,
                "category": category,
                "slice_name_pattern": slice_name_pattern,
            },
        )
        self._results.append(result)
        return result

    # ---------------------------------------------------------------------------
    # PERF-04: Assert slice count within range
    # ---------------------------------------------------------------------------

    def assert_slice_count(
        self,
        track_pattern: str | None = None,
        min_count: int | None = None,
        max_count: int | None = None,
        category: str | None = None,
    ) -> CheckResult:
        """Assert the number of slices is within a range, filtered by track and/or category.

        Pass ``category`` (with ``track_pattern=None``) to count slices of a
        given kind regardless of which track they live on — the production
        per-category correctness pattern (e.g. "exactly N kernel_dispatch slices").

        Args:
            track_pattern: Optional track name pattern (substring match).
            min_count: Minimum expected slice count (inclusive). None = no lower bound.
            max_count: Maximum expected slice count (inclusive). None = no upper bound.
            category: Optional slice category (exact match).

        Returns:
            CheckResult with details containing the actual count.
        """
        df = self._query(self._build_slice_query(track_pattern, category, None))
        count = int(df["cnt"].iloc[0]) if not df.empty else 0

        passed = (min_count is None or count >= min_count) and (
            max_count is None or count <= max_count
        )

        expected_parts = []
        if min_count is not None:
            expected_parts.append(f">= {min_count}")
        if max_count is not None:
            expected_parts.append(f"<= {max_count}")
        expected_str = " and ".join(expected_parts) if expected_parts else "any count"
        desc = self._slice_filter_desc(track_pattern, category)

        result = CheckResult(
            passed=passed,
            validator_name="assert_slice_count",
            message=(
                f"Slice count on {desc}: {count} (within range)"
                if passed
                else f"Slice count on {desc}: {count} (out of range)"
            ),
            expected=expected_str if not passed else None,
            actual=count if not passed else None,
            details={
                "track_pattern": track_pattern,
                "category": category,
                "count": count,
            },
        )
        self._results.append(result)
        return result

    def assert_slice_category_count(
        self,
        category: str,
        min_count: int | None = None,
        max_count: int | None = None,
    ) -> CheckResult:
        """Assert the number of slices of a given category is within a range.

        Category-first convenience over assert_slice_count — this is how
        rocprof-sys/rocprofiler output correctness is defined (per-category
        record counts: kernel_dispatch, memory_copy, hip_api, ...).

        Args:
            category: Slice category (exact match).
            min_count: Minimum expected count (inclusive). None = no lower bound.
            max_count: Maximum expected count (inclusive). None = no upper bound.

        Returns:
            CheckResult with the actual count in details.
        """
        df = self._query(self._build_slice_query(None, category, None))
        count = int(df["cnt"].iloc[0]) if not df.empty else 0
        passed = (min_count is None or count >= min_count) and (
            max_count is None or count <= max_count
        )

        expected_parts = []
        if min_count is not None:
            expected_parts.append(f">= {min_count}")
        if max_count is not None:
            expected_parts.append(f"<= {max_count}")
        expected_str = " and ".join(expected_parts) if expected_parts else "any count"

        result = CheckResult(
            passed=passed,
            validator_name="assert_slice_category_count",
            message=(
                f"Slice count for category {category!r}: {count} (within range)"
                if passed
                else f"Slice count for category {category!r}: {count} (out of range)"
            ),
            expected=expected_str if not passed else None,
            actual=count if not passed else None,
            details={"category": category, "count": count},
        )
        self._results.append(result)
        return result

    # ---------------------------------------------------------------------------
    # PERF-05: Assert maximum nesting depth
    # ---------------------------------------------------------------------------

    def assert_max_nesting_depth(
        self, track_pattern: str, max_depth: int
    ) -> CheckResult:
        """Assert the maximum slice nesting depth on a track does not exceed max_depth.

        Args:
            track_pattern: Track name pattern (substring match).
            max_depth: Maximum allowed nesting depth.

        Returns:
            CheckResult with actual depth in details.
        """
        safe_track = self._escape_like(track_pattern)
        df = self._query(
            "SELECT MAX(s.depth) AS d FROM slice s "
            "JOIN track t ON s.track_id = t.id "
            f"WHERE t.name LIKE '%{safe_track}%' ESCAPE '\\' AND t.name IS NOT NULL"
        )
        actual_depth = (
            int(df["d"].iloc[0]) if not df.empty and df["d"].iloc[0] is not None else 0
        )
        passed = actual_depth <= max_depth

        result = CheckResult(
            passed=passed,
            validator_name="assert_max_nesting_depth",
            message=(
                f"Max nesting depth on {track_pattern!r}: {actual_depth} (<= {max_depth})"
                if passed
                else f"Max nesting depth on {track_pattern!r}: {actual_depth} exceeds {max_depth}"
            ),
            expected=f"<= {max_depth}" if not passed else None,
            actual=actual_depth if not passed else None,
            details={"track_pattern": track_pattern, "max_depth": max_depth},
        )
        self._results.append(result)
        return result

    # ---------------------------------------------------------------------------
    # PERF-06: Assert non-overlapping slices
    # ---------------------------------------------------------------------------

    def assert_non_overlapping_slices(self, track_pattern: str) -> CheckResult:
        """Assert slices on the matching track do not overlap in time.

        Overlap detection is done in Python after fetching sorted (ts, dur) pairs.
        A slice at ts[i] overlaps with ts[i+1] when ts[i+1] < ts[i] + dur[i].

        Args:
            track_pattern: Track name pattern (substring match).

        Returns:
            CheckResult with overlap details when failed.
        """
        safe_track = self._escape_like(track_pattern)
        df = self._query(
            "SELECT s.ts, s.dur, s.track_id FROM slice s "
            "JOIN track t ON s.track_id = t.id "
            f"WHERE t.name LIKE '%{safe_track}%' ESCAPE '\\' AND t.name IS NOT NULL "
            "ORDER BY s.track_id, s.ts"
        )

        overlaps: list[dict[str, Any]] = []
        if not df.empty and len(df) > 1:
            for _, group in df.groupby("track_id"):
                ts_vals = group["ts"].tolist()
                dur_vals = group["dur"].tolist()
                for i in range(len(ts_vals) - 1):
                    if ts_vals[i + 1] < ts_vals[i] + dur_vals[i]:
                        overlaps.append(
                            {
                                "slice_a_ts": ts_vals[i],
                                "slice_a_end": ts_vals[i] + dur_vals[i],
                                "slice_b_ts": ts_vals[i + 1],
                            }
                        )

        passed = not overlaps
        result = CheckResult(
            passed=passed,
            validator_name="assert_non_overlapping_slices",
            message=(
                f"No overlapping slices on {track_pattern!r}"
                if passed
                else f"Found {len(overlaps)} overlap(s) on {track_pattern!r}"
            ),
            expected="no overlapping slices" if not passed else None,
            actual=f"{len(overlaps)} overlap(s)" if not passed else None,
            details={"track_pattern": track_pattern, "overlaps": overlaps[:5]},  # first 5
        )
        self._results.append(result)
        return result

    # ---------------------------------------------------------------------------
    # PERF-07: Assert counter track exists
    # ---------------------------------------------------------------------------

    def assert_counter_track_exists(
        self, track_pattern: str, match: str = "substring"
    ) -> CheckResult:
        """Assert at least one counter track matching the pattern exists.

        For GPU patterns like 'GPU [0] GFX Busy (S)', uses LIKE 'GPU % (S)' for
        SQL pre-filtering (GLOB would fail on the square brackets).

        Args:
            track_pattern: Counter track name pattern.
            match: 'exact', 'substring' (default), or 'regex'.

        Returns:
            CheckResult with passed=True if any matching counter track found.
        """
        found = self._find_counter_tracks(track_pattern, match_type=match)
        passed = bool(found)

        result = CheckResult(
            passed=passed,
            validator_name="assert_counter_track_exists",
            message=(
                f"Counter track matching {track_pattern!r} found: {found[0]!r}"
                if passed
                else f"No counter track matching {track_pattern!r} found"
            ),
            expected=track_pattern if not passed else None,
            actual="(none)" if not passed else None,
            details={"match_type": match, "trace": self._path},
        )
        self._results.append(result)
        return result

    # ---------------------------------------------------------------------------
    # PERF-08: Assert counter aggregate within range
    # ---------------------------------------------------------------------------

    def assert_counter_aggregate(
        self,
        track_pattern: str,
        metric: str,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> CheckResult:
        """Assert a counter track aggregate (sum/max/min) is within the given range.

        Args:
            track_pattern: Counter track name pattern (substring match).
            metric: Aggregation type — 'sum', 'max', or 'min'.
            min_val: Minimum expected value (inclusive). None = no lower bound.
            max_val: Maximum expected value (inclusive). None = no upper bound.

        Returns:
            CheckResult with the actual aggregate value in details.
        """
        if metric not in ("sum", "max", "min"):
            result = CheckResult(
                passed=False,
                validator_name="assert_counter_aggregate",
                message=f"Unknown metric {metric!r} — must be 'sum', 'max', or 'min'",
                expected="sum, max, or min",
                actual=metric,
            )
            self._results.append(result)
            return result

        agg_func = metric.upper()
        safe_track = self._escape_like(track_pattern)
        df = self._query(
            f"SELECT {agg_func}(c.value) AS result FROM counter c "
            "JOIN counter_track ct ON c.track_id = ct.id "
            f"WHERE ct.name LIKE '%{safe_track}%' ESCAPE '\\' AND ct.name IS NOT NULL"
        )

        actual = float(df["result"].iloc[0]) if not df.empty and df["result"].iloc[0] is not None else None

        if actual is None:
            result = CheckResult(
                passed=False,
                validator_name="assert_counter_aggregate",
                message=f"No counter data found for track matching {track_pattern!r}",
                expected=f"{metric} value in range",
                actual=None,
                details={"track_pattern": track_pattern, "metric": metric},
            )
            self._results.append(result)
            return result

        passed = (min_val is None or actual >= min_val) and (max_val is None or actual <= max_val)

        expected_parts = []
        if min_val is not None:
            expected_parts.append(f">= {min_val}")
        if max_val is not None:
            expected_parts.append(f"<= {max_val}")
        expected_str = " and ".join(expected_parts) if expected_parts else "any value"

        result = CheckResult(
            passed=passed,
            validator_name="assert_counter_aggregate",
            message=(
                f"Counter {metric} for {track_pattern!r}: {actual} (within range)"
                if passed
                else f"Counter {metric} for {track_pattern!r}: {actual} (out of range)"
            ),
            expected=expected_str if not passed else None,
            actual=actual if not passed else None,
            details={"track_pattern": track_pattern, "metric": metric, "value": actual},
        )
        self._results.append(result)
        return result

    # ---------------------------------------------------------------------------
    # PERF-09: Assert counter monotonicity
    # ---------------------------------------------------------------------------

    def assert_counter_monotonic(
        self, track_pattern: str, direction: str = "increasing"
    ) -> CheckResult:
        """Assert counter values on a track are monotonically increasing or decreasing.

        Comparison is done in Python after fetching (ts, value) pairs ordered by ts.

        Args:
            track_pattern: Counter track name pattern (substring match).
            direction: 'increasing' (default) or 'decreasing'.

        Returns:
            CheckResult with violation details when failed.
        """
        safe_track = self._escape_like(track_pattern)
        df = self._query(
            "SELECT c.ts, c.value, c.track_id FROM counter c "
            "JOIN counter_track ct ON c.track_id = ct.id "
            f"WHERE ct.name LIKE '%{safe_track}%' ESCAPE '\\' AND ct.name IS NOT NULL "
            "ORDER BY c.track_id, c.ts"
        )

        if df.empty or len(df) < 2:
            result = CheckResult(
                passed=True,
                validator_name="assert_counter_monotonic",
                message=f"Insufficient data on {track_pattern!r} for monotonicity check (< 2 samples)",
                details={"track_pattern": track_pattern, "direction": direction},
            )
            self._results.append(result)
            return result

        violations: list[dict[str, Any]] = []

        for _, group in df.groupby("track_id"):
            values = group["value"].tolist()
            for i in range(len(values) - 1):
                if direction == "increasing":
                    if values[i + 1] < values[i]:
                        violations.append({"index": i, "prev": values[i], "next": values[i + 1]})
                else:  # decreasing
                    if values[i + 1] > values[i]:
                        violations.append({"index": i, "prev": values[i], "next": values[i + 1]})

        passed = not violations
        result = CheckResult(
            passed=passed,
            validator_name="assert_counter_monotonic",
            message=(
                f"Counter on {track_pattern!r} is monotonically {direction}"
                if passed
                else f"Counter on {track_pattern!r} has {len(violations)} monotonicity violation(s)"
            ),
            expected=f"monotonically {direction}" if not passed else None,
            actual=f"{len(violations)} violation(s)" if not passed else None,
            details={
                "track_pattern": track_pattern,
                "direction": direction,
                "violations": violations[:5],
            },
        )
        self._results.append(result)
        return result

    # ---------------------------------------------------------------------------
    # PERF-10: Assert debug annotations on slices
    # ---------------------------------------------------------------------------

    def assert_debug_annotations(
        self,
        track_pattern: str,
        slice_name_pattern: str,
        expected_keys: list[str],
    ) -> CheckResult:
        """Assert that expected debug annotation keys are present on matching slices.

        Args:
            track_pattern: Track name pattern (substring match).
            slice_name_pattern: Slice name pattern (substring match).
            expected_keys: List of annotation keys expected to be present.

        Returns:
            CheckResult with missing_keys and found_keys in details.
        """
        safe_track = self._escape_like(track_pattern)
        safe_slice = self._escape_like(slice_name_pattern)
        df = self._query(
            "SELECT a.key FROM args a "
            "JOIN slice s ON a.arg_set_id = s.arg_set_id "
            "JOIN track t ON s.track_id = t.id "
            f"WHERE t.name LIKE '%{safe_track}%' ESCAPE '\\' "
            f"AND s.name LIKE '%{safe_slice}%' ESCAPE '\\'"
        )

        found_keys = set(df["key"].tolist()) if not df.empty else set()
        missing = [k for k in expected_keys if k not in found_keys]
        passed = not missing

        result = CheckResult(
            passed=passed,
            validator_name="assert_debug_annotations",
            message=(
                f"All {len(expected_keys)} expected annotation key(s) found on {slice_name_pattern!r}"
                if passed
                else f"Missing {len(missing)} annotation key(s) on {slice_name_pattern!r}: {missing}"
            ),
            expected=expected_keys if not passed else None,
            actual=sorted(found_keys) if not passed else None,
            details={
                "track_pattern": track_pattern,
                "slice_name_pattern": slice_name_pattern,
                "missing_keys": missing,
                "found_keys": sorted(found_keys),
            },
        )
        self._results.append(result)
        return result

    # ---------------------------------------------------------------------------
    # PERF-11: Assert flow events (conditional skip)
    # ---------------------------------------------------------------------------

    def assert_flow_events_connect_tracks(
        self,
        *,
        require: bool = False,
        min_count: int = 1,
        cross_track: bool = False,
    ) -> CheckResult:
        """Assert flow events are present in the trace.

        By default (``require=False``) this is a soft probe: a missing/empty
        ``flow`` table emits a warning and returns ``passed=True`` so older traces
        without flow data don't fail a suite. Set ``require=True`` to turn absence
        (or fewer than ``min_count`` events) into a real failure.

        Args:
            require: When True, fewer than ``min_count`` flow events — or a
                     missing/unqueryable ``flow`` table — is a hard failure
                     (``passed=False``) instead of a warned skip.
            min_count: Minimum number of flow events required (used when
                       ``require=True``). Default 1.
            cross_track: When True, count only flow events whose two endpoints
                         lie on *different* tracks — i.e. a genuine
                         "connects tracks" check rather than mere presence.

        Returns:
            CheckResult. ``passed=False`` only when ``require=True`` and the
            requirement is unmet; otherwise ``passed=True`` (with a skip message
            and warning when flow data is absent and ``require=False``).
        """
        if cross_track:
            sql = (
                "SELECT COUNT(*) AS cnt FROM flow f "
                "JOIN slice so ON f.slice_out = so.id "
                "JOIN slice si ON f.slice_in = si.id "
                "WHERE so.track_id != si.track_id"
            )
            kind = "cross-track flow events"
        else:
            sql = "SELECT COUNT(*) AS cnt FROM flow"
            kind = "flow events"

        try:
            df = self._query(sql)
            count = int(df["cnt"].iloc[0]) if not df.empty else 0
        except Exception as exc:
            # flow table may not exist in older trace formats
            if require:
                result = CheckResult(
                    passed=False,
                    validator_name="assert_flow_events_connect_tracks",
                    message=f"flow table absent or not queryable; {min_count}+ {kind} required",
                    expected=f">= {min_count} {kind}",
                    actual="(flow table absent)",
                    details={"reason": "flow table absent", "error": str(exc),
                             "cross_track": cross_track},
                )
                self._results.append(result)
                return result
            warnings.warn(
                "Flow table is absent or not queryable in this trace — skipping flow assertion.",
                stacklevel=2,
            )
            result = CheckResult(
                passed=True,
                validator_name="assert_flow_events_connect_tracks",
                message="No flow events found, skipping",
                details={"reason": "flow table absent", "error": str(exc)},
            )
            self._results.append(result)
            return result

        if count >= min_count:
            result = CheckResult(
                passed=True,
                validator_name="assert_flow_events_connect_tracks",
                message=f"{kind.capitalize()} present: {count}",
                details={"flow_count": count, "cross_track": cross_track},
            )
        elif require:
            result = CheckResult(
                passed=False,
                validator_name="assert_flow_events_connect_tracks",
                message=f"expected >= {min_count} {kind}, found {count}",
                expected=f">= {min_count} {kind}",
                actual=count,
                details={"flow_count": count, "cross_track": cross_track},
            )
        else:
            warnings.warn(
                f"No {kind} found in trace — skipping flow assertion.",
                stacklevel=2,
            )
            result = CheckResult(
                passed=True,
                validator_name="assert_flow_events_connect_tracks",
                message="No flow events found, skipping",
                details={"flow_count": count, "cross_track": cross_track},
            )
        self._results.append(result)
        return result

    def assert_flow_between_tracks(
        self,
        from_track: str,
        to_track: str,
        *,
        match: str = "substring",
        min_count: int = 1,
        directional: bool = True,
    ) -> CheckResult:
        """Assert flow events link a source track to a destination track.

        Resolves both track-name patterns to concrete tracks (same matching as
        ``assert_track_exists``), then counts flow events whose source slice
        (``slice_out``) sits on a ``from_track`` and whose destination slice
        (``slice_in``) sits on a ``to_track``.

        Args:
            from_track: Source track name pattern.
            to_track: Destination track name pattern.
            match: 'exact', 'substring' (default), or 'regex' — how the patterns
                   match against track names.
            min_count: Minimum number of qualifying flow events required.
            directional: When True (default) only count flows from a from_track
                         to a to_track; when False, also count the reverse
                         direction (``from <-> to``).

        Returns:
            CheckResult: passed=True if at least ``min_count`` qualifying flow
            events exist. passed=False if either pattern matches no track, the
            flow table is absent, or the count is below ``min_count``.
        """
        from_names = self._find_tracks(from_track, match_type=match)
        to_names = self._find_tracks(to_track, match_type=match)

        if not from_names or not to_names:
            missing = []
            if not from_names:
                missing.append(f"source {from_track!r}")
            if not to_names:
                missing.append(f"destination {to_track!r}")
            result = CheckResult(
                passed=False,
                validator_name="assert_flow_between_tracks",
                message=f"no track matched {' and '.join(missing)}",
                expected=f">= {min_count} flow events {from_track!r} -> {to_track!r}",
                actual="(track not found)",
                details={"from_track": from_track, "to_track": to_track,
                         "from_matched": from_names, "to_matched": to_names,
                         "trace": self._path},
            )
            self._results.append(result)
            return result

        from_in = self._sql_in_list(from_names)
        to_in = self._sql_in_list(to_names)
        forward = f"(t_out.name IN ({from_in}) AND t_in.name IN ({to_in}))"
        where = forward if directional else (
            f"{forward} OR (t_out.name IN ({to_in}) AND t_in.name IN ({from_in}))"
        )
        sql = (
            "SELECT COUNT(*) AS cnt FROM flow f "
            "JOIN slice so ON f.slice_out = so.id "
            "JOIN slice si ON f.slice_in = si.id "
            "JOIN track t_out ON so.track_id = t_out.id "
            "JOIN track t_in ON si.track_id = t_in.id "
            f"WHERE {where}"
        )

        try:
            df = self._query(sql)
            count = int(df["cnt"].iloc[0]) if not df.empty else 0
        except Exception as exc:
            result = CheckResult(
                passed=False,
                validator_name="assert_flow_between_tracks",
                message=f"flow table absent or not queryable: {exc}",
                expected=f">= {min_count} flow events {from_track!r} -> {to_track!r}",
                actual="(flow table absent)",
                details={"from_track": from_track, "to_track": to_track,
                         "error": str(exc), "trace": self._path},
            )
            self._results.append(result)
            return result

        arrow = "->" if directional else "<->"
        passed = count >= min_count
        result = CheckResult(
            passed=passed,
            validator_name="assert_flow_between_tracks",
            message=(
                f"{count} flow event(s) {from_track!r} {arrow} {to_track!r}"
                if passed
                else f"expected >= {min_count} flow event(s) "
                     f"{from_track!r} {arrow} {to_track!r}, found {count}"
            ),
            expected=None if passed else f">= {min_count}",
            actual=count,
            details={"from_track": from_track, "to_track": to_track,
                     "from_matched": from_names, "to_matched": to_names,
                     "flow_count": count, "directional": directional,
                     "trace": self._path},
        )
        self._results.append(result)
        return result

    def assert_slice_order(
        self,
        track_pattern: str,
        *steps: Any,
        match: str = "exact",
        track_match: str = "substring",
        depth: int | None = None,
    ) -> CheckResult:
        """Assert slices on a track occur in a given chronological order.

        The ``steps`` describe the **full** ordered sequence of slices on the
        track (ordered by timestamp). Each step is one of:

        - ``"name"``             — exactly one slice named ``name``
        - ``["name", count]``    — ``count`` consecutive slices named ``name``
        - ``...`` / ``ANYTHING`` — any run (>= 0) of slices you don't care about

        Because the match is anchored at both ends, put ``...`` wherever
        arbitrary slices may appear — including the very start or end.

        Example::

            r.assert_slice_order(
                "HIP",
                ["hipGetDevice", 1],
                ["hipSetDevice", 1],
                ...,                       # don't care in between
                ["transpose", 1000],
            )

        Args:
            track_pattern: Track to inspect.
            *steps: The ordered step list (see above).
            match: How step names compare to slice names: 'exact' (default),
                   'substring', or 'regex'.
            track_match: How ``track_pattern`` resolves to tracks: 'substring'
                         (default), 'exact', or 'regex'.
            depth: If given, only consider slices at this nesting depth
                   (e.g. ``depth=0`` for top-level slices). Default: all depths.

        Returns:
            CheckResult: passed=True iff the ordered slice names match the steps.
        """
        track_names = self._find_tracks(track_pattern, match_type=track_match)
        if not track_names:
            result = CheckResult(
                passed=False,
                validator_name="assert_slice_order",
                message=f"no track matched {track_pattern!r}",
                expected=_fmt_slice_order(steps),
                actual="(track not found)",
                details={"track": track_pattern, "trace": self._path},
            )
            self._results.append(result)
            return result

        try:
            tokens = _expand_slice_order_steps(steps)
        except ValueError as exc:
            result = CheckResult(
                passed=False,
                validator_name="assert_slice_order",
                message=str(exc),
                details={"track": track_pattern, "trace": self._path},
            )
            self._results.append(result)
            return result

        names_in = self._sql_in_list(track_names)
        depth_clause = f" AND s.depth = {int(depth)}" if depth is not None else ""
        sql = (
            "SELECT s.name AS name FROM slice s "
            "JOIN track t ON s.track_id = t.id "
            f"WHERE t.name IN ({names_in}){depth_clause} "
            "ORDER BY s.ts, s.depth, s.id"
        )
        df = self._query(sql)
        seq = df["name"].tolist() if not df.empty else []

        ok = _match_slice_order(seq, tokens, match)
        preview = seq[:25]
        result = CheckResult(
            passed=ok,
            validator_name="assert_slice_order",
            message=(
                f"slice order matches on track {track_pattern!r} ({len(seq)} slice(s))"
                if ok
                else f"slice order mismatch on track {track_pattern!r}"
            ),
            expected=None if ok else _fmt_slice_order(steps),
            actual=None if ok else (
                f"{len(seq)} slice(s): {preview}"
                + (" ..." if len(seq) > len(preview) else "")
            ),
            details={
                "track": track_pattern,
                "matched_tracks": track_names,
                "slice_count": len(seq),
                "match": match,
                "trace": self._path,
            },
        )
        self._results.append(result)
        return result

    # ---------------------------------------------------------------------------
    # Internal fetch helpers for the timeline / statistics validators
    # ---------------------------------------------------------------------------

    def _slice_rows(
        self,
        name: str,
        *,
        match: str = "exact",
        track_pattern: str | None = None,
        track_match: str = "substring",
        depth: int | None = None,
        marker: str = "",
    ):
        """Fetch (name, ts, dur, depth) rows for slices matching ``name``.

        Returns ``(rows, tracks)``. ``rows`` is None when ``track_pattern`` is
        given but matches no track; otherwise a list of namedtuples with
        attributes name/ts/dur/depth. ``regex`` matching is applied in Python.
        """
        where: list[str] = []
        if match == "exact":
            where.append("s.name = '" + name.replace("'", "''") + "'")
        elif match == "substring":
            where.append(f"s.name LIKE '%{self._escape_like(name)}%' ESCAPE '\\'")
        # regex: no SQL name filter; filtered after fetch

        join = ""
        tracks: list[str] | None = None
        if track_pattern is not None:
            tracks = self._find_tracks(track_pattern, match_type=track_match)
            if not tracks:
                return None, tracks
            join = "JOIN track t ON s.track_id = t.id"
            where.append(f"t.name IN ({self._sql_in_list(tracks)})")
        if depth is not None:
            where.append(f"s.depth = {int(depth)}")

        where_sql = " AND ".join(where) if where else "1=1"
        sql = (
            "SELECT s.name AS name, s.ts AS ts, s.dur AS dur, s.depth AS depth "
            f"FROM slice s {join} WHERE {where_sql} ORDER BY s.ts, s.id -- {marker}"
        )
        df = self._query(sql)
        rows = list(df.itertuples(index=False)) if not df.empty else []
        if match == "regex":
            pat = re.compile(name)
            rows = [r for r in rows if r.name is not None and pat.search(str(r.name))]
        return rows, tracks

    def _track_intervals(
        self,
        track_pattern: str,
        *,
        track_match: str = "substring",
        depth: int | None = None,
        marker: str = "",
    ):
        """Fetch (ts, dur) intervals for slices on the matched track(s).

        Returns ``(intervals, tracks)``; ``intervals`` is None when the pattern
        matches no track.
        """
        tracks = self._find_tracks(track_pattern, match_type=track_match)
        if not tracks:
            return None, tracks
        depth_clause = f" AND s.depth = {int(depth)}" if depth is not None else ""
        sql = (
            "SELECT s.ts AS ts, s.dur AS dur FROM slice s "
            "JOIN track t ON s.track_id = t.id "
            f"WHERE t.name IN ({self._sql_in_list(tracks)}){depth_clause} "
            f"ORDER BY s.ts -- {marker}"
        )
        df = self._query(sql)
        iv = (
            [(int(r.ts), int(r.dur)) for r in df.itertuples(index=False)]
            if not df.empty
            else []
        )
        return iv, tracks

    def _track_not_found(self, validator: str, pattern: str) -> CheckResult:
        result = CheckResult(
            passed=False,
            validator_name=validator,
            message=f"no track matched {pattern!r}",
            actual="(track not found)",
            details={"track": pattern, "trace": self._path},
        )
        self._results.append(result)
        return result

    # ---------------------------------------------------------------------------
    # Temporal / concurrency validators
    # ---------------------------------------------------------------------------

    def assert_gpu_utilization(
        self,
        track_pattern: str,
        min_pct: float,
        *,
        track_match: str = "substring",
        depth: int | None = 0,
    ) -> CheckResult:
        """Assert a track is busy at least ``min_pct``% of its active span.

        Utilization = union(slice durations) / (last_end - first_start). Defaults
        to ``depth=0`` so nested slices don't inflate busy time.
        """
        iv, _ = self._track_intervals(
            track_pattern, track_match=track_match, depth=depth, marker="gpu_utilization"
        )
        if iv is None:
            return self._track_not_found("assert_gpu_utilization", track_pattern)
        busy = _interval_union_ns(iv)
        span = _interval_span_ns(iv)
        util = (100.0 * busy / span) if span > 0 else 0.0
        passed = util >= min_pct
        result = CheckResult(
            passed=passed,
            validator_name="assert_gpu_utilization",
            message=(
                f"track {track_pattern!r} utilization {util:.1f}% (busy {busy} / span {span} ns)"
            ),
            expected=None if passed else f">= {min_pct}%",
            actual=None if passed else f"{util:.1f}%",
            details={"track": track_pattern, "utilization_pct": round(util, 3),
                     "busy_ns": busy, "span_ns": span, "trace": self._path},
        )
        self._results.append(result)
        return result

    def assert_max_idle_gap(
        self,
        track_pattern: str,
        max_ns: int,
        *,
        track_match: str = "substring",
        depth: int | None = 0,
    ) -> CheckResult:
        """Assert no idle gap between consecutive slices on a track exceeds ``max_ns``."""
        iv, _ = self._track_intervals(
            track_pattern, track_match=track_match, depth=depth, marker="max_idle_gap"
        )
        if iv is None:
            return self._track_not_found("assert_max_idle_gap", track_pattern)
        gap = _max_gap_ns(iv)
        passed = gap <= max_ns
        result = CheckResult(
            passed=passed,
            validator_name="assert_max_idle_gap",
            message=(
                f"max idle gap on {track_pattern!r} is {gap} ns"
                if passed
                else f"idle gap {gap} ns exceeds budget {max_ns} ns on {track_pattern!r}"
            ),
            expected=None if passed else f"<= {max_ns} ns",
            actual=None if passed else f"{gap} ns",
            details={"track": track_pattern, "max_gap_ns": gap, "trace": self._path},
        )
        self._results.append(result)
        return result

    def assert_overlap(
        self,
        name_a: str,
        name_b: str,
        *,
        min_overlap_pct: float | None = None,
        max_overlap_pct: float | None = None,
        match: str = "substring",
    ) -> CheckResult:
        """Assert two slice groups overlap in time within a percentage band.

        Overlap percent is relative to the smaller group's total covered time.
        Use ``min_overlap_pct`` to require overlap (e.g. copy/compute) or
        ``max_overlap_pct`` to forbid it.
        """
        rows_a, _ = self._slice_rows(name_a, match=match, marker="overlap_a")
        rows_b, _ = self._slice_rows(name_b, match=match, marker="overlap_b")
        iv_a = [(int(r.ts), int(r.dur)) for r in (rows_a or [])]
        iv_b = [(int(r.ts), int(r.dur)) for r in (rows_b or [])]
        overlap = _interval_intersect_ns(iv_a, iv_b)
        base = min(_interval_union_ns(iv_a), _interval_union_ns(iv_b))
        pct = (100.0 * overlap / base) if base > 0 else 0.0
        passed = True
        if min_overlap_pct is not None and pct < min_overlap_pct:
            passed = False
        if max_overlap_pct is not None and pct > max_overlap_pct:
            passed = False
        bounds = []
        if min_overlap_pct is not None:
            bounds.append(f">= {min_overlap_pct}%")
        if max_overlap_pct is not None:
            bounds.append(f"<= {max_overlap_pct}%")
        result = CheckResult(
            passed=passed,
            validator_name="assert_overlap",
            message=f"{name_a!r} vs {name_b!r} overlap {pct:.1f}% ({overlap} ns)",
            expected=None if passed else " and ".join(bounds),
            actual=None if passed else f"{pct:.1f}%",
            details={"name_a": name_a, "name_b": name_b, "overlap_ns": overlap,
                     "overlap_pct": round(pct, 3), "trace": self._path},
        )
        self._results.append(result)
        return result

    def assert_serial_on_track(
        self,
        track_pattern: str,
        *,
        track_match: str = "substring",
        depth: int | None = 0,
    ) -> CheckResult:
        """Assert slices on a track never overlap (strictly serial execution)."""
        iv, _ = self._track_intervals(
            track_pattern, track_match=track_match, depth=depth, marker="serial_on_track"
        )
        if iv is None:
            return self._track_not_found("assert_serial_on_track", track_pattern)
        violations = _count_overlapping(iv)
        passed = violations == 0
        result = CheckResult(
            passed=passed,
            validator_name="assert_serial_on_track",
            message=(
                f"{track_pattern!r} is serial ({len(iv)} slices, no overlaps)"
                if passed
                else f"{violations} overlapping slice(s) on {track_pattern!r}"
            ),
            expected=None if passed else "0 overlaps",
            actual=None if passed else f"{violations} overlaps",
            details={"track": track_pattern, "overlaps": violations,
                     "slice_count": len(iv), "trace": self._path},
        )
        self._results.append(result)
        return result

    def assert_flow_latency(
        self,
        from_track: str,
        to_track: str,
        max_ns: int,
        *,
        pctile: float = 99.0,
        match: str = "substring",
    ) -> CheckResult:
        """Assert flow latency (dest start - source start) stays within ``max_ns``.

        Considers flows from a ``from_track`` slice to a ``to_track`` slice and
        checks the given percentile of their latencies.
        """
        from_names = self._find_tracks(from_track, match_type=match)
        to_names = self._find_tracks(to_track, match_type=match)
        if not from_names or not to_names:
            missing = []
            if not from_names:
                missing.append(f"source {from_track!r}")
            if not to_names:
                missing.append(f"destination {to_track!r}")
            result = CheckResult(
                passed=False, validator_name="assert_flow_latency",
                message=f"no track matched {' and '.join(missing)}",
                actual="(track not found)",
                details={"from_track": from_track, "to_track": to_track,
                         "trace": self._path},
            )
            self._results.append(result)
            return result

        sql = (
            "SELECT (si.ts - so.ts) AS latency FROM flow f "
            "JOIN slice so ON f.slice_out = so.id "
            "JOIN slice si ON f.slice_in = si.id "
            "JOIN track t_out ON so.track_id = t_out.id "
            "JOIN track t_in ON si.track_id = t_in.id "
            f"WHERE t_out.name IN ({self._sql_in_list(from_names)}) "
            f"AND t_in.name IN ({self._sql_in_list(to_names)}) -- flow_latency"
        )
        try:
            df = self._query(sql)
            lats = [int(r.latency) for r in df.itertuples(index=False)] if not df.empty else []
        except Exception as exc:
            result = CheckResult(
                passed=False, validator_name="assert_flow_latency",
                message=f"flow table absent or not queryable: {exc}",
                details={"error": str(exc), "trace": self._path},
            )
            self._results.append(result)
            return result

        if not lats:
            result = CheckResult(
                passed=False, validator_name="assert_flow_latency",
                message=f"no flows {from_track!r} -> {to_track!r} to measure",
                actual="(no flows)",
                details={"from_track": from_track, "to_track": to_track,
                         "trace": self._path},
            )
            self._results.append(result)
            return result

        value = _percentile(lats, pctile)
        passed = value <= max_ns
        result = CheckResult(
            passed=passed,
            validator_name="assert_flow_latency",
            message=(
                f"p{pctile:g} flow latency {from_track!r} -> {to_track!r} = {value:.0f} ns "
                f"over {len(lats)} flow(s)"
            ),
            expected=None if passed else f"p{pctile:g} <= {max_ns} ns",
            actual=None if passed else f"{value:.0f} ns",
            details={"from_track": from_track, "to_track": to_track,
                     "pctile": pctile, "latency_ns": value, "flow_count": len(lats),
                     "trace": self._path},
        )
        self._results.append(result)
        return result

    def assert_iteration_consistency(
        self,
        name: str,
        *,
        count: int | None = None,
        max_cv: float | None = None,
        no_upward_trend: bool = False,
        trend_tol: float = 0.05,
        match: str = "exact",
        track_pattern: str | None = None,
        track_match: str = "substring",
    ) -> CheckResult:
        """Assert repeated slices form a stable iteration (count / variance / trend).

        Args:
            count: Exact number of iterations expected (if given).
            max_cv: Maximum coefficient of variation (std/mean) of durations.
            no_upward_trend: When True, the mean of the second half must not
                exceed the first half by more than ``trend_tol`` (catches
                throttling / fragmentation drift).
        """
        rows, tracks = self._slice_rows(
            name, match=match, track_pattern=track_pattern,
            track_match=track_match, marker="iterations",
        )
        if rows is None:
            return self._track_not_found("assert_iteration_consistency", track_pattern)
        durs = [int(r.dur) for r in rows]
        n = len(durs)
        issues: list[str] = []
        if count is not None and n != count:
            issues.append(f"count {n} != {count}")
        cv = None
        if max_cv is not None:
            if n < 2:
                issues.append(f"need >= 2 iterations for CV, found {n}")
            else:
                mean, std = _mean_std(durs)
                cv = (std / mean) if mean else float("inf")
                if cv > max_cv:
                    issues.append(f"CV {cv:.4f} > {max_cv}")
        second_first_ratio = None
        if no_upward_trend:
            if n < 4:
                issues.append(f"need >= 4 iterations for trend, found {n}")
            else:
                half = n // 2
                first_mean = sum(durs[:half]) / half
                second_mean = sum(durs[half:]) / (n - half)
                second_first_ratio = (second_mean / first_mean) if first_mean else float("inf")
                if second_first_ratio > 1.0 + trend_tol:
                    issues.append(
                        f"upward trend: second-half mean {second_mean:.0f} > "
                        f"first-half {first_mean:.0f} by >{trend_tol:.0%}"
                    )
        passed = not issues
        result = CheckResult(
            passed=passed,
            validator_name="assert_iteration_consistency",
            message=(
                f"{name!r} iterations consistent ({n} samples)"
                if passed
                else f"{name!r} iteration issues: " + "; ".join(issues)
            ),
            expected=None if passed else "stable iterations",
            actual=None if passed else "; ".join(issues),
            details={"name": name, "count": n, "cv": cv,
                     "second_first_ratio": second_first_ratio, "trace": self._path},
        )
        self._results.append(result)
        return result

    # ---------------------------------------------------------------------------
    # Statistical / distributional validators
    # ---------------------------------------------------------------------------

    def assert_slice_duration_distribution(
        self,
        name: str,
        *,
        p50_range: tuple[float, float] | None = None,
        p95_max_ns: float | None = None,
        p99_max_ns: float | None = None,
        min_ns: float | None = None,
        max_ns: float | None = None,
        match: str = "exact",
        track_pattern: str | None = None,
        track_match: str = "substring",
    ) -> CheckResult:
        """Assert percentile bounds on the duration distribution of matching slices."""
        rows, tracks = self._slice_rows(
            name, match=match, track_pattern=track_pattern,
            track_match=track_match, marker="slice_duration",
        )
        if rows is None:
            return self._track_not_found("assert_slice_duration_distribution", track_pattern)
        durs = [int(r.dur) for r in rows]
        if not durs:
            result = CheckResult(
                passed=False, validator_name="assert_slice_duration_distribution",
                message=f"no slices matched {name!r} to measure",
                actual="(no slices)", details={"name": name, "trace": self._path},
            )
            self._results.append(result)
            return result

        p50 = _percentile(durs, 50)
        p95 = _percentile(durs, 95)
        p99 = _percentile(durs, 99)
        lo, hi = min(durs), max(durs)
        issues: list[str] = []
        if p50_range is not None and not (p50_range[0] <= p50 <= p50_range[1]):
            issues.append(f"p50 {p50:.0f} not in [{p50_range[0]}, {p50_range[1]}]")
        if p95_max_ns is not None and p95 > p95_max_ns:
            issues.append(f"p95 {p95:.0f} > {p95_max_ns}")
        if p99_max_ns is not None and p99 > p99_max_ns:
            issues.append(f"p99 {p99:.0f} > {p99_max_ns}")
        if min_ns is not None and lo < min_ns:
            issues.append(f"min {lo} < {min_ns}")
        if max_ns is not None and hi > max_ns:
            issues.append(f"max {hi} > {max_ns}")
        passed = not issues
        result = CheckResult(
            passed=passed,
            validator_name="assert_slice_duration_distribution",
            message=(
                f"{name!r} durations: p50={p50:.0f} p95={p95:.0f} p99={p99:.0f} ns (n={len(durs)})"
                if passed
                else f"{name!r} distribution issues: " + "; ".join(issues)
            ),
            expected=None if passed else "; ".join(issues),
            actual=None if passed else f"p50={p50:.0f} p95={p95:.0f} p99={p99:.0f}",
            details={"name": name, "p50": p50, "p95": p95, "p99": p99,
                     "min": lo, "max": hi, "count": len(durs), "trace": self._path},
        )
        self._results.append(result)
        return result

    def assert_no_duration_outliers(
        self,
        name: str,
        *,
        sigma: float = 4.0,
        max_outliers: int = 0,
        match: str = "exact",
        track_pattern: str | None = None,
        track_match: str = "substring",
    ) -> CheckResult:
        """Assert at most ``max_outliers`` slice durations lie beyond ``sigma`` SD of the mean."""
        rows, tracks = self._slice_rows(
            name, match=match, track_pattern=track_pattern,
            track_match=track_match, marker="duration_outliers",
        )
        if rows is None:
            return self._track_not_found("assert_no_duration_outliers", track_pattern)
        durs = [int(r.dur) for r in rows]
        mean, std = _mean_std(durs)
        outliers = [d for d in durs if std > 0 and abs(d - mean) > sigma * std]
        passed = len(outliers) <= max_outliers
        result = CheckResult(
            passed=passed,
            validator_name="assert_no_duration_outliers",
            message=(
                f"{name!r}: {len(outliers)} outlier(s) beyond {sigma}σ (n={len(durs)})"
            ),
            expected=None if passed else f"<= {max_outliers} outliers",
            actual=None if passed else f"{len(outliers)} outliers",
            details={"name": name, "outliers": len(outliers), "sigma": sigma,
                     "mean": mean, "std": std, "count": len(durs), "trace": self._path},
        )
        self._results.append(result)
        return result

    def assert_counter_in_range(
        self,
        track_pattern: str,
        *,
        min_val: float | None = None,
        max_val: float | None = None,
        track_match: str = "substring",
    ) -> CheckResult:
        """Assert all values of a counter track stay within [min_val, max_val]."""
        tracks = self._find_counter_tracks(track_pattern, match_type=track_match)
        if not tracks:
            return self._track_not_found("assert_counter_in_range", track_pattern)
        sql = (
            "SELECT MIN(c.value) AS lo, MAX(c.value) AS hi FROM counter c "
            "JOIN counter_track ct ON c.track_id = ct.id "
            f"WHERE ct.name IN ({self._sql_in_list(tracks)}) -- counter_in_range"
        )
        df = self._query(sql)
        if df.empty or df["lo"].iloc[0] is None:
            result = CheckResult(
                passed=False, validator_name="assert_counter_in_range",
                message=f"no counter samples for {track_pattern!r}",
                actual="(no samples)", details={"track": track_pattern, "trace": self._path},
            )
            self._results.append(result)
            return result
        lo = float(df["lo"].iloc[0])
        hi = float(df["hi"].iloc[0])
        issues = []
        if min_val is not None and lo < min_val:
            issues.append(f"min {lo} < {min_val}")
        if max_val is not None and hi > max_val:
            issues.append(f"max {hi} > {max_val}")
        passed = not issues
        result = CheckResult(
            passed=passed,
            validator_name="assert_counter_in_range",
            message=(
                f"{track_pattern!r} in [{lo}, {hi}]"
                if passed
                else f"{track_pattern!r} out of range: " + "; ".join(issues)
            ),
            expected=None if passed else f"[{min_val}, {max_val}]",
            actual=None if passed else f"[{lo}, {hi}]",
            details={"track": track_pattern, "observed_min": lo, "observed_max": hi,
                     "trace": self._path},
        )
        self._results.append(result)
        return result

    def assert_counter_rate(
        self,
        track_pattern: str,
        *,
        max_per_sec: float | None = None,
        min_per_sec: float | None = None,
        track_match: str = "substring",
    ) -> CheckResult:
        """Assert the per-second rate of change of a counter stays within bounds."""
        tracks = self._find_counter_tracks(track_pattern, match_type=track_match)
        if not tracks:
            return self._track_not_found("assert_counter_rate", track_pattern)
        sql = (
            "SELECT c.ts AS ts, c.value AS value FROM counter c "
            "JOIN counter_track ct ON c.track_id = ct.id "
            f"WHERE ct.name IN ({self._sql_in_list(tracks)}) ORDER BY c.ts -- counter_rate"
        )
        df = self._query(sql)
        pts = [(int(r.ts), float(r.value)) for r in df.itertuples(index=False)] if not df.empty else []
        if len(pts) < 2:
            result = CheckResult(
                passed=False, validator_name="assert_counter_rate",
                message=f"need >= 2 counter samples for {track_pattern!r}, found {len(pts)}",
                actual=f"{len(pts)} samples", details={"track": track_pattern, "trace": self._path},
            )
            self._results.append(result)
            return result
        rates = []
        for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
            dt = (t1 - t0) / 1e9
            if dt > 0:
                rates.append((v1 - v0) / dt)
        max_rate = max(rates) if rates else 0.0
        min_rate = min(rates) if rates else 0.0
        issues = []
        if max_per_sec is not None and max_rate > max_per_sec:
            issues.append(f"max rate {max_rate:.3g} > {max_per_sec}")
        if min_per_sec is not None and min_rate < min_per_sec:
            issues.append(f"min rate {min_rate:.3g} < {min_per_sec}")
        passed = not issues
        result = CheckResult(
            passed=passed,
            validator_name="assert_counter_rate",
            message=(
                f"{track_pattern!r} rate in [{min_rate:.3g}, {max_rate:.3g}]/s"
                if passed
                else f"{track_pattern!r} rate out of bounds: " + "; ".join(issues)
            ),
            expected=None if passed else "; ".join(issues),
            actual=None if passed else f"[{min_rate:.3g}, {max_rate:.3g}]/s",
            details={"track": track_pattern, "max_rate": max_rate, "min_rate": min_rate,
                     "samples": len(pts), "trace": self._path},
        )
        self._results.append(result)
        return result

    # ---------------------------------------------------------------------------
    # Structural validators
    # ---------------------------------------------------------------------------

    def assert_call_tree(
        self,
        parent_pattern: str,
        *,
        contains: list[str] | None = None,
        max_depth: int | None = None,
        no_recursion: bool = False,
        match: str = "exact",
        track_pattern: str | None = None,
        track_match: str = "substring",
    ) -> CheckResult:
        """Assert structural properties of a slice subtree rooted at a parent.

        Children are determined by timestamp containment on the same track
        (a slice is a descendant if its [ts, ts+dur] is within the parent's and
        its depth is greater).

        Args:
            contains: child names that must appear somewhere under a parent.
            max_depth: maximum relative nesting depth below the parent.
            no_recursion: when True, no descendant may share the parent's name.
        """
        join = ""
        where = []
        tracks = None
        if track_pattern is not None:
            tracks = self._find_tracks(track_pattern, match_type=track_match)
            if not tracks:
                return self._track_not_found("assert_call_tree", track_pattern)
            join = "JOIN track t ON s.track_id = t.id"
            where.append(f"t.name IN ({self._sql_in_list(tracks)})")
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""
        sql = (
            "SELECT s.name AS name, s.ts AS ts, s.dur AS dur, s.depth AS depth, "
            f"s.track_id AS track_id FROM slice s {join}{where_sql} "
            "ORDER BY s.track_id, s.ts, s.depth -- call_tree"
        )
        df = self._query(sql)
        rows = list(df.itertuples(index=False)) if not df.empty else []

        def _name_ok(actual: str) -> bool:
            return _slice_name_match(actual, parent_pattern, match)

        parents = [r for r in rows if r.name is not None and _name_ok(str(r.name))]
        if not parents:
            result = CheckResult(
                passed=False, validator_name="assert_call_tree",
                message=f"no parent slice matched {parent_pattern!r}",
                actual="(parent not found)", details={"parent": parent_pattern, "trace": self._path},
            )
            self._results.append(result)
            return result

        contains = contains or []
        found_children: set[str] = set()
        max_rel_depth = 0
        recursion_hit = False
        for p in parents:
            p_end = int(p.ts) + int(p.dur)
            for r in rows:
                if r is p or r.track_id != p.track_id:
                    continue
                if int(r.ts) >= int(p.ts) and int(r.ts) + int(r.dur) <= p_end and int(r.depth) > int(p.depth):
                    found_children.add(str(r.name))
                    max_rel_depth = max(max_rel_depth, int(r.depth) - int(p.depth))
                    if no_recursion and r.name == p.name:
                        recursion_hit = True

        issues = []
        missing = [c for c in contains if not any(_slice_name_match(fc, c, match) for fc in found_children)]
        if missing:
            issues.append(f"missing children {missing}")
        if max_depth is not None and max_rel_depth > max_depth:
            issues.append(f"depth {max_rel_depth} > {max_depth}")
        if no_recursion and recursion_hit:
            issues.append("unexpected recursion (descendant shares parent name)")
        passed = not issues
        result = CheckResult(
            passed=passed,
            validator_name="assert_call_tree",
            message=(
                f"call tree under {parent_pattern!r} ok (depth {max_rel_depth})"
                if passed
                else f"call tree under {parent_pattern!r} issues: " + "; ".join(issues)
            ),
            expected=None if passed else "; ".join(issues),
            actual=None if passed else sorted(found_children),
            details={"parent": parent_pattern, "children_found": sorted(found_children),
                     "max_rel_depth": max_rel_depth, "trace": self._path},
        )
        self._results.append(result)
        return result

    def assert_slice_args(
        self,
        name: str,
        require: dict[str, type],
        *,
        match: str = "exact",
    ) -> CheckResult:
        """Assert every slice matching ``name`` carries the required arg keys/types.

        ``require`` maps an arg key to a Python type: ``int`` -> ``int_value``,
        ``float`` -> ``real_value``, ``str`` -> ``string_value`` must be present.
        """
        if match == "exact":
            pred = "s.name = '" + name.replace("'", "''") + "'"
        elif match == "substring":
            pred = f"s.name LIKE '%{self._escape_like(name)}%' ESCAPE '\\'"
        else:
            pred = "1=1"  # regex filtered in Python

        ids_df = self._query(f"SELECT s.id AS sid, s.name AS name FROM slice s WHERE {pred} -- slice_args_ids")
        id_rows = list(ids_df.itertuples(index=False)) if not ids_df.empty else []
        if match == "regex":
            pat = re.compile(name)
            id_rows = [r for r in id_rows if r.name is not None and pat.search(str(r.name))]
        sids = {int(r.sid) for r in id_rows}
        if not sids:
            result = CheckResult(
                passed=False, validator_name="assert_slice_args",
                message=f"no slices matched {name!r}",
                actual="(no slices)", details={"name": name, "trace": self._path},
            )
            self._results.append(result)
            return result

        args_df = self._query(
            "SELECT s.id AS sid, a.key AS key, a.int_value AS iv, a.string_value AS sv, "
            f"a.real_value AS rv FROM slice s JOIN args a ON s.arg_set_id = a.arg_set_id "
            f"WHERE {pred} -- slice_args"
        )
        per: dict[int, dict[str, tuple]] = {sid: {} for sid in sids}
        if not args_df.empty:
            for r in args_df.itertuples(index=False):
                if int(r.sid) in per:
                    per[int(r.sid)][str(r.key)] = (r.iv, r.sv, r.rv)

        def _has_type(triple, typ) -> bool:
            iv, sv, rv = triple
            if typ is int:
                return iv is not None
            if typ is float:
                return rv is not None or iv is not None
            if typ is str:
                return sv is not None
            return iv is not None or sv is not None or rv is not None

        missing_counts: dict[str, int] = {}
        for sid in sids:
            for key, typ in require.items():
                triple = per[sid].get(key)
                if triple is None or not _has_type(triple, typ):
                    missing_counts[key] = missing_counts.get(key, 0) + 1
        passed = not missing_counts
        result = CheckResult(
            passed=passed,
            validator_name="assert_slice_args",
            message=(
                f"all {len(sids)} {name!r} slices carry required args"
                if passed
                else f"{name!r} slices missing args: " + ", ".join(
                    f"{k} (×{v})" for k, v in missing_counts.items())
            ),
            expected=None if passed else {k: t.__name__ for k, t in require.items()},
            actual=None if passed else missing_counts,
            details={"name": name, "slice_count": len(sids),
                     "missing": missing_counts, "trace": self._path},
        )
        self._results.append(result)
        return result

    # ---------------------------------------------------------------------------
    # Anti-pattern validators ("this should never appear")
    # ---------------------------------------------------------------------------

    def _count_query(self, sql: str) -> int:
        df = self._query(sql)
        if df.empty or df["cnt"].iloc[0] is None:
            return 0
        return int(df["cnt"].iloc[0])

    def assert_no_anti_patterns(
        self,
        *,
        negative_durations: bool = True,
        duplicate_slices: bool = True,
        orphan_slices: bool = True,
        giant_slice: bool = True,
        giant_pct: float = 99.0,
        zero_duration_category: str | None = None,
    ) -> CheckResult:
        """Assert a curated bundle of timeline anti-patterns is absent.

        Toggle individual checks off as needed. Returns a single CheckResult that
        lists every anti-pattern found.
        """
        found: dict[str, Any] = {}

        if negative_durations:
            n = self._count_query("SELECT COUNT(*) AS cnt FROM slice WHERE dur < 0 -- anti_neg")
            if n:
                found["negative_durations"] = n

        if zero_duration_category is not None:
            esc = self._escape_like(zero_duration_category)
            n = self._count_query(
                "SELECT COUNT(*) AS cnt FROM slice WHERE dur = 0 AND category LIKE "
                f"'%{esc}%' ESCAPE '\\' -- anti_zero"
            )
            if n:
                found["zero_duration_slices"] = n

        if duplicate_slices:
            n = self._count_query(
                "SELECT COUNT(*) AS cnt FROM (SELECT track_id, ts, name, COUNT(*) AS c "
                "FROM slice GROUP BY track_id, ts, name HAVING c > 1) -- anti_dup"
            )
            if n:
                found["duplicate_slices"] = n

        if orphan_slices:
            df = self._query(
                "SELECT s.track_id AS track_id, s.ts AS ts, s.dur AS dur, s.depth AS depth "
                "FROM slice s WHERE s.dur >= 0 ORDER BY s.track_id, s.ts, s.depth -- anti_orphan"
            )
            rows = list(df.itertuples(index=False)) if not df.empty else []
            orphans = 0
            stack: list[tuple[int, int]] = []  # (end, depth) of open ancestors
            cur_track = None
            for r in rows:
                if r.track_id != cur_track:
                    stack = []
                    cur_track = r.track_id
                end = int(r.ts) + int(r.dur)
                while stack and (int(r.depth) <= stack[-1][1]):
                    stack.pop()
                if stack and end > stack[-1][0]:
                    orphans += 1  # extends beyond its parent's end → not properly nested
                stack.append((end, int(r.depth)))
            if orphans:
                found["orphan_slices"] = orphans

        if giant_slice:
            span_df = self._query(
                "SELECT MIN(ts) AS lo, MAX(ts + dur) AS hi FROM slice WHERE dur >= 0 -- anti_span"
            )
            if not span_df.empty and span_df["lo"].iloc[0] is not None:
                span = int(span_df["hi"].iloc[0]) - int(span_df["lo"].iloc[0])
                if span > 0:
                    thr = span * (giant_pct / 100.0)
                    n = self._count_query(
                        "SELECT COUNT(*) AS cnt FROM (SELECT track_id, COUNT(*) AS c, "
                        f"MAX(dur) AS md FROM slice WHERE dur >= 0 GROUP BY track_id "
                        f"HAVING c = 1 AND md >= {thr}) -- anti_giant"
                    )
                    if n:
                        found["single_giant_slice_tracks"] = n

        passed = not found
        result = CheckResult(
            passed=passed,
            validator_name="assert_no_anti_patterns",
            message=(
                "no anti-patterns detected"
                if passed
                else "anti-patterns detected: " + ", ".join(
                    f"{k}={v}" for k, v in found.items())
            ),
            expected=None if passed else "no anti-patterns",
            actual=None if passed else found,
            details={"found": found, "trace": self._path},
        )
        self._results.append(result)
        return result

    # ---------------------------------------------------------------------------
    # PERF-12: Assert sampling frequency
    # ---------------------------------------------------------------------------

    def assert_sampling_frequency(
        self,
        track_pattern: str,
        expected_hz: float,
        tolerance_pct: float = 20.0,
    ) -> CheckResult:
        """Assert the counter sampling frequency is within tolerance of expected_hz.

        Computes actual Hz from mean inter-sample interval (ts column in nanoseconds).

        Args:
            track_pattern: Counter track name pattern (substring match).
            expected_hz: Expected sampling frequency in Hz.
            tolerance_pct: Allowed deviation from expected_hz as a percentage (default 20%).

        Returns:
            CheckResult with actual_hz and tolerance in details.
        """
        safe_track = self._escape_like(track_pattern)
        df = self._query(
            "SELECT c.ts, c.track_id FROM counter c "
            "JOIN counter_track ct ON c.track_id = ct.id "
            f"WHERE ct.name LIKE '%{safe_track}%' ESCAPE '\\' AND ct.name IS NOT NULL "
            "ORDER BY c.track_id, c.ts"
        )

        if df.empty or len(df) < 2:
            result = CheckResult(
                passed=False,
                validator_name="assert_sampling_frequency",
                message=f"Insufficient samples on {track_pattern!r} for frequency check (need >= 2)",
                expected=f"{expected_hz} Hz",
                actual="< 2 samples",
                details={"track_pattern": track_pattern, "expected_hz": expected_hz},
            )
            self._results.append(result)
            return result

        tolerance = expected_hz * tolerance_pct / 100.0

        # Compute per-track mean frequency; all tracks must be within tolerance.
        per_track_hz: dict[int, float] = {}
        for track_id, group in df.groupby("track_id"):
            if len(group) < 2:
                continue
            ts_series = group["ts"].astype(float)
            mean_diff_ns = ts_series.diff().dropna().mean()
            per_track_hz[int(track_id)] = 1e9 / mean_diff_ns if mean_diff_ns > 0 else 0.0

        if not per_track_hz:
            result = CheckResult(
                passed=False,
                validator_name="assert_sampling_frequency",
                message=f"Insufficient samples on {track_pattern!r} for frequency check (need >= 2 per track)",
                expected=f"{expected_hz} Hz",
                actual="< 2 samples per track",
                details={"track_pattern": track_pattern, "expected_hz": expected_hz},
            )
            self._results.append(result)
            return result

        # Report using the mean across all per-track frequencies for the summary.
        actual_hz = sum(per_track_hz.values()) / len(per_track_hz)
        passed = all(abs(hz - expected_hz) <= tolerance for hz in per_track_hz.values())

        result = CheckResult(
            passed=passed,
            validator_name="assert_sampling_frequency",
            message=(
                f"Sampling frequency on {track_pattern!r}: {actual_hz:.1f} Hz (within {tolerance_pct}% of {expected_hz} Hz)"
                if passed
                else f"Sampling frequency on {track_pattern!r}: {actual_hz:.1f} Hz (outside {tolerance_pct}% of {expected_hz} Hz)"
            ),
            expected=f"{expected_hz} Hz ± {tolerance_pct}%" if not passed else None,
            actual=f"{actual_hz:.1f} Hz" if not passed else None,
            details={
                "track_pattern": track_pattern,
                "expected_hz": expected_hz,
                "actual_hz": actual_hz,
                "per_track_hz": per_track_hz,
                "tolerance_pct": tolerance_pct,
            },
        )
        self._results.append(result)
        return result

    # ---------------------------------------------------------------------------
    # PERF-16: Assert sampling frequency from slice timestamps
    # ---------------------------------------------------------------------------

    def assert_slice_sampling_frequency(
        self,
        track_pattern: str,
        expected_hz: float,
        tolerance_pct: float = 20.0,
    ) -> CheckResult:
        """Assert the sampling rate derived from slice timestamps matches expected_hz.

        Unlike assert_sampling_frequency (which reads the counter table), this
        derives the rate from the spacing of sampling *slices* (e.g. the
        timer_sampling track), which is how rocprof-sys records CPU sampling.
        Frequency is the inverse of the mean inter-slice interval, computed per
        track; every matched track must be within tolerance.

        Only root frames (depth=0) are counted: rocprof-sys call-stack sampling
        emits a full nested stack per sample (many slices sharing one timestamp),
        so counting every slice would inflate the apparent rate by the mean stack
        depth. One root frame per sampling tick gives the true frequency.

        Args:
            track_pattern: Track name pattern (substring match).
            expected_hz: Expected sampling frequency in Hz.
            tolerance_pct: Allowed deviation from expected_hz as a percentage.

        Returns:
            CheckResult with actual_hz and per-track frequencies in details.
        """
        safe_track = self._escape_like(track_pattern)
        df = self._query(
            "SELECT s.ts, s.track_id FROM slice s "
            "JOIN track t ON s.track_id = t.id "
            f"WHERE t.name LIKE '%{safe_track}%' ESCAPE '\\' AND t.name IS NOT NULL "
            "AND s.depth = 0 "
            "ORDER BY s.track_id, s.ts"
        )

        if df.empty or len(df) < 2:
            result = CheckResult(
                passed=False,
                validator_name="assert_slice_sampling_frequency",
                message=f"Insufficient slices on {track_pattern!r} for frequency check (need >= 2)",
                expected=f"{expected_hz} Hz",
                actual="< 2 slices",
                details={"track_pattern": track_pattern, "expected_hz": expected_hz},
            )
            self._results.append(result)
            return result

        tolerance = expected_hz * tolerance_pct / 100.0

        per_track_hz: dict[int, float] = {}
        for track_id, group in df.groupby("track_id"):
            if len(group) < 2:
                continue
            ts_series = group["ts"].astype(float)
            mean_diff_ns = ts_series.diff().dropna().mean()
            per_track_hz[int(track_id)] = 1e9 / mean_diff_ns if mean_diff_ns > 0 else 0.0

        if not per_track_hz:
            result = CheckResult(
                passed=False,
                validator_name="assert_slice_sampling_frequency",
                message=f"Insufficient slices on {track_pattern!r} for frequency check (need >= 2 per track)",
                expected=f"{expected_hz} Hz",
                actual="< 2 slices per track",
                details={"track_pattern": track_pattern, "expected_hz": expected_hz},
            )
            self._results.append(result)
            return result

        actual_hz = sum(per_track_hz.values()) / len(per_track_hz)
        passed = all(abs(hz - expected_hz) <= tolerance for hz in per_track_hz.values())

        result = CheckResult(
            passed=passed,
            validator_name="assert_slice_sampling_frequency",
            message=(
                f"Slice sampling frequency on {track_pattern!r}: {actual_hz:.1f} Hz "
                f"(within {tolerance_pct}% of {expected_hz} Hz)"
                if passed
                else f"Slice sampling frequency on {track_pattern!r}: {actual_hz:.1f} Hz "
                f"(outside {tolerance_pct}% of {expected_hz} Hz)"
            ),
            expected=f"{expected_hz} Hz ± {tolerance_pct}%" if not passed else None,
            actual=f"{actual_hz:.1f} Hz" if not passed else None,
            details={
                "track_pattern": track_pattern,
                "expected_hz": expected_hz,
                "actual_hz": actual_hz,
                "per_track_hz": per_track_hz,
                "tolerance_pct": tolerance_pct,
            },
        )
        self._results.append(result)
        return result

    # ---------------------------------------------------------------------------
    # PERF-13: Raw SQL execution
    # ---------------------------------------------------------------------------

    def execute_sql(self, sql: str) -> pd.DataFrame:
        """Execute arbitrary SQL against the trace and return results as a DataFrame.

        This is a trusted-caller API — sql is not user-supplied in normal operation.

        Args:
            sql: PerfettoSQL query string.

        Returns:
            pandas DataFrame with query results.
        """
        return _query_to_dataframe(self._tp, sql)

    # ---------------------------------------------------------------------------
    # PERF-14: Assert expected slice categories present
    # ---------------------------------------------------------------------------

    def assert_categories_present(self, expected_categories: list[str]) -> CheckResult:
        """Assert that all expected slice categories are present in the trace.

        Args:
            expected_categories: List of category names expected to be present.

        Returns:
            CheckResult with missing categories listed in details.
        """
        df = self._query("SELECT DISTINCT category FROM slice WHERE category IS NOT NULL")
        found = set(df["category"].tolist()) if not df.empty else set()
        missing = [c for c in expected_categories if c not in found]
        passed = not missing

        result = CheckResult(
            passed=passed,
            validator_name="assert_categories_present",
            message=(
                f"All {len(expected_categories)} expected category/categories found"
                if passed
                else f"Missing {len(missing)} category/categories: {missing}"
            ),
            expected=expected_categories if not passed else None,
            actual=sorted(found) if not passed else None,
            details={"missing_categories": missing, "found_categories": sorted(found)},
        )
        self._results.append(result)
        return result

    # ---------------------------------------------------------------------------
    # PERF-15: Assert process name matches pattern
    # ---------------------------------------------------------------------------

    def assert_process_name(
        self, pattern: str, match: str = "substring"
    ) -> CheckResult:
        """Assert at least one process name matches the given pattern.

        Args:
            pattern: Process name pattern.
            match: 'exact', 'substring' (default), or 'regex'.

        Returns:
            CheckResult with passed=True if any matching process name found.
        """
        df = self._query("SELECT name FROM process WHERE name IS NOT NULL")
        names = df["name"].tolist() if not df.empty else []

        if match == "exact":
            matched = [n for n in names if n == pattern]
        elif match == "substring":
            matched = [n for n in names if pattern in str(n)]
        else:  # regex
            compiled = re.compile(pattern)
            matched = [n for n in names if compiled.search(str(n) if n else "")]

        passed = bool(matched)
        result = CheckResult(
            passed=passed,
            validator_name="assert_process_name",
            message=(
                f"Process matching {pattern!r} found: {matched[0]!r}"
                if passed
                else f"No process matching {pattern!r} found"
            ),
            expected=pattern if not passed else None,
            actual="(none)" if not passed else None,
            details={"match_type": match, "trace": self._path},
        )
        self._results.append(result)
        return result
