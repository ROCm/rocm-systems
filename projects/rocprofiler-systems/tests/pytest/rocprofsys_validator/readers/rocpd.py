# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RocpdReader — wraps sqlite3.Connection for RocPD database validation.

RocPD is a SQLite database produced by rocprof-sys. Its schema uses hash-suffixed
physical tables (e.g., rocpd_region_abc123) with clean un-suffixed views (regions,
kernels, ...). Validators MUST target the views, not the raw tables.

Design decisions:
- ROCPD-03: Connection always opened read-only via URI mode (file:path?mode=ro, uri=True)
- ROCPD-13 Pitfall: region_args JOIN uses guid, NOT id — id is an independent auto-increment
- ROCPD-08/09: Conditional validators emit warnings.warn and return passed=True when absent
- Security (T-02-01-01): execute_sql() is for trusted callers only; no user input expected
- Security (T-02-01-02): function_name and other user args use parameterized queries (?), not
  string interpolation
"""
from __future__ import annotations

import re
import sqlite3
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from rocprofsys_validator.core import FormatReader, CheckResult
from rocprofsys_validator.registry import reader

# Reuse the timeline / statistics / sequence helpers proven for Perfetto so the
# two readers compute identical semantics (interval union, percentiles, the
# wildcard slice-order matcher, etc.).
from rocprofsys_validator.readers.perfetto import (
    _count_overlapping,
    _expand_slice_order_steps,
    _fmt_slice_order,
    _interval_span_ns,
    _interval_union_ns,
    _interval_intersect_ns,
    _match_slice_order,
    _max_gap_ns,
    _mean_std,
    _percentile,
    _slice_name_match,
)

# Identifier guard for view/column names interpolated into SQL (values are always
# bound as parameters; only structural identifiers are interpolated).
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Per-view partition columns used by duplicate / giant-record anti-pattern checks
# (e.g. the same kernel can legitimately start at the same time on two streams).
_VIEW_PARTITION: dict[str, str] = {
    "kernels": "stream_id",
    "memory_copies": "stream_id",
    "regions": "tid",
}

if TYPE_CHECKING:
    from rocprofsys_validator.gpu import GPUProfile

# The canonical view names that every RocPD database must expose.
# Confirmed against the real rocpd-3214910.db artifact. The sampling views
# (samples / regions_and_samples / sample_regions) are first-class in RocPD v3:
# rocprof-sys is primarily a sampling profiler, so a database missing them is
# structurally incomplete.
_EXPECTED_VIEWS: list[str] = [
    "regions",
    "kernels",
    "memory_copies",
    "memory_allocations",
    "pmc_events",
    "region_args",
    "rocpd_metadata",
    "rocpd_info_agent",
    "processes",
    "threads",
    "samples",
    "regions_and_samples",
    "sample_regions",
]

@reader("rocpd")
class RocpdReader(FormatReader):
    """Format reader for RocPD SQLite databases produced by rocprof-sys.

    Opens the database read-only and provides 14 validator methods covering
    ROCPD-01 through ROCPD-14. Each method appends its result to an internal
    list; calling .validate() returns all accumulated results.

    Usage::

        with RocpdReader("output.db") as r:
            r.assert_schema_valid()
            r.assert_hip_api_calls_present()
            results = r.validate()
            assert all(res.passed for res in results)
    """

    def __init__(self, path: str | Path) -> None:
        """Open a RocPD database read-only.

        Args:
            path: Path to the RocPD SQLite database file.

        Raises:
            FileNotFoundError: If the file does not exist (message includes the path).
            sqlite3.DatabaseError: If the file is not a valid SQLite database
                (message includes the path) — raised here at construction rather
                than as a pathless error on the first query.
        """
        self._path = str(path)
        if not Path(self._path).exists():
            raise FileNotFoundError(f"RocPD database not found: {self._path}")
        # ROCPD-03: URI mode with mode=ro prevents all write operations.
        # sqlite3 raises OperationalError on any INSERT/UPDATE/DELETE/CREATE attempt.
        self._conn = sqlite3.connect(f"file:{self._path}?mode=ro", uri=True)
        self._conn.row_factory = sqlite3.Row
        # sqlite3 opens lazily: a non-database file connects fine but raises a
        # pathless DatabaseError on first query. Probe now so the failure names
        # the offending file and the CheckResult contract is never broken later.
        try:
            self._conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
        except sqlite3.DatabaseError as exc:
            self._conn.close()
            raise sqlite3.DatabaseError(
                f"Not a valid RocPD SQLite database: {self._path} ({exc})"
            ) from exc
        self._results: list[CheckResult] = []

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()

    def validate(self) -> list[CheckResult]:
        """Return all accumulated validation results.

        Returns:
            list[CheckResult]: Results from all validator methods called so far.
        """
        return list(self._results)

    # -------------------------------------------------------------------------
    # ROCPD-11: Raw SQL execution
    # -------------------------------------------------------------------------

    def execute_sql(self, sql: str) -> list[sqlite3.Row]:
        """Execute arbitrary SQL against the database and return all rows.

        Security (T-02-01-01): This method is for trusted validator code only.
        No user-supplied input should flow through this method in Phase 2.

        Args:
            sql: SQL statement to execute (read-only — connection is mode=ro).

        Returns:
            list[sqlite3.Row]: All rows returned by the query.
        """
        return self._conn.execute(sql).fetchall()

    # -------------------------------------------------------------------------
    # ROCPD-01 + ROCPD-04: assert_schema_valid
    # -------------------------------------------------------------------------

    def assert_schema_valid(self) -> CheckResult:
        """Assert all expected views exist and foreign key constraints are clean.

        Checks for the 10 canonical views in sqlite_master and runs PRAGMA
        foreign_key_check to detect referential integrity violations.

        Returns:
            CheckResult: passed=True if all views present and no FK violations.
        """
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view'"
        ).fetchall()
        found = {r["name"] for r in rows}
        missing = [v for v in _EXPECTED_VIEWS if v not in found]

        fk_violations = self._conn.execute("PRAGMA foreign_key_check").fetchall()

        passed = not missing and not fk_violations
        result = CheckResult(
            passed=passed,
            validator_name="assert_schema_valid",
            message="Schema valid" if passed else "Schema issues detected",
            details={
                "missing_views": missing,
                "fk_violations": len(fk_violations),
            },
        )
        self._results.append(result)
        return result

    # -------------------------------------------------------------------------
    # ROCPD-02: assert_columns_exist
    # -------------------------------------------------------------------------

    def assert_columns_exist(
        self, view_name: str, expected_cols: list[str]
    ) -> CheckResult:
        """Assert that a view has all expected columns.

        Security (T-02-01-03): view_name is interpolated into PRAGMA — only trusted
        validator code calls this method; no external input path in Phase 2.

        Args:
            view_name: Name of the view or table to inspect.
            expected_cols: List of column names that must be present.

        Returns:
            CheckResult: passed=True if all columns found.
        """
        rows = self._conn.execute(
            f"PRAGMA table_info({view_name})"
        ).fetchall()
        found_cols = {r["name"] for r in rows}
        missing = [c for c in expected_cols if c not in found_cols]
        passed = not missing
        result = CheckResult(
            passed=passed,
            validator_name="assert_columns_exist",
            message=(
                f"All expected columns present in {view_name!r}"
                if passed
                else f"Missing columns in {view_name!r}"
            ),
            expected=sorted(expected_cols),
            actual=sorted(found_cols),
            details={
                "view_name": view_name,
                "missing_columns": missing,
            },
        )
        self._results.append(result)
        return result

    # -------------------------------------------------------------------------
    # ROCPD-05: assert_min_row_count
    # -------------------------------------------------------------------------

    def assert_min_row_count(
        self, view_name: str, min_count: int = 1
    ) -> CheckResult:
        """Assert a view has at least min_count rows.

        Security (T-02-01-03): view_name interpolated into SELECT — only trusted
        validator code calls this method.

        Args:
            view_name: Name of the view or table to count rows in.
            min_count: Minimum expected row count (default: 1).

        Returns:
            CheckResult: passed=True if COUNT(*) >= min_count.
        """
        count = self._conn.execute(
            f"SELECT COUNT(*) FROM {view_name}"
        ).fetchone()[0]
        passed = count >= min_count
        result = CheckResult(
            passed=passed,
            validator_name="assert_min_row_count",
            message=(
                f"{view_name!r} has {count} row(s)"
                if passed
                else f"Expected >= {min_count} rows in {view_name!r}, found {count}"
            ),
            expected=f">= {min_count}" if not passed else None,
            actual=count if not passed else None,
            details={"view_name": view_name, "count": count},
        )
        self._results.append(result)
        return result

    def assert_no_rows(self, view_name: str) -> CheckResult:
        """Assert a view has zero rows (negative test).

        The explicit complement of assert_min_row_count — for asserting absence,
        e.g. that a CPU-only run produced no kernel dispatches.

        Security (T-02-01-03): view_name interpolated into SELECT — trusted
        validator code only.

        Args:
            view_name: Name of the view or table to check.

        Returns:
            CheckResult: passed=True when COUNT(*) == 0.
        """
        count = self._conn.execute(
            f"SELECT COUNT(*) FROM {view_name}"
        ).fetchone()[0]
        passed = count == 0
        result = CheckResult(
            passed=passed,
            validator_name="assert_no_rows",
            message=(
                f"{view_name!r} is empty (as expected)"
                if passed
                else f"Expected {view_name!r} to be empty, found {count} row(s)"
            ),
            expected="0 rows" if not passed else None,
            actual=count if not passed else None,
            details={"view_name": view_name, "count": count},
        )
        self._results.append(result)
        return result

    # -------------------------------------------------------------------------
    # ROCPD-06: assert_hip_api_calls_present
    # -------------------------------------------------------------------------

    def assert_hip_api_calls_present(
        self,
        function_name: str | None = None,
        min_count: int = 1,
        category: str = "rocm_hip_api",
    ) -> CheckResult:
        """Assert HIP API calls are recorded in the regions view.

        Filters by a configurable region category (default 'rocm_hip_api').
        The category name varies by tool/version, so it is a parameter rather
        than a hardcoded literal (CLAUDE.md adaptability constraint). Optionally
        filters to a specific function name using a parameterized query
        (Security: T-02-01-02).

        Args:
            function_name: If provided, only count calls matching this name.
            min_count: Minimum expected call count (default: 1).
            category: Region category to filter on (default: 'rocm_hip_api').

        Returns:
            CheckResult: passed=True if count >= min_count.
        """
        # T-02-01-02: category and function_name are bound as parameters — no
        # string interpolation of values into the SQL text.
        if function_name:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM regions WHERE category=? AND name=?",
                (category, function_name),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM regions WHERE category=?",
                (category,),
            ).fetchone()
        count = row[0]
        passed = count >= min_count
        result = CheckResult(
            passed=passed,
            validator_name="assert_hip_api_calls_present",
            message=(
                f"HIP API calls found: {count}"
                if passed
                else f"Expected >= {min_count} HIP API calls, found {count}"
            ),
            expected=f">= {min_count}",
            actual=count,
            details={
                "function_name": function_name,
                "category": category,
            },
        )
        self._results.append(result)
        return result

    # -------------------------------------------------------------------------
    # ROCPD-15: assert_samples_present (sampling profiler output)
    # -------------------------------------------------------------------------

    def assert_samples_present(
        self,
        min_count: int = 1,
        category: str | None = None,
    ) -> CheckResult:
        """Assert point-sample records are present in the samples view.

        rocprof-sys is primarily a sampling profiler; for sampling runs the
        samples view (and regions_and_samples) carries the bulk of the data.
        This validator is the sampling counterpart to assert_hip_api_calls_present.

        Args:
            min_count: Minimum expected sample count (default: 1).
            category: If provided, count only samples whose category matches
                      (e.g. 'timer_sampling'). Bound as a parameter (no
                      interpolation).

        Returns:
            CheckResult: passed=True if sample count >= min_count.
        """
        if category is not None:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM samples WHERE category=?",
                (category,),
            ).fetchone()[0]
        else:
            count = self._conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        passed = count >= min_count
        result = CheckResult(
            passed=passed,
            validator_name="assert_samples_present",
            message=(
                f"Samples found: {count}"
                if passed
                else f"Expected >= {min_count} samples, found {count}"
            ),
            expected=f">= {min_count}",
            actual=count,
            details={"category": category, "count": count},
        )
        self._results.append(result)
        return result

    # -------------------------------------------------------------------------
    # ROCPD-07: assert_kernel_dispatches_valid
    # -------------------------------------------------------------------------

    def assert_kernel_dispatches_valid(self, min_count: int = 1) -> CheckResult:
        """Assert kernel dispatches in the kernels view are valid (non-null, positive grid).

        Checks that at least min_count kernels have a non-null name and
        positive grid dimensions (grid_x, grid_y, grid_z > 0).

        Args:
            min_count: Minimum expected valid kernel count (default: 1).

        Returns:
            CheckResult: passed=True if valid kernel count >= min_count.
        """
        count = self._conn.execute(
            "SELECT COUNT(*) FROM kernels WHERE name IS NOT NULL"
            " AND grid_x > 0 AND grid_y > 0 AND grid_z > 0"
        ).fetchone()[0]
        passed = count >= min_count
        result = CheckResult(
            passed=passed,
            validator_name="assert_kernel_dispatches_valid",
            message=(
                f"Valid kernel dispatches found: {count}"
                if passed
                else f"Expected >= {min_count} valid kernel dispatches, found {count}"
            ),
            expected=f">= {min_count}",
            actual=count,
        )
        self._results.append(result)
        return result

    # -------------------------------------------------------------------------
    # ROCPD-08: assert_memory_copies_present (conditional)
    # -------------------------------------------------------------------------

    def assert_memory_copies_present(
        self,
        direction: str | None = None,
        min_count: int = 1,
    ) -> CheckResult:
        """Assert memory copy operations are recorded (conditional skip when absent).

        If no memory copies are found, emits a warnings.warn and returns
        passed=True with a skip message. This reflects that memory copies are
        optional — not all runs perform DMA transfers.

        Args:
            direction: If provided, filter by name=direction (e.g., 'HtoD', 'DtoH').
            min_count: Minimum expected copy count when data IS present (default: 1).

        Returns:
            CheckResult: passed=True (skipped if empty, or count >= min_count if found).
        """
        if direction:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM memory_copies WHERE name=?",
                (direction,),
            ).fetchone()[0]
        else:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM memory_copies"
            ).fetchone()[0]

        if count == 0:
            warnings.warn(
                "memory_copies is empty — no memory copy operations recorded. Skipping.",
                stacklevel=2,
            )
            result = CheckResult(
                passed=True,
                validator_name="assert_memory_copies_present",
                message="Skipped: no memory copies found",
            )
            self._results.append(result)
            return result

        passed = count >= min_count
        result = CheckResult(
            passed=passed,
            validator_name="assert_memory_copies_present",
            message=(
                f"Memory copies found: {count}"
                if passed
                else f"Expected >= {min_count} memory copies, found {count}"
            ),
            expected=f">= {min_count}",
            actual=count,
            details={"direction": direction},
        )
        self._results.append(result)
        return result

    # -------------------------------------------------------------------------
    # ROCPD-09: assert_pmc_events_present (conditional)
    # -------------------------------------------------------------------------

    def assert_pmc_events_present(self, min_count: int = 1) -> CheckResult:
        """Assert PMC counter events are present (conditional skip when not collected).

        If pmc_events is empty, emits a warnings.warn and returns passed=True.
        The pmc_events view always exists in RocPD schema v3 but is empty when
        PMC collection was not enabled for the profiling run.

        Args:
            min_count: Minimum expected PMC event count when data IS present (default: 1).

        Returns:
            CheckResult: passed=True (skipped if empty, or count >= min_count if found).
        """
        count = self._conn.execute(
            "SELECT COUNT(*) FROM pmc_events"
        ).fetchone()[0]

        if count == 0:
            warnings.warn(
                "pmc_events is empty — PMC collection not enabled. Skipping.",
                stacklevel=2,
            )
            result = CheckResult(
                passed=True,
                validator_name="assert_pmc_events_present",
                message="Skipped: PMC collection not detected (pmc_events is empty)",
            )
            self._results.append(result)
            return result

        passed = count >= min_count
        result = CheckResult(
            passed=passed,
            validator_name="assert_pmc_events_present",
            message=(
                f"PMC events found: {count}"
                if passed
                else f"Expected >= {min_count} PMC events, found {count}"
            ),
            expected=f">= {min_count}",
            actual=count,
        )
        self._results.append(result)
        return result

    # -------------------------------------------------------------------------
    # ROCPD-10: assert_agent_info
    # -------------------------------------------------------------------------

    def assert_agent_info(
        self,
        expected_type: str = "GPU",
        expected_vendor: str = "AMD",
    ) -> CheckResult:
        """Assert agent/device information is present with expected type and vendor.

        Uses parameterized query to check rocpd_info_agent view for an entry
        matching (type, vendor_name) (Security: T-02-01-02).

        Args:
            expected_type: Expected device type (default: 'GPU').
            expected_vendor: Expected vendor name (default: 'AMD').

        Returns:
            CheckResult: passed=True if at least one matching agent row found.
        """
        rows = self._conn.execute(
            "SELECT type, vendor_name FROM rocpd_info_agent"
            " WHERE type=? AND vendor_name=?",
            (expected_type, expected_vendor),
        ).fetchall()
        passed = len(rows) > 0
        result = CheckResult(
            passed=passed,
            validator_name="assert_agent_info",
            message=(
                f"Agent info found: type={expected_type!r}, vendor={expected_vendor!r}"
                if passed
                else f"No agent entry with type={expected_type!r}, vendor_name={expected_vendor!r}"
            ),
            expected={"type": expected_type, "vendor_name": expected_vendor},
            actual={"row_count": len(rows)},
        )
        self._results.append(result)
        return result

    # -------------------------------------------------------------------------
    # ROCPD-12: assert_gpu_conditional
    # -------------------------------------------------------------------------

    def assert_gpu_conditional(
        self,
        counter_name: str,
        gpu_profile: GPUProfile | None = None,
    ) -> CheckResult:
        """Assert a GPU counter is available, skipping gracefully on unknown GPU.

        If gpu_profile is None, arch is 'unknown', or the counter is not in the
        profile's counter_names, emits warnings.warn and returns passed=True.

        Args:
            counter_name: Hardware counter name to check (e.g., 'SQ_WAVES').
            gpu_profile: GPUProfile instance. If None, skips automatically.

        Returns:
            CheckResult: passed=True (either counter available or skipped).
        """
        if (
            gpu_profile is None
            or gpu_profile.arch == "unknown"
            or not gpu_profile.has_counter(counter_name)
        ):
            warnings.warn(
                f"Counter {counter_name!r} not available on this GPU. Skipping.",
                stacklevel=2,
            )
            result = CheckResult(
                passed=True,
                validator_name="assert_gpu_conditional",
                message=f"Skipped: counter {counter_name!r} not available",
            )
            self._results.append(result)
            return result

        result = CheckResult(
            passed=True,
            validator_name="assert_gpu_conditional",
            message=f"Counter {counter_name!r} available on {gpu_profile.arch}",
        )
        self._results.append(result)
        return result

    # -------------------------------------------------------------------------
    # ROCPD-13: assert_region_args
    # -------------------------------------------------------------------------

    def assert_region_args(
        self,
        region_name: str,
        arg_name: str,
        expected_value: str | None = None,
    ) -> CheckResult:
        """Assert region argument annotations are present for a named region.

        CRITICAL: Joins region_args to regions ON guid (NOT id). The id columns
        are independent auto-increment values — joining on id would produce
        coincidental matches (Pitfall 3 in RESEARCH.md).

        Args:
            region_name: Region name to look up in the regions view.
            arg_name: Argument name to find in region_args.
            expected_value: If provided, also check that the arg value matches.

        Returns:
            CheckResult: passed=True if matching arg found (and value matches).
        """
        # CRITICAL: do NOT join the `regions` and `region_args` VIEWS directly.
        # Both are complex multi-join views (with correlated subqueries to
        # rocpd_string); joining the two expanded views on guid over the full
        # regions set produces a catastrophic, literal-dependent query plan that
        # can hang for minutes. Each view scans cheaply on its own, so resolve
        # the region guids first, then filter region_args by that bounded set —
        # preserving the guid-join semantics without the pathological plan.
        guid_rows = self._conn.execute(
            "SELECT guid FROM regions WHERE name = ? LIMIT 1000",
            (region_name,),
        ).fetchall()
        guids = [gr["guid"] for gr in guid_rows]

        if not guids:
            rows: list = []
        else:
            placeholders = ",".join("?" * len(guids))
            rows = self._conn.execute(
                f"SELECT ra.name, ra.type, ra.value FROM region_args ra "
                f"WHERE ra.name = ? AND ra.guid IN ({placeholders}) LIMIT 100",
                (arg_name, *guids),
            ).fetchall()

        passed = len(rows) > 0
        if passed and expected_value is not None:
            passed = rows[0]["value"] == expected_value

        result = CheckResult(
            passed=passed,
            validator_name="assert_region_args",
            message=(
                f"Region arg {arg_name!r} found for region {region_name!r}"
                if passed
                else (
                    f"Region arg {arg_name!r} not found for region {region_name!r}"
                    if len(rows) == 0
                    else f"Region arg {arg_name!r} value mismatch"
                )
            ),
            expected=expected_value,
            actual=rows[0]["value"] if rows else None,
            details={
                "region_name": region_name,
                "arg_name": arg_name,
                "row_count": len(rows),
                "join_key": "guid",  # Documents the critical join key
            },
        )
        self._results.append(result)
        return result

    # -------------------------------------------------------------------------
    # ROCPD-14: assert_schema_version
    # -------------------------------------------------------------------------

    def assert_schema_version(self, min_version: int = 1) -> CheckResult:
        """Assert the RocPD schema version meets a minimum requirement.

        Reads the 'schema_version' tag from rocpd_metadata and compares it
        to min_version. Returns passed=False if the tag is absent or the
        version is below the minimum.

        Args:
            min_version: Minimum acceptable schema version integer (default: 1).

        Returns:
            CheckResult: passed=True if actual_version >= min_version.
        """
        row = self._conn.execute(
            "SELECT value FROM rocpd_metadata WHERE tag='schema_version'"
        ).fetchone()

        if row is None:
            result = CheckResult(
                passed=False,
                validator_name="assert_schema_version",
                message="schema_version not found in rocpd_metadata",
                expected=f">= {min_version}",
                actual=None,
            )
            self._results.append(result)
            return result

        actual_version = int(row["value"])
        passed = actual_version >= min_version
        result = CheckResult(
            passed=passed,
            validator_name="assert_schema_version",
            message=(
                f"Schema version {actual_version} meets minimum {min_version}"
                if passed
                else f"Schema version {actual_version} is below minimum {min_version}"
            ),
            expected=f">= {min_version}",
            actual=actual_version,
        )
        self._results.append(result)
        return result

    # =========================================================================
    # Timeline / statistical / structural validators — mirror of PerfettoReader.
    # RocPD partitions by view (regions/kernels/memory_copies/pmc_events) plus
    # optional name/category/stream filters, using start/duration (ns) columns.
    # =========================================================================

    def _safe_view(self, view: str) -> str:
        """Validate a view/identifier before interpolating it into SQL."""
        if not _IDENT_RE.fullmatch(view):
            raise ValueError(f"invalid view name: {view!r}")
        return view

    def _view_rows(
        self,
        view: str,
        *,
        name: str | None = None,
        match: str = "exact",
        category: str | None = None,
        stream: str | None = None,
        marker: str = "",
    ) -> list:
        """Fetch (name, start, dur) rows from a view with bound-parameter filters."""
        v = self._safe_view(view)
        where: list[str] = []
        params: list = []
        if name is not None and match == "exact":
            where.append("name = ?")
            params.append(name)
        elif name is not None and match == "substring":
            esc = name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            where.append("name LIKE ? ESCAPE '\\'")
            params.append(f"%{esc}%")
        if category is not None:
            where.append("category = ?")
            params.append(category)
        if stream is not None:
            where.append("stream = ?")
            params.append(stream)
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""
        sql = (
            f"SELECT name AS name, start AS start, duration AS dur FROM {v}"
            f"{where_sql} ORDER BY start -- {marker}"
        )
        rows = self._conn.execute(sql, params).fetchall()
        if name is not None and match == "regex":
            pat = re.compile(name)
            rows = [r for r in rows if r["name"] is not None and pat.search(str(r["name"]))]
        return rows

    def _fail(self, validator: str, message: str, **details) -> CheckResult:
        result = CheckResult(
            passed=False, validator_name=validator, message=message,
            actual=details.pop("actual", None), details={**details, "db": self._path},
        )
        self._results.append(result)
        return result

    # --- temporal / concurrency ---------------------------------------------

    def assert_gpu_utilization(
        self,
        min_pct: float,
        *,
        view: str = "kernels",
        name: str | None = None,
        match: str = "exact",
        category: str | None = None,
        stream: str | None = None,
    ) -> CheckResult:
        """Assert records in a view cover at least ``min_pct``% of their active span."""
        rows = self._view_rows(view, name=name, match=match, category=category,
                               stream=stream, marker="rocpd_gpu_util")
        iv = [(int(r["start"]), int(r["dur"])) for r in rows if r["dur"] is not None]
        if not iv:
            return self._fail("assert_gpu_utilization", f"no records in {view!r}",
                              view=view, actual="(no records)")
        busy, span = _interval_union_ns(iv), _interval_span_ns(iv)
        util = (100.0 * busy / span) if span > 0 else 0.0
        passed = util >= min_pct
        result = CheckResult(
            passed=passed, validator_name="assert_gpu_utilization",
            message=f"{view!r} utilization {util:.1f}% (busy {busy} / span {span} ns)",
            expected=None if passed else f">= {min_pct}%",
            actual=None if passed else f"{util:.1f}%",
            details={"view": view, "utilization_pct": round(util, 3),
                     "busy_ns": busy, "span_ns": span, "db": self._path},
        )
        self._results.append(result)
        return result

    def assert_max_idle_gap(
        self,
        max_ns: int,
        *,
        view: str = "kernels",
        name: str | None = None,
        match: str = "exact",
        category: str | None = None,
        stream: str | None = None,
    ) -> CheckResult:
        """Assert no idle gap between consecutive records in a view exceeds ``max_ns``."""
        rows = self._view_rows(view, name=name, match=match, category=category,
                               stream=stream, marker="rocpd_idle_gap")
        iv = [(int(r["start"]), int(r["dur"])) for r in rows if r["dur"] is not None]
        if not iv:
            return self._fail("assert_max_idle_gap", f"no records in {view!r}",
                              view=view, actual="(no records)")
        gap = _max_gap_ns(iv)
        passed = gap <= max_ns
        result = CheckResult(
            passed=passed, validator_name="assert_max_idle_gap",
            message=(f"max idle gap in {view!r} is {gap} ns" if passed
                     else f"idle gap {gap} ns exceeds budget {max_ns} ns in {view!r}"),
            expected=None if passed else f"<= {max_ns} ns",
            actual=None if passed else f"{gap} ns",
            details={"view": view, "max_gap_ns": gap, "db": self._path},
        )
        self._results.append(result)
        return result

    def assert_overlap(
        self,
        name_a: str,
        name_b: str,
        *,
        view: str = "kernels",
        view_b: str | None = None,
        min_overlap_pct: float | None = None,
        max_overlap_pct: float | None = None,
        match: str = "exact",
    ) -> CheckResult:
        """Assert two record groups overlap in time within a percentage band."""
        rows_a = self._view_rows(view, name=name_a, match=match, marker="rocpd_overlap_a")
        rows_b = self._view_rows(view_b or view, name=name_b, match=match, marker="rocpd_overlap_b")
        iv_a = [(int(r["start"]), int(r["dur"])) for r in rows_a if r["dur"] is not None]
        iv_b = [(int(r["start"]), int(r["dur"])) for r in rows_b if r["dur"] is not None]
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
            passed=passed, validator_name="assert_overlap",
            message=f"{name_a!r} vs {name_b!r} overlap {pct:.1f}% ({overlap} ns)",
            expected=None if passed else " and ".join(bounds),
            actual=None if passed else f"{pct:.1f}%",
            details={"name_a": name_a, "name_b": name_b, "overlap_ns": overlap,
                     "overlap_pct": round(pct, 3), "db": self._path},
        )
        self._results.append(result)
        return result

    def assert_serial_on_stream(
        self,
        stream: str,
        *,
        view: str = "kernels",
    ) -> CheckResult:
        """Assert records on a given stream never overlap (strictly serial)."""
        rows = self._view_rows(view, stream=stream, marker="rocpd_serial")
        iv = [(int(r["start"]), int(r["dur"])) for r in rows if r["dur"] is not None]
        if not iv:
            return self._fail("assert_serial_on_stream",
                              f"no records on stream {stream!r} in {view!r}",
                              stream=stream, actual="(no records)")
        violations = _count_overlapping(iv)
        passed = violations == 0
        result = CheckResult(
            passed=passed, validator_name="assert_serial_on_stream",
            message=(f"stream {stream!r} is serial ({len(iv)} records)" if passed
                     else f"{violations} overlapping record(s) on stream {stream!r}"),
            expected=None if passed else "0 overlaps",
            actual=None if passed else f"{violations} overlaps",
            details={"stream": stream, "view": view, "overlaps": violations,
                     "record_count": len(iv), "db": self._path},
        )
        self._results.append(result)
        return result

    def assert_flow_latency(
        self,
        max_ns: int,
        *,
        from_category: str = "rocm_hip_api",
        pctile: float = 99.0,
    ) -> CheckResult:
        """Assert API-call → kernel-dispatch latency stays within ``max_ns``.

        Correlates ``regions`` to ``kernels`` via ``corr_id`` (the RocPD analog of
        a Perfetto flow), measuring ``kernel.start - region.start``. Rows with a
        non-positive ``corr_id`` (unset) are excluded.
        """
        try:
            rows = self._conn.execute(
                "SELECT (k.start - r.start) AS latency FROM regions r "
                "JOIN kernels k ON r.corr_id = k.corr_id "
                "WHERE r.corr_id > 0 AND r.category = ? -- rocpd_flow_latency",
                (from_category,),
            ).fetchall()
        except sqlite3.Error as exc:
            return self._fail("assert_flow_latency", f"correlation query failed: {exc}")
        lats = [int(r["latency"]) for r in rows if r["latency"] is not None]
        if not lats:
            return self._fail("assert_flow_latency",
                              f"no correlated {from_category!r} -> kernel pairs (corr_id)",
                              from_category=from_category, actual="(no correlated pairs)")
        value = _percentile(lats, pctile)
        passed = value <= max_ns
        result = CheckResult(
            passed=passed, validator_name="assert_flow_latency",
            message=f"p{pctile:g} {from_category!r}->kernel latency = {value:.0f} ns over {len(lats)} pair(s)",
            expected=None if passed else f"p{pctile:g} <= {max_ns} ns",
            actual=None if passed else f"{value:.0f} ns",
            details={"from_category": from_category, "pctile": pctile,
                     "latency_ns": value, "pair_count": len(lats), "db": self._path},
        )
        self._results.append(result)
        return result

    def assert_iteration_consistency(
        self,
        name: str,
        *,
        view: str = "kernels",
        count: int | None = None,
        max_cv: float | None = None,
        no_upward_trend: bool = False,
        trend_tol: float = 0.05,
        match: str = "exact",
    ) -> CheckResult:
        """Assert repeated records form a stable iteration (count / CV / trend)."""
        rows = self._view_rows(view, name=name, match=match, marker="rocpd_iterations")
        durs = [int(r["dur"]) for r in rows if r["dur"] is not None]
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
        ratio = None
        if no_upward_trend:
            if n < 4:
                issues.append(f"need >= 4 iterations for trend, found {n}")
            else:
                half = n // 2
                first = sum(durs[:half]) / half
                second = sum(durs[half:]) / (n - half)
                ratio = (second / first) if first else float("inf")
                if ratio > 1.0 + trend_tol:
                    issues.append(
                        f"upward trend: second-half {second:.0f} > first-half {first:.0f} "
                        f"by >{trend_tol:.0%}"
                    )
        passed = not issues
        result = CheckResult(
            passed=passed, validator_name="assert_iteration_consistency",
            message=(f"{name!r} iterations consistent ({n} samples)" if passed
                     else f"{name!r} iteration issues: " + "; ".join(issues)),
            expected=None if passed else "stable iterations",
            actual=None if passed else "; ".join(issues),
            details={"name": name, "view": view, "count": n, "cv": cv,
                     "second_first_ratio": ratio, "db": self._path},
        )
        self._results.append(result)
        return result

    # --- statistical / distributional ---------------------------------------

    def assert_duration_distribution(
        self,
        name: str,
        *,
        view: str = "kernels",
        p50_range: tuple[float, float] | None = None,
        p95_max_ns: float | None = None,
        p99_max_ns: float | None = None,
        min_ns: float | None = None,
        max_ns: float | None = None,
        match: str = "exact",
    ) -> CheckResult:
        """Assert percentile bounds on the duration distribution of matching records."""
        rows = self._view_rows(view, name=name, match=match, marker="rocpd_distribution")
        durs = [int(r["dur"]) for r in rows if r["dur"] is not None]
        if not durs:
            return self._fail("assert_duration_distribution",
                              f"no records matched {name!r} in {view!r}",
                              name=name, actual="(no records)")
        p50, p95, p99 = _percentile(durs, 50), _percentile(durs, 95), _percentile(durs, 99)
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
            passed=passed, validator_name="assert_duration_distribution",
            message=(f"{name!r} durations: p50={p50:.0f} p95={p95:.0f} p99={p99:.0f} ns (n={len(durs)})"
                     if passed else f"{name!r} distribution issues: " + "; ".join(issues)),
            expected=None if passed else "; ".join(issues),
            actual=None if passed else f"p50={p50:.0f} p95={p95:.0f} p99={p99:.0f}",
            details={"name": name, "view": view, "p50": p50, "p95": p95, "p99": p99,
                     "min": lo, "max": hi, "count": len(durs), "db": self._path},
        )
        self._results.append(result)
        return result

    def assert_no_duration_outliers(
        self,
        name: str,
        *,
        view: str = "kernels",
        sigma: float = 4.0,
        max_outliers: int = 0,
        match: str = "exact",
    ) -> CheckResult:
        """Assert at most ``max_outliers`` durations lie beyond ``sigma`` SD of the mean."""
        rows = self._view_rows(view, name=name, match=match, marker="rocpd_outliers")
        durs = [int(r["dur"]) for r in rows if r["dur"] is not None]
        mean, std = _mean_std(durs)
        outliers = [d for d in durs if std > 0 and abs(d - mean) > sigma * std]
        passed = len(outliers) <= max_outliers
        result = CheckResult(
            passed=passed, validator_name="assert_no_duration_outliers",
            message=f"{name!r}: {len(outliers)} outlier(s) beyond {sigma}σ (n={len(durs)})",
            expected=None if passed else f"<= {max_outliers} outliers",
            actual=None if passed else f"{len(outliers)} outliers",
            details={"name": name, "view": view, "outliers": len(outliers),
                     "sigma": sigma, "mean": mean, "std": std, "count": len(durs),
                     "db": self._path},
        )
        self._results.append(result)
        return result

    def assert_counter_in_range(
        self,
        counter_name: str,
        *,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> CheckResult:
        """Assert all values of a PMC counter stay within [min_val, max_val]."""
        row = self._conn.execute(
            "SELECT MIN(counter_value) AS lo, MAX(counter_value) AS hi FROM pmc_events "
            "WHERE counter_name = ? -- rocpd_counter_range",
            (counter_name,),
        ).fetchone()
        if row is None or row["lo"] is None:
            return self._fail("assert_counter_in_range",
                              f"no PMC samples for {counter_name!r}",
                              counter_name=counter_name, actual="(no samples)")
        lo, hi = float(row["lo"]), float(row["hi"])
        issues = []
        if min_val is not None and lo < min_val:
            issues.append(f"min {lo} < {min_val}")
        if max_val is not None and hi > max_val:
            issues.append(f"max {hi} > {max_val}")
        passed = not issues
        result = CheckResult(
            passed=passed, validator_name="assert_counter_in_range",
            message=(f"{counter_name!r} in [{lo}, {hi}]" if passed
                     else f"{counter_name!r} out of range: " + "; ".join(issues)),
            expected=None if passed else f"[{min_val}, {max_val}]",
            actual=None if passed else f"[{lo}, {hi}]",
            details={"counter_name": counter_name, "observed_min": lo,
                     "observed_max": hi, "db": self._path},
        )
        self._results.append(result)
        return result

    def assert_counter_rate(
        self,
        counter_name: str,
        *,
        max_per_sec: float | None = None,
        min_per_sec: float | None = None,
    ) -> CheckResult:
        """Assert the per-second rate of change of a PMC counter stays within bounds."""
        rows = self._conn.execute(
            "SELECT start AS ts, counter_value AS value FROM pmc_events "
            "WHERE counter_name = ? ORDER BY start -- rocpd_counter_rate",
            (counter_name,),
        ).fetchall()
        pts = [(int(r["ts"]), float(r["value"])) for r in rows if r["value"] is not None]
        if len(pts) < 2:
            return self._fail("assert_counter_rate",
                              f"need >= 2 PMC samples for {counter_name!r}, found {len(pts)}",
                              counter_name=counter_name, actual=f"{len(pts)} samples")
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
            passed=passed, validator_name="assert_counter_rate",
            message=(f"{counter_name!r} rate in [{min_rate:.3g}, {max_rate:.3g}]/s" if passed
                     else f"{counter_name!r} rate out of bounds: " + "; ".join(issues)),
            expected=None if passed else "; ".join(issues),
            actual=None if passed else f"[{min_rate:.3g}, {max_rate:.3g}]/s",
            details={"counter_name": counter_name, "max_rate": max_rate,
                     "min_rate": min_rate, "samples": len(pts), "db": self._path},
        )
        self._results.append(result)
        return result

    # --- sequence / structural ----------------------------------------------

    def assert_record_order(
        self,
        view: str,
        *steps,
        match: str = "exact",
        category: str | None = None,
        stream: str | None = None,
    ) -> CheckResult:
        """Assert records in a view occur in a given order (mirror of slice_order).

        Steps are ``"name"``, ``["name", count]``, or ``...`` / ``ANYTHING``.
        """
        rows = self._view_rows(view, category=category, stream=stream, marker="rocpd_record_order")
        seq = [r["name"] for r in rows]
        try:
            tokens = _expand_slice_order_steps(steps)
        except ValueError as exc:
            return self._fail("assert_record_order", str(exc), view=view)
        ok = _match_slice_order(seq, tokens, match)
        preview = seq[:25]
        result = CheckResult(
            passed=ok, validator_name="assert_record_order",
            message=(f"record order matches in {view!r} ({len(seq)} records)" if ok
                     else f"record order mismatch in {view!r}"),
            expected=None if ok else _fmt_slice_order(steps),
            actual=None if ok else f"{len(seq)} records: {preview}{' ...' if len(seq) > 25 else ''}",
            details={"view": view, "record_count": len(seq), "match": match, "db": self._path},
        )
        self._results.append(result)
        return result

    def assert_call_tree(
        self,
        parent_name: str,
        *,
        contains: list[str] | None = None,
        max_depth: int | None = None,
        no_recursion: bool = False,
        match: str = "exact",
        max_parents: int = 1000,
    ) -> CheckResult:
        """Assert structural properties of the region subtree rooted at a parent.

        Children are determined by timestamp containment within the same
        (pid, tid); relative depth is the nesting level below the parent.
        """
        if match == "exact":
            parents = self._conn.execute(
                "SELECT name, start, duration, pid, tid FROM regions WHERE name = ? "
                "LIMIT ? -- rocpd_call_tree_parents",
                (parent_name, max_parents),
            ).fetchall()
        else:
            allp = self._conn.execute(
                "SELECT name, start, duration, pid, tid FROM regions "
                "LIMIT 200000 -- rocpd_call_tree_parents"
            ).fetchall()
            if match == "substring":
                parents = [r for r in allp if r["name"] and parent_name in r["name"]][:max_parents]
            else:
                pat = re.compile(parent_name)
                parents = [r for r in allp if r["name"] and pat.search(str(r["name"]))][:max_parents]

        if not parents:
            return self._fail("assert_call_tree", f"no region matched {parent_name!r}",
                              parent=parent_name, actual="(parent not found)")

        contains = contains or []
        found_children: set[str] = set()
        max_rel_depth = 0
        recursion_hit = False
        for p in parents:
            p_start, p_end = int(p["start"]), int(p["start"]) + int(p["duration"])
            sub = self._conn.execute(
                "SELECT name, start, duration FROM regions WHERE pid = ? AND tid = ? "
                "AND start >= ? AND (start + duration) <= ? "
                "AND NOT (start = ? AND duration = ?) -- rocpd_call_tree_sub",
                (p["pid"], p["tid"], p_start, p_end, p_start, int(p["duration"])),
            ).fetchall()
            spans = [(int(s["start"]), int(s["start"]) + int(s["duration"]), s["name"]) for s in sub]
            for s_start, s_end, s_name in spans:
                # relative depth = 1 + number of other sub-regions that strictly contain it
                depth = 1 + sum(
                    1 for o_s, o_e, _ in spans
                    if (o_s <= s_start and s_end <= o_e) and (o_s, o_e) != (s_start, s_end)
                )
                found_children.add(s_name)
                max_rel_depth = max(max_rel_depth, depth)
                if no_recursion and s_name == p["name"]:
                    recursion_hit = True

        issues = []
        missing = [c for c in contains
                   if not any(_slice_name_match(fc, c, match) for fc in found_children)]
        if missing:
            issues.append(f"missing children {missing}")
        if max_depth is not None and max_rel_depth > max_depth:
            issues.append(f"depth {max_rel_depth} > {max_depth}")
        if no_recursion and recursion_hit:
            issues.append("unexpected recursion (descendant shares parent name)")
        passed = not issues
        result = CheckResult(
            passed=passed, validator_name="assert_call_tree",
            message=(f"call tree under {parent_name!r} ok (depth {max_rel_depth})" if passed
                     else f"call tree under {parent_name!r} issues: " + "; ".join(issues)),
            expected=None if passed else "; ".join(issues),
            actual=None if passed else sorted(found_children),
            details={"parent": parent_name, "children_found": sorted(found_children),
                     "max_rel_depth": max_rel_depth, "db": self._path},
        )
        self._results.append(result)
        return result

    def assert_region_args_present(
        self,
        region_name: str,
        require: list[str],
        *,
        types: dict[str, str] | None = None,
        max_regions: int = 1000,
    ) -> CheckResult:
        """Assert every instance of a region carries the required arg names.

        Mirror of Perfetto ``assert_slice_args``. Joins ``region_args`` to
        ``regions`` on ``guid`` (per the RocPD pitfall). ``types`` optionally maps
        an arg name to a substring that must appear in its recorded ``type``.
        """
        guid_rows = self._conn.execute(
            "SELECT guid FROM regions WHERE name = ? LIMIT ? -- rocpd_args_guids",
            (region_name, max_regions),
        ).fetchall()
        guids = [g["guid"] for g in guid_rows]
        if not guids:
            return self._fail("assert_region_args_present",
                              f"no regions named {region_name!r}",
                              region_name=region_name, actual="(no regions)")
        gph = ",".join("?" * len(guids))
        nph = ",".join("?" * len(require))
        rows = self._conn.execute(
            f"SELECT guid, name, type FROM region_args "
            f"WHERE guid IN ({gph}) AND name IN ({nph}) -- rocpd_args",
            (*guids, *require),
        ).fetchall()
        per: dict = {g: {} for g in guids}
        for r in rows:
            per[r["guid"]][r["name"]] = r["type"]
        missing: dict[str, int] = {}
        type_mismatch: dict[str, int] = {}
        for g in guids:
            for nm in require:
                if nm not in per[g]:
                    missing[nm] = missing.get(nm, 0) + 1
                elif types and nm in types and types[nm] not in (per[g][nm] or ""):
                    type_mismatch[nm] = type_mismatch.get(nm, 0) + 1
        passed = not missing and not type_mismatch
        parts = []
        if missing:
            parts.append("missing " + ", ".join(f"{k}(×{v})" for k, v in missing.items()))
        if type_mismatch:
            parts.append("type-mismatch " + ", ".join(f"{k}(×{v})" for k, v in type_mismatch.items()))
        result = CheckResult(
            passed=passed, validator_name="assert_region_args_present",
            message=(f"all {len(guids)} {region_name!r} regions carry required args" if passed
                     else f"{region_name!r} arg issues: " + "; ".join(parts)),
            expected=None if passed else require,
            actual=None if passed else {"missing": missing, "type_mismatch": type_mismatch},
            details={"region_name": region_name, "region_count": len(guids),
                     "missing": missing, "type_mismatch": type_mismatch,
                     "truncated": len(guids) >= max_regions, "db": self._path},
        )
        self._results.append(result)
        return result

    # --- anti-patterns -------------------------------------------------------

    def assert_no_anti_patterns(
        self,
        *,
        negative_durations: bool = True,
        duplicate_records: bool = True,
        zero_duration: bool = False,
        giant_record: bool = True,
        giant_pct: float = 99.0,
        views: tuple[str, ...] = ("regions", "kernels"),
    ) -> CheckResult:
        """Assert a curated bundle of timeline anti-patterns is absent across views."""
        found: dict[str, int] = {}
        for view in views:
            v = self._safe_view(view)
            part = _VIEW_PARTITION.get(view)
            try:
                if negative_durations:
                    n = self._conn.execute(
                        f"SELECT COUNT(*) FROM {v} WHERE duration < 0 -- rocpd_anti_neg"
                    ).fetchone()[0]
                    if n:
                        found[f"{view}.negative_durations"] = n
                if zero_duration:
                    n = self._conn.execute(
                        f"SELECT COUNT(*) FROM {v} WHERE duration = 0 -- rocpd_anti_zero"
                    ).fetchone()[0]
                    if n:
                        found[f"{view}.zero_duration"] = n
                if duplicate_records:
                    key = "start, name" + (f", {part}" if part else "")
                    n = self._conn.execute(
                        f"SELECT COUNT(*) FROM (SELECT {key}, COUNT(*) AS c FROM {v} "
                        f"GROUP BY {key} HAVING c > 1) -- rocpd_anti_dup"
                    ).fetchone()[0]
                    if n:
                        found[f"{view}.duplicate_records"] = n
                if giant_record and part:
                    sp = self._conn.execute(
                        f"SELECT MIN(start) AS lo, MAX(start + duration) AS hi FROM {v} "
                        f"WHERE duration >= 0 -- rocpd_anti_span"
                    ).fetchone()
                    if sp["lo"] is not None:
                        span = int(sp["hi"]) - int(sp["lo"])
                        if span > 0:
                            thr = span * (giant_pct / 100.0)
                            n = self._conn.execute(
                                f"SELECT COUNT(*) FROM (SELECT {part}, COUNT(*) AS c, "
                                f"MAX(duration) AS md FROM {v} WHERE duration >= 0 "
                                f"GROUP BY {part} HAVING c = 1 AND md >= {thr}) -- rocpd_anti_giant"
                            ).fetchone()[0]
                            if n:
                                found[f"{view}.single_giant_record"] = n
            except sqlite3.Error:
                continue  # view absent or lacks expected columns in this DB
        passed = not found
        result = CheckResult(
            passed=passed, validator_name="assert_no_anti_patterns",
            message=("no anti-patterns detected" if passed
                     else "anti-patterns detected: " + ", ".join(f"{k}={v}" for k, v in found.items())),
            expected=None if passed else "no anti-patterns",
            actual=None if passed else found,
            details={"found": found, "db": self._path},
        )
        self._results.append(result)
        return result
