# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ROCPD validators for events outside ROCPROFSYS_SELECTED_REGIONS windows."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .region_filter_leakage import parse_selected_regions
from .validators import ValidationResult

_DEFAULT_TOLERANCE_NS = 1_000_000


@dataclass(frozen=True)
class RocpdLeakageCheck:
    label: str
    table: str
    category_glob: Optional[str] = None
    name_glob: Optional[str] = None
    name_not_glob: Optional[str] = None
    exclude_name_globs: tuple[str, ...] = ()
    start_column: str = "start"
    end_column: str = "end"


DEFAULT_ROCPD_LEAKAGE_CHECKS: tuple[RocpdLeakageCheck, ...] = (
    RocpdLeakageCheck(
        label="pthread regions",
        table="regions",
        category_glob="*pthread*",
    ),
    RocpdLeakageCheck(
        label="HIP API regions",
        table="regions",
        category_glob="*hip*",
        exclude_name_globs=("CodeBlock_*",),
    ),
    RocpdLeakageCheck(
        label="filtered-out ROCTx markers",
        table="regions",
        category_glob="*marker*",
        name_glob="Region*",
    ),
    RocpdLeakageCheck(
        label="kernel dispatches",
        table="kernels",
        name_glob="CodeBlock_*",
    ),
    RocpdLeakageCheck(
        label="timer sampling",
        table="samples",
        category_glob="*timer_sampling*",
        start_column="timestamp",
        end_column="timestamp",
    ),
)


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_iv[0]]
    for start, end in sorted_iv[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _fully_inside(
    start: int, end: int, allowed: list[tuple[int, int]], tolerance_ns: int
) -> bool:
    if not allowed:
        return False
    for win_start, win_end in allowed:
        if start >= win_start - tolerance_ns and end <= win_end + tolerance_ns:
            return True
    return False


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _fetch_marker_windows(
    conn: sqlite3.Connection, selected_regions: list[str]
) -> list[tuple[int, int]]:
    if not _table_exists(conn, "regions"):
        return []
    placeholders = ", ".join("?" for _ in selected_regions)
    query = f"""
        SELECT start, end
        FROM regions
        WHERE category LIKE '%marker%'
          AND name IN ({placeholders})
        ORDER BY start
    """
    rows = conn.execute(query, selected_regions).fetchall()
    intervals = []
    for start, end in rows:
        s, e = int(start), int(end)
        if e <= s:
            e = s + 1
        intervals.append((s, e))
    return _merge_intervals(intervals)


def _violations_for_check(
    conn: sqlite3.Connection,
    check: RocpdLeakageCheck,
    allowed: list[tuple[int, int]],
    selected_regions: list[str],
    tolerance_ns: int,
    max_report: int,
) -> list[str]:
    if not _table_exists(conn, check.table):
        return [f"  - missing table/view: {check.table}"]

    clauses = ["1=1"]
    params: list[str] = []
    if check.category_glob:
        clauses.append("category GLOB ?")
        params.append(check.category_glob)
    if check.name_glob:
        clauses.append("name GLOB ?")
        params.append(check.name_glob)
    if check.name_not_glob:
        clauses.append("name NOT GLOB ?")
        params.append(check.name_not_glob)
    if check.label == "filtered-out ROCTx markers" and selected_regions:
        placeholders = ", ".join("?" for _ in selected_regions)
        clauses.append(f"name NOT IN ({placeholders})")
        params.extend(selected_regions)

    for glob in check.exclude_name_globs:
        clauses.append("name NOT GLOB ?")
        params.append(glob)

    start_col = check.start_column
    end_col = check.end_column
    query = f"""
        SELECT category, name, {start_col}, {end_col}
        FROM {check.table}
        WHERE {' AND '.join(clauses)}
        ORDER BY {start_col}
        LIMIT 500
    """
    violations: list[str] = []
    for row in conn.execute(query, params).fetchall():
        category, name, start, end = row[0], row[1], int(row[2]), int(row[3])
        slice_end = end if end > start else start + 1
        if _fully_inside(int(start), slice_end, allowed, tolerance_ns):
            continue
        violations.append(
            f"  - [{category}] {name} @ {start}..{slice_end}"
        )
        if len(violations) >= max_report:
            violations.append("  - ... (truncated)")
            break
    return violations


def validate_region_filter_rocpd_leakage(
    db_path: Path,
    selected_regions: str | list[str],
    checks: Optional[tuple[RocpdLeakageCheck, ...]] = None,
    *,
    tolerance_ns: int = _DEFAULT_TOLERANCE_NS,
) -> ValidationResult:
    """Fail when ROCPD rows fall outside selected ROCTx region time windows."""
    if not db_path.exists():
        return ValidationResult(False, f"ROCPD database not found: {db_path}")

    regions = parse_selected_regions(selected_regions)
    if not regions:
        return ValidationResult(False, "No selected regions provided")

    check_list = checks if checks is not None else DEFAULT_ROCPD_LEAKAGE_CHECKS

    try:
        conn = sqlite3.connect(str(db_path))
    except sqlite3.Error as exc:
        return ValidationResult(False, f"Could not open ROCPD database: {exc}")

    try:
        allowed = _fetch_marker_windows(conn, regions)
        if not allowed:
            return ValidationResult(
                False,
                f"No marker regions rows found for {regions!r} in ROCPD",
            )

        sections: list[str] = [
            f"Selected regions: {', '.join(regions)}",
            f"ROCPD allowed windows: {len(allowed)} merged interval(s)",
        ]
        failed = False
        for check in check_list:
            violations = _violations_for_check(
                conn, check, allowed, regions, tolerance_ns, max_report=8
            )
            if violations:
                failed = True
                sections.append(f"{check.label} outside region windows:")
                sections.extend(violations)

        if failed:
            return ValidationResult(False, "\n".join(sections))
        return ValidationResult(
            True,
            "No ROCPD leakage outside selected region windows for configured checks",
        )
    finally:
        conn.close()
