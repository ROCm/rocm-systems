# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Baseline snapshot system for trace regression detection.

Provides `assert_baseline()` for capturing and comparing stable structural
invariants (track names, event counts, categories) of Perfetto, RocPD, and
timemory traces to JSON files.

Captured invariants intentionally exclude non-deterministic fields:
  - No `ts` (absolute timestamps)
  - No `dur` (durations)
  - No `start` / `end` timestamps
  - No `pid` / `tid` (process/thread IDs)

Typical workflow::

    # First run: capture the baseline
    assert_baseline(reader, Path(__file__).parent / "__snapshots__" / "my_test.snap.json", capture=True)

    # Subsequent runs: compare against baseline
    assert_baseline(reader, Path(__file__).parent / "__snapshots__" / "my_test.snap.json")
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rocprofsys_validator.readers.perfetto import PerfettoReader
    from rocprofsys_validator.readers.rocpd import RocpdReader
    from rocprofsys_validator.readers.timemory import TimemoryReader

def _update_mode_requested() -> bool:
    """True when baseline update mode is active (env var set by --baseline-update)."""
    return bool(os.environ.get("ROCPROFSYS_BASELINE_UPDATE"))

def assert_baseline(
    reader: object,
    snapshot_path: str | Path,
    capture: bool = False,
    *,
    rel_tol: float = 0.0,
    abs_tol: float = 0.0,
) -> None:
    """Assert a trace matches its captured baseline snapshot.

    IMPORTANT: snapshot_path must be an absolute path. Relative paths resolve
    against the current working directory which may not be the test file's
    directory when pytest changes directories. Use::

        Path(__file__).parent / "__snapshots__" / "my_test.snap.json"

    Numeric values (slice counts, counter min/max/count) are compared with the
    given tolerances so inherently noisy metrics do not produce flaky failures
    while still catching real drift. Names and categories are always compared
    exactly.

    Update workflow: pass ``capture=True`` to (re)write the snapshot, or run the
    whole suite with ``--baseline-update`` (sets ROCPROFSYS_BASELINE_UPDATE),
    which makes every assert_baseline call (re)capture instead of compare.

    Args:
        reader: A PerfettoReader, RocpdReader, or TimemoryReader instance.
        snapshot_path: Path to the .snap.json file (absolute path recommended).
        capture: If True, write the snapshot file (overwriting any existing file);
                 parent directories are created automatically.
                 If False (default), compare against existing snapshot.
        rel_tol: Relative tolerance for numeric comparisons (e.g. 0.05 = 5%).
        abs_tol: Absolute tolerance for numeric comparisons.

    Raises:
        FileNotFoundError: If comparing and no snapshot file exists (and update
            mode is off).
        AssertionError: If capture=False and the trace differs from the stored snapshot.
        TypeError: If reader is not a supported reader type.
    """
    path = Path(snapshot_path)
    if capture or _update_mode_requested():
        _capture_snapshot(reader, path)
    else:
        _compare_snapshot(reader, path, rel_tol=rel_tol, abs_tol=abs_tol)

def _build_snapshot_dict(reader: object) -> dict:
    """Build the snapshot dict for the given reader without writing to disk.

    This private helper is used by both _capture_snapshot and _compare_snapshot
    to avoid duplicating dispatch logic.

    Args:
        reader: A PerfettoReader, RocpdReader, or TimemoryReader instance.

    Returns:
        dict: Snapshot data containing stable structural invariants.

    Raises:
        TypeError: If reader is not a supported type.
    """
    from rocprofsys_validator.readers.perfetto import PerfettoReader
    from rocprofsys_validator.readers.rocpd import RocpdReader
    from rocprofsys_validator.readers.timemory import TimemoryReader

    if isinstance(reader, PerfettoReader):
        return _perfetto_snapshot(reader)
    elif isinstance(reader, RocpdReader):
        return _rocpd_snapshot(reader)
    elif isinstance(reader, TimemoryReader):
        return _timemory_snapshot(reader)
    raise TypeError(
        f"assert_baseline() does not support reader type: {type(reader).__name__}"
    )

def _capture_snapshot(reader: object, path: Path) -> None:
    """Write a structural snapshot of the reader's trace to path.

    Parent directories are created automatically. If the file already exists,
    it is overwritten.

    Args:
        reader: A supported reader instance.
        path: Absolute path to write the snapshot JSON file.
    """
    data = _build_snapshot_dict(reader)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False))

def _values_match(stored: object, current: object, rel_tol: float, abs_tol: float) -> bool:
    """Compare two snapshot values, applying tolerance to numbers (recursively).

    Numbers use math.isclose(rel_tol, abs_tol). Dicts compare key-wise (a missing
    key on either side is a mismatch). Everything else (lists of names, strings,
    None) is compared for exact equality.
    """
    if isinstance(stored, bool) or isinstance(current, bool):
        return stored == current
    if isinstance(stored, (int, float)) and isinstance(current, (int, float)):
        return math.isclose(stored, current, rel_tol=rel_tol, abs_tol=abs_tol)
    if isinstance(stored, dict) and isinstance(current, dict):
        if set(stored) != set(current):
            return False
        return all(
            _values_match(stored[k], current[k], rel_tol, abs_tol) for k in stored
        )
    return stored == current

def _compare_snapshot(
    reader: object,
    path: Path,
    rel_tol: float = 0.0,
    abs_tol: float = 0.0,
) -> None:
    """Compare reader's current structural state against a stored snapshot.

    Args:
        reader: A supported reader instance.
        path: Path to the .snap.json snapshot file.
        rel_tol: Relative tolerance for numeric comparisons.
        abs_tol: Absolute tolerance for numeric comparisons.

    Raises:
        FileNotFoundError: If no snapshot file exists at path.
        AssertionError: If the current trace state differs from the stored snapshot.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No snapshot found at {path}. "
            f"Run with capture=True (or --baseline-update) first to create it."
        )
    stored = json.loads(path.read_text())
    current = _build_snapshot_dict(reader)

    diffs: list[str] = []
    for key in stored:
        if key not in current:
            diffs.append(f"  {key!r}: in snapshot but missing in current trace")
        elif not _values_match(stored[key], current[key], rel_tol, abs_tol):
            diffs.append(f"  {key!r}:\n    was: {stored[key]!r}\n    now: {current[key]!r}")
    for key in current:
        if key not in stored:
            diffs.append(f"  {key!r}: new key in current trace, absent from snapshot")

    if diffs:
        raise AssertionError(
            f"Trace does not match baseline snapshot {path}:\n" + "\n".join(diffs)
        )

def _perfetto_snapshot(reader: "PerfettoReader") -> dict:
    """Capture stable structural invariants from a Perfetto trace.

    Queries track names, slice categories with counts, and counter track names.
    Never includes timestamps (ts), durations (dur), or process/thread IDs (pid/tid).

    Args:
        reader: A PerfettoReader with an active TraceProcessor connection.

    Returns:
        dict: Snapshot with keys: format, track_names, slice_categories,
              slice_counts_by_category, counter_track_names.
    """
    # Use execute_sql for structural queries — the assert_* methods append to
    # reader._results which we must not pollute with baseline capture calls.
    track_df = reader.execute_sql(
        "SELECT DISTINCT name FROM track WHERE name IS NOT NULL ORDER BY name"
    )
    cat_df = reader.execute_sql(
        "SELECT category, COUNT(*) AS cnt FROM slice "
        "WHERE category IS NOT NULL GROUP BY category ORDER BY category"
    )
    counter_df = reader.execute_sql(
        "SELECT DISTINCT name FROM counter_track WHERE name IS NOT NULL ORDER BY name"
    )
    # Per-counter value range (min/max/count). This is what catches a counter
    # saturating to 0 or a metric range collapsing — regressions that name-only
    # snapshots miss. Values are rounded to damp floating-point noise; the
    # tolerance-aware comparison handles legitimate run-to-run drift.
    counter_range_df = reader.execute_sql(
        "SELECT ct.name AS name, MIN(c.value) AS minv, MAX(c.value) AS maxv, "
        "COUNT(*) AS cnt FROM counter c "
        "JOIN counter_track ct ON c.track_id = ct.id "
        "WHERE ct.name IS NOT NULL GROUP BY ct.name ORDER BY ct.name"
    )

    track_names = sorted(track_df["name"].tolist() if not track_df.empty else [])
    categories = sorted(cat_df["category"].tolist() if not cat_df.empty else [])
    counts: dict[str, int] = (
        dict(zip(cat_df["category"].tolist(), [int(c) for c in cat_df["cnt"].tolist()]))
        if not cat_df.empty
        else {}
    )
    counter_names = sorted(counter_df["name"].tolist() if not counter_df.empty else [])

    counter_value_ranges: dict[str, dict[str, float]] = {}
    if not counter_range_df.empty:
        for _, row in counter_range_df.iterrows():
            counter_value_ranges[str(row["name"])] = {
                "min": round(float(row["minv"]), 6),
                "max": round(float(row["maxv"]), 6),
                "count": int(row["cnt"]),
            }

    return {
        "format": "perfetto",
        "track_names": track_names,
        "slice_categories": categories,
        "slice_counts_by_category": counts,
        "counter_track_names": counter_names,
        "counter_value_ranges": counter_value_ranges,
    }

def _rocpd_snapshot(reader: "RocpdReader") -> dict:
    """Capture stable structural invariants from a RocPD SQLite database.

    Queries view presence, row counts, HIP API call count, kernel dispatch count,
    agent types, and schema version. Never includes timestamps or raw IDs.

    Args:
        reader: A RocpdReader with an active SQLite connection.

    Returns:
        dict: Snapshot with keys: format, views_present, row_counts,
              hip_api_call_count, kernel_dispatch_count, agent_types, schema_version.
    """
    import sqlite3

    from rocprofsys_validator.readers.rocpd import _EXPECTED_VIEWS

    # 1. Views present
    views_rows = reader.execute_sql(
        "SELECT name FROM sqlite_master WHERE type='view' ORDER BY name"
    )
    views_present = sorted([row["name"] for row in views_rows])

    # 2. Row counts for each of the 10 canonical views
    row_counts: dict[str, int] = {}
    for view in _EXPECTED_VIEWS:
        try:
            rows = reader.execute_sql(f"SELECT COUNT(*) AS cnt FROM {view}")
            row_counts[view] = int(rows[0]["cnt"]) if rows else 0
        except sqlite3.OperationalError:
            row_counts[view] = 0

    # 3. HIP API call count
    try:
        hip_rows = reader.execute_sql(
            "SELECT COUNT(*) AS cnt FROM regions WHERE category = 'rocm_hip_api'"
        )
        hip_api_call_count = int(hip_rows[0]["cnt"]) if hip_rows else 0
    except sqlite3.OperationalError:
        hip_api_call_count = 0

    # 4. Kernel dispatch count
    try:
        kernel_rows = reader.execute_sql("SELECT COUNT(*) AS cnt FROM kernels")
        kernel_dispatch_count = int(kernel_rows[0]["cnt"]) if kernel_rows else 0
    except sqlite3.OperationalError:
        kernel_dispatch_count = 0

    # 5. Agent types
    try:
        agent_rows = reader.execute_sql(
            "SELECT DISTINCT type FROM rocpd_info_agent WHERE type IS NOT NULL ORDER BY type"
        )
        agent_types = sorted([row["type"] for row in agent_rows])
    except sqlite3.OperationalError:
        agent_types = []

    # 6. Schema version
    try:
        ver_rows = reader.execute_sql(
            "SELECT value FROM rocpd_metadata WHERE tag = 'schema_version' LIMIT 1"
        )
        schema_version = int(ver_rows[0]["value"]) if ver_rows else None
    except (sqlite3.OperationalError, ValueError, TypeError):
        schema_version = None

    return {
        "format": "rocpd",
        "views_present": views_present,
        "row_counts": row_counts,
        "hip_api_call_count": hip_api_call_count,
        "kernel_dispatch_count": kernel_dispatch_count,
        "agent_types": agent_types,
        "schema_version": schema_version,
    }

def _timemory_snapshot(reader: "TimemoryReader") -> dict:
    """Capture stable structural invariants from a timemory metric directory.

    Checks which .txt files are present and counts labels in each file.
    Only captures integer row counts — no float metric values.

    Args:
        reader: A TimemoryReader with a valid directory path.

    Returns:
        dict: Snapshot with keys: format, files_present, label_counts.
    """
    from rocprofsys_validator.readers.timemory import EXPECTED_STEMS

    present_stems: list[str] = []
    label_counts: dict[str, int] = {}

    for stem in EXPECTED_STEMS:
        # NOTE: _dir and _parse_file() are private TimemoryReader attributes.
        # This coupling is intentional — baseline snapshot logic requires low-level
        # access to the reader's directory path and raw parsing method to check file
        # presence and count labels without triggering the public validate() API.
        # If TimemoryReader is refactored, update these references accordingly.
        txt_path = reader._dir / f"{stem}.txt"
        if txt_path.exists():
            df = reader._parse_file(stem)
            if len(df) > 0:
                present_stems.append(stem)
                label_counts[stem] = len(df)
            # If df is empty, the file exists but has no data — skip it

    return {
        "format": "timemory",
        "files_present": sorted(present_stems),
        "label_counts": {stem: label_counts[stem] for stem in sorted(present_stems)},
    }
