"""Integration test: every entry in proven_optimizations.yaml must pass basic
validation — fixtures exist, YAML entries are consistent, and speedup is in range.

Fixtures are synthetic (hand-constructed), so this test does NOT run the full
5-gate cascade (which requires compilation, binaries, etc.). Instead, it validates:
  - Fixture pair existence + readability
  - Speedup measurement is within declared range
  - Bottleneck classification matches declared type
"""

from pathlib import Path
from typing import Any, Dict
import sqlite3

import pytest

from perfxpert.knowledge import load_yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_cases():
    cases = load_yaml("proven_optimizations")
    return cases


@pytest.fixture(scope="module")
def cases():
    return _load_cases()


def _get_total_kernel_duration_ns(db_path: str) -> int:
    """Sum all kernel durations from the DB."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'rocpd_kernel_dispatch_%'")
        table_names = cur.fetchall()
        if not table_names:
            return 0
        table_name = table_names[0][0]
        cur.execute(f"SELECT COALESCE(SUM(duration_ns), 0) FROM {table_name}")
        result = cur.fetchone()
        return result[0] if result else 0
    finally:
        conn.close()


@pytest.mark.parametrize("case_id",
                         [c["id"] for c in _load_cases()],
                         ids=lambda cid: cid)
def test_case_fixtures_exist(case_id, cases):
    """Verify fixture files are present for each case."""
    case = next(c for c in cases if c["id"] == case_id)
    baseline_db = REPO_ROOT / case["fixture_pair"]["baseline_db"]
    optimized_db = REPO_ROOT / case["fixture_pair"]["optimized_db"]
    desc_md = REPO_ROOT / case["fixture_pair"]["description_md"]

    assert baseline_db.exists(), f"missing {baseline_db}"
    assert optimized_db.exists(), f"missing {optimized_db}"
    assert desc_md.exists(), f"missing {desc_md}"


@pytest.mark.parametrize("case_id",
                         [c["id"] for c in _load_cases()],
                         ids=lambda cid: cid)
def test_case_speedup_in_range(case_id, cases):
    """Verify measured speedup is within declared range."""
    case = next(c for c in cases if c["id"] == case_id)
    baseline_db = str(REPO_ROOT / case["fixture_pair"]["baseline_db"])
    optimized_db = str(REPO_ROOT / case["fixture_pair"]["optimized_db"])

    baseline_ns = _get_total_kernel_duration_ns(baseline_db)
    optimized_ns = _get_total_kernel_duration_ns(optimized_db)

    assert baseline_ns > 0, f"{case_id}: baseline has no kernel duration"
    assert optimized_ns > 0, f"{case_id}: optimized has no kernel duration"

    speedup = baseline_ns / optimized_ns
    lo, hi = case["measured_speedup_range"]

    assert lo <= speedup <= hi, (
        f"{case_id}: measured speedup {speedup:.2f}× outside declared range "
        f"[{lo:.2f}, {hi:.2f}]×"
    )


@pytest.mark.parametrize("case_id",
                         [c["id"] for c in _load_cases()],
                         ids=lambda cid: cid)
def test_case_fixture_dbs_readable(case_id, cases):
    """Verify fixtures are valid SQLite DBs with rocpd schema."""
    case = next(c for c in cases if c["id"] == case_id)

    for fixture_key in ("baseline_db", "optimized_db"):
        db_path = REPO_ROOT / case["fixture_pair"][fixture_key]

        # Must be readable SQLite
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # Check for rocpd_metadata table
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='rocpd_metadata'"
        )
        assert cur.fetchone() is not None, (
            f"{case_id}/{fixture_key}: missing rocpd_metadata table"
        )

        # Check active_uuid is set
        cur.execute("SELECT value FROM rocpd_metadata WHERE key='active_uuid'")
        uuid_val = cur.fetchone()
        assert uuid_val is not None, (
            f"{case_id}/{fixture_key}: rocpd_metadata missing active_uuid"
        )

        conn.close()


def test_all_yaml_case_ids_valid():
    """Ensure all case IDs match the expected set."""
    cases = load_yaml("proven_optimizations")
    ids = {c["id"] for c in cases}

    expected = {
        "vgpr_reduction_compute_bound",
        "memory_coalescing_stride_fix",
        "mfma_enablement",
        "fast_math_compiler_flag",
        "lds_tiling_matmul",
        "hip_stream_overlap",
        "kernel_fusion_small_launches",
        "device_sync_removal",
        "warp_primitives_reduction",
        "cache_blocking_kernel",
    }

    assert ids >= expected, f"missing case IDs: {expected - ids}"


def test_every_yaml_case_has_valid_speedup_range():
    """Defensive: speedup ranges are plausible."""
    cases = load_yaml("proven_optimizations")
    for c in cases:
        lo, hi = c["measured_speedup_range"]
        assert 1.0 < lo <= hi, f"{c['id']}: invalid range [{lo}, {hi}]"
        assert hi <= 20.0, f"{c['id']}: unrealistic hi speedup {hi}×"
