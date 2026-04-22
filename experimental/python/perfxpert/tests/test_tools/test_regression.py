"""Tests for perfxpert.tools.regression — with Gate 4 'hot kernel' definition."""

import sqlite3
from pathlib import Path

import pytest

from perfxpert.tools import regression
from perfxpert.tools._class import ToolClass


FIX = Path(__file__).parent.parent / "fixtures"
BASELINE = FIX / "regression_baseline.db"
IMPROVED = FIX / "regression_improved.db"
TAIL_HURT = FIX / "regression_tail_hurt.db"


def _write_dispatch_db(path: Path, rows):
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE rocpd_kernel_dispatch (name TEXT NOT NULL, duration_ns INTEGER NOT NULL)"
        )
        conn.executemany(
            "INSERT INTO rocpd_kernel_dispatch (name, duration_ns) VALUES (?, ?)",
            rows,
        )


def test_compare_detects_improvement():
    r = regression.compare_runs(str(BASELINE), str(IMPROVED), threshold_pct=3.0)
    assert r["verdict"] == "improved"
    # matmul was 70% of runtime and got 20% faster → total ~14% speedup.
    # total_delta_pct is a fraction (negative = improvement); -0.14 ≈ -14%.
    assert r["total_delta_pct"] < -0.10


def test_compare_detects_tail_hurt():
    """Small kernels regressing 15% individually trigger weighted-geomean fail.

    Spec §5 Gate 4: hot = top-K covering 80% cumulative OR ≥ 3% individually.
    conv2d is 20% of runtime → hot. It regressed 15% → > 10% threshold → fail.
    """
    r = regression.compare_runs(str(BASELINE), str(TAIL_HURT), threshold_pct=3.0)
    assert r["verdict"] == "regressed"
    assert any(k["kernel"] == "conv2d" and k["delta_pct"] > 0.10 for k in r["per_kernel_deltas"])


def test_hot_kernel_definition():
    """Explicit: hot = top-K where K covers 80% cumulative OR kernel ≥ 3% individually."""
    hot = regression.identify_hot_kernels(str(BASELINE))
    names = {k["name"] for k in hot}
    # matmul (70%) and conv2d (20%) together cover 90% > 80% → both hot
    # add (10%) is individually ≥ 3% → also hot
    assert "matmul" in names
    assert "conv2d" in names
    assert "add" in names  # because 10% ≥ 3%


def test_noise_threshold():
    """Baseline vs itself → verdict = neutral."""
    r = regression.compare_runs(str(BASELINE), str(BASELINE), threshold_pct=3.0)
    assert r["verdict"] == "neutral"


def test_deleted_kernels_do_not_underweight_remaining_regressions(tmp_path: Path):
    before = tmp_path / "before.db"
    after = tmp_path / "after.db"
    _write_dispatch_db(before, [("A", 50), ("B", 50)])
    _write_dispatch_db(after, [("A", 55)])

    result = regression.compare_runs(str(before), str(after), threshold_pct=3.0)
    assert result["verdict"] == "regressed"
    assert result["weighted_geomean_delta_pct"] > 0.09


def test_kernel_durations_closes_connection_on_exception(monkeypatch):
    class BrokenConnection:
        def __init__(self):
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.closed = True
            return False

        def execute(self, *_args, **_kwargs):
            raise sqlite3.DatabaseError("boom")

    broken = BrokenConnection()
    monkeypatch.setattr(regression.sqlite3, "connect", lambda _path: broken)

    with pytest.raises(sqlite3.DatabaseError):
        regression._kernel_durations("broken.db")
    assert broken.closed is True


def test_is_read_only_class():
    assert regression.compare_runs.__tool_class__ == ToolClass.READ_ONLY
    assert regression.identify_hot_kernels.__tool_class__ == ToolClass.READ_ONLY
