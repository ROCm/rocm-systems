"""Tests for perfxpert.tools.roofline.plot_points (Phase 10 Live Roofline).

Regression guards:
  - schema_version + ridge_point are always present.
  - dtype detection picks up bf16 from the kernel name.
  - dtype_confidence falls back to "default" when no regex matches.
  - kernels with zero bytes are dropped from the plot.
  - kernels with only one of FETCH_SIZE / WRITE_SIZE populated are flagged
    ``confidence: "low"``.
  - Tier-1 traces (no pmc_events) return an empty kernels list without
    raising.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from perfxpert.tools import roofline


_FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "compute_bound.db"
)


def _build_db(tmp_path: Path, rows: list[tuple]) -> Path:
    """Write a minimal SQLite file with a ``pmc_events`` table + agent row.

    Each row: ``(kernel_name, counter_name, counter_value, duration_ns)``.
    """
    p = tmp_path / "fake.db"
    conn = sqlite3.connect(str(p))
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE rocpd_info_agent (id INTEGER PRIMARY KEY, name TEXT, "
        "type TEXT)"
    )
    cur.execute(
        "INSERT INTO rocpd_info_agent (name, type) VALUES ('gfx942', 'GPU')"
    )
    cur.execute(
        "CREATE TABLE pmc_events ("
        "name TEXT, counter_name TEXT, counter_value REAL, "
        "start INTEGER, end INTEGER, dispatch_id INTEGER)"
    )
    for i, (name, cname, val, dur) in enumerate(rows):
        cur.execute(
            "INSERT INTO pmc_events (name, counter_name, counter_value, "
            "start, end, dispatch_id) VALUES (?, ?, ?, ?, ?, ?)",
            (name, cname, val, 0, dur, i),
        )
    conn.commit()
    conn.close()
    return p


def test_plot_points_returns_ridge_point() -> None:
    """The schema guarantees ``ridge_point.ai`` + ``.flops_per_s``."""
    rf = roofline.plot_points(str(_FIXTURE), top_k=5)

    assert rf["schema_version"].startswith("0.3")
    assert rf["arch"] == "gfx942"

    ridge = rf["ridge_point"]
    assert "ai" in ridge
    assert "flops_per_s" in ridge
    # MI300X: peak FP32 163.4 TF/s / 5.3 TB/s ~= 30.8 FLOPs/Byte
    assert ridge["ai"] == pytest.approx(30.8, rel=0.02)
    assert ridge["flops_per_s"] == pytest.approx(1.634e14, rel=0.02)


def test_plot_points_detects_bf16_dtype_from_kernel_name(tmp_path: Path) -> None:
    db = _build_db(
        tmp_path,
        [
            ("gemm_bf16_kernel", "SQ_INSTS_VALU", 1_000_000, 1_000_000),
            ("gemm_bf16_kernel", "SQ_INSTS_VALU_MFMA", 10_000, 1_000_000),
            ("gemm_bf16_kernel", "FETCH_SIZE", 1024, 1_000_000),
            ("gemm_bf16_kernel", "WRITE_SIZE", 512, 1_000_000),
        ],
    )
    rf = roofline.plot_points(str(db), top_k=5)

    assert rf["kernels"], "kernel was unexpectedly filtered out"
    assert rf["kernels"][0]["fp_type"] == "bf16"
    assert rf["dtype"] == "bf16"
    assert rf["dtype_confidence"] == "from_kernel_name"


def test_plot_points_prefers_mops_counters_over_legacy_mfma(
    tmp_path: Path,
) -> None:
    db = _build_db(
        tmp_path,
        [
            ("gemm_bf16_kernel", "SQ_INSTS_VALU", 1_000, 1_000_000),
            ("gemm_bf16_kernel", "SQ_INSTS_VALU_MFMA", 123_456, 1_000_000),
            ("gemm_bf16_kernel", "SQ_INSTS_VALU_MFMA_MOPS_BF16", 10_000, 1_000_000),
            ("gemm_bf16_kernel", "FETCH_SIZE", 1024, 1_000_000),
            ("gemm_bf16_kernel", "WRITE_SIZE", 512, 1_000_000),
        ],
    )
    rf = roofline.plot_points(str(db), top_k=5)

    assert rf["kernels"], "kernel was unexpectedly filtered out"
    entry = rf["kernels"][0]
    assert entry["fp_type"] == "bf16"
    assert entry["flops"] == pytest.approx((1_000 * 64) + (10_000 * 512))


def test_plot_points_falls_back_to_legacy_mfma_counter(tmp_path: Path) -> None:
    db = _build_db(
        tmp_path,
        [
            ("gemm_bf16_kernel", "SQ_INSTS_VALU", 1_000, 1_000_000),
            ("gemm_bf16_kernel", "SQ_INSTS_VALU_MFMA", 10, 1_000_000),
            ("gemm_bf16_kernel", "FETCH_SIZE", 1024, 1_000_000),
            ("gemm_bf16_kernel", "WRITE_SIZE", 512, 1_000_000),
        ],
    )
    rf = roofline.plot_points(str(db), top_k=5)

    assert rf["kernels"], "kernel was unexpectedly filtered out"
    entry = rf["kernels"][0]
    assert entry["fp_type"] == "bf16"
    assert entry["flops"] == pytest.approx((1_000 * 64) + (10 * 4096))


def test_plot_points_default_fp32_when_no_regex_match(tmp_path: Path) -> None:
    db = _build_db(
        tmp_path,
        [
            ("matmul_kernel", "SQ_INSTS_VALU", 1_000_000, 1_000_000),
            ("matmul_kernel", "FETCH_SIZE", 2048, 1_000_000),
            ("matmul_kernel", "WRITE_SIZE", 1024, 1_000_000),
        ],
    )
    rf = roofline.plot_points(str(db), top_k=5)

    assert rf["kernels"], "kernel was unexpectedly filtered out"
    assert rf["kernels"][0]["fp_type"] == "fp32"
    assert rf["dtype"] == "fp32"
    assert rf["dtype_confidence"] == "default"


def test_plot_points_drops_kernel_with_zero_bytes(tmp_path: Path) -> None:
    db = _build_db(
        tmp_path,
        [
            ("pure_compute_kernel", "SQ_INSTS_VALU", 1_000_000, 1_000_000),
            # no FETCH_SIZE, no WRITE_SIZE -> AI undefined, drop the kernel.
            ("keep_me_kernel", "SQ_INSTS_VALU", 500_000, 500_000),
            ("keep_me_kernel", "FETCH_SIZE", 128, 500_000),
            ("keep_me_kernel", "WRITE_SIZE", 64, 500_000),
        ],
    )
    rf = roofline.plot_points(str(db), top_k=5)

    names = {k["name"] for k in rf["kernels"]}
    assert "pure_compute_kernel" not in names
    assert "keep_me_kernel" in names


def test_plot_points_marks_low_confidence_with_single_side_bytes(
    tmp_path: Path,
) -> None:
    db = _build_db(
        tmp_path,
        [
            ("readonly_kernel", "SQ_INSTS_VALU", 1_000_000, 1_000_000),
            ("readonly_kernel", "FETCH_SIZE", 2048, 1_000_000),
            # intentionally no WRITE_SIZE row
        ],
    )
    rf = roofline.plot_points(str(db), top_k=5)

    assert rf["kernels"]
    entry = rf["kernels"][0]
    assert entry["confidence"] == "low"


def test_plot_points_tier1_no_counters_returns_empty_kernels(
    tmp_path: Path,
) -> None:
    p = tmp_path / "tier1.db"
    conn = sqlite3.connect(str(p))
    conn.execute("CREATE TABLE rocpd_info_agent (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO rocpd_info_agent (name) VALUES ('gfx942')")
    conn.commit()
    conn.close()

    rf = roofline.plot_points(str(p), top_k=10)

    assert rf["kernels"] == []
    assert "ridge_point" in rf
