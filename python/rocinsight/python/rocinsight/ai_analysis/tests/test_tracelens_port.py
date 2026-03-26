#!/usr/bin/env python3
"""
Unit tests for rocpd/tracelens_port.py.

Run:
    ROCPD_SYS=$(python3 -c "import site; print(site.getsitepackages()[-1])")
    PYTHONPATH="${ROCPD_SYS}" pytest --noconftest test_tracelens_port.py -v

Integration test (requires real merged_db.db):
    ROCPD_TEST_DB=/path/to/merged_db.db pytest --noconftest test_tracelens_port.py -v
"""

from unittest.mock import MagicMock, patch
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_cursor(rows):
    """Return a mock cursor whose .fetchall() returns rows."""
    cur = MagicMock()
    cur.fetchall.return_value = rows
    return cur


def _mock_conn():
    return MagicMock()


# ===========================================================================
# Task 1 Tests: pure functions (no DB)
# ===========================================================================


class TestMergeIntervals:
    def test_empty(self):
        from rocpd.tracelens_port import _merge_intervals

        assert _merge_intervals([]) == []

    def test_single(self):
        from rocpd.tracelens_port import _merge_intervals

        assert _merge_intervals([(0, 100)]) == [(0, 100)]

    def test_non_overlapping(self):
        from rocpd.tracelens_port import _merge_intervals

        result = _merge_intervals([(0, 50), (100, 150)])
        assert result == [(0, 50), (100, 150)]

    def test_overlapping(self):
        from rocpd.tracelens_port import _merge_intervals

        result = _merge_intervals([(0, 100), (50, 150)])
        assert result == [(0, 150)]

    def test_adjacent(self):
        from rocpd.tracelens_port import _merge_intervals

        result = _merge_intervals([(0, 100), (100, 200)])
        assert result == [(0, 200)]

    def test_contained(self):
        from rocpd.tracelens_port import _merge_intervals

        result = _merge_intervals([(0, 200), (50, 100)])
        assert result == [(0, 200)]

    def test_unsorted_input(self):
        from rocpd.tracelens_port import _merge_intervals

        result = _merge_intervals([(100, 200), (0, 50)])
        assert result == [(0, 50), (100, 200)]


class TestCategorizeKernelName:
    def test_gemm(self):
        from rocpd.tracelens_port import categorize_kernel_name

        assert categorize_kernel_name("sgemm_nn_kernel") == "GEMM"
        assert categorize_kernel_name("Cijk_Alik_Bljk_HHS_BH_SRVB") == "GEMM"
        assert categorize_kernel_name("rocblas_gemm_kernel") == "GEMM"

    def test_conv(self):
        from rocpd.tracelens_port import categorize_kernel_name

        assert categorize_kernel_name("conv2d_fwd_kernel") == "CONV"
        assert categorize_kernel_name("implicit_gemm_conv_v4r1") == "CONV"

    def test_sdpa(self):
        from rocpd.tracelens_port import categorize_kernel_name

        assert categorize_kernel_name("flash_attention_fwd") == "SDPA"
        assert categorize_kernel_name("fmha_v2_flash_attn") == "SDPA"
        assert categorize_kernel_name("scaled_dot_product_attention") == "SDPA"

    def test_nccl(self):
        from rocpd.tracelens_port import categorize_kernel_name

        assert categorize_kernel_name("ncclKernel_AllReduce_RING_LL") == "NCCL"
        assert categorize_kernel_name("rccl_AllGather_kernel") == "NCCL"

    def test_elementwise(self):
        from rocpd.tracelens_port import categorize_kernel_name

        assert categorize_kernel_name("vectorized_elementwise_kernel") == "Elementwise"
        assert categorize_kernel_name("gelu_activation_kernel") == "Elementwise"

    def test_normalization(self):
        from rocpd.tracelens_port import categorize_kernel_name

        assert categorize_kernel_name("layer_norm_fwd") == "Normalization"
        assert categorize_kernel_name("rms_norm_kernel") == "Normalization"

    def test_reduction(self):
        from rocpd.tracelens_port import categorize_kernel_name

        assert categorize_kernel_name("reduce_kernel") == "Reduction"
        assert categorize_kernel_name("softmax_fwd") == "Reduction"

    def test_other(self):
        from rocpd.tracelens_port import categorize_kernel_name

        assert categorize_kernel_name("reproducible_dispatch_count") == "Other"
        assert categorize_kernel_name("custom_kernel_xyz") == "Other"

    def test_case_insensitive(self):
        from rocpd.tracelens_port import categorize_kernel_name

        assert categorize_kernel_name("SGEMM_KERNEL") == "GEMM"
        assert categorize_kernel_name("Flash_Attention") == "SDPA"


class TestSubtractIntervals:
    def test_empty_a(self):
        from rocpd.tracelens_port import _subtract_intervals

        assert _subtract_intervals([], [(0, 100)]) == []

    def test_empty_b(self):
        from rocpd.tracelens_port import _subtract_intervals

        result = _subtract_intervals([(0, 100)], [])
        assert result == [(0, 100)]

    def test_b_fully_covers_a(self):
        from rocpd.tracelens_port import _subtract_intervals

        result = _subtract_intervals([(0, 100)], [(0, 100)])
        assert result == []

    def test_b_partially_overlaps_left(self):
        from rocpd.tracelens_port import _subtract_intervals

        # b covers [0,50], a is [0,100] → remaining [50, 100]
        result = _subtract_intervals([(0, 100)], [(0, 50)])
        assert result == [(50, 100)]

    def test_b_partially_overlaps_right(self):
        from rocpd.tracelens_port import _subtract_intervals

        # b covers [50,100], a is [0,100] → remaining [0, 50]
        result = _subtract_intervals([(0, 100)], [(50, 100)])
        assert result == [(0, 50)]

    def test_b_cuts_middle(self):
        from rocpd.tracelens_port import _subtract_intervals

        # b covers [40,60], a is [0,100] → remaining [0,40] and [60,100]
        result = _subtract_intervals([(0, 100)], [(40, 60)])
        assert result == [(0, 40), (60, 100)]

    def test_multiple_a_intervals(self):
        from rocpd.tracelens_port import _subtract_intervals

        # a=[0,50],[100,150], b=[25,125] → [0,25] and [125,150]
        result = _subtract_intervals([(0, 50), (100, 150)], [(25, 125)])
        assert result == [(0, 25), (125, 150)]

    def test_adjacent_boundary(self):
        from rocpd.tracelens_port import _subtract_intervals

        # b ends exactly at a_start → no overlap, a is preserved
        result = _subtract_intervals([(100, 200)], [(0, 100)])
        assert result == [(100, 200)]

    def test_no_overlap(self):
        from rocpd.tracelens_port import _subtract_intervals

        # b is entirely before a
        result = _subtract_intervals([(200, 300)], [(0, 100)])
        assert result == [(200, 300)]


# ===========================================================================
# Task 2 Tests: DB-dependent functions (mocked execute_statement)
# ===========================================================================


class TestComputeIntervalTimeline:
    def test_overlapping_kernels(self):
        """Two overlapping kernel intervals: true_compute < sum of durations."""
        from rocpd.tracelens_port import compute_interval_timeline

        conn = _mock_conn()
        # Kernels: [0, 100] and [50, 150] → merged [0, 150] = 150ns
        # Memcpy: empty
        kernel_rows = [(0, 100), (50, 150)]
        memcpy_rows = []
        with patch("rocpd.tracelens_port.execute_statement") as mock_es:
            mock_es.side_effect = [
                _mock_cursor(kernel_rows),
                _mock_cursor(memcpy_rows),
            ]
            result = compute_interval_timeline(conn)
        assert result["true_compute_ns"] == 150
        assert result["total_wall_ns"] == 150
        assert result["true_compute_pct"] == 100.0
        assert result["exposed_memcpy_ns"] == 0
        assert result["idle_ns"] == 0

    def test_non_overlapping_intervals(self):
        """Non-overlapping: true_compute == sum of durations."""
        from rocpd.tracelens_port import compute_interval_timeline

        conn = _mock_conn()
        kernel_rows = [(0, 50), (100, 150)]  # 50 + 50 = 100ns compute
        memcpy_rows = []
        with patch("rocpd.tracelens_port.execute_statement") as mock_es:
            mock_es.side_effect = [
                _mock_cursor(kernel_rows),
                _mock_cursor(memcpy_rows),
            ]
            result = compute_interval_timeline(conn)
        assert result["true_compute_ns"] == 100
        assert result["total_wall_ns"] == 150
        assert result["idle_ns"] == 50

    def test_empty_kernels(self):
        """Empty kernels table → compute=0, idle=0, no crash."""
        from rocpd.tracelens_port import compute_interval_timeline

        conn = _mock_conn()
        with patch("rocpd.tracelens_port.execute_statement") as mock_es:
            mock_es.side_effect = [
                _mock_cursor([]),  # kernels
                _mock_cursor([]),  # memory_copies
            ]
            result = compute_interval_timeline(conn)
        assert result["true_compute_ns"] == 0
        assert result["true_compute_pct"] == 0.0
        assert result["idle_pct"] == 0.0

    def test_empty_memcpy(self):
        """No memory copies → exposed_memcpy_ns=0, no crash."""
        from rocpd.tracelens_port import compute_interval_timeline

        conn = _mock_conn()
        with patch("rocpd.tracelens_port.execute_statement") as mock_es:
            mock_es.side_effect = [
                _mock_cursor([(0, 100)]),
                _mock_cursor([]),
            ]
            result = compute_interval_timeline(conn)
        assert result["exposed_memcpy_ns"] == 0
        assert result["exposed_memcpy_pct"] == 0.0

    def test_zero_wall_time(self):
        """Single-point trace → all pct fields are 0.0 (no division by zero)."""
        from rocpd.tracelens_port import compute_interval_timeline

        conn = _mock_conn()
        with patch("rocpd.tracelens_port.execute_statement") as mock_es:
            mock_es.side_effect = [
                _mock_cursor([(100, 100)]),  # zero-length interval
                _mock_cursor([]),
            ]
            result = compute_interval_timeline(conn)
        assert result["true_compute_pct"] == 0.0
        assert result["idle_pct"] == 0.0


class TestAnalyzeKernelsByCategory:
    def test_basic(self):
        """Known kernel names → correct category aggregation."""
        from rocpd.tracelens_port import analyze_kernels_by_category

        conn = _mock_conn()
        rows = [
            ("sgemm_kernel", 1000),
            ("sgemm_kernel", 2000),
            ("gelu_kernel", 500),
        ]
        with patch("rocpd.tracelens_port.execute_statement") as mock_es:
            mock_es.return_value = _mock_cursor(rows)
            result = analyze_kernels_by_category(conn, total_wall_ns=10000)
        cats = {r["category"]: r for r in result}
        assert "GEMM" in cats
        assert cats["GEMM"]["count"] == 2
        assert cats["GEMM"]["total_ns"] == 3000
        assert cats["Elementwise"]["count"] == 1
        # Sorted by total_ns desc: GEMM first
        assert result[0]["category"] == "GEMM"

    def test_empty_table(self):
        """Empty kernels table → []."""
        from rocpd.tracelens_port import analyze_kernels_by_category

        conn = _mock_conn()
        with patch("rocpd.tracelens_port.execute_statement") as mock_es:
            mock_es.return_value = _mock_cursor([])
            result = analyze_kernels_by_category(conn, total_wall_ns=0)
        assert result == []

    def test_all_other(self):
        """Unrecognized kernel names → single Other entry."""
        from rocpd.tracelens_port import analyze_kernels_by_category

        conn = _mock_conn()
        rows = [("reproducible_dispatch_count", 100)] * 5
        with patch("rocpd.tracelens_port.execute_statement") as mock_es:
            mock_es.return_value = _mock_cursor(rows)
            result = analyze_kernels_by_category(conn, total_wall_ns=1000)
        assert len(result) == 1
        assert result[0]["category"] == "Other"
        assert result[0]["count"] == 5

    def test_zero_wall_pct_guard(self):
        """total_wall_ns=0 → pct_of_total_time=0.0, no crash."""
        from rocpd.tracelens_port import analyze_kernels_by_category

        conn = _mock_conn()
        rows = [("sgemm", 100)]
        with patch("rocpd.tracelens_port.execute_statement") as mock_es:
            mock_es.return_value = _mock_cursor(rows)
            result = analyze_kernels_by_category(conn, total_wall_ns=0)
        assert result[0]["pct_of_total_time"] == 0.0


class TestAnalyzeShortKernels:
    def test_basic(self):
        """Mix of short and long kernels → correct counts and histogram."""
        from rocpd.tracelens_port import analyze_short_kernels

        conn = _mock_conn()
        rows = [
            ("fast_k", 500),  # 0.5μs < 10μs
            ("fast_k", 2000),  # 2μs < 10μs
            ("slow_k", 50000),  # 50μs > 10μs
            ("slow_k", 80000),  # 80μs > 10μs
        ]
        with patch("rocpd.tracelens_port.execute_statement") as mock_es:
            mock_es.return_value = _mock_cursor(rows)
            result = analyze_short_kernels(conn, threshold_us=10.0)
        assert result["total_kernels"] == 4
        assert result["short_kernel_count"] == 2
        assert result["wasted_ns"] == 2500
        assert len(result["top_offenders"]) == 1
        assert result["top_offenders"][0]["name"] == "fast_k"
        assert result["top_offenders"][0]["count"] == 2

    def test_none_below_threshold(self):
        """All kernels above threshold → short_kernel_count=0, histogram=[]."""
        from rocpd.tracelens_port import analyze_short_kernels

        conn = _mock_conn()
        rows = [("slow_k", 50000), ("slow_k", 80000)]
        with patch("rocpd.tracelens_port.execute_statement") as mock_es:
            mock_es.return_value = _mock_cursor(rows)
            result = analyze_short_kernels(conn)
        assert result["short_kernel_count"] == 0
        assert result["histogram"] == []
        assert result["top_offenders"] == []

    def test_empty_table(self):
        """Empty kernels table → all-zero result, no crash."""
        from rocpd.tracelens_port import analyze_short_kernels

        conn = _mock_conn()
        with patch("rocpd.tracelens_port.execute_statement") as mock_es:
            mock_es.return_value = _mock_cursor([])
            result = analyze_short_kernels(conn)
        assert result["short_kernel_count"] == 0
        assert result["wasted_pct_of_kernel_time"] == 0.0


# ===========================================================================
# Integration test (requires real merged_db.db)
# ===========================================================================

MERGED_DB = __import__("os").environ.get(
    "ROCPD_TEST_DB",
    "",
)


@pytest.mark.skipif(
    not (MERGED_DB and __import__("os").path.exists(MERGED_DB)),
    reason="merged_db.db not found; set ROCPD_TEST_DB env var to enable",
)
def test_integration_tracelens_with_real_db():
    """End-to-end: all three functions return valid dicts from merged_db.db."""
    from rocpd.importer import RocpdImportData
    from rocpd.tracelens_port import (
        compute_interval_timeline,
        analyze_kernels_by_category,
        analyze_short_kernels,
    )

    conn = RocpdImportData([MERGED_DB])

    timeline = compute_interval_timeline(conn)
    assert timeline["total_wall_ns"] > 0
    assert 0.0 <= timeline["true_compute_pct"] <= 100.0
    assert 0.0 <= timeline["idle_pct"] <= 100.0

    categories = analyze_kernels_by_category(conn, timeline["total_wall_ns"])
    assert isinstance(categories, list)
    assert len(categories) > 0
    assert all("category" in c and "count" in c and "total_ns" in c for c in categories)

    short = analyze_short_kernels(conn)
    assert "short_kernel_count" in short
    assert "top_offenders" in short
    assert isinstance(short["histogram"], list)
